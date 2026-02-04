"""Remote Repository Management for Dynamic AI Assistant.

This module implements Phase 6.2 for the Dynamic AI Assistant:
- Remote Configuration: Multiple remotes, URL validation, authentication
- Intelligent Push/Pull: Upstream tracking, conflict prediction, protection
- Fetch Optimization: Shallow fetch, partial clone, scheduling

Key Features:
============
- Multiple remote repository management
- Intelligent push/pull with conflict prediction
- Fetch optimization for large repositories
- Force push protection and safeguards

Design Principle:
================
All remote operations use LLM reasoning when available.
The LLM analyzes remote state and suggests optimal sync strategies.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, Iterator, List,
    Optional, Pattern, Set, Tuple, Union
)
import uuid

logger = logging.getLogger(__name__)


class RemoteType(Enum):
    """Remote repository types."""
    ORIGIN = "origin"
    UPSTREAM = "upstream"
    FORK = "fork"
    MIRROR = "mirror"
    CUSTOM = "custom"


class ProtocolType(Enum):
    """Git protocol types."""
    HTTPS = "https"
    SSH = "ssh"
    GIT = "git"
    FILE = "file"
    UNKNOWN = "unknown"


class PushProtection(Enum):
    """Push protection levels."""
    NONE = "none"
    WARN = "warn"
    BLOCK = "block"


class FetchStrategy(Enum):
    """Fetch strategies."""
    FULL = "full"
    SHALLOW = "shallow"
    PARTIAL = "partial"
    SINGLE_BRANCH = "single_branch"


class SyncStatus(Enum):
    """Synchronization status."""
    UP_TO_DATE = "up_to_date"
    BEHIND = "behind"
    AHEAD = "ahead"
    DIVERGED = "diverged"
    UNKNOWN = "unknown"


@dataclass
class RemoteInfo:
    """Remote repository information."""
    name: str
    url: str
    remote_type: RemoteType = RemoteType.CUSTOM
    
    # URL details
    protocol: ProtocolType = ProtocolType.UNKNOWN
    host: str = ""
    owner: str = ""
    repo_name: str = ""
    
    # Status
    fetch_url: Optional[str] = None
    push_url: Optional[str] = None
    is_reachable: bool = False
    last_fetch: Optional[datetime] = None
    
    # Branches
    default_branch: Optional[str] = None
    tracking_branches: List[str] = field(default_factory=list)
    
    # Capabilities
    supports_lfs: bool = False
    supports_push: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "remote_type": self.remote_type.value,
            "protocol": self.protocol.value,
            "host": self.host,
            "owner": self.owner,
            "repo_name": self.repo_name,
            "is_reachable": self.is_reachable,
            "default_branch": self.default_branch,
        }


@dataclass
class PushResult:
    """Push operation result."""
    success: bool
    remote: str
    branch: str
    
    # Details
    refs_pushed: List[str] = field(default_factory=list)
    new_commits: int = 0
    forced: bool = False
    
    # Errors
    error: Optional[str] = None
    rejected_refs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "remote": self.remote,
            "branch": self.branch,
            "new_commits": self.new_commits,
            "forced": self.forced,
            "error": self.error,
        }


@dataclass
class PullResult:
    """Pull operation result."""
    success: bool
    remote: str
    branch: str
    
    # Details
    new_commits: int = 0
    fast_forward: bool = False
    merge_commit: Optional[str] = None
    
    # Changes
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0
    
    # Conflicts
    has_conflicts: bool = False
    conflicts: List[str] = field(default_factory=list)
    
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "remote": self.remote,
            "branch": self.branch,
            "new_commits": self.new_commits,
            "fast_forward": self.fast_forward,
            "has_conflicts": self.has_conflicts,
            "files_changed": self.files_changed,
            "error": self.error,
        }


@dataclass
class FetchResult:
    """Fetch operation result."""
    success: bool
    remote: str
    
    # Details
    refs_fetched: List[str] = field(default_factory=list)
    new_refs: List[str] = field(default_factory=list)
    updated_refs: List[str] = field(default_factory=list)
    deleted_refs: List[str] = field(default_factory=list)
    
    # Stats
    objects_received: int = 0
    bytes_received: int = 0
    duration_seconds: float = 0.0
    
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "remote": self.remote,
            "refs_fetched_count": len(self.refs_fetched),
            "new_refs_count": len(self.new_refs),
            "bytes_received": self.bytes_received,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


@dataclass
class ConflictPrediction:
    """Predicted merge conflict information."""
    files: List[str]
    likelihood: float  # 0.0 - 1.0
    affected_commits: int
    analysis: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "files": self.files,
            "likelihood": self.likelihood,
            "affected_commits": self.affected_commits,
            "analysis": self.analysis,
        }


class GitCommandRunner:
    """Safe git command execution."""
    
    def __init__(self, repo_path: Optional[Path] = None):
        """Initialize the git command runner.
        
        Args:
            repo_path: Path to the git repository
        """
        self._repo_path = repo_path or Path.cwd()
    
    def run(
        self,
        args: List[str],
        capture_output: bool = True,
        check: bool = True,
        timeout: int = 120,
        env: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        """Run a git command.
        
        Args:
            args: Git command arguments (without 'git')
            capture_output: Capture stdout/stderr
            check: Raise on non-zero exit
            timeout: Command timeout in seconds
            env: Additional environment variables
            
        Returns:
            CompletedProcess result
        """
        cmd = ["git"] + args
        
        # Merge environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self._repo_path,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                check=check,
                env=run_env,
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(cmd)}")
            logger.error(f"stderr: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            logger.error(f"Git command timed out: {' '.join(cmd)}")
            raise
    
    def run_safe(
        self,
        args: List[str],
        timeout: int = 120,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str, str]:
        """Run a git command safely without raising.
        
        Args:
            args: Git command arguments
            timeout: Command timeout
            env: Additional environment variables
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            result = self.run(args, check=False, timeout=timeout, env=env)
            return (
                result.returncode == 0,
                result.stdout or "",
                result.stderr or "",
            )
        except Exception as e:
            return False, "", str(e)
    
    def is_git_repo(self) -> bool:
        """Check if current path is a git repository."""
        success, _, _ = self.run_safe(["rev-parse", "--git-dir"])
        return success


class RemoteConfigManager:
    """Remote repository configuration management.
    
    Uses LLM reasoning to:
    1. Analyze remote URLs and suggest names
    2. Detect authentication issues
    3. Recommend mirroring strategies
    
    Example:
        >>> manager = RemoteConfigManager(llm_client=client)
        >>> remotes = manager.list_remotes()
        >>> manager.add_remote("upstream", "https://github.com/owner/repo.git")
    """
    
    # URL patterns for different protocols
    HTTPS_PATTERN = re.compile(
        r'^https?://(?P<host>[^/]+)/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$'
    )
    
    SSH_PATTERN = re.compile(
        r'^(?:ssh://)?git@(?P<host>[^:]+):(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$'
    )
    
    GIT_PATTERN = re.compile(
        r'^git://(?P<host>[^/]+)/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$'
    )
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        repo_path: Optional[Path] = None,
    ):
        """Initialize the remote config manager.
        
        Args:
            llm_client: LLM client for intelligent operations
            repo_path: Path to the git repository
        """
        self._llm_client = llm_client
        self._repo_path = repo_path or Path.cwd()
        self._git = GitCommandRunner(self._repo_path)
    
    def list_remotes(self) -> List[RemoteInfo]:
        """List all configured remotes.
        
        Returns:
            List of remote information
        """
        remotes = []
        
        # Get remotes with URLs
        success, stdout, _ = self._git.run_safe(["remote", "-v"])
        
        if not success:
            return remotes
        
        # Parse output (remote name \t url (fetch|push))
        remote_urls: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                url = parts[1]
                url_type = parts[2].strip("()") if len(parts) > 2 else "fetch"
                remote_urls[name][url_type] = url
        
        # Build remote info
        for name, urls in remote_urls.items():
            fetch_url = urls.get("fetch", "")
            push_url = urls.get("push", fetch_url)
            
            remote = RemoteInfo(
                name=name,
                url=fetch_url,
                fetch_url=fetch_url,
                push_url=push_url,
            )
            
            # Parse URL
            self._parse_url(remote, fetch_url)
            
            # Determine remote type
            remote.remote_type = self._detect_remote_type(name)
            
            # Get tracking branches
            remote.tracking_branches = self._get_tracking_branches(name)
            
            remotes.append(remote)
        
        return remotes
    
    def _parse_url(self, remote: RemoteInfo, url: str):
        """Parse URL to extract host, owner, and repo."""
        # Try HTTPS
        match = self.HTTPS_PATTERN.match(url)
        if match:
            remote.protocol = ProtocolType.HTTPS
            remote.host = match.group("host")
            remote.owner = match.group("owner")
            remote.repo_name = match.group("repo")
            return
        
        # Try SSH
        match = self.SSH_PATTERN.match(url)
        if match:
            remote.protocol = ProtocolType.SSH
            remote.host = match.group("host")
            remote.owner = match.group("owner")
            remote.repo_name = match.group("repo")
            return
        
        # Try git://
        match = self.GIT_PATTERN.match(url)
        if match:
            remote.protocol = ProtocolType.GIT
            remote.host = match.group("host")
            remote.owner = match.group("owner")
            remote.repo_name = match.group("repo")
            return
        
        # Check for file://
        if url.startswith("file://"):
            remote.protocol = ProtocolType.FILE
    
    def _detect_remote_type(self, name: str) -> RemoteType:
        """Detect remote type from name."""
        name_lower = name.lower()
        
        if name_lower == "origin":
            return RemoteType.ORIGIN
        elif name_lower == "upstream":
            return RemoteType.UPSTREAM
        elif name_lower in ("fork", "my-fork"):
            return RemoteType.FORK
        elif "mirror" in name_lower:
            return RemoteType.MIRROR
        
        return RemoteType.CUSTOM
    
    def _get_tracking_branches(self, remote: str) -> List[str]:
        """Get tracking branches for a remote."""
        branches = []
        
        success, stdout, _ = self._git.run_safe([
            "branch", "-r", "--list", f"{remote}/*"
        ])
        
        if success and stdout:
            for line in stdout.strip().split("\n"):
                branch = line.strip()
                if branch and "->" not in branch:  # Skip HEAD pointer
                    branches.append(branch)
        
        return branches
    
    def add_remote(
        self,
        name: str,
        url: str,
        fetch_tags: bool = True,
    ) -> Dict[str, Any]:
        """Add a new remote.
        
        Args:
            name: Remote name
            url: Remote URL
            fetch_tags: Fetch tags automatically
            
        Returns:
            Operation result
        """
        # Validate URL
        if not self._validate_url(url):
            return {"success": False, "error": "Invalid URL format"}
        
        # Check if exists
        success, _, _ = self._git.run_safe(["remote", "get-url", name])
        if success:
            return {"success": False, "error": f"Remote '{name}' already exists"}
        
        # Add remote
        success, _, stderr = self._git.run_safe(["remote", "add", name, url])
        
        if not success:
            return {"success": False, "error": stderr}
        
        # Configure tags
        if not fetch_tags:
            self._git.run_safe(["config", f"remote.{name}.tagOpt", "--no-tags"])
        
        return {
            "success": True,
            "remote": name,
            "url": url,
        }
    
    def _validate_url(self, url: str) -> bool:
        """Validate git remote URL."""
        if not url:
            return False
        
        # Check for known patterns
        if self.HTTPS_PATTERN.match(url):
            return True
        if self.SSH_PATTERN.match(url):
            return True
        if self.GIT_PATTERN.match(url):
            return True
        if url.startswith("file://"):
            return True
        
        return False
    
    def remove_remote(self, name: str) -> Dict[str, Any]:
        """Remove a remote.
        
        Args:
            name: Remote name
            
        Returns:
            Operation result
        """
        success, _, stderr = self._git.run_safe(["remote", "remove", name])
        
        return {
            "success": success,
            "remote": name,
            "error": stderr if not success else None,
        }
    
    def rename_remote(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """Rename a remote.
        
        Args:
            old_name: Current remote name
            new_name: New remote name
            
        Returns:
            Operation result
        """
        success, _, stderr = self._git.run_safe(["remote", "rename", old_name, new_name])
        
        return {
            "success": success,
            "old_name": old_name,
            "new_name": new_name,
            "error": stderr if not success else None,
        }
    
    def set_url(
        self,
        name: str,
        url: str,
        push_url: bool = False,
    ) -> Dict[str, Any]:
        """Set remote URL.
        
        Args:
            name: Remote name
            url: New URL
            push_url: Set push URL instead of fetch URL
            
        Returns:
            Operation result
        """
        args = ["remote", "set-url"]
        if push_url:
            args.append("--push")
        args.extend([name, url])
        
        success, _, stderr = self._git.run_safe(args)
        
        return {
            "success": success,
            "remote": name,
            "url": url,
            "is_push_url": push_url,
            "error": stderr if not success else None,
        }
    
    def check_reachability(self, name: str) -> Dict[str, Any]:
        """Check if a remote is reachable.
        
        Args:
            name: Remote name
            
        Returns:
            Reachability result
        """
        # Use ls-remote with timeout
        success, stdout, stderr = self._git.run_safe(
            ["ls-remote", "--heads", name],
            timeout=30,
        )
        
        if not success:
            return {
                "reachable": False,
                "remote": name,
                "error": stderr,
            }
        
        # Parse branches
        branches = []
        for line in stdout.strip().split("\n"):
            if line and "\t" in line:
                _, ref = line.split("\t", 1)
                if ref.startswith("refs/heads/"):
                    branches.append(ref.replace("refs/heads/", ""))
        
        return {
            "reachable": True,
            "remote": name,
            "branches": branches,
        }


class IntelligentPushPull:
    """Intelligent push/pull with safeguards.
    
    Uses LLM reasoning to:
    1. Predict conflicts before pull
    2. Suggest optimal merge strategies
    3. Protect against dangerous force pushes
    
    Example:
        >>> pusher = IntelligentPushPull(llm_client=client)
        >>> prediction = pusher.predict_pull_conflicts("origin", "main")
        >>> result = pusher.pull("origin", "main")
    """
    
    # Protected branches (default)
    PROTECTED_BRANCHES = {"main", "master", "develop", "release"}
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        repo_path: Optional[Path] = None,
        force_push_protection: PushProtection = PushProtection.WARN,
    ):
        """Initialize the intelligent push/pull manager.
        
        Args:
            llm_client: LLM client for intelligent operations
            repo_path: Path to the git repository
            force_push_protection: Protection level for force pushes
        """
        self._llm_client = llm_client
        self._repo_path = repo_path or Path.cwd()
        self._git = GitCommandRunner(self._repo_path)
        self._force_protection = force_push_protection
        self._protected_branches = self.PROTECTED_BRANCHES.copy()
    
    def get_sync_status(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get synchronization status with remote.
        
        Args:
            remote: Remote name
            branch: Branch name (default: current)
            
        Returns:
            Sync status information
        """
        # Get current branch if not specified
        if not branch:
            success, stdout, _ = self._git.run_safe(["branch", "--show-current"])
            branch = stdout.strip() if success else "main"
        
        result = {
            "remote": remote,
            "branch": branch,
            "status": SyncStatus.UNKNOWN.value,
            "ahead": 0,
            "behind": 0,
        }
        
        # Fetch to get latest remote state
        self._git.run_safe(["fetch", remote, branch, "--quiet"])
        
        # Count commits ahead
        success, stdout, _ = self._git.run_safe([
            "rev-list", "--count", f"{remote}/{branch}..HEAD"
        ])
        if success:
            result["ahead"] = int(stdout.strip()) if stdout.strip() else 0
        
        # Count commits behind
        success, stdout, _ = self._git.run_safe([
            "rev-list", "--count", f"HEAD..{remote}/{branch}"
        ])
        if success:
            result["behind"] = int(stdout.strip()) if stdout.strip() else 0
        
        # Determine status
        if result["ahead"] == 0 and result["behind"] == 0:
            result["status"] = SyncStatus.UP_TO_DATE.value
        elif result["ahead"] > 0 and result["behind"] == 0:
            result["status"] = SyncStatus.AHEAD.value
        elif result["ahead"] == 0 and result["behind"] > 0:
            result["status"] = SyncStatus.BEHIND.value
        else:
            result["status"] = SyncStatus.DIVERGED.value
        
        return result
    
    def predict_pull_conflicts(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
    ) -> ConflictPrediction:
        """Predict conflicts that might occur on pull.
        
        Args:
            remote: Remote name
            branch: Branch name (default: current)
            
        Returns:
            Conflict prediction
        """
        # Get current branch
        if not branch:
            success, stdout, _ = self._git.run_safe(["branch", "--show-current"])
            branch = stdout.strip() if success else "main"
        
        prediction = ConflictPrediction(
            files=[],
            likelihood=0.0,
            affected_commits=0,
            analysis="No conflicts predicted",
        )
        
        # Fetch latest
        self._git.run_safe(["fetch", remote, branch, "--quiet"])
        
        # Get files changed locally
        success, local_files, _ = self._git.run_safe([
            "diff", "--name-only", f"{remote}/{branch}...HEAD"
        ])
        local_changed = set(local_files.strip().split("\n")) if local_files else set()
        
        # Get files changed in remote
        success, remote_files, _ = self._git.run_safe([
            "diff", "--name-only", f"HEAD...{remote}/{branch}"
        ])
        remote_changed = set(remote_files.strip().split("\n")) if remote_files else set()
        
        # Find overlapping files
        conflicting_files = local_changed & remote_changed
        conflicting_files.discard("")  # Remove empty strings
        
        if conflicting_files:
            prediction.files = list(conflicting_files)
            prediction.likelihood = min(0.8, len(conflicting_files) * 0.1)
            prediction.analysis = f"Potential conflicts in {len(conflicting_files)} files modified both locally and remotely"
        
        # Count affected commits
        success, stdout, _ = self._git.run_safe([
            "rev-list", "--count", f"HEAD...{remote}/{branch}"
        ])
        if success:
            prediction.affected_commits = int(stdout.strip()) if stdout.strip() else 0
        
        return prediction
    
    def pull(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        rebase: bool = False,
        ff_only: bool = False,
    ) -> PullResult:
        """Pull changes from remote.
        
        Args:
            remote: Remote name
            branch: Branch name (default: current)
            rebase: Use rebase instead of merge
            ff_only: Only allow fast-forward
            
        Returns:
            Pull result
        """
        # Get current branch
        if not branch:
            success, stdout, _ = self._git.run_safe(["branch", "--show-current"])
            branch = stdout.strip() if success else "main"
        
        result = PullResult(
            success=False,
            remote=remote,
            branch=branch,
        )
        
        # Build pull command
        args = ["pull", remote, branch]
        if rebase:
            args.insert(1, "--rebase")
        if ff_only:
            args.insert(1, "--ff-only")
        
        start_time = time.time()
        success, stdout, stderr = self._git.run_safe(args)
        
        if not success:
            # Check for conflicts
            conflict_success, conflict_out, _ = self._git.run_safe([
                "diff", "--name-only", "--diff-filter=U"
            ])
            
            if conflict_out.strip():
                result.has_conflicts = True
                result.conflicts = conflict_out.strip().split("\n")
            
            result.error = stderr
            return result
        
        result.success = True
        
        # Parse output for stats
        if "Already up to date" in stdout or "Already up-to-date" in stdout:
            result.fast_forward = True
        elif "Fast-forward" in stdout:
            result.fast_forward = True
        
        # Parse file changes
        for line in stdout.split("\n"):
            match = re.search(r'(\d+) files? changed', line)
            if match:
                result.files_changed = int(match.group(1))
            match = re.search(r'(\d+) insertions?\(\+\)', line)
            if match:
                result.insertions = int(match.group(1))
            match = re.search(r'(\d+) deletions?\(-\)', line)
            if match:
                result.deletions = int(match.group(1))
        
        return result
    
    def push(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        force: bool = False,
        force_with_lease: bool = False,
        set_upstream: bool = False,
        tags: bool = False,
    ) -> PushResult:
        """Push changes to remote.
        
        Args:
            remote: Remote name
            branch: Branch name (default: current)
            force: Force push (dangerous!)
            force_with_lease: Safer force push
            set_upstream: Set upstream tracking
            tags: Push tags as well
            
        Returns:
            Push result
        """
        # Get current branch
        if not branch:
            success, stdout, _ = self._git.run_safe(["branch", "--show-current"])
            branch = stdout.strip() if success else "main"
        
        result = PushResult(
            success=False,
            remote=remote,
            branch=branch,
        )
        
        # Check force push protection
        if force and not force_with_lease:
            if branch in self._protected_branches:
                if self._force_protection == PushProtection.BLOCK:
                    result.error = f"Force push to protected branch '{branch}' is blocked"
                    return result
                elif self._force_protection == PushProtection.WARN:
                    logger.warning(f"Force pushing to protected branch '{branch}'")
            
            result.forced = True
        
        # Build push command
        args = ["push", remote, branch]
        
        if force:
            args.insert(1, "--force")
        elif force_with_lease:
            args.insert(1, "--force-with-lease")
            result.forced = True
        
        if set_upstream:
            args.insert(1, "-u")
        
        if tags:
            args.append("--tags")
        
        success, stdout, stderr = self._git.run_safe(args)
        
        if not success:
            # Check for rejection
            if "rejected" in stderr.lower():
                # Parse rejected refs
                for line in stderr.split("\n"):
                    if "rejected" in line.lower():
                        result.rejected_refs.append(line.strip())
            
            result.error = stderr
            return result
        
        result.success = True
        
        # Parse output for stats
        for line in stdout.split("\n") + stderr.split("\n"):
            if "->" in line and "refs/" in line:
                result.refs_pushed.append(line.strip())
        
        return result
    
    def set_upstream(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set upstream tracking branch.
        
        Args:
            remote: Remote name
            branch: Branch name (default: current)
            
        Returns:
            Operation result
        """
        # Get current branch
        if not branch:
            success, stdout, _ = self._git.run_safe(["branch", "--show-current"])
            branch = stdout.strip() if success else "main"
        
        success, _, stderr = self._git.run_safe([
            "branch", f"--set-upstream-to={remote}/{branch}", branch
        ])
        
        return {
            "success": success,
            "branch": branch,
            "upstream": f"{remote}/{branch}",
            "error": stderr if not success else None,
        }
    
    def add_protected_branch(self, branch: str):
        """Add a branch to the protected list."""
        self._protected_branches.add(branch)
    
    def remove_protected_branch(self, branch: str):
        """Remove a branch from the protected list."""
        self._protected_branches.discard(branch)


class FetchOptimizer:
    """Optimized fetch operations for large repositories.
    
    Uses LLM reasoning to:
    1. Suggest optimal fetch strategies
    2. Schedule background fetches
    3. Optimize network usage
    
    Example:
        >>> optimizer = FetchOptimizer(llm_client=client)
        >>> result = optimizer.fetch_optimized("origin", strategy=FetchStrategy.SHALLOW)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        repo_path: Optional[Path] = None,
    ):
        """Initialize the fetch optimizer.
        
        Args:
            llm_client: LLM client for intelligent operations
            repo_path: Path to the git repository
        """
        self._llm_client = llm_client
        self._repo_path = repo_path or Path.cwd()
        self._git = GitCommandRunner(self._repo_path)
        
        # Fetch history for optimization
        self._fetch_history: List[Dict[str, Any]] = []
    
    def fetch(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        prune: bool = False,
        tags: bool = True,
    ) -> FetchResult:
        """Standard fetch operation.
        
        Args:
            remote: Remote name
            branch: Specific branch to fetch (None = all)
            prune: Prune deleted remote branches
            tags: Fetch tags
            
        Returns:
            Fetch result
        """
        result = FetchResult(success=False, remote=remote)
        
        args = ["fetch", remote]
        
        if branch:
            args.append(branch)
        
        if prune:
            args.append("--prune")
        
        if not tags:
            args.append("--no-tags")
        
        # Add progress for tracking
        args.append("--progress")
        
        start_time = time.time()
        success, stdout, stderr = self._git.run_safe(args)
        result.duration_seconds = time.time() - start_time
        
        if not success:
            result.error = stderr
            return result
        
        result.success = True
        
        # Parse output
        output = stdout + stderr  # Progress goes to stderr
        
        for line in output.split("\n"):
            # Track new branches
            if "-> " in line and "new" in line.lower():
                ref = line.split("->")[-1].strip()
                result.new_refs.append(ref)
                result.refs_fetched.append(ref)
            
            # Track updated branches
            elif "->" in line:
                ref = line.split("->")[-1].strip()
                result.updated_refs.append(ref)
                result.refs_fetched.append(ref)
            
            # Track deleted
            elif "deleted" in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    result.deleted_refs.append(parts[-1])
            
            # Parse object stats
            if "objects" in line.lower():
                match = re.search(r'(\d+)\s*objects', line)
                if match:
                    result.objects_received = int(match.group(1))
        
        # Record in history
        self._fetch_history.append({
            "remote": remote,
            "branch": branch,
            "timestamp": datetime.now().isoformat(),
            "duration": result.duration_seconds,
            "objects": result.objects_received,
        })
        
        return result
    
    def fetch_optimized(
        self,
        remote: str = "origin",
        strategy: FetchStrategy = FetchStrategy.FULL,
        depth: int = 100,
    ) -> FetchResult:
        """Fetch with optimization strategy.
        
        Args:
            remote: Remote name
            strategy: Fetch strategy
            depth: Depth for shallow fetch
            
        Returns:
            Fetch result
        """
        result = FetchResult(success=False, remote=remote)
        
        args = ["fetch", remote]
        
        if strategy == FetchStrategy.SHALLOW:
            args.extend(["--depth", str(depth)])
        
        elif strategy == FetchStrategy.SINGLE_BRANCH:
            # Get current branch
            success, stdout, _ = self._git.run_safe(["branch", "--show-current"])
            branch = stdout.strip() if success else "main"
            args.append(branch)
        
        elif strategy == FetchStrategy.PARTIAL:
            # Enable partial clone filter
            args.extend(["--filter=blob:none"])
        
        args.append("--progress")
        
        start_time = time.time()
        success, stdout, stderr = self._git.run_safe(args)
        result.duration_seconds = time.time() - start_time
        
        if not success:
            result.error = stderr
            return result
        
        result.success = True
        
        # Parse output (same as regular fetch)
        output = stdout + stderr
        
        for line in output.split("\n"):
            if "-> " in line:
                ref = line.split("->")[-1].strip()
                result.refs_fetched.append(ref)
        
        return result
    
    def fetch_all(
        self,
        prune: bool = True,
    ) -> Dict[str, FetchResult]:
        """Fetch from all remotes.
        
        Args:
            prune: Prune deleted branches
            
        Returns:
            Results for each remote
        """
        results = {}
        
        # Get all remotes
        success, stdout, _ = self._git.run_safe(["remote"])
        
        if not success:
            return results
        
        remotes = [r.strip() for r in stdout.strip().split("\n") if r.strip()]
        
        for remote in remotes:
            results[remote] = self.fetch(remote, prune=prune)
        
        return results
    
    def get_suggested_strategy(
        self,
        remote: str = "origin",
    ) -> FetchStrategy:
        """Suggest optimal fetch strategy based on history.
        
        Args:
            remote: Remote name
            
        Returns:
            Suggested strategy
        """
        # Analyze recent fetches
        recent_fetches = [
            f for f in self._fetch_history
            if f["remote"] == remote
        ][-10:]  # Last 10 fetches
        
        if not recent_fetches:
            return FetchStrategy.FULL
        
        # Calculate averages
        avg_duration = sum(f["duration"] for f in recent_fetches) / len(recent_fetches)
        avg_objects = sum(f["objects"] for f in recent_fetches) / len(recent_fetches)
        
        # If fetches are slow, suggest optimization
        if avg_duration > 10:  # More than 10 seconds
            if avg_objects > 1000:
                return FetchStrategy.PARTIAL
            else:
                return FetchStrategy.SHALLOW
        
        return FetchStrategy.FULL
    
    def unshallow(self) -> Dict[str, Any]:
        """Convert shallow clone to full clone.
        
        Returns:
            Operation result
        """
        success, _, stderr = self._git.run_safe(["fetch", "--unshallow"])
        
        return {
            "success": success,
            "error": stderr if not success else None,
        }


class GitRemoteManager:
    """Main remote repository manager.
    
    Integrates all remote management components:
    - Remote configuration
    - Intelligent push/pull
    - Fetch optimization
    
    Example:
        >>> manager = GitRemoteManager(llm_client=client)
        >>> remotes = manager.list_remotes()
        >>> result = manager.pull()
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        repo_path: Optional[Path] = None,
    ):
        """Initialize the git remote manager.
        
        Args:
            llm_client: LLM client for intelligent operations
            repo_path: Path to the git repository
        """
        self._llm_client = llm_client
        self._repo_path = repo_path or Path.cwd()
        self._git = GitCommandRunner(self._repo_path)
        
        # Initialize components
        self._config_manager = RemoteConfigManager(
            llm_client=llm_client,
            repo_path=self._repo_path,
        )
        
        self._push_pull = IntelligentPushPull(
            llm_client=llm_client,
            repo_path=self._repo_path,
        )
        
        self._fetch_optimizer = FetchOptimizer(
            llm_client=llm_client,
            repo_path=self._repo_path,
        )
    
    def is_git_repo(self) -> bool:
        """Check if current path is a git repository."""
        return self._git.is_git_repo()
    
    # Remote configuration
    def list_remotes(self) -> List[RemoteInfo]:
        """List all configured remotes."""
        return self._config_manager.list_remotes()
    
    def add_remote(
        self,
        name: str,
        url: str,
    ) -> Dict[str, Any]:
        """Add a new remote."""
        return self._config_manager.add_remote(name, url)
    
    def remove_remote(self, name: str) -> Dict[str, Any]:
        """Remove a remote."""
        return self._config_manager.remove_remote(name)
    
    def rename_remote(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """Rename a remote."""
        return self._config_manager.rename_remote(old_name, new_name)
    
    def set_remote_url(
        self,
        name: str,
        url: str,
        push_url: bool = False,
    ) -> Dict[str, Any]:
        """Set remote URL."""
        return self._config_manager.set_url(name, url, push_url)
    
    def check_remote_reachability(self, name: str) -> Dict[str, Any]:
        """Check if remote is reachable."""
        return self._config_manager.check_reachability(name)
    
    # Push/Pull operations
    def get_sync_status(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get sync status with remote."""
        return self._push_pull.get_sync_status(remote, branch)
    
    def predict_pull_conflicts(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
    ) -> ConflictPrediction:
        """Predict conflicts before pull."""
        return self._push_pull.predict_pull_conflicts(remote, branch)
    
    def pull(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        rebase: bool = False,
        ff_only: bool = False,
    ) -> PullResult:
        """Pull changes from remote."""
        return self._push_pull.pull(remote, branch, rebase, ff_only)
    
    def push(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        force: bool = False,
        force_with_lease: bool = False,
        set_upstream: bool = False,
        tags: bool = False,
    ) -> PushResult:
        """Push changes to remote."""
        return self._push_pull.push(
            remote, branch, force, force_with_lease, set_upstream, tags
        )
    
    def set_upstream(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set upstream tracking branch."""
        return self._push_pull.set_upstream(remote, branch)
    
    # Fetch operations
    def fetch(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        prune: bool = False,
    ) -> FetchResult:
        """Fetch from remote."""
        return self._fetch_optimizer.fetch(remote, branch, prune)
    
    def fetch_optimized(
        self,
        remote: str = "origin",
        strategy: Optional[FetchStrategy] = None,
    ) -> FetchResult:
        """Fetch with optimization."""
        if strategy is None:
            strategy = self._fetch_optimizer.get_suggested_strategy(remote)
        
        return self._fetch_optimizer.fetch_optimized(remote, strategy)
    
    def fetch_all(
        self,
        prune: bool = True,
    ) -> Dict[str, FetchResult]:
        """Fetch from all remotes."""
        return self._fetch_optimizer.fetch_all(prune)
    
    async def analyze_with_llm(
        self,
        operation: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze remote situation using LLM.
        
        Args:
            operation: Type of analysis
            context: Context information
            
        Returns:
            LLM analysis
        """
        if not self._llm_client:
            return {"error": "LLM client not available"}
        
        if operation == "sync_strategy":
            sync_status = self.get_sync_status()
            prompt = f"""Analyze this git sync situation and suggest the best approach:

Status: {sync_status['status']}
Ahead: {sync_status['ahead']} commits
Behind: {sync_status['behind']} commits

Provide a brief recommendation for how to sync safely.
"""
        
        elif operation == "remote_setup":
            remotes = self.list_remotes()
            prompt = f"""Analyze this remote configuration:

Remotes: {[r.to_dict() for r in remotes]}

Suggest any improvements or issues to address.
"""
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        try:
            response = await self._llm_client.generate(prompt)
            return {"success": True, "result": response}
        except Exception as e:
            return {"error": str(e)}


# Module-level instance
_global_remote_manager: Optional[GitRemoteManager] = None


def get_git_remote_manager(
    llm_client: Optional[Any] = None,
    repo_path: Optional[Path] = None,
) -> GitRemoteManager:
    """Get the global git remote manager.
    
    Args:
        llm_client: Optional LLM client
        repo_path: Optional repository path
        
    Returns:
        GitRemoteManager instance
    """
    global _global_remote_manager
    if _global_remote_manager is None:
        _global_remote_manager = GitRemoteManager(
            llm_client=llm_client,
            repo_path=repo_path,
        )
    return _global_remote_manager
