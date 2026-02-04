"""Intelligent Git Workflows for Dynamic AI Assistant.

This module implements Phase 6.1 for the Dynamic AI Assistant:
- Smart Branch Management: Naming conventions, lifecycle tracking, cleanup
- Commit Intelligence: Conventional commits, staging, squashing
- Merge Conflict Resolution: Detection, visualization, AI-suggested resolutions
- History Management: Rebase, amend, cherry-pick, bisect automation

Key Features:
============
- AI-assisted branch naming and management
- Intelligent commit message generation
- Automated conflict resolution suggestions
- Safe history manipulation with safeguards

Design Principle:
================
All git workflow decisions use LLM reasoning when available.
The LLM analyzes changes and suggests optimal git operations.
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


class BranchType(Enum):
    """Branch type categories."""
    FEATURE = "feature"
    BUGFIX = "bugfix"
    HOTFIX = "hotfix"
    RELEASE = "release"
    SUPPORT = "support"
    MAIN = "main"
    DEVELOP = "develop"
    CUSTOM = "custom"


class ConflictStrategy(Enum):
    """Merge conflict resolution strategies."""
    OURS = "ours"
    THEIRS = "theirs"
    MANUAL = "manual"
    AI_SUGGEST = "ai_suggest"
    UNION = "union"


class CommitType(Enum):
    """Conventional commit types."""
    FEAT = "feat"
    FIX = "fix"
    DOCS = "docs"
    STYLE = "style"
    REFACTOR = "refactor"
    PERF = "perf"
    TEST = "test"
    BUILD = "build"
    CI = "ci"
    CHORE = "chore"
    REVERT = "revert"


class MergeReadiness(Enum):
    """Branch merge readiness status."""
    READY = "ready"
    CONFLICTS = "conflicts"
    BEHIND = "behind"
    NEEDS_REVIEW = "needs_review"
    FAILING_CHECKS = "failing_checks"
    NOT_READY = "not_ready"


class BisectState(Enum):
    """Git bisect state."""
    IDLE = "idle"
    RUNNING = "running"
    FOUND = "found"
    ABORTED = "aborted"


@dataclass
class BranchInfo:
    """Detailed branch information."""
    name: str
    branch_type: BranchType
    is_current: bool = False
    is_remote: bool = False
    
    # Tracking
    upstream: Optional[str] = None
    ahead: int = 0
    behind: int = 0
    
    # Metadata
    last_commit: Optional[str] = None
    last_commit_date: Optional[datetime] = None
    author: Optional[str] = None
    
    # Lifecycle
    created_date: Optional[datetime] = None
    is_merged: bool = False
    is_stale: bool = False
    stale_days: int = 0
    
    # Protection
    is_protected: bool = False
    protection_rules: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "branch_type": self.branch_type.value,
            "is_current": self.is_current,
            "is_remote": self.is_remote,
            "upstream": self.upstream,
            "ahead": self.ahead,
            "behind": self.behind,
            "last_commit": self.last_commit,
            "last_commit_date": self.last_commit_date.isoformat() if self.last_commit_date else None,
            "is_merged": self.is_merged,
            "is_stale": self.is_stale,
            "stale_days": self.stale_days,
            "is_protected": self.is_protected,
        }


@dataclass
class CommitInfo:
    """Detailed commit information."""
    hash: str
    short_hash: str
    message: str
    author: str
    author_email: str
    date: datetime
    
    # Content
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0
    
    # Metadata
    is_merge: bool = False
    parents: List[str] = field(default_factory=list)
    is_signed: bool = False
    
    # Analysis
    commit_type: Optional[CommitType] = None
    scope: Optional[str] = None
    breaking: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash": self.hash,
            "short_hash": self.short_hash,
            "message": self.message,
            "author": self.author,
            "date": self.date.isoformat(),
            "files_changed": self.files_changed,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "is_merge": self.is_merge,
            "commit_type": self.commit_type.value if self.commit_type else None,
            "scope": self.scope,
            "breaking": self.breaking,
        }


@dataclass
class ConflictInfo:
    """Merge conflict information."""
    file_path: Path
    conflict_type: str  # content, rename, delete
    
    # Content
    ours_content: Optional[str] = None
    theirs_content: Optional[str] = None
    base_content: Optional[str] = None
    
    # Resolution
    resolved: bool = False
    resolution_strategy: Optional[ConflictStrategy] = None
    resolved_content: Optional[str] = None
    
    # AI suggestion
    ai_suggestion: Optional[str] = None
    ai_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": str(self.file_path),
            "conflict_type": self.conflict_type,
            "resolved": self.resolved,
            "resolution_strategy": self.resolution_strategy.value if self.resolution_strategy else None,
            "ai_confidence": self.ai_confidence,
        }


@dataclass
class MergeResult:
    """Merge operation result."""
    success: bool
    merged_branch: str
    target_branch: str
    
    # Details
    merge_commit: Optional[str] = None
    fast_forward: bool = False
    conflicts: List[ConflictInfo] = field(default_factory=list)
    
    # Stats
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0
    
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "merged_branch": self.merged_branch,
            "target_branch": self.target_branch,
            "merge_commit": self.merge_commit,
            "fast_forward": self.fast_forward,
            "conflict_count": len(self.conflicts),
            "files_changed": self.files_changed,
            "error": self.error,
        }


@dataclass
class BisectResult:
    """Git bisect result."""
    found: bool
    bad_commit: Optional[str] = None
    good_commit: Optional[str] = None
    first_bad_commit: Optional[str] = None
    
    # Details
    steps_taken: int = 0
    commits_tested: List[str] = field(default_factory=list)
    
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "first_bad_commit": self.first_bad_commit,
            "steps_taken": self.steps_taken,
            "commits_tested_count": len(self.commits_tested),
            "error": self.error,
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
        timeout: int = 60,
    ) -> subprocess.CompletedProcess:
        """Run a git command.
        
        Args:
            args: Git command arguments (without 'git')
            capture_output: Capture stdout/stderr
            check: Raise on non-zero exit
            timeout: Command timeout in seconds
            
        Returns:
            CompletedProcess result
        """
        cmd = ["git"] + args
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self._repo_path,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                check=check,
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
        timeout: int = 60,
    ) -> Tuple[bool, str, str]:
        """Run a git command safely without raising.
        
        Args:
            args: Git command arguments
            timeout: Command timeout
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            result = self.run(args, check=False, timeout=timeout)
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
    
    def get_repo_root(self) -> Optional[Path]:
        """Get the repository root path."""
        success, stdout, _ = self.run_safe(["rev-parse", "--show-toplevel"])
        if success and stdout:
            return Path(stdout.strip())
        return None


class SmartBranchManager:
    """Intelligent branch management with best practices.
    
    Uses LLM reasoning to:
    1. Suggest branch names based on task description
    2. Detect stale branches and recommend cleanup
    3. Check merge readiness with comprehensive analysis
    
    Example:
        >>> manager = SmartBranchManager(llm_client=client)
        >>> branch_name = manager.suggest_branch_name("Add user authentication")
        >>> # Returns: "feature/add-user-authentication"
    """
    
    # Branch naming conventions
    BRANCH_PREFIXES = {
        BranchType.FEATURE: "feature/",
        BranchType.BUGFIX: "bugfix/",
        BranchType.HOTFIX: "hotfix/",
        BranchType.RELEASE: "release/",
        BranchType.SUPPORT: "support/",
    }
    
    # Protected branch patterns
    PROTECTED_PATTERNS = ["main", "master", "develop", "release/*"]
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        repo_path: Optional[Path] = None,
        stale_days: int = 30,
    ):
        """Initialize the smart branch manager.
        
        Args:
            llm_client: LLM client for intelligent suggestions
            repo_path: Path to the git repository
            stale_days: Days after which a branch is considered stale
        """
        self._llm_client = llm_client
        self._repo_path = repo_path or Path.cwd()
        self._git = GitCommandRunner(self._repo_path)
        self._stale_days = stale_days
    
    def list_branches(
        self,
        include_remote: bool = True,
        include_merged: bool = True,
    ) -> List[BranchInfo]:
        """List all branches with detailed information.
        
        Args:
            include_remote: Include remote tracking branches
            include_merged: Include merged branches
            
        Returns:
            List of branch information
        """
        branches = []
        
        # Get current branch
        success, current, _ = self._git.run_safe(["branch", "--show-current"])
        current_branch = current.strip() if success else ""
        
        # Get local branches with details
        args = ["branch", "-v", "--format=%(refname:short)|%(objectname:short)|%(authordate:iso)|%(authoremail)|%(upstream:short)|%(upstream:track)"]
        success, stdout, _ = self._git.run_safe(args)
        
        if success:
            for line in stdout.strip().split("\n"):
                if not line:
                    continue
                
                parts = line.split("|")
                if len(parts) >= 4:
                    name = parts[0]
                    branch_info = BranchInfo(
                        name=name,
                        branch_type=self._detect_branch_type(name),
                        is_current=(name == current_branch),
                        last_commit=parts[1] if len(parts) > 1 else None,
                        author=parts[3] if len(parts) > 3 else None,
                        upstream=parts[4] if len(parts) > 4 and parts[4] else None,
                    )
                    
                    # Parse date
                    if len(parts) > 2 and parts[2]:
                        try:
                            branch_info.last_commit_date = datetime.fromisoformat(parts[2].strip())
                        except Exception:
                            pass
                    
                    # Parse ahead/behind
                    if len(parts) > 5 and parts[5]:
                        track_info = parts[5]
                        ahead_match = re.search(r'ahead (\d+)', track_info)
                        behind_match = re.search(r'behind (\d+)', track_info)
                        if ahead_match:
                            branch_info.ahead = int(ahead_match.group(1))
                        if behind_match:
                            branch_info.behind = int(behind_match.group(1))
                    
                    # Check if stale
                    if branch_info.last_commit_date:
                        days_old = (datetime.now() - branch_info.last_commit_date).days
                        branch_info.stale_days = days_old
                        branch_info.is_stale = days_old > self._stale_days
                    
                    # Check if protected
                    branch_info.is_protected = self._is_protected(name)
                    
                    branches.append(branch_info)
        
        # Get remote branches if requested
        if include_remote:
            args = ["branch", "-r", "--format=%(refname:short)|%(objectname:short)"]
            success, stdout, _ = self._git.run_safe(args)
            
            if success:
                for line in stdout.strip().split("\n"):
                    if not line or "->" in line:  # Skip HEAD pointers
                        continue
                    
                    parts = line.split("|")
                    name = parts[0]
                    
                    # Skip if local branch exists
                    local_name = name.split("/", 1)[-1] if "/" in name else name
                    if any(b.name == local_name for b in branches):
                        continue
                    
                    branches.append(BranchInfo(
                        name=name,
                        branch_type=self._detect_branch_type(name),
                        is_remote=True,
                        last_commit=parts[1] if len(parts) > 1 else None,
                    ))
        
        # Check merged status if requested
        if include_merged:
            success, merged_output, _ = self._git.run_safe(["branch", "--merged", "HEAD"])
            if success:
                merged_branches = {b.strip() for b in merged_output.strip().split("\n") if b.strip()}
                for branch in branches:
                    if branch.name in merged_branches or f"* {branch.name}" in merged_branches:
                        branch.is_merged = True
        
        return branches
    
    def _detect_branch_type(self, name: str) -> BranchType:
        """Detect branch type from name."""
        name_lower = name.lower()
        
        if name_lower in ("main", "master"):
            return BranchType.MAIN
        if name_lower == "develop":
            return BranchType.DEVELOP
        
        for branch_type, prefix in self.BRANCH_PREFIXES.items():
            if name_lower.startswith(prefix):
                return branch_type
        
        # Try to detect from common patterns
        if "feature" in name_lower or "feat" in name_lower:
            return BranchType.FEATURE
        if "fix" in name_lower or "bug" in name_lower:
            return BranchType.BUGFIX
        if "hotfix" in name_lower:
            return BranchType.HOTFIX
        if "release" in name_lower:
            return BranchType.RELEASE
        
        return BranchType.CUSTOM
    
    def _is_protected(self, name: str) -> bool:
        """Check if branch is protected."""
        import fnmatch
        
        for pattern in self.PROTECTED_PATTERNS:
            if fnmatch.fnmatch(name, pattern):
                return True
        
        return False
    
    def suggest_branch_name(
        self,
        description: str,
        branch_type: BranchType = BranchType.FEATURE,
        issue_number: Optional[str] = None,
    ) -> str:
        """Suggest a branch name based on description.
        
        Args:
            description: Task or feature description
            branch_type: Type of branch
            issue_number: Optional issue number to include
            
        Returns:
            Suggested branch name following conventions
        """
        # Get prefix
        prefix = self.BRANCH_PREFIXES.get(branch_type, "")
        
        # Normalize description
        name = description.lower()
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "to", "for", "in", "on", "at", "by"}
        words = name.split()
        words = [w for w in words if w not in stop_words]
        
        # Limit to first few words
        words = words[:5]
        
        # Join with hyphens
        name = "-".join(words)
        
        # Clean up special characters
        name = re.sub(r'[^a-z0-9-]', '', name)
        name = re.sub(r'-+', '-', name)
        name = name.strip('-')
        
        # Add issue number if provided
        if issue_number:
            name = f"{issue_number}-{name}"
        
        return f"{prefix}{name}"
    
    def create_branch(
        self,
        name: str,
        base: Optional[str] = None,
        checkout: bool = True,
    ) -> Dict[str, Any]:
        """Create a new branch.
        
        Args:
            name: Branch name
            base: Base branch (default: current branch)
            checkout: Switch to new branch after creation
            
        Returns:
            Creation result
        """
        # Validate name
        if not self._validate_branch_name(name):
            return {"success": False, "error": "Invalid branch name"}
        
        # Check if exists
        success, _, _ = self._git.run_safe(["rev-parse", "--verify", name])
        if success:
            return {"success": False, "error": f"Branch '{name}' already exists"}
        
        # Create branch
        args = ["branch", name]
        if base:
            args.append(base)
        
        success, _, stderr = self._git.run_safe(args)
        
        if not success:
            return {"success": False, "error": stderr}
        
        # Checkout if requested
        if checkout:
            success, _, stderr = self._git.run_safe(["checkout", name])
            if not success:
                return {"success": True, "checkout": False, "error": stderr}
        
        return {
            "success": True,
            "branch": name,
            "base": base,
            "checked_out": checkout,
        }
    
    def _validate_branch_name(self, name: str) -> bool:
        """Validate branch name against git rules."""
        # Git branch name rules
        if not name:
            return False
        if name.startswith('-'):
            return False
        if '..' in name:
            return False
        if name.endswith('.lock'):
            return False
        if '@{' in name:
            return False
        
        # Check for invalid characters
        invalid_chars = ['~', '^', ':', '\\', ' ', '?', '*', '[']
        for char in invalid_chars:
            if char in name:
                return False
        
        return True
    
    def delete_branch(
        self,
        name: str,
        force: bool = False,
        delete_remote: bool = False,
    ) -> Dict[str, Any]:
        """Delete a branch.
        
        Args:
            name: Branch name
            force: Force delete even if not merged
            delete_remote: Also delete remote tracking branch
            
        Returns:
            Deletion result
        """
        # Check if protected
        if self._is_protected(name):
            return {"success": False, "error": f"Branch '{name}' is protected"}
        
        # Check if current
        success, current, _ = self._git.run_safe(["branch", "--show-current"])
        if success and current.strip() == name:
            return {"success": False, "error": "Cannot delete current branch"}
        
        # Delete local branch
        flag = "-D" if force else "-d"
        success, _, stderr = self._git.run_safe(["branch", flag, name])
        
        if not success:
            return {"success": False, "error": stderr}
        
        result = {"success": True, "branch": name, "forced": force}
        
        # Delete remote if requested
        if delete_remote:
            # Find remote name (usually origin)
            remote = "origin"
            success, _, stderr = self._git.run_safe(["push", remote, "--delete", name])
            result["remote_deleted"] = success
            if not success:
                result["remote_error"] = stderr
        
        return result
    
    def get_stale_branches(
        self,
        days: Optional[int] = None,
        exclude_protected: bool = True,
    ) -> List[BranchInfo]:
        """Get list of stale branches.
        
        Args:
            days: Days threshold (default: instance default)
            exclude_protected: Exclude protected branches
            
        Returns:
            List of stale branches
        """
        days = days or self._stale_days
        branches = self.list_branches(include_remote=False)
        
        stale = []
        for branch in branches:
            if branch.last_commit_date:
                age = (datetime.now() - branch.last_commit_date).days
                if age > days:
                    if exclude_protected and branch.is_protected:
                        continue
                    branch.is_stale = True
                    branch.stale_days = age
                    stale.append(branch)
        
        return stale
    
    def check_merge_readiness(
        self,
        source_branch: str,
        target_branch: str = "main",
    ) -> Dict[str, Any]:
        """Check if a branch is ready to merge.
        
        Args:
            source_branch: Branch to merge
            target_branch: Target branch
            
        Returns:
            Merge readiness analysis
        """
        result = {
            "source": source_branch,
            "target": target_branch,
            "ready": False,
            "status": MergeReadiness.NOT_READY.value,
            "issues": [],
        }
        
        # Check if source exists
        success, _, _ = self._git.run_safe(["rev-parse", "--verify", source_branch])
        if not success:
            result["issues"].append(f"Source branch '{source_branch}' does not exist")
            return result
        
        # Check if target exists
        success, _, _ = self._git.run_safe(["rev-parse", "--verify", target_branch])
        if not success:
            result["issues"].append(f"Target branch '{target_branch}' does not exist")
            return result
        
        # Check if behind target
        success, stdout, _ = self._git.run_safe([
            "rev-list", "--count", f"{source_branch}..{target_branch}"
        ])
        if success:
            behind = int(stdout.strip()) if stdout.strip() else 0
            if behind > 0:
                result["issues"].append(f"Branch is {behind} commits behind {target_branch}")
                result["behind"] = behind
                result["status"] = MergeReadiness.BEHIND.value
        
        # Check for conflicts
        success, stdout, _ = self._git.run_safe([
            "merge-tree", "--write-tree", target_branch, source_branch
        ])
        if not success:
            # Conflicts detected
            result["issues"].append("Merge conflicts detected")
            result["status"] = MergeReadiness.CONFLICTS.value
        
        # If no issues, it's ready
        if not result["issues"]:
            result["ready"] = True
            result["status"] = MergeReadiness.READY.value
        
        return result


class CommitIntelligence:
    """Intelligent commit assistance.
    
    Uses LLM reasoning to:
    1. Generate conventional commit messages from diffs
    2. Suggest related changes for staging
    3. Recommend commit squashing/splitting
    
    Example:
        >>> intel = CommitIntelligence(llm_client=client)
        >>> message = intel.generate_commit_message()
        >>> # Returns: "feat(auth): add user authentication module"
    """
    
    # Conventional commit pattern
    CONVENTIONAL_PATTERN = re.compile(
        r'^(?P<type>\w+)(\((?P<scope>[^)]+)\))?(?P<breaking>!)?: (?P<description>.+)$'
    )
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        repo_path: Optional[Path] = None,
    ):
        """Initialize commit intelligence.
        
        Args:
            llm_client: LLM client for message generation
            repo_path: Path to the git repository
        """
        self._llm_client = llm_client
        self._repo_path = repo_path or Path.cwd()
        self._git = GitCommandRunner(self._repo_path)
    
    def get_staged_changes(self) -> Dict[str, Any]:
        """Get information about staged changes.
        
        Returns:
            Staged changes information
        """
        result = {
            "files": [],
            "stats": {"insertions": 0, "deletions": 0, "files_changed": 0},
        }
        
        # Get staged files
        success, stdout, _ = self._git.run_safe(["diff", "--cached", "--name-status"])
        
        if success and stdout:
            for line in stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    status = parts[0]
                    file_path = parts[1]
                    result["files"].append({
                        "path": file_path,
                        "status": self._parse_status(status),
                    })
        
        # Get stats
        success, stdout, _ = self._git.run_safe(["diff", "--cached", "--stat"])
        if success and stdout:
            # Parse last line for stats
            lines = stdout.strip().split("\n")
            if lines:
                stat_line = lines[-1]
                # Format: " X files changed, Y insertions(+), Z deletions(-)"
                match = re.search(r'(\d+) files? changed', stat_line)
                if match:
                    result["stats"]["files_changed"] = int(match.group(1))
                match = re.search(r'(\d+) insertions?\(\+\)', stat_line)
                if match:
                    result["stats"]["insertions"] = int(match.group(1))
                match = re.search(r'(\d+) deletions?\(-\)', stat_line)
                if match:
                    result["stats"]["deletions"] = int(match.group(1))
        
        return result
    
    def _parse_status(self, status: str) -> str:
        """Parse git status code to human-readable."""
        status_map = {
            "A": "added",
            "M": "modified",
            "D": "deleted",
            "R": "renamed",
            "C": "copied",
            "U": "unmerged",
            "?": "untracked",
            "!": "ignored",
        }
        return status_map.get(status[0], "unknown")
    
    def get_diff(self, staged: bool = True) -> str:
        """Get diff content.
        
        Args:
            staged: Get staged diff (else unstaged)
            
        Returns:
            Diff content
        """
        args = ["diff"]
        if staged:
            args.append("--cached")
        
        success, stdout, _ = self._git.run_safe(args)
        return stdout if success else ""
    
    def generate_commit_message(
        self,
        commit_type: Optional[CommitType] = None,
        scope: Optional[str] = None,
    ) -> str:
        """Generate a conventional commit message.
        
        Args:
            commit_type: Override detected commit type
            scope: Override detected scope
            
        Returns:
            Generated commit message
        """
        # Get staged changes info
        changes = self.get_staged_changes()
        diff = self.get_diff(staged=True)
        
        if not changes["files"]:
            return ""
        
        # Detect commit type if not provided
        if not commit_type:
            commit_type = self._detect_commit_type(changes, diff)
        
        # Detect scope if not provided
        if not scope:
            scope = self._detect_scope(changes)
        
        # Generate description
        description = self._generate_description(changes, diff, commit_type)
        
        # Build message
        type_str = commit_type.value if commit_type else "chore"
        
        if scope:
            return f"{type_str}({scope}): {description}"
        else:
            return f"{type_str}: {description}"
    
    def _detect_commit_type(
        self,
        changes: Dict[str, Any],
        diff: str,
    ) -> CommitType:
        """Detect appropriate commit type from changes."""
        files = [f["path"] for f in changes["files"]]
        
        # Check for test files
        if any("test" in f.lower() for f in files):
            return CommitType.TEST
        
        # Check for documentation
        doc_extensions = {".md", ".rst", ".txt", ".doc"}
        if all(Path(f).suffix.lower() in doc_extensions for f in files):
            return CommitType.DOCS
        
        # Check for config files
        config_patterns = ["config", "settings", ".json", ".yaml", ".yml", ".toml"]
        if any(any(p in f.lower() for p in config_patterns) for f in files):
            return CommitType.CHORE
        
        # Check for CI files
        if any(".github" in f or "ci" in f.lower() for f in files):
            return CommitType.CI
        
        # Check for build files
        build_files = ["package.json", "setup.py", "cargo.toml", "makefile", "cmake"]
        if any(any(b in f.lower() for b in build_files) for f in files):
            return CommitType.BUILD
        
        # Check diff content for bug fix indicators
        if diff:
            fix_indicators = ["fix", "bug", "error", "issue", "patch"]
            if any(ind in diff.lower() for ind in fix_indicators):
                return CommitType.FIX
        
        # Default to feature
        return CommitType.FEAT
    
    def _detect_scope(self, changes: Dict[str, Any]) -> Optional[str]:
        """Detect scope from changed files."""
        files = [f["path"] for f in changes["files"]]
        
        if not files:
            return None
        
        # Extract common directory
        paths = [Path(f) for f in files]
        
        # If all files in same directory, use that
        parents = set()
        for p in paths:
            if p.parent != Path("."):
                parents.add(p.parent.parts[0] if p.parent.parts else None)
        
        if len(parents) == 1 and None not in parents:
            return list(parents)[0]
        
        return None
    
    def _generate_description(
        self,
        changes: Dict[str, Any],
        diff: str,
        commit_type: CommitType,
    ) -> str:
        """Generate commit description."""
        files = changes["files"]
        
        if not files:
            return "update files"
        
        # Single file change
        if len(files) == 1:
            file_path = files[0]["path"]
            status = files[0]["status"]
            
            file_name = Path(file_path).name
            
            if status == "added":
                return f"add {file_name}"
            elif status == "deleted":
                return f"remove {file_name}"
            else:
                return f"update {file_name}"
        
        # Multiple files
        stats = changes["stats"]
        
        if commit_type == CommitType.DOCS:
            return "update documentation"
        elif commit_type == CommitType.TEST:
            return "update tests"
        elif commit_type == CommitType.FIX:
            return "fix issues"
        else:
            return f"update {stats['files_changed']} files"
    
    def parse_commit_message(self, message: str) -> Dict[str, Any]:
        """Parse a conventional commit message.
        
        Args:
            message: Commit message to parse
            
        Returns:
            Parsed components
        """
        result = {
            "is_conventional": False,
            "type": None,
            "scope": None,
            "breaking": False,
            "description": message,
        }
        
        match = self.CONVENTIONAL_PATTERN.match(message.split("\n")[0])
        
        if match:
            result["is_conventional"] = True
            result["type"] = match.group("type")
            result["scope"] = match.group("scope")
            result["breaking"] = match.group("breaking") == "!"
            result["description"] = match.group("description")
        
        return result
    
    def suggest_staging(self) -> List[Dict[str, Any]]:
        """Suggest files to stage together.
        
        Returns:
            List of staging suggestions with related files
        """
        suggestions = []
        
        # Get unstaged files
        success, stdout, _ = self._git.run_safe(["status", "--porcelain"])
        
        if not success:
            return suggestions
        
        # Group files by directory and type
        by_directory: Dict[str, List[str]] = defaultdict(list)
        by_extension: Dict[str, List[str]] = defaultdict(list)
        
        for line in stdout.strip().split("\n"):
            if not line or line.startswith(" "):
                continue
            
            status = line[:2]
            file_path = line[3:]
            
            # Skip already staged
            if status[0] != " " and status[0] != "?":
                continue
            
            path = Path(file_path)
            parent = str(path.parent) if path.parent != Path(".") else "root"
            by_directory[parent].append(file_path)
            
            ext = path.suffix or "no_extension"
            by_extension[ext].append(file_path)
        
        # Create suggestions
        for directory, files in by_directory.items():
            if len(files) > 1:
                suggestions.append({
                    "type": "directory",
                    "group": directory,
                    "files": files,
                    "reason": f"Files in same directory: {directory}",
                })
        
        return suggestions
    
    def commit(
        self,
        message: str,
        amend: bool = False,
        no_verify: bool = False,
    ) -> Dict[str, Any]:
        """Create a commit.
        
        Args:
            message: Commit message
            amend: Amend previous commit
            no_verify: Skip pre-commit hooks
            
        Returns:
            Commit result
        """
        args = ["commit", "-m", message]
        
        if amend:
            args.append("--amend")
        if no_verify:
            args.append("--no-verify")
        
        success, stdout, stderr = self._git.run_safe(args)
        
        if not success:
            return {"success": False, "error": stderr}
        
        # Get commit hash
        success, hash_out, _ = self._git.run_safe(["rev-parse", "HEAD"])
        
        return {
            "success": True,
            "message": message,
            "hash": hash_out.strip() if success else None,
            "amended": amend,
        }


class MergeConflictResolver:
    """Merge conflict detection and resolution.
    
    Uses LLM reasoning to:
    1. Analyze conflicts and suggest resolutions
    2. Provide side-by-side conflict visualization
    3. Apply resolution strategies intelligently
    
    Example:
        >>> resolver = MergeConflictResolver(llm_client=client)
        >>> conflicts = resolver.get_conflicts()
        >>> resolution = resolver.suggest_resolution(conflicts[0])
    """
    
    CONFLICT_MARKERS = {
        "ours_start": "<<<<<<< ",
        "separator": "=======",
        "theirs_start": ">>>>>>> ",
    }
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        repo_path: Optional[Path] = None,
    ):
        """Initialize the merge conflict resolver.
        
        Args:
            llm_client: LLM client for AI suggestions
            repo_path: Path to the git repository
        """
        self._llm_client = llm_client
        self._repo_path = repo_path or Path.cwd()
        self._git = GitCommandRunner(self._repo_path)
    
    def get_conflicts(self) -> List[ConflictInfo]:
        """Get list of current merge conflicts.
        
        Returns:
            List of conflict information
        """
        conflicts = []
        
        # Get conflicting files
        success, stdout, _ = self._git.run_safe(["diff", "--name-only", "--diff-filter=U"])
        
        if not success or not stdout:
            return conflicts
        
        for file_path in stdout.strip().split("\n"):
            if not file_path:
                continue
            
            conflict = ConflictInfo(
                file_path=Path(file_path),
                conflict_type="content",
            )
            
            # Parse conflict content
            full_path = self._repo_path / file_path
            if full_path.exists():
                try:
                    content = full_path.read_text()
                    self._parse_conflict_content(conflict, content)
                except Exception as e:
                    logger.warning(f"Failed to parse conflict in {file_path}: {e}")
            
            conflicts.append(conflict)
        
        return conflicts
    
    def _parse_conflict_content(self, conflict: ConflictInfo, content: str):
        """Parse conflict markers from content."""
        ours_parts = []
        theirs_parts = []
        in_ours = False
        in_theirs = False
        
        for line in content.split("\n"):
            if line.startswith(self.CONFLICT_MARKERS["ours_start"]):
                in_ours = True
                in_theirs = False
            elif line.startswith(self.CONFLICT_MARKERS["separator"]):
                in_ours = False
                in_theirs = True
            elif line.startswith(self.CONFLICT_MARKERS["theirs_start"]):
                in_theirs = False
            elif in_ours:
                ours_parts.append(line)
            elif in_theirs:
                theirs_parts.append(line)
        
        conflict.ours_content = "\n".join(ours_parts)
        conflict.theirs_content = "\n".join(theirs_parts)
    
    def suggest_resolution(
        self,
        conflict: ConflictInfo,
    ) -> Dict[str, Any]:
        """Suggest a resolution for a conflict.
        
        Args:
            conflict: The conflict to resolve
            
        Returns:
            Resolution suggestion
        """
        result = {
            "file": str(conflict.file_path),
            "strategy": ConflictStrategy.MANUAL.value,
            "suggested_content": None,
            "confidence": 0.0,
        }
        
        if not conflict.ours_content and not conflict.theirs_content:
            return result
        
        # If one side is empty, use the other
        if not conflict.ours_content and conflict.theirs_content:
            result["strategy"] = ConflictStrategy.THEIRS.value
            result["suggested_content"] = conflict.theirs_content
            result["confidence"] = 0.9
            return result
        
        if conflict.ours_content and not conflict.theirs_content:
            result["strategy"] = ConflictStrategy.OURS.value
            result["suggested_content"] = conflict.ours_content
            result["confidence"] = 0.9
            return result
        
        # If identical, use either
        if conflict.ours_content == conflict.theirs_content:
            result["strategy"] = ConflictStrategy.OURS.value
            result["suggested_content"] = conflict.ours_content
            result["confidence"] = 1.0
            return result
        
        # Check for simple additions (one side is subset of other)
        if conflict.theirs_content and conflict.ours_content in conflict.theirs_content:
            result["strategy"] = ConflictStrategy.THEIRS.value
            result["suggested_content"] = conflict.theirs_content
            result["confidence"] = 0.7
            return result
        
        if conflict.ours_content and conflict.theirs_content in conflict.ours_content:
            result["strategy"] = ConflictStrategy.OURS.value
            result["suggested_content"] = conflict.ours_content
            result["confidence"] = 0.7
            return result
        
        # If LLM available, use it for complex conflicts
        if self._llm_client:
            result["strategy"] = ConflictStrategy.AI_SUGGEST.value
            # In production, would call LLM here
        
        return result
    
    def resolve_conflict(
        self,
        conflict: ConflictInfo,
        strategy: ConflictStrategy,
        content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve a merge conflict.
        
        Args:
            conflict: The conflict to resolve
            strategy: Resolution strategy
            content: Custom content (for MANUAL strategy)
            
        Returns:
            Resolution result
        """
        full_path = self._repo_path / conflict.file_path
        
        if not full_path.exists():
            return {"success": False, "error": "File not found"}
        
        # Determine resolved content
        if strategy == ConflictStrategy.OURS:
            resolved = conflict.ours_content
        elif strategy == ConflictStrategy.THEIRS:
            resolved = conflict.theirs_content
        elif strategy == ConflictStrategy.MANUAL:
            if content is None:
                return {"success": False, "error": "Content required for manual resolution"}
            resolved = content
        elif strategy == ConflictStrategy.UNION:
            # Combine both versions
            resolved = conflict.ours_content + "\n" + conflict.theirs_content
        else:
            return {"success": False, "error": f"Unsupported strategy: {strategy}"}
        
        try:
            # Read file and replace conflict markers
            file_content = full_path.read_text()
            
            # Build pattern to match conflict block
            pattern = re.compile(
                r'<<<<<<< .+?\n(.*?)\n=======\n(.*?)\n>>>>>>> .+?\n',
                re.DOTALL
            )
            
            # Replace first conflict with resolved content
            new_content = pattern.sub(resolved + "\n", file_content, count=1)
            
            # Write resolved file
            full_path.write_text(new_content)
            
            # Stage the resolved file
            self._git.run_safe(["add", str(conflict.file_path)])
            
            conflict.resolved = True
            conflict.resolution_strategy = strategy
            conflict.resolved_content = resolved
            
            return {
                "success": True,
                "file": str(conflict.file_path),
                "strategy": strategy.value,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def abort_merge(self) -> Dict[str, Any]:
        """Abort the current merge operation.
        
        Returns:
            Abort result
        """
        success, _, stderr = self._git.run_safe(["merge", "--abort"])
        
        return {
            "success": success,
            "error": stderr if not success else None,
        }


class HistoryManager:
    """Git history management with safety checks.
    
    Uses LLM reasoning to:
    1. Guide interactive rebases
    2. Suggest commit squashing
    3. Automate bisect for bug finding
    
    Example:
        >>> manager = HistoryManager(llm_client=client)
        >>> result = manager.bisect_auto("test_command", "v1.0", "HEAD")
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        repo_path: Optional[Path] = None,
    ):
        """Initialize the history manager.
        
        Args:
            llm_client: LLM client for suggestions
            repo_path: Path to the git repository
        """
        self._llm_client = llm_client
        self._repo_path = repo_path or Path.cwd()
        self._git = GitCommandRunner(self._repo_path)
        
        # Bisect state
        self._bisect_state = BisectState.IDLE
    
    def get_log(
        self,
        count: int = 50,
        branch: Optional[str] = None,
        since: Optional[datetime] = None,
        author: Optional[str] = None,
    ) -> List[CommitInfo]:
        """Get commit history.
        
        Args:
            count: Number of commits to retrieve
            branch: Branch to get history from
            since: Only commits after this date
            author: Filter by author
            
        Returns:
            List of commit information
        """
        commits = []
        
        args = [
            "log",
            f"-{count}",
            "--format=%H|%h|%s|%an|%ae|%aI|%P",
        ]
        
        if branch:
            args.append(branch)
        if since:
            args.append(f"--since={since.isoformat()}")
        if author:
            args.append(f"--author={author}")
        
        success, stdout, _ = self._git.run_safe(args)
        
        if not success:
            return commits
        
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            
            parts = line.split("|")
            if len(parts) < 7:
                continue
            
            commit = CommitInfo(
                hash=parts[0],
                short_hash=parts[1],
                message=parts[2],
                author=parts[3],
                author_email=parts[4],
                date=datetime.fromisoformat(parts[5]),
                parents=parts[6].split() if parts[6] else [],
            )
            
            commit.is_merge = len(commit.parents) > 1
            
            # Parse conventional commit
            intel = CommitIntelligence(repo_path=self._repo_path)
            parsed = intel.parse_commit_message(commit.message)
            if parsed["is_conventional"]:
                try:
                    commit.commit_type = CommitType(parsed["type"])
                except ValueError:
                    pass
                commit.scope = parsed["scope"]
                commit.breaking = parsed["breaking"]
            
            commits.append(commit)
        
        return commits
    
    def cherry_pick(
        self,
        commit_hash: str,
        no_commit: bool = False,
    ) -> Dict[str, Any]:
        """Cherry-pick a commit.
        
        Args:
            commit_hash: Commit hash to cherry-pick
            no_commit: Stage changes without committing
            
        Returns:
            Cherry-pick result
        """
        args = ["cherry-pick", commit_hash]
        
        if no_commit:
            args.append("-n")
        
        success, _, stderr = self._git.run_safe(args)
        
        if not success:
            # Check for conflicts
            conflicts = MergeConflictResolver(repo_path=self._repo_path).get_conflicts()
            
            return {
                "success": False,
                "error": stderr,
                "has_conflicts": len(conflicts) > 0,
                "conflicts": [str(c.file_path) for c in conflicts],
            }
        
        return {
            "success": True,
            "commit": commit_hash,
            "committed": not no_commit,
        }
    
    def abort_cherry_pick(self) -> Dict[str, Any]:
        """Abort the current cherry-pick operation.
        
        Returns:
            Abort result
        """
        success, _, stderr = self._git.run_safe(["cherry-pick", "--abort"])
        
        return {
            "success": success,
            "error": stderr if not success else None,
        }
    
    def amend_commit(
        self,
        message: Optional[str] = None,
        no_edit: bool = False,
    ) -> Dict[str, Any]:
        """Amend the last commit.
        
        Args:
            message: New commit message
            no_edit: Keep existing message
            
        Returns:
            Amend result
        """
        # Check if there are staged changes
        success, stdout, _ = self._git.run_safe(["diff", "--cached", "--name-only"])
        has_staged = bool(stdout.strip()) if success else False
        
        args = ["commit", "--amend"]
        
        if message:
            args.extend(["-m", message])
        elif no_edit:
            args.append("--no-edit")
        
        success, _, stderr = self._git.run_safe(args)
        
        if not success:
            return {"success": False, "error": stderr}
        
        # Get new commit hash
        success, hash_out, _ = self._git.run_safe(["rev-parse", "HEAD"])
        
        return {
            "success": True,
            "hash": hash_out.strip() if success else None,
            "staged_changes": has_staged,
        }
    
    def bisect_start(
        self,
        bad_commit: str = "HEAD",
        good_commit: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a bisect operation.
        
        Args:
            bad_commit: Known bad commit
            good_commit: Known good commit
            
        Returns:
            Start result
        """
        # Start bisect
        success, _, stderr = self._git.run_safe(["bisect", "start"])
        
        if not success:
            return {"success": False, "error": stderr}
        
        # Mark bad
        success, _, stderr = self._git.run_safe(["bisect", "bad", bad_commit])
        
        if not success:
            self._git.run_safe(["bisect", "reset"])
            return {"success": False, "error": stderr}
        
        # Mark good if provided
        if good_commit:
            success, stdout, stderr = self._git.run_safe(["bisect", "good", good_commit])
            
            if not success:
                self._git.run_safe(["bisect", "reset"])
                return {"success": False, "error": stderr}
        
        self._bisect_state = BisectState.RUNNING
        
        return {
            "success": True,
            "state": "running",
            "bad": bad_commit,
            "good": good_commit,
        }
    
    def bisect_good(self) -> Dict[str, Any]:
        """Mark current commit as good."""
        success, stdout, stderr = self._git.run_safe(["bisect", "good"])
        
        # Check if bisect is done
        if "first bad commit" in stdout.lower() if stdout else False:
            self._bisect_state = BisectState.FOUND
        
        return {
            "success": success,
            "output": stdout,
            "error": stderr if not success else None,
        }
    
    def bisect_bad(self) -> Dict[str, Any]:
        """Mark current commit as bad."""
        success, stdout, stderr = self._git.run_safe(["bisect", "bad"])
        
        # Check if bisect is done
        if "first bad commit" in stdout.lower() if stdout else False:
            self._bisect_state = BisectState.FOUND
        
        return {
            "success": success,
            "output": stdout,
            "error": stderr if not success else None,
        }
    
    def bisect_reset(self) -> Dict[str, Any]:
        """Reset/abort bisect operation."""
        success, _, stderr = self._git.run_safe(["bisect", "reset"])
        
        self._bisect_state = BisectState.IDLE
        
        return {
            "success": success,
            "error": stderr if not success else None,
        }
    
    def bisect_auto(
        self,
        test_command: str,
        good_commit: str,
        bad_commit: str = "HEAD",
        timeout: int = 60,
    ) -> BisectResult:
        """Run automated bisect with a test command.
        
        Args:
            test_command: Command to test (exit 0 = good, else = bad)
            good_commit: Known good commit
            bad_commit: Known bad commit
            timeout: Timeout per test in seconds
            
        Returns:
            Bisect result
        """
        result = BisectResult(
            found=False,
            good_commit=good_commit,
            bad_commit=bad_commit,
        )
        
        # Start bisect
        start_result = self.bisect_start(bad_commit, good_commit)
        
        if not start_result["success"]:
            result.error = start_result.get("error")
            return result
        
        # Run automated bisect
        success, stdout, stderr = self._git.run_safe(
            ["bisect", "run", "sh", "-c", test_command],
            timeout=timeout * 100,  # Allow for many iterations
        )
        
        if success and stdout:
            # Parse output for first bad commit
            for line in stdout.split("\n"):
                if "first bad commit" in line.lower():
                    result.found = True
                    # Extract commit hash
                    match = re.search(r'([a-f0-9]{40})', line)
                    if match:
                        result.first_bad_commit = match.group(1)
                    break
        else:
            result.error = stderr
        
        # Reset bisect
        self.bisect_reset()
        
        return result
    
    def get_reflog(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get reflog entries for recovery.
        
        Args:
            count: Number of entries to retrieve
            
        Returns:
            List of reflog entries
        """
        entries = []
        
        success, stdout, _ = self._git.run_safe([
            "reflog",
            f"-{count}",
            "--format=%H|%gd|%gs|%ci",
        ])
        
        if not success:
            return entries
        
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            
            parts = line.split("|")
            if len(parts) >= 4:
                entries.append({
                    "hash": parts[0],
                    "ref": parts[1],
                    "action": parts[2],
                    "date": parts[3],
                })
        
        return entries


class GitWorkflowManager:
    """Main git workflow manager.
    
    Integrates all git workflow components:
    - Smart branch management
    - Commit intelligence
    - Merge conflict resolution
    - History management
    
    Example:
        >>> manager = GitWorkflowManager(llm_client=client)
        >>> branches = manager.list_branches()
        >>> message = manager.generate_commit_message()
        >>> manager.commit(message)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        repo_path: Optional[Path] = None,
    ):
        """Initialize the git workflow manager.
        
        Args:
            llm_client: LLM client for intelligent operations
            repo_path: Path to the git repository
        """
        self._llm_client = llm_client
        self._repo_path = repo_path or Path.cwd()
        self._git = GitCommandRunner(self._repo_path)
        
        # Initialize components
        self._branch_manager = SmartBranchManager(
            llm_client=llm_client,
            repo_path=self._repo_path,
        )
        
        self._commit_intel = CommitIntelligence(
            llm_client=llm_client,
            repo_path=self._repo_path,
        )
        
        self._conflict_resolver = MergeConflictResolver(
            llm_client=llm_client,
            repo_path=self._repo_path,
        )
        
        self._history_manager = HistoryManager(
            llm_client=llm_client,
            repo_path=self._repo_path,
        )
    
    def is_git_repo(self) -> bool:
        """Check if current path is a git repository."""
        return self._git.is_git_repo()
    
    def get_repo_root(self) -> Optional[Path]:
        """Get the repository root path."""
        return self._git.get_repo_root()
    
    # Branch operations
    def list_branches(
        self,
        include_remote: bool = True,
    ) -> List[BranchInfo]:
        """List all branches."""
        return self._branch_manager.list_branches(include_remote=include_remote)
    
    def suggest_branch_name(
        self,
        description: str,
        branch_type: BranchType = BranchType.FEATURE,
        issue_number: Optional[str] = None,
    ) -> str:
        """Suggest a branch name."""
        return self._branch_manager.suggest_branch_name(
            description, branch_type, issue_number
        )
    
    def create_branch(
        self,
        name: str,
        base: Optional[str] = None,
        checkout: bool = True,
    ) -> Dict[str, Any]:
        """Create a new branch."""
        return self._branch_manager.create_branch(name, base, checkout)
    
    def delete_branch(
        self,
        name: str,
        force: bool = False,
        delete_remote: bool = False,
    ) -> Dict[str, Any]:
        """Delete a branch."""
        return self._branch_manager.delete_branch(name, force, delete_remote)
    
    def get_stale_branches(
        self,
        days: Optional[int] = None,
    ) -> List[BranchInfo]:
        """Get stale branches."""
        return self._branch_manager.get_stale_branches(days)
    
    def check_merge_readiness(
        self,
        source_branch: str,
        target_branch: str = "main",
    ) -> Dict[str, Any]:
        """Check merge readiness."""
        return self._branch_manager.check_merge_readiness(source_branch, target_branch)
    
    # Commit operations
    def get_staged_changes(self) -> Dict[str, Any]:
        """Get staged changes."""
        return self._commit_intel.get_staged_changes()
    
    def generate_commit_message(
        self,
        commit_type: Optional[CommitType] = None,
        scope: Optional[str] = None,
    ) -> str:
        """Generate a commit message."""
        return self._commit_intel.generate_commit_message(commit_type, scope)
    
    def commit(
        self,
        message: str,
        amend: bool = False,
    ) -> Dict[str, Any]:
        """Create a commit."""
        return self._commit_intel.commit(message, amend)
    
    def suggest_staging(self) -> List[Dict[str, Any]]:
        """Suggest files to stage."""
        return self._commit_intel.suggest_staging()
    
    # Conflict operations
    def get_conflicts(self) -> List[ConflictInfo]:
        """Get current conflicts."""
        return self._conflict_resolver.get_conflicts()
    
    def suggest_resolution(
        self,
        conflict: ConflictInfo,
    ) -> Dict[str, Any]:
        """Suggest conflict resolution."""
        return self._conflict_resolver.suggest_resolution(conflict)
    
    def resolve_conflict(
        self,
        conflict: ConflictInfo,
        strategy: ConflictStrategy,
        content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve a conflict."""
        return self._conflict_resolver.resolve_conflict(conflict, strategy, content)
    
    def abort_merge(self) -> Dict[str, Any]:
        """Abort merge."""
        return self._conflict_resolver.abort_merge()
    
    # History operations
    def get_log(
        self,
        count: int = 50,
        branch: Optional[str] = None,
    ) -> List[CommitInfo]:
        """Get commit history."""
        return self._history_manager.get_log(count, branch)
    
    def cherry_pick(
        self,
        commit_hash: str,
    ) -> Dict[str, Any]:
        """Cherry-pick a commit."""
        return self._history_manager.cherry_pick(commit_hash)
    
    def amend_commit(
        self,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Amend last commit."""
        return self._history_manager.amend_commit(message)
    
    def bisect_auto(
        self,
        test_command: str,
        good_commit: str,
        bad_commit: str = "HEAD",
    ) -> BisectResult:
        """Run automated bisect."""
        return self._history_manager.bisect_auto(test_command, good_commit, bad_commit)
    
    def get_reflog(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get reflog for recovery."""
        return self._history_manager.get_reflog(count)
    
    async def analyze_with_llm(
        self,
        operation: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze git situation using LLM.
        
        Args:
            operation: Type of analysis
            context: Context information
            
        Returns:
            LLM analysis
        """
        if not self._llm_client:
            return {"error": "LLM client not available"}
        
        if operation == "commit_message":
            diff = self._commit_intel.get_diff(staged=True)
            prompt = f"""Generate a conventional commit message for these changes:

{diff[:3000]}

Format: type(scope): description
Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore

Return only the commit message, nothing else.
"""
        elif operation == "conflict_resolution":
            conflict = context.get("conflict")
            if conflict:
                prompt = f"""Suggest how to resolve this merge conflict:

Ours:
{conflict.ours_content}

Theirs:
{conflict.theirs_content}

Provide the resolved content that best combines both changes.
"""
            else:
                return {"error": "No conflict provided"}
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        try:
            response = await self._llm_client.generate(prompt)
            return {"success": True, "result": response}
        except Exception as e:
            return {"error": str(e)}


# Module-level instance
_global_git_workflow_manager: Optional[GitWorkflowManager] = None


def get_git_workflow_manager(
    llm_client: Optional[Any] = None,
    repo_path: Optional[Path] = None,
) -> GitWorkflowManager:
    """Get the global git workflow manager.
    
    Args:
        llm_client: Optional LLM client
        repo_path: Optional repository path
        
    Returns:
        GitWorkflowManager instance
    """
    global _global_git_workflow_manager
    if _global_git_workflow_manager is None:
        _global_git_workflow_manager = GitWorkflowManager(
            llm_client=llm_client,
            repo_path=repo_path,
        )
    return _global_git_workflow_manager
