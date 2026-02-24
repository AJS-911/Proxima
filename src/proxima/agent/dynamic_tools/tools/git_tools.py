"""Git Tools for Dynamic Tool System.

This module provides Git operation tools that can be
dynamically discovered and used by the LLM.

Tools included:
- GitStatus: Get repository status
- GitCommit: Create commits (with optional LLM commit message generation)
- GitBranch: Branch management
- GitLog: View commit history
- GitDiff: View changes
- GitCheckout: Switch branches/restore files
- GitStash: Stash management
- GitAdd: Stage files
- GitPull: Pull from remote
- GitPush: Push to remote (with auto-fix for upstream failures)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..tool_interface import (
    BaseTool,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolCategory,
    PermissionLevel,
    RiskLevel,
    ParameterType,
)
from ..execution_context import ExecutionContext, GitState
from ..tool_registry import register_tool


def run_git_command(
    args: List[str],
    cwd: str,
    capture_output: bool = True,
) -> Dict[str, Any]:
    """Run a git command and return the result."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            timeout=30,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip() if result.stdout else "",
            "stderr": result.stderr.strip() if result.stderr else "",
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Git command timed out",
            "returncode": -1,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "Git is not installed or not in PATH",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "returncode": -1,
        }


def find_git_root(path: Path) -> Optional[Path]:
    """Find the root of the git repository."""
    current = path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None


@register_tool
class GitStatusTool(BaseTool):
    """Get the status of a Git repository."""
    
    @property
    def name(self) -> str:
        return "git_status"
    
    @property
    def description(self) -> str:
        return (
            "Get the current status of a Git repository, including branch name, "
            "staged changes, unstaged changes, and untracked files."
        )
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT
    
    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_ONLY
    
    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.NONE
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                description="Path to the repository (default: current directory)",
                param_type=ParameterType.PATH,
                required=False,
                default=".",
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["git", "status", "changes", "branch", "modified"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "What is the git status?",
            "Show me what files have changed",
            "What branch am I on?",
            "Are there any uncommitted changes?",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        path = Path(parameters.get("path", "."))
        
        if not path.is_absolute():
            path = Path(context.current_directory) / path
        
        git_root = find_git_root(path)
        if not git_root:
            return ToolResult(
                success=False,
                error="Not a git repository",
            )
        
        # Get branch name
        branch_result = run_git_command(["branch", "--show-current"], str(git_root))
        branch = branch_result.get("stdout", "unknown") if branch_result["success"] else "unknown"
        
        # Get status
        status_result = run_git_command(["status", "--porcelain"], str(git_root))
        if not status_result["success"]:
            return ToolResult(
                success=False,
                error=status_result.get("error", "Failed to get status"),
            )
        
        # Parse status
        staged = []
        unstaged = []
        untracked = []
        
        for line in status_result["stdout"].split("\n"):
            if not line:
                continue
            
            index_status = line[0]
            worktree_status = line[1]
            filename = line[3:]
            
            if index_status == "?":
                untracked.append(filename)
            else:
                if index_status not in " ?":
                    staged.append({"file": filename, "status": index_status})
                if worktree_status not in " ?":
                    unstaged.append({"file": filename, "status": worktree_status})
        
        # Check for remote
        remote_result = run_git_command(["remote", "-v"], str(git_root))
        has_remote = bool(remote_result.get("stdout"))
        
        # Get remote URL
        remote_url = None
        if has_remote:
            url_result = run_git_command(["remote", "get-url", "origin"], str(git_root))
            remote_url = url_result.get("stdout") if url_result["success"] else None
        
        # Update context with git state
        git_state = GitState(
            repo_path=str(git_root),
            current_branch=branch,
            is_dirty=bool(staged or unstaged or untracked),
            has_remote=has_remote,
            remote_url=remote_url,
            uncommitted_changes=[f["file"] for f in unstaged] + untracked,
        )
        context.update_git_state(git_state)
        
        status_data = {
            "branch": branch,
            "repo_path": str(git_root),
            "is_clean": not (staged or unstaged or untracked),
            "staged_files": staged,
            "unstaged_files": unstaged,
            "untracked_files": untracked,
            "has_remote": has_remote,
            "remote_url": remote_url,
        }
        
        # Generate summary message
        parts = [f"On branch {branch}"]
        if staged:
            parts.append(f"{len(staged)} staged")
        if unstaged:
            parts.append(f"{len(unstaged)} modified")
        if untracked:
            parts.append(f"{len(untracked)} untracked")
        if not (staged or unstaged or untracked):
            parts.append("working tree clean")
        
        return ToolResult(
            success=True,
            result=status_data,
            message=", ".join(parts),
        )


@register_tool
class GitCommitTool(BaseTool):
    """Create a Git commit."""
    
    @property
    def name(self) -> str:
        return "git_commit"
    
    @property
    def description(self) -> str:
        return (
            "Create a new Git commit with the staged changes. "
            "Optionally stage all changes before committing."
        )
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT
    
    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_WRITE
    
    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="message",
                description=(
                    "Commit message. If omitted and an LLM is available, "
                    "a message is auto-generated from the staged diff."
                ),
                param_type=ParameterType.STRING,
                required=False,
            ),
            ToolParameter(
                name="stage_all",
                description="Stage all changes before committing (git add -A)",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=False,
            ),
            ToolParameter(
                name="path",
                description="Path to the repository",
                param_type=ParameterType.PATH,
                required=False,
                default=".",
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["git", "commit", "save", "snapshot", "changes"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "Commit with message 'Fix bug in login'",
            "Stage all changes and commit as 'Initial commit'",
            "Create a commit for the changes",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        message = parameters.get("message", "")
        stage_all = parameters.get("stage_all", False)
        path = Path(parameters.get("path", "."))
        
        if not path.is_absolute():
            path = Path(context.current_directory) / path
        
        git_root = find_git_root(path)
        if not git_root:
            return ToolResult(
                success=False,
                error="Not a git repository",
            )
        
        # Stage all if requested
        if stage_all:
            add_result = run_git_command(["add", "-A"], str(git_root))
            if not add_result["success"]:
                return ToolResult(
                    success=False,
                    error=f"Failed to stage changes: {add_result.get('stderr', '')}",
                )
        
        # ── LLM commit message generation (Phase 8, Step 8.4) ────────
        if not message:
            # Try to generate a commit message from the staged diff
            diff_result = run_git_command(
                ["diff", "--staged", "--stat"], str(git_root)
            )
            diff_detail = run_git_command(
                ["diff", "--staged"], str(git_root)
            )
            staged_diff = diff_detail.get("stdout", "")
            diff_summary = diff_result.get("stdout", "")
            
            if not staged_diff.strip():
                return ToolResult(
                    success=False,
                    error=(
                        "Nothing staged to commit and no message provided. "
                        "Stage some changes first."
                    ),
                )
            
            # Attempt LLM generation if available via context
            llm_generated = False
            if hasattr(context, "llm_client") and context.llm_client is not None:
                try:
                    # Truncate diff to avoid exceeding token limits
                    truncated = staged_diff[:3000]
                    prompt = (
                        "Generate a concise commit message for these changes. "
                        "Return ONLY the commit message, no explanation:\n\n"
                        f"{truncated}"
                    )
                    llm_response = context.llm_client.generate(prompt)
                    if llm_response and llm_response.strip():
                        message = llm_response.strip().strip('"').strip("'")
                        llm_generated = True
                except Exception:
                    pass  # Fall through to auto-generated message
            
            if not message:
                # Fallback: auto-generate from diff summary
                files_changed = diff_summary.strip().split("\n")[-1] if diff_summary.strip() else ""
                message = f"Update: {files_changed}" if files_changed else "Update changes"
                llm_generated = False
        
        # Create commit
        commit_result = run_git_command(["commit", "-m", message], str(git_root))
        
        if not commit_result["success"]:
            stderr = commit_result.get("stderr", "")
            if "nothing to commit" in stderr.lower():
                return ToolResult(
                    success=False,
                    error="Nothing to commit. Stage some changes first.",
                )
            return ToolResult(
                success=False,
                error=f"Commit failed: {stderr}",
            )
        
        # Get commit hash
        hash_result = run_git_command(["rev-parse", "--short", "HEAD"], str(git_root))
        commit_hash = hash_result.get("stdout", "unknown")
        
        return ToolResult(
            success=True,
            message=f"Created commit {commit_hash}: {message}",
            result={
                "commit_hash": commit_hash,
                "message": message,
            },
        )


@register_tool
class GitLogTool(BaseTool):
    """View Git commit history."""
    
    @property
    def name(self) -> str:
        return "git_log"
    
    @property
    def description(self) -> str:
        return "View the commit history of a Git repository."
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT
    
    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_ONLY
    
    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.NONE
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="count",
                description="Number of commits to show (default: 10)",
                param_type=ParameterType.INTEGER,
                required=False,
                default=10,
            ),
            ToolParameter(
                name="branch",
                description="Branch to show history for",
                param_type=ParameterType.BRANCH_NAME,
                required=False,
            ),
            ToolParameter(
                name="path",
                description="Path to the repository",
                param_type=ParameterType.PATH,
                required=False,
                default=".",
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["git", "log", "history", "commits"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "Show the last 5 commits",
            "View git history",
            "What are the recent commits?",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        count = parameters.get("count", 10)
        branch = parameters.get("branch")
        path = Path(parameters.get("path", "."))
        
        if not path.is_absolute():
            path = Path(context.current_directory) / path
        
        git_root = find_git_root(path)
        if not git_root:
            return ToolResult(
                success=False,
                error="Not a git repository",
            )
        
        # Build command
        format_str = "%H|%h|%an|%ae|%at|%s"
        args = ["log", f"--format={format_str}", f"-n{count}"]
        
        if branch:
            args.append(branch)
        
        result = run_git_command(args, str(git_root))
        
        if not result["success"]:
            return ToolResult(
                success=False,
                error=f"Failed to get log: {result.get('stderr', '')}",
            )
        
        commits = []
        for line in result["stdout"].split("\n"):
            if not line:
                continue
            
            parts = line.split("|")
            if len(parts) >= 6:
                commits.append({
                    "hash": parts[0],
                    "short_hash": parts[1],
                    "author": parts[2],
                    "email": parts[3],
                    "timestamp": datetime.fromtimestamp(int(parts[4])).isoformat(),
                    "message": parts[5],
                })
        
        return ToolResult(
            success=True,
            result=commits,
            message=f"Showing {len(commits)} commits",
        )


@register_tool
class GitDiffTool(BaseTool):
    """View Git diff."""
    
    @property
    def name(self) -> str:
        return "git_diff"
    
    @property
    def description(self) -> str:
        return (
            "View the differences between commits, branches, or the working tree. "
            "Shows what changes have been made."
        )
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT
    
    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_ONLY
    
    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.NONE
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="staged",
                description="Show staged changes (--cached)",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=False,
            ),
            ToolParameter(
                name="file",
                description="Specific file to diff",
                param_type=ParameterType.PATH,
                required=False,
            ),
            ToolParameter(
                name="commit",
                description="Compare with specific commit",
                param_type=ParameterType.STRING,
                required=False,
            ),
            ToolParameter(
                name="path",
                description="Path to the repository",
                param_type=ParameterType.PATH,
                required=False,
                default=".",
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["git", "diff", "changes", "compare"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "Show me the diff",
            "What changes are staged?",
            "Show diff for main.py",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        staged = parameters.get("staged", False)
        file_path = parameters.get("file")
        commit = parameters.get("commit")
        path = Path(parameters.get("path", "."))
        
        if not path.is_absolute():
            path = Path(context.current_directory) / path
        
        git_root = find_git_root(path)
        if not git_root:
            return ToolResult(
                success=False,
                error="Not a git repository",
            )
        
        args = ["diff"]
        
        if staged:
            args.append("--cached")
        
        if commit:
            args.append(commit)
        
        if file_path:
            args.append("--")
            args.append(file_path)
        
        result = run_git_command(args, str(git_root))
        
        if not result["success"]:
            return ToolResult(
                success=False,
                error=f"Failed to get diff: {result.get('stderr', '')}",
            )
        
        diff_output = result["stdout"]
        
        if not diff_output:
            return ToolResult(
                success=True,
                message="No differences found",
                result="",
            )
        
        # Count changes
        additions = diff_output.count("\n+") - diff_output.count("\n+++")
        deletions = diff_output.count("\n-") - diff_output.count("\n---")
        
        return ToolResult(
            success=True,
            result=diff_output,
            message=f"+{additions} -{deletions} lines",
        )


@register_tool
class GitBranchTool(BaseTool):
    """Manage Git branches."""
    
    @property
    def name(self) -> str:
        return "git_branch"
    
    @property
    def description(self) -> str:
        return (
            "List, create, or delete Git branches. "
            "Can also switch to a different branch."
        )
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT
    
    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_WRITE
    
    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                description="Action: list, create, delete, or switch",
                param_type=ParameterType.STRING,
                required=False,
                default="list",
            ),
            ToolParameter(
                name="name",
                description="Branch name (for create/delete/switch)",
                param_type=ParameterType.BRANCH_NAME,
                required=False,
            ),
            ToolParameter(
                name="path",
                description="Path to the repository",
                param_type=ParameterType.PATH,
                required=False,
                default=".",
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["git", "branch", "checkout", "switch", "create"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "List all branches",
            "Create a new branch called feature-x",
            "Switch to the main branch",
            "Delete the old-feature branch",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        action = parameters.get("action", "list")
        name = parameters.get("name")
        path = Path(parameters.get("path", "."))
        
        if not path.is_absolute():
            path = Path(context.current_directory) / path
        
        git_root = find_git_root(path)
        if not git_root:
            return ToolResult(
                success=False,
                error="Not a git repository",
            )
        
        if action == "list":
            result = run_git_command(["branch", "-a"], str(git_root))
            if result["success"]:
                branches = []
                current = None
                for line in result["stdout"].split("\n"):
                    if not line:
                        continue
                    is_current = line.startswith("*")
                    branch_name = line.strip().lstrip("* ")
                    branches.append(branch_name)
                    if is_current:
                        current = branch_name
                
                return ToolResult(
                    success=True,
                    result={"branches": branches, "current": current},
                    message=f"Current branch: {current}, {len(branches)} total branches",
                )
        
        elif action == "create":
            if not name:
                return ToolResult(success=False, error="Branch name required")
            
            result = run_git_command(["branch", name], str(git_root))
            if result["success"]:
                return ToolResult(
                    success=True,
                    message=f"Created branch: {name}",
                )
            return ToolResult(
                success=False,
                error=result.get("stderr", "Failed to create branch"),
            )
        
        elif action == "delete":
            if not name:
                return ToolResult(success=False, error="Branch name required")
            
            result = run_git_command(["branch", "-d", name], str(git_root))
            if result["success"]:
                return ToolResult(
                    success=True,
                    message=f"Deleted branch: {name}",
                )
            return ToolResult(
                success=False,
                error=result.get("stderr", "Failed to delete branch"),
            )
        
        elif action == "switch":
            if not name:
                return ToolResult(success=False, error="Branch name required")
            
            result = run_git_command(["checkout", name], str(git_root))
            if result["success"]:
                return ToolResult(
                    success=True,
                    message=f"Switched to branch: {name}",
                )
            return ToolResult(
                success=False,
                error=result.get("stderr", "Failed to switch branch"),
            )
        
        return ToolResult(
            success=False,
            error=f"Unknown action: {action}",
        )


@register_tool
class GitAddTool(BaseTool):
    """Stage files for commit."""
    
    @property
    def name(self) -> str:
        return "git_add"
    
    @property
    def description(self) -> str:
        return "Stage files for the next commit."
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT
    
    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_WRITE
    
    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.NONE
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="files",
                description="Files to stage (use '.' for all files)",
                param_type=ParameterType.STRING,
                required=True,
            ),
            ToolParameter(
                name="path",
                description="Path to the repository",
                param_type=ParameterType.PATH,
                required=False,
                default=".",
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["git", "add", "stage"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "Stage all changes",
            "Add main.py to staging",
            "Stage the config files",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        files = parameters["files"]
        path = Path(parameters.get("path", "."))
        
        if not path.is_absolute():
            path = Path(context.current_directory) / path
        
        git_root = find_git_root(path)
        if not git_root:
            return ToolResult(
                success=False,
                error="Not a git repository",
            )
        
        # Handle multiple files
        file_list = files.split() if isinstance(files, str) else files
        
        result = run_git_command(["add"] + file_list, str(git_root))
        
        if result["success"]:
            return ToolResult(
                success=True,
                message=f"Staged: {files}",
            )
        
        return ToolResult(
            success=False,
            error=result.get("stderr", "Failed to stage files"),
        )


# ═══════════════════════════════════════════════════════════════════════
# Phase 8 — Git Pull and Push tools
# ═══════════════════════════════════════════════════════════════════════


@register_tool
class GitPullTool(BaseTool):
    """Pull changes from a remote repository."""

    @property
    def name(self) -> str:
        return "git_pull"

    @property
    def description(self) -> str:
        return (
            "Pull the latest changes from a remote repository. "
            "Verifies the directory is a git repo before pulling."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT

    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_WRITE

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="remote",
                description="Remote name (default: origin)",
                param_type=ParameterType.STRING,
                required=False,
                default="origin",
            ),
            ToolParameter(
                name="branch",
                description="Branch to pull (default: current tracking branch)",
                param_type=ParameterType.STRING,
                required=False,
            ),
            ToolParameter(
                name="path",
                description="Path to the repository",
                param_type=ParameterType.PATH,
                required=False,
                default=".",
            ),
        ]

    @property
    def tags(self) -> List[str]:
        return ["git", "pull", "fetch", "update", "remote"]

    @property
    def examples(self) -> List[str]:
        return [
            "Pull the latest changes",
            "Pull from upstream main",
            "Update the repo",
        ]

    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        remote = parameters.get("remote", "origin")
        branch = parameters.get("branch")
        path = Path(parameters.get("path", "."))

        if not path.is_absolute():
            path = Path(context.current_directory) / path

        git_root = find_git_root(path)
        if not git_root:
            return ToolResult(
                success=False,
                error="Not a git repository",
            )

        # Build pull command
        cmd = ["pull", remote]
        if branch:
            cmd.append(branch)

        result = run_git_command(cmd, str(git_root))

        if result["success"]:
            stdout = result.get("stdout", "")
            if "Already up to date" in stdout:
                return ToolResult(
                    success=True,
                    message="Already up to date.",
                )
            return ToolResult(
                success=True,
                message=f"Pull complete.\n{stdout}",
                result={"stdout": stdout},
            )

        stderr = result.get("stderr", "")
        # Detect merge conflicts
        if "CONFLICT" in stderr or "CONFLICT" in result.get("stdout", ""):
            return ToolResult(
                success=False,
                error=(
                    f"Pull resulted in merge conflicts:\n{stderr}\n"
                    "Resolve conflicts manually or use 'resolve merge conflicts'."
                ),
            )
        return ToolResult(
            success=False,
            error=f"Pull failed: {stderr}",
        )


@register_tool
class GitPushTool(BaseTool):
    """Push commits to a remote repository.

    Handles common push failures automatically:
    - No upstream branch → auto-sets upstream with ``--set-upstream``
    - Rejected (non-fast-forward) → suggests ``git pull --rebase``
    """

    @property
    def name(self) -> str:
        return "git_push"

    @property
    def description(self) -> str:
        return (
            "Push local commits to a remote repository. "
            "Automatically handles 'no upstream branch' errors by "
            "setting the upstream. Requires consent."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT

    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.EXECUTE

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.MEDIUM

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="remote",
                description="Remote name (default: origin)",
                param_type=ParameterType.STRING,
                required=False,
                default="origin",
            ),
            ToolParameter(
                name="branch",
                description="Branch to push (default: current branch)",
                param_type=ParameterType.STRING,
                required=False,
            ),
            ToolParameter(
                name="force",
                description="Force push (--force-with-lease for safety)",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=False,
            ),
            ToolParameter(
                name="path",
                description="Path to the repository",
                param_type=ParameterType.PATH,
                required=False,
                default=".",
            ),
        ]

    @property
    def tags(self) -> List[str]:
        return ["git", "push", "remote", "publish"]

    @property
    def examples(self) -> List[str]:
        return [
            "Push changes to remote",
            "Push to origin main",
            "Publish my branch",
        ]

    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        remote = parameters.get("remote", "origin")
        branch = parameters.get("branch")
        force = parameters.get("force", False)
        path = Path(parameters.get("path", "."))

        if not path.is_absolute():
            path = Path(context.current_directory) / path

        git_root = find_git_root(path)
        if not git_root:
            return ToolResult(
                success=False,
                error="Not a git repository",
            )

        # Get current branch name if not specified
        if not branch:
            branch_result = run_git_command(
                ["rev-parse", "--abbrev-ref", "HEAD"], str(git_root)
            )
            branch = branch_result.get("stdout", "").strip() or None

        # Build push command
        cmd = ["push", remote]
        if branch:
            cmd.append(branch)
        if force:
            cmd.append("--force-with-lease")

        result = run_git_command(cmd, str(git_root))

        if result["success"]:
            return ToolResult(
                success=True,
                message=f"Pushed to {remote}/{branch or 'HEAD'} successfully.",
                result={
                    "remote": remote,
                    "branch": branch,
                },
            )

        stderr = result.get("stderr", "")

        # ── Auto-fix: no upstream branch ──────────────────────────────
        if (
            "no upstream branch" in stderr.lower()
            or "has no upstream branch" in stderr.lower()
            or "--set-upstream" in stderr
        ):
            if branch:
                retry = run_git_command(
                    ["push", "--set-upstream", remote, branch],
                    str(git_root),
                )
                if retry["success"]:
                    return ToolResult(
                        success=True,
                        message=(
                            f"Set upstream and pushed {branch} to {remote}."
                        ),
                        result={
                            "remote": remote,
                            "branch": branch,
                            "set_upstream": True,
                        },
                    )
                stderr = retry.get("stderr", stderr)

        # ── Non-fast-forward / rejected push ──────────────────────────
        if "rejected" in stderr.lower() or "non-fast-forward" in stderr.lower():
            return ToolResult(
                success=False,
                error=(
                    f"Push rejected (non-fast-forward). Remote has changes "
                    f"not present locally.\n{stderr}\n\n"
                    f"Suggested fix: run 'git pull --rebase' first, then push again."
                ),
            )

        # ── Authentication error ──────────────────────────────────────
        if "authentication" in stderr.lower() or "403" in stderr or "401" in stderr:
            return ToolResult(
                success=False,
                error=(
                    f"Push failed due to authentication error.\n{stderr}\n\n"
                    "Set up credentials with 'git credential-manager' or "
                    "a personal access token."
                ),
            )

        return ToolResult(
            success=False,
            error=f"Push failed: {stderr}",
        )
