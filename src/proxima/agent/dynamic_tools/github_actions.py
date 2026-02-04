"""GitHub Actions Integration for Dynamic AI Assistant.

This module implements Phase 4.3 for the Dynamic AI Assistant:
- Workflow Monitoring: List workflows, watch runs, get status
- Workflow Triggering: Trigger workflows, dispatch events
- Artifact Management: Download artifacts, manage workflow outputs

Key Features:
============
- Real-time workflow run monitoring
- Intelligent workflow triggering
- Artifact download and management
- Job and step analysis
- Log retrieval and parsing

Design Principle:
================
All workflow decisions use LLM reasoning for intelligent automation.
The LLM analyzes workflow patterns and suggests optimizations.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow run status."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ACTION_REQUIRED = "action_required"
    CANCELLED = "cancelled"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    SKIPPED = "skipped"
    STALE = "stale"
    SUCCESS = "success"
    TIMED_OUT = "timed_out"
    WAITING = "waiting"
    PENDING = "pending"
    REQUESTED = "requested"


class WorkflowConclusion(Enum):
    """Workflow run conclusion."""
    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    ACTION_REQUIRED = "action_required"
    STALE = "stale"
    STARTUP_FAILURE = "startup_failure"


class JobStatus(Enum):
    """Job status within a workflow run."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    WAITING = "waiting"


class ArtifactExpiration(Enum):
    """Artifact expiration status."""
    ACTIVE = "active"
    EXPIRED = "expired"


@dataclass
class WorkflowInfo:
    """Information about a GitHub Actions workflow."""
    id: int
    name: str
    path: str  # e.g., ".github/workflows/ci.yml"
    state: str = "active"  # active, disabled_manually, disabled_inactivity
    
    # URLs
    html_url: str = ""
    badge_url: str = ""
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "state": self.state,
            "html_url": self.html_url,
            "badge_url": self.badge_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class WorkflowRunInfo:
    """Information about a workflow run."""
    id: int
    name: str
    workflow_id: int
    
    # Status
    status: WorkflowStatus = WorkflowStatus.QUEUED
    conclusion: Optional[WorkflowConclusion] = None
    
    # Trigger
    event: str = ""  # push, pull_request, workflow_dispatch, etc.
    head_branch: str = ""
    head_sha: str = ""
    
    # URLs
    html_url: str = ""
    jobs_url: str = ""
    logs_url: str = ""
    artifacts_url: str = ""
    
    # Attempt info
    run_number: int = 0
    run_attempt: int = 1
    
    # Actor
    actor: Optional[str] = None
    triggering_actor: Optional[str] = None
    
    # Repository
    repository: Optional[str] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    run_started_at: Optional[datetime] = None
    
    @property
    def is_completed(self) -> bool:
        return self.status == WorkflowStatus.COMPLETED
    
    @property
    def is_successful(self) -> bool:
        return self.conclusion == WorkflowConclusion.SUCCESS
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.run_started_at and self.updated_at:
            return self.updated_at - self.run_started_at
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "conclusion": self.conclusion.value if self.conclusion else None,
            "event": self.event,
            "head_branch": self.head_branch,
            "head_sha": self.head_sha[:8] if self.head_sha else None,
            "html_url": self.html_url,
            "run_number": self.run_number,
            "actor": self.actor,
            "repository": self.repository,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
        }


@dataclass
class JobInfo:
    """Information about a job in a workflow run."""
    id: int
    run_id: int
    name: str
    
    # Status
    status: JobStatus = JobStatus.QUEUED
    conclusion: Optional[WorkflowConclusion] = None
    
    # URLs
    html_url: str = ""
    
    # Steps
    steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Runner
    runner_id: Optional[int] = None
    runner_name: Optional[str] = None
    runner_group_id: Optional[int] = None
    runner_group_name: Optional[str] = None
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status.value,
            "conclusion": self.conclusion.value if self.conclusion else None,
            "html_url": self.html_url,
            "steps_count": len(self.steps),
            "runner_name": self.runner_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
        }


@dataclass
class ArtifactInfo:
    """Information about a workflow artifact."""
    id: int
    name: str
    
    # Size
    size_in_bytes: int = 0
    
    # URLs
    archive_download_url: str = ""
    
    # Status
    expired: bool = False
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Associated workflow run
    workflow_run_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "size_bytes": self.size_in_bytes,
            "size_mb": round(self.size_in_bytes / (1024 * 1024), 2),
            "expired": self.expired,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "workflow_run_id": self.workflow_run_id,
        }


@dataclass
class WorkflowDispatchInput:
    """Input for workflow dispatch trigger."""
    name: str
    value: str
    type: str = "string"  # string, boolean, choice
    required: bool = False
    description: str = ""


class GitHubActionsAPI:
    """GitHub Actions API client."""
    
    def __init__(self, access_token: Optional[str] = None):
        """Initialize API client.
        
        Args:
            access_token: GitHub access token
        """
        self._access_token = access_token
        self._base_url = "https://api.github.com"
    
    def set_token(self, token: str):
        """Set or update access token."""
        self._access_token = token
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        return headers
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        raw_response: bool = False,
    ) -> Any:
        """Make an API request."""
        url = f"{self._base_url}{endpoint}"
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    url,
                    headers=self._get_headers(),
                    params=params,
                    json=json_data,
                ) as response:
                    if raw_response:
                        return await response.read(), response.status
                    
                    if response.status >= 400:
                        error_text = await response.text()
                        return {"error": error_text, "status": response.status}
                    
                    if response.status == 204:
                        return {"success": True}
                    
                    return await response.json()
                    
        except ImportError:
            # Fallback to requests
            import requests
            
            response = requests.request(
                method,
                url,
                headers=self._get_headers(),
                params=params,
                json=json_data,
            )
            
            if raw_response:
                return response.content, response.status_code
            
            if response.status_code >= 400:
                return {"error": response.text, "status": response.status_code}
            
            if response.status_code == 204:
                return {"success": True}
            
            return response.json()
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self._request("GET", endpoint, params=params)
    
    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self._request("POST", endpoint, json_data=data or {})
    
    async def delete(self, endpoint: str) -> Dict[str, Any]:
        return await self._request("DELETE", endpoint)
    
    async def get_raw(self, endpoint: str) -> Tuple[bytes, int]:
        return await self._request("GET", endpoint, raw_response=True)


class GitHubActionsManager:
    """GitHub Actions workflow manager.
    
    Uses LLM reasoning to:
    1. Monitor and analyze workflow runs
    2. Intelligently trigger workflows
    3. Manage and analyze artifacts
    4. Diagnose workflow failures
    
    Example:
        >>> actions = GitHubActionsManager(llm_client=client, auth=authenticator)
        >>> workflows = await actions.list_workflows("owner/repo")
        >>> runs = await actions.list_workflow_runs("owner/repo")
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        authenticator: Optional[Any] = None,
    ):
        """Initialize actions manager.
        
        Args:
            llm_client: LLM client for reasoning
            authenticator: GitHubAuthenticator instance
        """
        self._llm_client = llm_client
        self._authenticator = authenticator
        self._api_client = GitHubActionsAPI()
        
        # Monitoring state
        self._watched_runs: Dict[int, WorkflowRunInfo] = {}
        self._run_callbacks: Dict[int, List[Callable]] = {}
    
    def _get_access_token(self) -> Optional[str]:
        """Get current access token."""
        if self._authenticator:
            session = self._authenticator.get_current_session()
            if session:
                return session.token_info.access_token
        return None
    
    def _ensure_authenticated(self) -> bool:
        """Ensure user is authenticated."""
        token = self._get_access_token()
        if token:
            self._api_client.set_token(token)
            return True
        return False
    
    # ========== Workflow Management ==========
    
    async def list_workflows(
        self,
        repo_full_name: str,
        per_page: int = 30,
        page: int = 1,
    ) -> List[WorkflowInfo]:
        """List repository workflows.
        
        Args:
            repo_full_name: Repository name (owner/repo)
            per_page: Results per page
            page: Page number
            
        Returns:
            List of workflows
        """
        self._ensure_authenticated()
        
        result = await self._api_client.get(
            f"/repos/{repo_full_name}/actions/workflows",
            params={"per_page": per_page, "page": page},
        )
        
        if "error" in result:
            logger.error(f"Failed to list workflows: {result['error']}")
            return []
        
        workflows = []
        for w in result.get("workflows", []):
            workflow = WorkflowInfo(
                id=w.get("id"),
                name=w.get("name", ""),
                path=w.get("path", ""),
                state=w.get("state", "active"),
                html_url=w.get("html_url", ""),
                badge_url=w.get("badge_url", ""),
                created_at=datetime.fromisoformat(w["created_at"].replace("Z", "+00:00")) if w.get("created_at") else None,
                updated_at=datetime.fromisoformat(w["updated_at"].replace("Z", "+00:00")) if w.get("updated_at") else None,
            )
            workflows.append(workflow)
        
        return workflows
    
    async def get_workflow(
        self,
        repo_full_name: str,
        workflow_id: Union[int, str],
    ) -> Optional[WorkflowInfo]:
        """Get a specific workflow.
        
        Args:
            repo_full_name: Repository name
            workflow_id: Workflow ID or file name
            
        Returns:
            Workflow info
        """
        self._ensure_authenticated()
        
        result = await self._api_client.get(
            f"/repos/{repo_full_name}/actions/workflows/{workflow_id}"
        )
        
        if "error" in result:
            return None
        
        return WorkflowInfo(
            id=result.get("id"),
            name=result.get("name", ""),
            path=result.get("path", ""),
            state=result.get("state", "active"),
            html_url=result.get("html_url", ""),
            badge_url=result.get("badge_url", ""),
        )
    
    async def enable_workflow(
        self,
        repo_full_name: str,
        workflow_id: Union[int, str],
    ) -> bool:
        """Enable a workflow.
        
        Args:
            repo_full_name: Repository name
            workflow_id: Workflow ID or file name
            
        Returns:
            True if enabled
        """
        if not self._ensure_authenticated():
            return False
        
        result = await self._api_client._request(
            "PUT",
            f"/repos/{repo_full_name}/actions/workflows/{workflow_id}/enable",
        )
        
        return "error" not in result
    
    async def disable_workflow(
        self,
        repo_full_name: str,
        workflow_id: Union[int, str],
    ) -> bool:
        """Disable a workflow.
        
        Args:
            repo_full_name: Repository name
            workflow_id: Workflow ID or file name
            
        Returns:
            True if disabled
        """
        if not self._ensure_authenticated():
            return False
        
        result = await self._api_client._request(
            "PUT",
            f"/repos/{repo_full_name}/actions/workflows/{workflow_id}/disable",
        )
        
        return "error" not in result
    
    # ========== Workflow Runs ==========
    
    async def list_workflow_runs(
        self,
        repo_full_name: str,
        workflow_id: Optional[Union[int, str]] = None,
        branch: Optional[str] = None,
        event: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
        actor: Optional[str] = None,
        per_page: int = 30,
        page: int = 1,
    ) -> List[WorkflowRunInfo]:
        """List workflow runs.
        
        Args:
            repo_full_name: Repository name
            workflow_id: Filter by workflow
            branch: Filter by branch
            event: Filter by event type
            status: Filter by status
            actor: Filter by actor
            per_page: Results per page
            page: Page number
            
        Returns:
            List of workflow runs
        """
        self._ensure_authenticated()
        
        params = {"per_page": per_page, "page": page}
        
        if branch:
            params["branch"] = branch
        if event:
            params["event"] = event
        if status:
            params["status"] = status.value
        if actor:
            params["actor"] = actor
        
        if workflow_id:
            endpoint = f"/repos/{repo_full_name}/actions/workflows/{workflow_id}/runs"
        else:
            endpoint = f"/repos/{repo_full_name}/actions/runs"
        
        result = await self._api_client.get(endpoint, params=params)
        
        if "error" in result:
            logger.error(f"Failed to list workflow runs: {result['error']}")
            return []
        
        runs = []
        for r in result.get("workflow_runs", []):
            run = self._parse_workflow_run(r)
            run.repository = repo_full_name
            runs.append(run)
        
        return runs
    
    async def get_workflow_run(
        self,
        repo_full_name: str,
        run_id: int,
    ) -> Optional[WorkflowRunInfo]:
        """Get a specific workflow run.
        
        Args:
            repo_full_name: Repository name
            run_id: Run ID
            
        Returns:
            Workflow run info
        """
        self._ensure_authenticated()
        
        result = await self._api_client.get(
            f"/repos/{repo_full_name}/actions/runs/{run_id}"
        )
        
        if "error" in result:
            return None
        
        run = self._parse_workflow_run(result)
        run.repository = repo_full_name
        return run
    
    async def cancel_workflow_run(
        self,
        repo_full_name: str,
        run_id: int,
    ) -> bool:
        """Cancel a workflow run.
        
        Args:
            repo_full_name: Repository name
            run_id: Run ID
            
        Returns:
            True if cancelled
        """
        if not self._ensure_authenticated():
            return False
        
        result = await self._api_client.post(
            f"/repos/{repo_full_name}/actions/runs/{run_id}/cancel"
        )
        
        return "error" not in result
    
    async def rerun_workflow(
        self,
        repo_full_name: str,
        run_id: int,
        enable_debug_logging: bool = False,
    ) -> bool:
        """Re-run a workflow.
        
        Args:
            repo_full_name: Repository name
            run_id: Run ID
            enable_debug_logging: Enable debug logging
            
        Returns:
            True if re-run started
        """
        if not self._ensure_authenticated():
            return False
        
        data = {}
        if enable_debug_logging:
            data["enable_debug_logging"] = True
        
        result = await self._api_client.post(
            f"/repos/{repo_full_name}/actions/runs/{run_id}/rerun",
            data if data else None,
        )
        
        return "error" not in result
    
    async def rerun_failed_jobs(
        self,
        repo_full_name: str,
        run_id: int,
    ) -> bool:
        """Re-run only failed jobs in a workflow run.
        
        Args:
            repo_full_name: Repository name
            run_id: Run ID
            
        Returns:
            True if re-run started
        """
        if not self._ensure_authenticated():
            return False
        
        result = await self._api_client.post(
            f"/repos/{repo_full_name}/actions/runs/{run_id}/rerun-failed-jobs"
        )
        
        return "error" not in result
    
    async def delete_workflow_run(
        self,
        repo_full_name: str,
        run_id: int,
    ) -> bool:
        """Delete a workflow run.
        
        Args:
            repo_full_name: Repository name
            run_id: Run ID
            
        Returns:
            True if deleted
        """
        if not self._ensure_authenticated():
            return False
        
        result = await self._api_client.delete(
            f"/repos/{repo_full_name}/actions/runs/{run_id}"
        )
        
        return "error" not in result
    
    # ========== Workflow Triggering ==========
    
    async def trigger_workflow(
        self,
        repo_full_name: str,
        workflow_id: Union[int, str],
        ref: str,
        inputs: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Trigger a workflow dispatch event.
        
        Args:
            repo_full_name: Repository name
            workflow_id: Workflow ID or file name
            ref: Branch or tag reference
            inputs: Workflow inputs
            
        Returns:
            Trigger result
        """
        if not self._ensure_authenticated():
            return {"success": False, "error": "Not authenticated"}
        
        data = {"ref": ref}
        if inputs:
            data["inputs"] = inputs
        
        result = await self._api_client.post(
            f"/repos/{repo_full_name}/actions/workflows/{workflow_id}/dispatches",
            data,
        )
        
        if "error" in result:
            return {"success": False, "error": result["error"]}
        
        return {"success": True, "message": "Workflow dispatch triggered"}
    
    async def trigger_repository_dispatch(
        self,
        repo_full_name: str,
        event_type: str,
        client_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Trigger a repository dispatch event.
        
        Args:
            repo_full_name: Repository name
            event_type: Event type
            client_payload: Custom payload
            
        Returns:
            Dispatch result
        """
        if not self._ensure_authenticated():
            return {"success": False, "error": "Not authenticated"}
        
        data = {"event_type": event_type}
        if client_payload:
            data["client_payload"] = client_payload
        
        result = await self._api_client.post(
            f"/repos/{repo_full_name}/dispatches",
            data,
        )
        
        if "error" in result:
            return {"success": False, "error": result["error"]}
        
        return {"success": True, "message": f"Repository dispatch '{event_type}' triggered"}
    
    # ========== Jobs ==========
    
    async def list_jobs_for_workflow_run(
        self,
        repo_full_name: str,
        run_id: int,
        filter_status: Optional[str] = None,
    ) -> List[JobInfo]:
        """List jobs for a workflow run.
        
        Args:
            repo_full_name: Repository name
            run_id: Workflow run ID
            filter_status: Filter by status (latest, all)
            
        Returns:
            List of jobs
        """
        self._ensure_authenticated()
        
        params = {}
        if filter_status:
            params["filter"] = filter_status
        
        result = await self._api_client.get(
            f"/repos/{repo_full_name}/actions/runs/{run_id}/jobs",
            params=params if params else None,
        )
        
        if "error" in result:
            return []
        
        jobs = []
        for j in result.get("jobs", []):
            job = self._parse_job(j)
            jobs.append(job)
        
        return jobs
    
    async def get_job(
        self,
        repo_full_name: str,
        job_id: int,
    ) -> Optional[JobInfo]:
        """Get a specific job.
        
        Args:
            repo_full_name: Repository name
            job_id: Job ID
            
        Returns:
            Job info
        """
        self._ensure_authenticated()
        
        result = await self._api_client.get(
            f"/repos/{repo_full_name}/actions/jobs/{job_id}"
        )
        
        if "error" in result:
            return None
        
        return self._parse_job(result)
    
    async def get_job_logs(
        self,
        repo_full_name: str,
        job_id: int,
    ) -> Optional[str]:
        """Get logs for a job.
        
        Args:
            repo_full_name: Repository name
            job_id: Job ID
            
        Returns:
            Job logs as string
        """
        self._ensure_authenticated()
        
        try:
            content, status = await self._api_client.get_raw(
                f"/repos/{repo_full_name}/actions/jobs/{job_id}/logs"
            )
            
            if status == 200:
                return content.decode("utf-8")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get job logs: {e}")
            return None
    
    async def get_workflow_run_logs(
        self,
        repo_full_name: str,
        run_id: int,
        download_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Download logs for a workflow run.
        
        Args:
            repo_full_name: Repository name
            run_id: Workflow run ID
            download_path: Path to save logs
            
        Returns:
            Download result
        """
        self._ensure_authenticated()
        
        try:
            content, status = await self._api_client.get_raw(
                f"/repos/{repo_full_name}/actions/runs/{run_id}/logs"
            )
            
            if status != 200:
                return {"success": False, "error": "Failed to download logs"}
            
            if download_path:
                download_path = Path(download_path)
                download_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(download_path, "wb") as f:
                    f.write(content)
                
                return {"success": True, "path": str(download_path)}
            else:
                # Return as zip file contents info
                try:
                    with zipfile.ZipFile(io.BytesIO(content)) as zf:
                        return {
                            "success": True,
                            "files": zf.namelist(),
                            "size_bytes": len(content),
                        }
                except zipfile.BadZipFile:
                    return {"success": True, "size_bytes": len(content)}
                    
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ========== Artifacts ==========
    
    async def list_artifacts(
        self,
        repo_full_name: str,
        run_id: Optional[int] = None,
        per_page: int = 30,
    ) -> List[ArtifactInfo]:
        """List artifacts.
        
        Args:
            repo_full_name: Repository name
            run_id: Optional workflow run ID filter
            per_page: Results per page
            
        Returns:
            List of artifacts
        """
        self._ensure_authenticated()
        
        if run_id:
            endpoint = f"/repos/{repo_full_name}/actions/runs/{run_id}/artifacts"
        else:
            endpoint = f"/repos/{repo_full_name}/actions/artifacts"
        
        result = await self._api_client.get(endpoint, params={"per_page": per_page})
        
        if "error" in result:
            return []
        
        artifacts = []
        for a in result.get("artifacts", []):
            artifact = self._parse_artifact(a)
            artifacts.append(artifact)
        
        return artifacts
    
    async def get_artifact(
        self,
        repo_full_name: str,
        artifact_id: int,
    ) -> Optional[ArtifactInfo]:
        """Get a specific artifact.
        
        Args:
            repo_full_name: Repository name
            artifact_id: Artifact ID
            
        Returns:
            Artifact info
        """
        self._ensure_authenticated()
        
        result = await self._api_client.get(
            f"/repos/{repo_full_name}/actions/artifacts/{artifact_id}"
        )
        
        if "error" in result:
            return None
        
        return self._parse_artifact(result)
    
    async def download_artifact(
        self,
        repo_full_name: str,
        artifact_id: int,
        download_path: Path,
        extract: bool = True,
    ) -> Dict[str, Any]:
        """Download an artifact.
        
        Args:
            repo_full_name: Repository name
            artifact_id: Artifact ID
            download_path: Download destination
            extract: Extract zip contents
            
        Returns:
            Download result
        """
        self._ensure_authenticated()
        
        try:
            content, status = await self._api_client.get_raw(
                f"/repos/{repo_full_name}/actions/artifacts/{artifact_id}/zip"
            )
            
            if status != 200:
                return {"success": False, "error": "Failed to download artifact"}
            
            download_path = Path(download_path)
            download_path.parent.mkdir(parents=True, exist_ok=True)
            
            if extract:
                # Extract zip contents
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    zf.extractall(download_path)
                
                return {
                    "success": True,
                    "path": str(download_path),
                    "files": os.listdir(download_path),
                }
            else:
                # Save as zip
                zip_path = download_path.with_suffix(".zip")
                with open(zip_path, "wb") as f:
                    f.write(content)
                
                return {"success": True, "path": str(zip_path)}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def delete_artifact(
        self,
        repo_full_name: str,
        artifact_id: int,
    ) -> bool:
        """Delete an artifact.
        
        Args:
            repo_full_name: Repository name
            artifact_id: Artifact ID
            
        Returns:
            True if deleted
        """
        if not self._ensure_authenticated():
            return False
        
        result = await self._api_client.delete(
            f"/repos/{repo_full_name}/actions/artifacts/{artifact_id}"
        )
        
        return "error" not in result
    
    # ========== Monitoring ==========
    
    async def watch_workflow_run(
        self,
        repo_full_name: str,
        run_id: int,
        poll_interval: int = 30,
        timeout: int = 3600,
        callback: Optional[Callable[[WorkflowRunInfo], None]] = None,
    ) -> WorkflowRunInfo:
        """Watch a workflow run until completion.
        
        Args:
            repo_full_name: Repository name
            run_id: Run ID to watch
            poll_interval: Seconds between polls
            timeout: Max seconds to wait
            callback: Optional callback for status updates
            
        Returns:
            Final run status
        """
        start_time = datetime.now()
        
        while True:
            run = await self.get_workflow_run(repo_full_name, run_id)
            
            if not run:
                raise ValueError(f"Workflow run {run_id} not found")
            
            # Update watched runs
            self._watched_runs[run_id] = run
            
            # Call callback if provided
            if callback:
                callback(run)
            
            # Check if completed
            if run.is_completed:
                return run
            
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                raise TimeoutError(f"Workflow run {run_id} timed out after {timeout}s")
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
    
    async def get_workflow_run_summary(
        self,
        repo_full_name: str,
        run_id: int,
    ) -> Dict[str, Any]:
        """Get a summary of a workflow run.
        
        Args:
            repo_full_name: Repository name
            run_id: Run ID
            
        Returns:
            Run summary with jobs and status
        """
        run = await self.get_workflow_run(repo_full_name, run_id)
        if not run:
            return {"error": "Run not found"}
        
        jobs = await self.list_jobs_for_workflow_run(repo_full_name, run_id)
        artifacts = await self.list_artifacts(repo_full_name, run_id)
        
        summary = {
            "run": run.to_dict(),
            "jobs": [j.to_dict() for j in jobs],
            "artifacts": [a.to_dict() for a in artifacts],
            "stats": {
                "total_jobs": len(jobs),
                "completed_jobs": sum(1 for j in jobs if j.status == JobStatus.COMPLETED),
                "failed_jobs": sum(1 for j in jobs if j.conclusion == WorkflowConclusion.FAILURE),
                "total_artifacts": len(artifacts),
                "total_artifact_size_mb": round(
                    sum(a.size_in_bytes for a in artifacts) / (1024 * 1024), 2
                ),
            },
        }
        
        return summary
    
    # ========== LLM Integration ==========
    
    async def analyze_workflow_failure(
        self,
        repo_full_name: str,
        run_id: int,
    ) -> Dict[str, Any]:
        """Analyze a failed workflow run using LLM.
        
        Args:
            repo_full_name: Repository name
            run_id: Run ID
            
        Returns:
            Analysis with diagnosis and suggestions
        """
        run = await self.get_workflow_run(repo_full_name, run_id)
        if not run:
            return {"error": "Run not found"}
        
        if run.conclusion != WorkflowConclusion.FAILURE:
            return {"error": "Run did not fail", "conclusion": run.conclusion.value if run.conclusion else None}
        
        # Get failed jobs
        jobs = await self.list_jobs_for_workflow_run(repo_full_name, run_id)
        failed_jobs = [j for j in jobs if j.conclusion == WorkflowConclusion.FAILURE]
        
        # Get logs for failed jobs
        logs_info = []
        for job in failed_jobs[:3]:  # Limit to 3 jobs
            logs = await self.get_job_logs(repo_full_name, job.id)
            if logs:
                # Get last 100 lines
                log_lines = logs.strip().split("\n")[-100:]
                logs_info.append({
                    "job": job.name,
                    "logs": "\n".join(log_lines),
                })
        
        analysis = {
            "run": run.to_dict(),
            "failed_jobs": [j.to_dict() for j in failed_jobs],
            "logs_available": len(logs_info),
        }
        
        if self._llm_client and logs_info:
            # Use LLM to analyze failure
            prompt = f"""Analyze this GitHub Actions workflow failure.

Workflow: {run.name}
Event: {run.event}
Branch: {run.head_branch}
Status: {run.status.value}
Conclusion: {run.conclusion.value if run.conclusion else 'N/A'}

Failed jobs:
{json.dumps([j.to_dict() for j in failed_jobs], indent=2)}

Logs from first failed job ({logs_info[0]['job']}):
```
{logs_info[0]['logs'][:2000]}
```

Provide:
1. Root cause analysis
2. Specific error identified
3. Suggested fix
4. Prevention recommendations
"""
            
            try:
                llm_analysis = await self._llm_client.generate(prompt)
                analysis["llm_diagnosis"] = llm_analysis
            except Exception as e:
                analysis["llm_diagnosis"] = f"Analysis failed: {e}"
        
        return analysis
    
    async def suggest_workflow_improvements(
        self,
        repo_full_name: str,
    ) -> Dict[str, Any]:
        """Suggest workflow improvements using LLM.
        
        Args:
            repo_full_name: Repository name
            
        Returns:
            Improvement suggestions
        """
        # Get recent workflow runs
        runs = await self.list_workflow_runs(repo_full_name, per_page=20)
        workflows = await self.list_workflows(repo_full_name)
        
        # Calculate statistics
        success_count = sum(1 for r in runs if r.conclusion == WorkflowConclusion.SUCCESS)
        failure_count = sum(1 for r in runs if r.conclusion == WorkflowConclusion.FAILURE)
        
        avg_duration = None
        durations = [r.duration.total_seconds() for r in runs if r.duration]
        if durations:
            avg_duration = sum(durations) / len(durations)
        
        stats = {
            "total_runs": len(runs),
            "success_rate": success_count / len(runs) if runs else 0,
            "failure_rate": failure_count / len(runs) if runs else 0,
            "avg_duration_seconds": avg_duration,
            "workflows_count": len(workflows),
        }
        
        suggestions = {
            "stats": stats,
            "workflows": [w.to_dict() for w in workflows],
        }
        
        # Format average duration
        avg_duration_str = f"{avg_duration/60:.1f} minutes" if avg_duration else "N/A"
        
        if self._llm_client:
            prompt = f"""Analyze these GitHub Actions workflows and suggest improvements.

Repository: {repo_full_name}

Workflows:
{json.dumps([w.to_dict() for w in workflows], indent=2)}

Recent runs statistics:
- Total runs: {stats['total_runs']}
- Success rate: {stats['success_rate']*100:.1f}%
- Failure rate: {stats['failure_rate']*100:.1f}%
- Average duration: {avg_duration_str}

Recent run events: {set(r.event for r in runs)}
Recent branches: {set(r.head_branch for r in runs[:10])}

Provide specific suggestions for:
1. Improving reliability (reduce failures)
2. Optimizing performance (reduce duration)
3. Better workflow organization
4. Security improvements
5. Cost optimization (if using paid runners)
"""
            
            try:
                llm_suggestions = await self._llm_client.generate(prompt)
                suggestions["llm_suggestions"] = llm_suggestions
            except Exception as e:
                suggestions["llm_suggestions"] = f"Analysis failed: {e}"
        
        return suggestions
    
    # ========== Helper Methods ==========
    
    def _parse_workflow_run(self, data: Dict[str, Any]) -> WorkflowRunInfo:
        """Parse workflow run API response."""
        status = WorkflowStatus.QUEUED
        try:
            status = WorkflowStatus(data.get("status", "queued"))
        except ValueError:
            pass
        
        conclusion = None
        if data.get("conclusion"):
            try:
                conclusion = WorkflowConclusion(data["conclusion"])
            except ValueError:
                pass
        
        return WorkflowRunInfo(
            id=data.get("id"),
            name=data.get("name", ""),
            workflow_id=data.get("workflow_id"),
            status=status,
            conclusion=conclusion,
            event=data.get("event", ""),
            head_branch=data.get("head_branch", ""),
            head_sha=data.get("head_sha", ""),
            html_url=data.get("html_url", ""),
            jobs_url=data.get("jobs_url", ""),
            logs_url=data.get("logs_url", ""),
            artifacts_url=data.get("artifacts_url", ""),
            run_number=data.get("run_number", 0),
            run_attempt=data.get("run_attempt", 1),
            actor=data.get("actor", {}).get("login") if data.get("actor") else None,
            triggering_actor=data.get("triggering_actor", {}).get("login") if data.get("triggering_actor") else None,
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")) if data.get("updated_at") else None,
            run_started_at=datetime.fromisoformat(data["run_started_at"].replace("Z", "+00:00")) if data.get("run_started_at") else None,
        )
    
    def _parse_job(self, data: Dict[str, Any]) -> JobInfo:
        """Parse job API response."""
        status = JobStatus.QUEUED
        try:
            status = JobStatus(data.get("status", "queued"))
        except ValueError:
            pass
        
        conclusion = None
        if data.get("conclusion"):
            try:
                conclusion = WorkflowConclusion(data["conclusion"])
            except ValueError:
                pass
        
        steps = []
        for step in data.get("steps", []):
            steps.append({
                "name": step.get("name"),
                "status": step.get("status"),
                "conclusion": step.get("conclusion"),
                "number": step.get("number"),
            })
        
        return JobInfo(
            id=data.get("id"),
            run_id=data.get("run_id"),
            name=data.get("name", ""),
            status=status,
            conclusion=conclusion,
            html_url=data.get("html_url", ""),
            steps=steps,
            runner_id=data.get("runner_id"),
            runner_name=data.get("runner_name"),
            runner_group_id=data.get("runner_group_id"),
            runner_group_name=data.get("runner_group_name"),
            started_at=datetime.fromisoformat(data["started_at"].replace("Z", "+00:00")) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"].replace("Z", "+00:00")) if data.get("completed_at") else None,
        )
    
    def _parse_artifact(self, data: Dict[str, Any]) -> ArtifactInfo:
        """Parse artifact API response."""
        return ArtifactInfo(
            id=data.get("id"),
            name=data.get("name", ""),
            size_in_bytes=data.get("size_in_bytes", 0),
            archive_download_url=data.get("archive_download_url", ""),
            expired=data.get("expired", False),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")) if data.get("updated_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00")) if data.get("expires_at") else None,
            workflow_run_id=data.get("workflow_run", {}).get("id") if data.get("workflow_run") else None,
        )


# Module-level instance
_global_actions_manager: Optional[GitHubActionsManager] = None


def get_github_actions_manager(
    llm_client: Optional[Any] = None,
    authenticator: Optional[Any] = None,
) -> GitHubActionsManager:
    """Get the global GitHub Actions manager.
    
    Args:
        llm_client: Optional LLM client
        authenticator: Optional authenticator
        
    Returns:
        GitHubActionsManager instance
    """
    global _global_actions_manager
    if _global_actions_manager is None:
        _global_actions_manager = GitHubActionsManager(
            llm_client=llm_client,
            authenticator=authenticator,
        )
    return _global_actions_manager
