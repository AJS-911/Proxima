"""GitHub Repository Operations for Dynamic AI Assistant.

This module implements Phase 4.2 for the Dynamic AI Assistant:
- Repository Discovery and Search: Find repositories based on criteria
- Repository Cloning and Management: Clone, fork, create repositories
- Repository Information Retrieval: Metadata, stats, contributors
- Issue and PR Management: Create, update, search issues and PRs
- Release and Asset Management: Manage releases and release assets

Key Features:
============
- Comprehensive repository operations
- LLM-guided search and discovery
- Issue and PR workflow automation
- Release management with asset handling
- Repository statistics and insights

Design Principle:
================
All repository operations use LLM reasoning for intelligent decisions.
The LLM analyzes repositories and suggests actions dynamically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

logger = logging.getLogger(__name__)


class IssueState(Enum):
    """Issue/PR state."""
    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


class PRMergeMethod(Enum):
    """Pull request merge methods."""
    MERGE = "merge"
    SQUASH = "squash"
    REBASE = "rebase"


class RepoVisibility(Enum):
    """Repository visibility."""
    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"


class SortDirection(Enum):
    """Sort direction."""
    ASC = "asc"
    DESC = "desc"


class RepoSearchSort(Enum):
    """Repository search sort criteria."""
    STARS = "stars"
    FORKS = "forks"
    HELP_WANTED_ISSUES = "help-wanted-issues"
    UPDATED = "updated"


@dataclass
class RepositoryInfo:
    """Information about a GitHub repository."""
    owner: str
    name: str
    full_name: str
    
    # Basic info
    description: Optional[str] = None
    url: str = ""
    html_url: str = ""
    clone_url: str = ""
    ssh_url: str = ""
    
    # Stats
    stars: int = 0
    forks: int = 0
    watchers: int = 0
    open_issues: int = 0
    size_kb: int = 0
    
    # Metadata
    language: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    license_name: Optional[str] = None
    
    # Settings
    visibility: RepoVisibility = RepoVisibility.PUBLIC
    is_fork: bool = False
    is_archived: bool = False
    is_template: bool = False
    has_issues: bool = True
    has_wiki: bool = True
    has_discussions: bool = False
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    pushed_at: Optional[datetime] = None
    
    # Default branch
    default_branch: str = "main"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "owner": self.owner,
            "name": self.name,
            "full_name": self.full_name,
            "description": self.description,
            "url": self.url,
            "html_url": self.html_url,
            "clone_url": self.clone_url,
            "stars": self.stars,
            "forks": self.forks,
            "language": self.language,
            "topics": self.topics,
            "visibility": self.visibility.value,
            "is_fork": self.is_fork,
            "default_branch": self.default_branch,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class IssueInfo:
    """Information about a GitHub issue or PR."""
    number: int
    title: str
    
    # Content
    body: Optional[str] = None
    state: IssueState = IssueState.OPEN
    
    # Type
    is_pull_request: bool = False
    
    # Labels and assignments
    labels: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    milestone: Optional[str] = None
    
    # Author and repo
    author: Optional[str] = None
    repository: Optional[str] = None
    
    # URLs
    html_url: str = ""
    api_url: str = ""
    
    # PR specific
    pr_merged: bool = False
    pr_mergeable: Optional[bool] = None
    pr_draft: bool = False
    head_branch: Optional[str] = None
    base_branch: Optional[str] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    # Comments
    comments_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "state": self.state.value,
            "is_pull_request": self.is_pull_request,
            "labels": self.labels,
            "assignees": self.assignees,
            "author": self.author,
            "html_url": self.html_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "comments_count": self.comments_count,
        }


@dataclass
class ReleaseInfo:
    """Information about a GitHub release."""
    tag_name: str
    name: str
    
    # Content
    body: Optional[str] = None
    
    # Status
    draft: bool = False
    prerelease: bool = False
    
    # Author
    author: Optional[str] = None
    
    # URLs
    html_url: str = ""
    tarball_url: str = ""
    zipball_url: str = ""
    
    # Assets
    assets: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timestamps
    created_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    
    # IDs
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tag_name": self.tag_name,
            "name": self.name,
            "body": self.body,
            "draft": self.draft,
            "prerelease": self.prerelease,
            "author": self.author,
            "html_url": self.html_url,
            "assets": self.assets,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "published_at": self.published_at.isoformat() if self.published_at else None,
        }


@dataclass
class SearchResult:
    """Search result with items and metadata."""
    total_count: int
    items: List[Any]
    incomplete_results: bool = False
    search_query: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_count": self.total_count,
            "incomplete_results": self.incomplete_results,
            "search_query": self.search_query,
            "items": [item.to_dict() if hasattr(item, 'to_dict') else item for item in self.items],
        }


class GitHubAPIClient:
    """GitHub API client for repository operations."""
    
    def __init__(self, access_token: Optional[str] = None):
        """Initialize API client.
        
        Args:
            access_token: GitHub access token
        """
        self._access_token = access_token
        self._base_url = "https://api.github.com"
        self._session = None
    
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
    ) -> Dict[str, Any]:
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
            
            if response.status_code >= 400:
                return {"error": response.text, "status": response.status_code}
            
            if response.status_code == 204:
                return {"success": True}
            
            return response.json()
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self._request("GET", endpoint, params=params)
    
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", endpoint, json_data=data)
    
    async def patch(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("PATCH", endpoint, json_data=data)
    
    async def delete(self, endpoint: str) -> Dict[str, Any]:
        return await self._request("DELETE", endpoint)


class GitHubRepoOperations:
    """GitHub repository operations manager.
    
    Uses LLM reasoning to:
    1. Search and discover relevant repositories
    2. Analyze repository structure and content
    3. Manage issues and pull requests intelligently
    4. Handle releases and versioning
    
    Example:
        >>> ops = GitHubRepoOperations(llm_client=client, auth=authenticator)
        >>> repos = await ops.search_repositories("quantum computing python")
        >>> info = await ops.get_repository_info("owner/repo")
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        authenticator: Optional[Any] = None,
    ):
        """Initialize repository operations.
        
        Args:
            llm_client: LLM client for reasoning
            authenticator: GitHubAuthenticator instance
        """
        self._llm_client = llm_client
        self._authenticator = authenticator
        self._api_client = GitHubAPIClient()
        
        # Cache for repository info
        self._repo_cache: Dict[str, RepositoryInfo] = {}
        self._cache_ttl = timedelta(minutes=10)
        self._cache_timestamps: Dict[str, datetime] = {}
    
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
    
    # ========== Repository Search ==========
    
    async def search_repositories(
        self,
        query: str,
        sort: Optional[RepoSearchSort] = None,
        order: SortDirection = SortDirection.DESC,
        per_page: int = 30,
        page: int = 1,
        language: Optional[str] = None,
        user: Optional[str] = None,
        org: Optional[str] = None,
    ) -> SearchResult:
        """Search for repositories.
        
        Args:
            query: Search query
            sort: Sort criteria
            order: Sort direction
            per_page: Results per page
            page: Page number
            language: Filter by language
            user: Filter by user
            org: Filter by organization
            
        Returns:
            Search results
        """
        self._ensure_authenticated()
        
        # Build query with filters
        full_query = query
        if language:
            full_query += f" language:{language}"
        if user:
            full_query += f" user:{user}"
        if org:
            full_query += f" org:{org}"
        
        params = {
            "q": full_query,
            "per_page": min(per_page, 100),
            "page": page,
            "order": order.value,
        }
        
        if sort:
            params["sort"] = sort.value
        
        result = await self._api_client.get("/search/repositories", params=params)
        
        if "error" in result:
            logger.error(f"Search failed: {result['error']}")
            return SearchResult(total_count=0, items=[], search_query=full_query)
        
        items = []
        for repo in result.get("items", []):
            repo_info = self._parse_repository(repo)
            items.append(repo_info)
        
        return SearchResult(
            total_count=result.get("total_count", 0),
            items=items,
            incomplete_results=result.get("incomplete_results", False),
            search_query=full_query,
        )
    
    async def search_repositories_with_llm(
        self,
        natural_query: str,
    ) -> SearchResult:
        """Search repositories using LLM to interpret natural language.
        
        Args:
            natural_query: Natural language search request
            
        Returns:
            Search results
        """
        if self._llm_client is None:
            # Fallback to direct search
            return await self.search_repositories(natural_query)
        
        prompt = f"""Convert this natural language repository search into GitHub search syntax.

Natural query: {natural_query}

GitHub search supports:
- language:python, language:javascript etc.
- stars:>100, stars:50..100
- forks:>10
- user:username or org:orgname
- topic:topic-name
- license:mit, license:apache-2.0
- created:>2024-01-01
- pushed:>2024-01-01
- is:public or is:private

Return ONLY the search query string, nothing else.
Example: "machine learning language:python stars:>100"
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            search_query = response.strip().strip('"')
            
            return await self.search_repositories(search_query)
            
        except Exception as e:
            logger.warning(f"LLM search failed, using direct query: {e}")
            return await self.search_repositories(natural_query)
    
    # ========== Repository Information ==========
    
    async def get_repository_info(
        self,
        repo_full_name: str,
        use_cache: bool = True,
    ) -> Optional[RepositoryInfo]:
        """Get detailed repository information.
        
        Args:
            repo_full_name: Full repository name (owner/repo)
            use_cache: Whether to use cached data
            
        Returns:
            Repository information
        """
        # Check cache
        if use_cache and repo_full_name in self._repo_cache:
            cache_time = self._cache_timestamps.get(repo_full_name)
            if cache_time and datetime.now() - cache_time < self._cache_ttl:
                return self._repo_cache[repo_full_name]
        
        self._ensure_authenticated()
        
        result = await self._api_client.get(f"/repos/{repo_full_name}")
        
        if "error" in result:
            logger.error(f"Failed to get repo info: {result['error']}")
            return None
        
        repo_info = self._parse_repository(result)
        
        # Update cache
        self._repo_cache[repo_full_name] = repo_info
        self._cache_timestamps[repo_full_name] = datetime.now()
        
        return repo_info
    
    async def get_repository_languages(self, repo_full_name: str) -> Dict[str, int]:
        """Get repository language breakdown.
        
        Args:
            repo_full_name: Full repository name
            
        Returns:
            Dict of language -> bytes
        """
        self._ensure_authenticated()
        
        result = await self._api_client.get(f"/repos/{repo_full_name}/languages")
        
        if "error" in result:
            return {}
        
        return result
    
    async def get_repository_contributors(
        self,
        repo_full_name: str,
        per_page: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get repository contributors.
        
        Args:
            repo_full_name: Full repository name
            per_page: Results per page
            
        Returns:
            List of contributors
        """
        self._ensure_authenticated()
        
        result = await self._api_client.get(
            f"/repos/{repo_full_name}/contributors",
            params={"per_page": per_page},
        )
        
        if "error" in result or not isinstance(result, list):
            return []
        
        return [
            {
                "login": c.get("login"),
                "contributions": c.get("contributions"),
                "avatar_url": c.get("avatar_url"),
                "html_url": c.get("html_url"),
            }
            for c in result
        ]
    
    async def get_repository_readme(self, repo_full_name: str) -> Optional[str]:
        """Get repository README content.
        
        Args:
            repo_full_name: Full repository name
            
        Returns:
            README content
        """
        self._ensure_authenticated()
        
        result = await self._api_client.get(f"/repos/{repo_full_name}/readme")
        
        if "error" in result:
            return None
        
        import base64
        content = result.get("content", "")
        if content:
            try:
                return base64.b64decode(content).decode("utf-8")
            except Exception:
                return content
        
        return None
    
    async def get_repository_topics(self, repo_full_name: str) -> List[str]:
        """Get repository topics.
        
        Args:
            repo_full_name: Full repository name
            
        Returns:
            List of topics
        """
        self._ensure_authenticated()
        
        # Set accept header for topics preview
        result = await self._api_client.get(f"/repos/{repo_full_name}/topics")
        
        if "error" in result:
            return []
        
        return result.get("names", [])
    
    # ========== Repository Management ==========
    
    async def create_repository(
        self,
        name: str,
        description: Optional[str] = None,
        visibility: RepoVisibility = RepoVisibility.PUBLIC,
        auto_init: bool = True,
        gitignore_template: Optional[str] = None,
        license_template: Optional[str] = None,
    ) -> Optional[RepositoryInfo]:
        """Create a new repository.
        
        Args:
            name: Repository name
            description: Repository description
            visibility: Public or private
            auto_init: Initialize with README
            gitignore_template: Gitignore template name
            license_template: License template name
            
        Returns:
            Created repository info
        """
        if not self._ensure_authenticated():
            logger.error("Not authenticated")
            return None
        
        data = {
            "name": name,
            "private": visibility == RepoVisibility.PRIVATE,
            "auto_init": auto_init,
        }
        
        if description:
            data["description"] = description
        if gitignore_template:
            data["gitignore_template"] = gitignore_template
        if license_template:
            data["license_template"] = license_template
        
        result = await self._api_client.post("/user/repos", data)
        
        if "error" in result:
            logger.error(f"Failed to create repository: {result['error']}")
            return None
        
        return self._parse_repository(result)
    
    async def fork_repository(
        self,
        repo_full_name: str,
        organization: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Optional[RepositoryInfo]:
        """Fork a repository.
        
        Args:
            repo_full_name: Repository to fork
            organization: Target organization
            name: New repository name
            
        Returns:
            Forked repository info
        """
        if not self._ensure_authenticated():
            return None
        
        data = {}
        if organization:
            data["organization"] = organization
        if name:
            data["name"] = name
        
        result = await self._api_client.post(f"/repos/{repo_full_name}/forks", data)
        
        if "error" in result:
            logger.error(f"Failed to fork repository: {result['error']}")
            return None
        
        return self._parse_repository(result)
    
    async def delete_repository(self, repo_full_name: str) -> bool:
        """Delete a repository.
        
        Args:
            repo_full_name: Repository to delete
            
        Returns:
            True if deleted
        """
        if not self._ensure_authenticated():
            return False
        
        result = await self._api_client.delete(f"/repos/{repo_full_name}")
        
        if "error" in result:
            logger.error(f"Failed to delete repository: {result['error']}")
            return False
        
        # Clear from cache
        if repo_full_name in self._repo_cache:
            del self._repo_cache[repo_full_name]
        
        return True
    
    async def clone_repository(
        self,
        repo_full_name: str,
        destination: Optional[Path] = None,
        use_ssh: bool = False,
        depth: Optional[int] = None,
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Clone a repository to local filesystem.
        
        Args:
            repo_full_name: Repository to clone
            destination: Destination path
            use_ssh: Use SSH instead of HTTPS
            depth: Shallow clone depth
            branch: Specific branch to clone
            
        Returns:
            Clone result
        """
        # Get repository info
        repo_info = await self.get_repository_info(repo_full_name)
        
        if not repo_info:
            return {"success": False, "error": "Repository not found"}
        
        # Determine clone URL
        if use_ssh:
            clone_url = repo_info.ssh_url
        else:
            clone_url = repo_info.clone_url
            
            # Add token for private repos
            token = self._get_access_token()
            if token and repo_info.visibility == RepoVisibility.PRIVATE:
                clone_url = clone_url.replace(
                    "https://",
                    f"https://{token}@",
                )
        
        # Determine destination
        if destination is None:
            destination = Path.cwd() / repo_info.name
        
        destination = Path(destination)
        
        # Build clone command
        cmd = ["git", "clone"]
        
        if depth:
            cmd.extend(["--depth", str(depth)])
        
        if branch:
            cmd.extend(["--branch", branch])
        
        cmd.extend([clone_url, str(destination)])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "path": str(destination),
                    "repository": repo_full_name,
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Clone operation timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ========== Issues ==========
    
    async def list_issues(
        self,
        repo_full_name: str,
        state: IssueState = IssueState.OPEN,
        labels: Optional[List[str]] = None,
        assignee: Optional[str] = None,
        creator: Optional[str] = None,
        per_page: int = 30,
        page: int = 1,
    ) -> List[IssueInfo]:
        """List repository issues.
        
        Args:
            repo_full_name: Repository name
            state: Issue state filter
            labels: Filter by labels
            assignee: Filter by assignee
            creator: Filter by creator
            per_page: Results per page
            page: Page number
            
        Returns:
            List of issues
        """
        self._ensure_authenticated()
        
        params = {
            "state": state.value,
            "per_page": per_page,
            "page": page,
        }
        
        if labels:
            params["labels"] = ",".join(labels)
        if assignee:
            params["assignee"] = assignee
        if creator:
            params["creator"] = creator
        
        result = await self._api_client.get(
            f"/repos/{repo_full_name}/issues",
            params=params,
        )
        
        if "error" in result or not isinstance(result, list):
            return []
        
        issues = []
        for item in result:
            issue = self._parse_issue(item)
            issue.repository = repo_full_name
            issues.append(issue)
        
        return issues
    
    async def get_issue(
        self,
        repo_full_name: str,
        issue_number: int,
    ) -> Optional[IssueInfo]:
        """Get a specific issue.
        
        Args:
            repo_full_name: Repository name
            issue_number: Issue number
            
        Returns:
            Issue information
        """
        self._ensure_authenticated()
        
        result = await self._api_client.get(
            f"/repos/{repo_full_name}/issues/{issue_number}"
        )
        
        if "error" in result:
            return None
        
        issue = self._parse_issue(result)
        issue.repository = repo_full_name
        return issue
    
    async def create_issue(
        self,
        repo_full_name: str,
        title: str,
        body: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
        milestone: Optional[int] = None,
    ) -> Optional[IssueInfo]:
        """Create a new issue.
        
        Args:
            repo_full_name: Repository name
            title: Issue title
            body: Issue body
            labels: Labels to add
            assignees: Assignees
            milestone: Milestone number
            
        Returns:
            Created issue
        """
        if not self._ensure_authenticated():
            return None
        
        data = {"title": title}
        
        if body:
            data["body"] = body
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees
        if milestone:
            data["milestone"] = milestone
        
        result = await self._api_client.post(
            f"/repos/{repo_full_name}/issues",
            data,
        )
        
        if "error" in result:
            logger.error(f"Failed to create issue: {result['error']}")
            return None
        
        issue = self._parse_issue(result)
        issue.repository = repo_full_name
        return issue
    
    async def update_issue(
        self,
        repo_full_name: str,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[IssueState] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> Optional[IssueInfo]:
        """Update an existing issue.
        
        Args:
            repo_full_name: Repository name
            issue_number: Issue number
            title: New title
            body: New body
            state: New state
            labels: New labels
            assignees: New assignees
            
        Returns:
            Updated issue
        """
        if not self._ensure_authenticated():
            return None
        
        data = {}
        if title:
            data["title"] = title
        if body:
            data["body"] = body
        if state:
            data["state"] = state.value
        if labels is not None:
            data["labels"] = labels
        if assignees is not None:
            data["assignees"] = assignees
        
        result = await self._api_client.patch(
            f"/repos/{repo_full_name}/issues/{issue_number}",
            data,
        )
        
        if "error" in result:
            logger.error(f"Failed to update issue: {result['error']}")
            return None
        
        issue = self._parse_issue(result)
        issue.repository = repo_full_name
        return issue
    
    async def add_issue_comment(
        self,
        repo_full_name: str,
        issue_number: int,
        body: str,
    ) -> Optional[Dict[str, Any]]:
        """Add a comment to an issue.
        
        Args:
            repo_full_name: Repository name
            issue_number: Issue number
            body: Comment body
            
        Returns:
            Created comment
        """
        if not self._ensure_authenticated():
            return None
        
        result = await self._api_client.post(
            f"/repos/{repo_full_name}/issues/{issue_number}/comments",
            {"body": body},
        )
        
        if "error" in result:
            return None
        
        return {
            "id": result.get("id"),
            "body": result.get("body"),
            "user": result.get("user", {}).get("login"),
            "created_at": result.get("created_at"),
        }
    
    async def search_issues(
        self,
        query: str,
        repo_full_name: Optional[str] = None,
        state: Optional[IssueState] = None,
        is_pr: Optional[bool] = None,
        per_page: int = 30,
    ) -> SearchResult:
        """Search for issues and PRs.
        
        Args:
            query: Search query
            repo_full_name: Optional repository filter
            state: State filter
            is_pr: Filter for PRs only
            per_page: Results per page
            
        Returns:
            Search results
        """
        self._ensure_authenticated()
        
        full_query = query
        
        if repo_full_name:
            full_query += f" repo:{repo_full_name}"
        if state:
            full_query += f" is:{state.value}"
        if is_pr is not None:
            full_query += " is:pr" if is_pr else " is:issue"
        
        params = {
            "q": full_query,
            "per_page": per_page,
        }
        
        result = await self._api_client.get("/search/issues", params=params)
        
        if "error" in result:
            return SearchResult(total_count=0, items=[], search_query=full_query)
        
        items = []
        for item in result.get("items", []):
            issue = self._parse_issue(item)
            items.append(issue)
        
        return SearchResult(
            total_count=result.get("total_count", 0),
            items=items,
            incomplete_results=result.get("incomplete_results", False),
            search_query=full_query,
        )
    
    # ========== Pull Requests ==========
    
    async def list_pull_requests(
        self,
        repo_full_name: str,
        state: IssueState = IssueState.OPEN,
        head: Optional[str] = None,
        base: Optional[str] = None,
        per_page: int = 30,
    ) -> List[IssueInfo]:
        """List repository pull requests.
        
        Args:
            repo_full_name: Repository name
            state: PR state filter
            head: Filter by head branch
            base: Filter by base branch
            per_page: Results per page
            
        Returns:
            List of PRs
        """
        self._ensure_authenticated()
        
        params = {
            "state": state.value,
            "per_page": per_page,
        }
        
        if head:
            params["head"] = head
        if base:
            params["base"] = base
        
        result = await self._api_client.get(
            f"/repos/{repo_full_name}/pulls",
            params=params,
        )
        
        if "error" in result or not isinstance(result, list):
            return []
        
        prs = []
        for item in result:
            pr = self._parse_pr(item)
            pr.repository = repo_full_name
            prs.append(pr)
        
        return prs
    
    async def create_pull_request(
        self,
        repo_full_name: str,
        title: str,
        head: str,
        base: str,
        body: Optional[str] = None,
        draft: bool = False,
    ) -> Optional[IssueInfo]:
        """Create a pull request.
        
        Args:
            repo_full_name: Repository name
            title: PR title
            head: Head branch
            base: Base branch
            body: PR body
            draft: Create as draft
            
        Returns:
            Created PR
        """
        if not self._ensure_authenticated():
            return None
        
        data = {
            "title": title,
            "head": head,
            "base": base,
            "draft": draft,
        }
        
        if body:
            data["body"] = body
        
        result = await self._api_client.post(
            f"/repos/{repo_full_name}/pulls",
            data,
        )
        
        if "error" in result:
            logger.error(f"Failed to create PR: {result['error']}")
            return None
        
        pr = self._parse_pr(result)
        pr.repository = repo_full_name
        return pr
    
    async def merge_pull_request(
        self,
        repo_full_name: str,
        pr_number: int,
        merge_method: PRMergeMethod = PRMergeMethod.MERGE,
        commit_title: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Merge a pull request.
        
        Args:
            repo_full_name: Repository name
            pr_number: PR number
            merge_method: Merge method
            commit_title: Commit title
            commit_message: Commit message
            
        Returns:
            Merge result
        """
        if not self._ensure_authenticated():
            return {"success": False, "error": "Not authenticated"}
        
        data = {"merge_method": merge_method.value}
        
        if commit_title:
            data["commit_title"] = commit_title
        if commit_message:
            data["commit_message"] = commit_message
        
        result = await self._api_client._request(
            "PUT",
            f"/repos/{repo_full_name}/pulls/{pr_number}/merge",
            json_data=data,
        )
        
        if "error" in result:
            return {"success": False, "error": result["error"]}
        
        return {
            "success": True,
            "sha": result.get("sha"),
            "message": result.get("message"),
        }
    
    # ========== Releases ==========
    
    async def list_releases(
        self,
        repo_full_name: str,
        per_page: int = 30,
    ) -> List[ReleaseInfo]:
        """List repository releases.
        
        Args:
            repo_full_name: Repository name
            per_page: Results per page
            
        Returns:
            List of releases
        """
        self._ensure_authenticated()
        
        result = await self._api_client.get(
            f"/repos/{repo_full_name}/releases",
            params={"per_page": per_page},
        )
        
        if "error" in result or not isinstance(result, list):
            return []
        
        return [self._parse_release(r) for r in result]
    
    async def get_latest_release(
        self,
        repo_full_name: str,
    ) -> Optional[ReleaseInfo]:
        """Get the latest release.
        
        Args:
            repo_full_name: Repository name
            
        Returns:
            Latest release info
        """
        self._ensure_authenticated()
        
        result = await self._api_client.get(
            f"/repos/{repo_full_name}/releases/latest"
        )
        
        if "error" in result:
            return None
        
        return self._parse_release(result)
    
    async def create_release(
        self,
        repo_full_name: str,
        tag_name: str,
        name: Optional[str] = None,
        body: Optional[str] = None,
        draft: bool = False,
        prerelease: bool = False,
        target_commitish: Optional[str] = None,
    ) -> Optional[ReleaseInfo]:
        """Create a new release.
        
        Args:
            repo_full_name: Repository name
            tag_name: Tag name
            name: Release name
            body: Release notes
            draft: Create as draft
            prerelease: Mark as prerelease
            target_commitish: Target commit/branch
            
        Returns:
            Created release
        """
        if not self._ensure_authenticated():
            return None
        
        data = {
            "tag_name": tag_name,
            "draft": draft,
            "prerelease": prerelease,
        }
        
        if name:
            data["name"] = name
        if body:
            data["body"] = body
        if target_commitish:
            data["target_commitish"] = target_commitish
        
        result = await self._api_client.post(
            f"/repos/{repo_full_name}/releases",
            data,
        )
        
        if "error" in result:
            logger.error(f"Failed to create release: {result['error']}")
            return None
        
        return self._parse_release(result)
    
    async def upload_release_asset(
        self,
        repo_full_name: str,
        release_id: int,
        file_path: Path,
        name: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload an asset to a release.
        
        Args:
            repo_full_name: Repository name
            release_id: Release ID
            file_path: Path to asset file
            name: Asset name
            label: Asset label
            
        Returns:
            Upload result
        """
        if not self._ensure_authenticated():
            return {"success": False, "error": "Not authenticated"}
        
        file_path = Path(file_path)
        if not file_path.exists():
            return {"success": False, "error": "File not found"}
        
        asset_name = name or file_path.name
        
        try:
            import aiohttp
            
            token = self._get_access_token()
            
            upload_url = f"https://uploads.github.com/repos/{repo_full_name}/releases/{release_id}/assets"
            
            params = {"name": asset_name}
            if label:
                params["label"] = label
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream",
            }
            
            with open(file_path, "rb") as f:
                content = f.read()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    upload_url,
                    params=params,
                    headers=headers,
                    data=content,
                ) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        return {
                            "success": True,
                            "id": result.get("id"),
                            "name": result.get("name"),
                            "download_url": result.get("browser_download_url"),
                        }
                    else:
                        error = await response.text()
                        return {"success": False, "error": error}
                        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ========== Helper Methods ==========
    
    def _parse_repository(self, data: Dict[str, Any]) -> RepositoryInfo:
        """Parse repository API response."""
        owner = data.get("owner", {})
        
        visibility = RepoVisibility.PUBLIC
        if data.get("private"):
            visibility = RepoVisibility.PRIVATE
        elif data.get("visibility") == "internal":
            visibility = RepoVisibility.INTERNAL
        
        return RepositoryInfo(
            owner=owner.get("login", ""),
            name=data.get("name", ""),
            full_name=data.get("full_name", ""),
            description=data.get("description"),
            url=data.get("url", ""),
            html_url=data.get("html_url", ""),
            clone_url=data.get("clone_url", ""),
            ssh_url=data.get("ssh_url", ""),
            stars=data.get("stargazers_count", 0),
            forks=data.get("forks_count", 0),
            watchers=data.get("watchers_count", 0),
            open_issues=data.get("open_issues_count", 0),
            size_kb=data.get("size", 0),
            language=data.get("language"),
            topics=data.get("topics", []),
            license_name=data.get("license", {}).get("name") if data.get("license") else None,
            visibility=visibility,
            is_fork=data.get("fork", False),
            is_archived=data.get("archived", False),
            is_template=data.get("is_template", False),
            has_issues=data.get("has_issues", True),
            has_wiki=data.get("has_wiki", True),
            has_discussions=data.get("has_discussions", False),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")) if data.get("updated_at") else None,
            pushed_at=datetime.fromisoformat(data["pushed_at"].replace("Z", "+00:00")) if data.get("pushed_at") else None,
            default_branch=data.get("default_branch", "main"),
        )
    
    def _parse_issue(self, data: Dict[str, Any]) -> IssueInfo:
        """Parse issue API response."""
        state = IssueState.OPEN
        if data.get("state") == "closed":
            state = IssueState.CLOSED
        
        return IssueInfo(
            number=data.get("number", 0),
            title=data.get("title", ""),
            body=data.get("body"),
            state=state,
            is_pull_request="pull_request" in data,
            labels=[l.get("name", "") for l in data.get("labels", [])],
            assignees=[a.get("login", "") for a in data.get("assignees", [])],
            milestone=data.get("milestone", {}).get("title") if data.get("milestone") else None,
            author=data.get("user", {}).get("login"),
            html_url=data.get("html_url", ""),
            api_url=data.get("url", ""),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")) if data.get("updated_at") else None,
            closed_at=datetime.fromisoformat(data["closed_at"].replace("Z", "+00:00")) if data.get("closed_at") else None,
            comments_count=data.get("comments", 0),
        )
    
    def _parse_pr(self, data: Dict[str, Any]) -> IssueInfo:
        """Parse PR API response."""
        issue = self._parse_issue(data)
        issue.is_pull_request = True
        issue.pr_merged = data.get("merged", False)
        issue.pr_mergeable = data.get("mergeable")
        issue.pr_draft = data.get("draft", False)
        
        head = data.get("head", {})
        base = data.get("base", {})
        issue.head_branch = head.get("ref")
        issue.base_branch = base.get("ref")
        
        return issue
    
    def _parse_release(self, data: Dict[str, Any]) -> ReleaseInfo:
        """Parse release API response."""
        assets = []
        for asset in data.get("assets", []):
            assets.append({
                "id": asset.get("id"),
                "name": asset.get("name"),
                "size": asset.get("size"),
                "download_count": asset.get("download_count"),
                "download_url": asset.get("browser_download_url"),
            })
        
        return ReleaseInfo(
            tag_name=data.get("tag_name", ""),
            name=data.get("name", ""),
            body=data.get("body"),
            draft=data.get("draft", False),
            prerelease=data.get("prerelease", False),
            author=data.get("author", {}).get("login"),
            html_url=data.get("html_url", ""),
            tarball_url=data.get("tarball_url", ""),
            zipball_url=data.get("zipball_url", ""),
            assets=assets,
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")) if data.get("created_at") else None,
            published_at=datetime.fromisoformat(data["published_at"].replace("Z", "+00:00")) if data.get("published_at") else None,
            id=data.get("id"),
        )
    
    async def analyze_repository_with_llm(
        self,
        repo_full_name: str,
    ) -> Dict[str, Any]:
        """Analyze a repository using LLM reasoning.
        
        Args:
            repo_full_name: Repository to analyze
            
        Returns:
            Analysis results
        """
        # Get comprehensive repo info
        repo_info = await self.get_repository_info(repo_full_name)
        if not repo_info:
            return {"error": "Repository not found"}
        
        languages = await self.get_repository_languages(repo_full_name)
        contributors = await self.get_repository_contributors(repo_full_name, per_page=10)
        readme = await self.get_repository_readme(repo_full_name)
        
        # Recent issues and PRs
        recent_issues = await self.list_issues(repo_full_name, per_page=5)
        recent_prs = await self.list_pull_requests(repo_full_name, per_page=5)
        
        analysis = {
            "repository": repo_info.to_dict(),
            "languages": languages,
            "top_contributors": contributors[:5] if contributors else [],
            "recent_issues": [i.to_dict() for i in recent_issues],
            "recent_prs": [pr.to_dict() for pr in recent_prs],
        }
        
        if self._llm_client:
            # Get LLM insights
            prompt = f"""Analyze this GitHub repository and provide insights.

Repository: {repo_info.full_name}
Description: {repo_info.description}
Language: {repo_info.language}
Stars: {repo_info.stars}
Forks: {repo_info.forks}
Open Issues: {repo_info.open_issues}
Topics: {', '.join(repo_info.topics)}

README excerpt (first 500 chars):
{readme[:500] if readme else 'No README'}

Languages breakdown: {json.dumps(languages)}

Provide:
1. Summary of what this project does
2. Tech stack assessment
3. Project health indicators
4. Suggestions for contributors
"""
            
            try:
                insights = await self._llm_client.generate(prompt)
                analysis["llm_insights"] = insights
            except Exception as e:
                analysis["llm_insights"] = f"Analysis failed: {e}"
        
        return analysis


# Module-level instance
_global_repo_ops: Optional[GitHubRepoOperations] = None


def get_github_repo_operations(
    llm_client: Optional[Any] = None,
    authenticator: Optional[Any] = None,
) -> GitHubRepoOperations:
    """Get the global GitHub repository operations manager.
    
    Args:
        llm_client: Optional LLM client
        authenticator: Optional authenticator
        
    Returns:
        GitHubRepoOperations instance
    """
    global _global_repo_ops
    if _global_repo_ops is None:
        _global_repo_ops = GitHubRepoOperations(
            llm_client=llm_client,
            authenticator=authenticator,
        )
    return _global_repo_ops
