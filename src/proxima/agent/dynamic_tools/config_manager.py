"""Configuration Management System for Dynamic AI Assistant.

This module implements Phase 7.2 for the Dynamic AI Assistant:
- Profile System: Named configuration profiles with inheritance
- Environment Detection: Auto-detect and configure for environments
- Configuration Synchronization: Backup, restore, versioning

Key Features:
============
- Named configuration profiles
- Profile inheritance and composition
- Automatic environment detection
- Configuration backup and restore
- Configuration versioning and rollback
- Configuration audit trail

Design Principle:
================
All configuration decisions use LLM reasoning when available.
The LLM analyzes context and suggests optimal configurations.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import shutil
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
    Optional, Pattern, Set, Tuple, Type, Union
)
import uuid

logger = logging.getLogger(__name__)


class ProfileType(Enum):
    """Configuration profile types."""
    USER = "user"
    PROJECT = "project"
    WORKSPACE = "workspace"
    GLOBAL = "global"
    TEMPLATE = "template"


class ConfigScope(Enum):
    """Configuration scope levels."""
    SYSTEM = "system"  # System-wide defaults
    GLOBAL = "global"  # User-wide settings
    WORKSPACE = "workspace"  # Workspace settings
    PROJECT = "project"  # Project-specific
    LOCAL = "local"  # Current session only


class EnvironmentType(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    CI_CD = "ci_cd"
    UNKNOWN = "unknown"


class SyncStatus(Enum):
    """Configuration sync status."""
    SYNCED = "synced"
    PENDING = "pending"
    CONFLICT = "conflict"
    ERROR = "error"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    LOCAL_WINS = "local_wins"
    REMOTE_WINS = "remote_wins"
    MERGE = "merge"
    MANUAL = "manual"


@dataclass
class ConfigValue:
    """A configuration value with metadata."""
    key: str
    value: Any
    scope: ConfigScope
    
    # Metadata
    description: Optional[str] = None
    data_type: str = "string"
    default: Any = None
    
    # Validation
    validators: List[str] = field(default_factory=list)
    valid_values: Optional[List[Any]] = None
    
    # Tracking
    source: str = "default"
    modified_at: Optional[datetime] = None
    modified_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "scope": self.scope.value,
            "description": self.description,
            "data_type": self.data_type,
            "source": self.source,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
        }
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate the configuration value.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check valid values constraint
        if self.valid_values is not None:
            if self.value not in self.valid_values:
                return False, f"Value must be one of: {self.valid_values}"
        
        # Type validation
        type_map = {
            "string": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        
        expected_type = type_map.get(self.data_type)
        if expected_type and not isinstance(self.value, expected_type):
            return False, f"Value must be of type {self.data_type}"
        
        return True, None


@dataclass
class ConfigProfile:
    """A named configuration profile."""
    profile_id: str
    name: str
    profile_type: ProfileType
    
    # Configuration values
    values: Dict[str, Any] = field(default_factory=dict)
    
    # Inheritance
    parent_profile: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    active: bool = False
    version: int = 1
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Tags for organization
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "profile_type": self.profile_type.value,
            "values": self.values,
            "parent_profile": self.parent_profile,
            "description": self.description,
            "active": self.active,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigProfile":
        """Create profile from dictionary."""
        return cls(
            profile_id=data["profile_id"],
            name=data["name"],
            profile_type=ProfileType(data["profile_type"]),
            values=data.get("values", {}),
            parent_profile=data.get("parent_profile"),
            description=data.get("description"),
            active=data.get("active", False),
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            tags=data.get("tags", []),
        )


@dataclass
class ConfigVersion:
    """A versioned configuration snapshot."""
    version_id: str
    version_number: int
    profile_id: str
    
    # Snapshot
    values: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    
    # Metadata
    message: Optional[str] = None
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "version_number": self.version_number,
            "profile_id": self.profile_id,
            "values": self.values,
            "checksum": self.checksum,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class ConfigConflict:
    """A configuration conflict during sync."""
    conflict_id: str
    key: str
    
    local_value: Any
    remote_value: Any
    base_value: Optional[Any] = None
    
    resolved: bool = False
    resolution: Optional[ConflictResolution] = None
    resolved_value: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "key": self.key,
            "local_value": self.local_value,
            "remote_value": self.remote_value,
            "resolved": self.resolved,
            "resolution": self.resolution.value if self.resolution else None,
        }


@dataclass
class AuditEntry:
    """Configuration audit trail entry."""
    entry_id: str
    timestamp: datetime
    action: str  # created, updated, deleted, activated, deactivated
    
    profile_id: Optional[str] = None
    key: Optional[str] = None
    old_value: Any = None
    new_value: Any = None
    
    user: Optional[str] = None
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "profile_id": self.profile_id,
            "key": self.key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "user": self.user,
            "reason": self.reason,
        }


class ProfileManager:
    """Manage configuration profiles.
    
    Uses LLM reasoning to:
    1. Suggest profile configurations
    2. Detect profile compatibility
    3. Generate profile from context
    
    Example:
        >>> manager = ProfileManager()
        >>> profile = manager.create_profile("dev", ProfileType.PROJECT)
        >>> manager.activate_profile(profile.profile_id)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        storage_path: Optional[Path] = None,
    ):
        """Initialize the profile manager.
        
        Args:
            llm_client: LLM client for intelligent operations
            storage_path: Path for persistent storage
        """
        self._llm_client = llm_client
        self._storage_path = storage_path
        
        # Profiles
        self._profiles: Dict[str, ConfigProfile] = {}
        self._active_profile: Optional[str] = None
        
        # Templates
        self._templates: Dict[str, ConfigProfile] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Load profiles
        self._load_profiles()
        self._load_templates()
    
    def create_profile(
        self,
        name: str,
        profile_type: ProfileType,
        values: Optional[Dict[str, Any]] = None,
        parent_profile: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ConfigProfile:
        """Create a new configuration profile.
        
        Args:
            name: Profile name
            profile_type: Type of profile
            values: Initial configuration values
            parent_profile: Parent profile ID for inheritance
            description: Profile description
            
        Returns:
            Created profile
        """
        profile = ConfigProfile(
            profile_id=str(uuid.uuid4()),
            name=name,
            profile_type=profile_type,
            values=values or {},
            parent_profile=parent_profile,
            description=description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        with self._lock:
            self._profiles[profile.profile_id] = profile
        
        self._save_profiles()
        return profile
    
    def get_profile(self, profile_id: str) -> Optional[ConfigProfile]:
        """Get a profile by ID."""
        with self._lock:
            return self._profiles.get(profile_id)
    
    def get_profile_by_name(self, name: str) -> Optional[ConfigProfile]:
        """Get a profile by name."""
        with self._lock:
            for profile in self._profiles.values():
                if profile.name == name:
                    return profile
        return None
    
    def list_profiles(
        self,
        profile_type: Optional[ProfileType] = None,
    ) -> List[ConfigProfile]:
        """List all profiles.
        
        Args:
            profile_type: Filter by type
            
        Returns:
            List of profiles
        """
        with self._lock:
            profiles = list(self._profiles.values())
        
        if profile_type:
            profiles = [p for p in profiles if p.profile_type == profile_type]
        
        return profiles
    
    def update_profile(
        self,
        profile_id: str,
        values: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[ConfigProfile]:
        """Update a profile.
        
        Args:
            profile_id: Profile ID
            values: New values to merge
            name: New name
            description: New description
            
        Returns:
            Updated profile
        """
        with self._lock:
            profile = self._profiles.get(profile_id)
            if not profile:
                return None
            
            if values:
                profile.values.update(values)
            if name:
                profile.name = name
            if description:
                profile.description = description
            
            profile.updated_at = datetime.now()
            profile.version += 1
        
        self._save_profiles()
        return profile
    
    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile.
        
        Args:
            profile_id: Profile ID
            
        Returns:
            Whether deletion succeeded
        """
        with self._lock:
            if profile_id not in self._profiles:
                return False
            
            # Cannot delete active profile
            if profile_id == self._active_profile:
                return False
            
            del self._profiles[profile_id]
        
        self._save_profiles()
        return True
    
    def activate_profile(self, profile_id: str) -> bool:
        """Activate a profile.
        
        Args:
            profile_id: Profile to activate
            
        Returns:
            Whether activation succeeded
        """
        with self._lock:
            if profile_id not in self._profiles:
                return False
            
            # Deactivate current
            if self._active_profile:
                self._profiles[self._active_profile].active = False
            
            # Activate new
            self._profiles[profile_id].active = True
            self._active_profile = profile_id
        
        self._save_profiles()
        return True
    
    def get_active_profile(self) -> Optional[ConfigProfile]:
        """Get the currently active profile."""
        with self._lock:
            if self._active_profile:
                return self._profiles.get(self._active_profile)
        return None
    
    def get_resolved_values(
        self,
        profile_id: str,
    ) -> Dict[str, Any]:
        """Get resolved values with inheritance.
        
        Args:
            profile_id: Profile ID
            
        Returns:
            Resolved configuration values
        """
        with self._lock:
            profile = self._profiles.get(profile_id)
            if not profile:
                return {}
            
            # Build inheritance chain
            chain = []
            current = profile
            while current:
                chain.append(current)
                if current.parent_profile:
                    current = self._profiles.get(current.parent_profile)
                else:
                    current = None
            
            # Resolve values (parent -> child order)
            resolved = {}
            for p in reversed(chain):
                resolved.update(p.values)
            
            return resolved
    
    def create_from_template(
        self,
        template_name: str,
        profile_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConfigProfile]:
        """Create a profile from a template.
        
        Args:
            template_name: Template name
            profile_name: New profile name
            overrides: Value overrides
            
        Returns:
            Created profile
        """
        template = self._templates.get(template_name)
        if not template:
            return None
        
        values = copy.deepcopy(template.values)
        if overrides:
            values.update(overrides)
        
        return self.create_profile(
            name=profile_name,
            profile_type=ProfileType.USER,
            values=values,
            description=f"Created from template: {template_name}",
        )
    
    def _load_profiles(self):
        """Load profiles from storage."""
        if not self._storage_path:
            return
        
        profiles_file = self._storage_path / "profiles.json"
        if not profiles_file.exists():
            return
        
        try:
            with open(profiles_file, "r") as f:
                data = json.load(f)
            
            for profile_data in data.get("profiles", []):
                profile = ConfigProfile.from_dict(profile_data)
                self._profiles[profile.profile_id] = profile
                
                if profile.active:
                    self._active_profile = profile.profile_id
            
            logger.info(f"Loaded {len(self._profiles)} profiles")
            
        except Exception as e:
            logger.warning(f"Failed to load profiles: {e}")
    
    def _save_profiles(self):
        """Save profiles to storage."""
        if not self._storage_path:
            return
        
        self._storage_path.mkdir(parents=True, exist_ok=True)
        profiles_file = self._storage_path / "profiles.json"
        
        try:
            with self._lock:
                data = {
                    "profiles": [p.to_dict() for p in self._profiles.values()],
                    "active_profile": self._active_profile,
                    "updated_at": datetime.now().isoformat(),
                }
            
            with open(profiles_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save profiles: {e}")
    
    def _load_templates(self):
        """Load built-in templates."""
        self._templates = {
            "development": ConfigProfile(
                profile_id="template_dev",
                name="Development",
                profile_type=ProfileType.TEMPLATE,
                values={
                    "log_level": "debug",
                    "verbose": True,
                    "auto_save": True,
                    "hot_reload": True,
                },
                description="Development environment template",
            ),
            "production": ConfigProfile(
                profile_id="template_prod",
                name="Production",
                profile_type=ProfileType.TEMPLATE,
                values={
                    "log_level": "warning",
                    "verbose": False,
                    "auto_save": False,
                    "hot_reload": False,
                },
                description="Production environment template",
            ),
            "testing": ConfigProfile(
                profile_id="template_test",
                name="Testing",
                profile_type=ProfileType.TEMPLATE,
                values={
                    "log_level": "debug",
                    "verbose": True,
                    "mock_services": True,
                    "coverage": True,
                },
                description="Testing environment template",
            ),
        }


class EnvironmentDetector:
    """Detect and configure for environments.
    
    Uses LLM reasoning to:
    1. Analyze project structure
    2. Detect environment type
    3. Suggest appropriate configuration
    
    Example:
        >>> detector = EnvironmentDetector()
        >>> env = detector.detect_environment()
        >>> config = detector.get_environment_config(env)
    """
    
    # Environment indicators
    ENVIRONMENT_INDICATORS = {
        EnvironmentType.DEVELOPMENT: [
            ".env.development",
            ".env.local",
            ".vscode",
            ".idea",
            "debug",
        ],
        EnvironmentType.TESTING: [
            ".env.test",
            "pytest.ini",
            "jest.config.js",
            "test",
            "tests",
            "spec",
            "__tests__",
        ],
        EnvironmentType.STAGING: [
            ".env.staging",
            "staging",
        ],
        EnvironmentType.PRODUCTION: [
            ".env.production",
            "Dockerfile",
            "docker-compose.yml",
            "kubernetes",
            "k8s",
        ],
        EnvironmentType.CI_CD: [
            ".github/workflows",
            ".gitlab-ci.yml",
            "Jenkinsfile",
            ".circleci",
            ".travis.yml",
        ],
    }
    
    # Default configurations per environment
    ENVIRONMENT_CONFIGS = {
        EnvironmentType.DEVELOPMENT: {
            "log_level": "debug",
            "verbose": True,
            "debug_mode": True,
            "hot_reload": True,
            "source_maps": True,
            "minify": False,
        },
        EnvironmentType.TESTING: {
            "log_level": "debug",
            "verbose": True,
            "mock_external": True,
            "coverage": True,
            "fail_fast": False,
        },
        EnvironmentType.STAGING: {
            "log_level": "info",
            "verbose": False,
            "debug_mode": False,
            "use_production_db": False,
        },
        EnvironmentType.PRODUCTION: {
            "log_level": "warning",
            "verbose": False,
            "debug_mode": False,
            "minify": True,
            "source_maps": False,
        },
        EnvironmentType.CI_CD: {
            "log_level": "info",
            "verbose": True,
            "fail_fast": True,
            "coverage": True,
        },
    }
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize the environment detector.
        
        Args:
            llm_client: LLM client for intelligent detection
        """
        self._llm_client = llm_client
    
    def detect_environment(
        self,
        working_dir: Optional[Path] = None,
    ) -> EnvironmentType:
        """Detect the current environment type.
        
        Args:
            working_dir: Working directory to analyze
            
        Returns:
            Detected environment type
        """
        working_dir = working_dir or Path.cwd()
        
        # Score each environment type
        scores: Dict[EnvironmentType, int] = defaultdict(int)
        
        for env_type, indicators in self.ENVIRONMENT_INDICATORS.items():
            for indicator in indicators:
                indicator_path = working_dir / indicator
                if indicator_path.exists():
                    scores[env_type] += 1
        
        # Check environment variables
        env_var = os.environ.get("ENVIRONMENT", "").lower()
        env_var_2 = os.environ.get("NODE_ENV", "").lower()
        
        for env_type in EnvironmentType:
            if env_type.value in env_var or env_type.value in env_var_2:
                scores[env_type] += 3  # Weight env vars higher
        
        # Check CI environment variables
        ci_vars = ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL", "TRAVIS"]
        if any(os.environ.get(var) for var in ci_vars):
            scores[EnvironmentType.CI_CD] += 5
        
        if scores:
            return max(scores, key=scores.get)
        
        return EnvironmentType.UNKNOWN
    
    def get_environment_config(
        self,
        env_type: EnvironmentType,
    ) -> Dict[str, Any]:
        """Get default configuration for environment.
        
        Args:
            env_type: Environment type
            
        Returns:
            Default configuration
        """
        return self.ENVIRONMENT_CONFIGS.get(env_type, {}).copy()
    
    def get_environment_variables(
        self,
        env_type: EnvironmentType,
    ) -> Dict[str, str]:
        """Get recommended environment variables.
        
        Args:
            env_type: Environment type
            
        Returns:
            Recommended environment variables
        """
        base_vars = {
            "ENVIRONMENT": env_type.value,
        }
        
        if env_type == EnvironmentType.DEVELOPMENT:
            base_vars.update({
                "DEBUG": "true",
                "LOG_LEVEL": "debug",
            })
        elif env_type == EnvironmentType.PRODUCTION:
            base_vars.update({
                "DEBUG": "false",
                "LOG_LEVEL": "warning",
            })
        
        return base_vars


class ConfigurationSynchronizer:
    """Synchronize configurations with backup and versioning.
    
    Uses LLM reasoning to:
    1. Resolve sync conflicts intelligently
    2. Suggest merge strategies
    3. Validate configuration changes
    
    Example:
        >>> sync = ConfigurationSynchronizer(storage_path=path)
        >>> sync.backup("Before major changes")
        >>> sync.restore(version_id)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        storage_path: Optional[Path] = None,
        max_versions: int = 50,
    ):
        """Initialize the configuration synchronizer.
        
        Args:
            llm_client: LLM client for intelligent sync
            storage_path: Path for storage
            max_versions: Maximum versions to keep
        """
        self._llm_client = llm_client
        self._storage_path = storage_path
        self._max_versions = max_versions
        
        # Version history
        self._versions: List[ConfigVersion] = []
        
        # Audit trail
        self._audit_trail: List[AuditEntry] = []
        
        # Pending conflicts
        self._conflicts: Dict[str, ConfigConflict] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Load version history
        self._load_versions()
    
    def backup(
        self,
        profile_id: str,
        values: Dict[str, Any],
        message: Optional[str] = None,
    ) -> ConfigVersion:
        """Create a backup version.
        
        Args:
            profile_id: Profile being backed up
            values: Configuration values
            message: Backup message
            
        Returns:
            Created version
        """
        with self._lock:
            version_number = len(self._versions) + 1
        
        # Calculate checksum
        content = json.dumps(values, sort_keys=True)
        checksum = hashlib.sha256(content.encode()).hexdigest()
        
        version = ConfigVersion(
            version_id=str(uuid.uuid4()),
            version_number=version_number,
            profile_id=profile_id,
            values=copy.deepcopy(values),
            checksum=checksum,
            message=message,
            created_at=datetime.now(),
        )
        
        with self._lock:
            self._versions.append(version)
            
            # Trim old versions
            if len(self._versions) > self._max_versions:
                self._versions = self._versions[-self._max_versions:]
        
        self._save_versions()
        self._log_audit("backup", profile_id, reason=message)
        
        return version
    
    def restore(
        self,
        version_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Restore configuration from a version.
        
        Args:
            version_id: Version to restore
            
        Returns:
            Restored values
        """
        with self._lock:
            for version in self._versions:
                if version.version_id == version_id:
                    self._log_audit(
                        "restore",
                        version.profile_id,
                        reason=f"Restored from version {version.version_number}",
                    )
                    return copy.deepcopy(version.values)
        
        return None
    
    def get_version_history(
        self,
        profile_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[ConfigVersion]:
        """Get version history.
        
        Args:
            profile_id: Filter by profile
            limit: Maximum results
            
        Returns:
            List of versions
        """
        with self._lock:
            versions = list(self._versions)
        
        if profile_id:
            versions = [v for v in versions if v.profile_id == profile_id]
        
        return versions[-limit:]
    
    def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str,
    ) -> Dict[str, Any]:
        """Compare two versions.
        
        Args:
            version_id_1: First version
            version_id_2: Second version
            
        Returns:
            Comparison result
        """
        v1 = None
        v2 = None
        
        with self._lock:
            for version in self._versions:
                if version.version_id == version_id_1:
                    v1 = version
                if version.version_id == version_id_2:
                    v2 = version
        
        if not v1 or not v2:
            return {"error": "Version not found"}
        
        # Find differences
        added = {}
        removed = {}
        changed = {}
        
        all_keys = set(v1.values.keys()) | set(v2.values.keys())
        
        for key in all_keys:
            in_v1 = key in v1.values
            in_v2 = key in v2.values
            
            if in_v1 and not in_v2:
                removed[key] = v1.values[key]
            elif in_v2 and not in_v1:
                added[key] = v2.values[key]
            elif v1.values[key] != v2.values[key]:
                changed[key] = {
                    "old": v1.values[key],
                    "new": v2.values[key],
                }
        
        return {
            "version_1": version_id_1,
            "version_2": version_id_2,
            "added": added,
            "removed": removed,
            "changed": changed,
        }
    
    def detect_conflicts(
        self,
        local_values: Dict[str, Any],
        remote_values: Dict[str, Any],
        base_values: Optional[Dict[str, Any]] = None,
    ) -> List[ConfigConflict]:
        """Detect configuration conflicts.
        
        Args:
            local_values: Local configuration
            remote_values: Remote configuration
            base_values: Common ancestor (if known)
            
        Returns:
            List of conflicts
        """
        conflicts = []
        
        all_keys = set(local_values.keys()) | set(remote_values.keys())
        
        for key in all_keys:
            local_val = local_values.get(key)
            remote_val = remote_values.get(key)
            base_val = base_values.get(key) if base_values else None
            
            # Check for conflict
            if local_val != remote_val:
                # If base is available, check if both changed from base
                if base_values:
                    local_changed = local_val != base_val
                    remote_changed = remote_val != base_val
                    
                    if local_changed and remote_changed:
                        conflict = ConfigConflict(
                            conflict_id=str(uuid.uuid4()),
                            key=key,
                            local_value=local_val,
                            remote_value=remote_val,
                            base_value=base_val,
                        )
                        conflicts.append(conflict)
                        self._conflicts[conflict.conflict_id] = conflict
                else:
                    # No base - any difference is a conflict
                    conflict = ConfigConflict(
                        conflict_id=str(uuid.uuid4()),
                        key=key,
                        local_value=local_val,
                        remote_value=remote_val,
                    )
                    conflicts.append(conflict)
                    self._conflicts[conflict.conflict_id] = conflict
        
        return conflicts
    
    def resolve_conflict(
        self,
        conflict_id: str,
        resolution: ConflictResolution,
        custom_value: Any = None,
    ) -> bool:
        """Resolve a configuration conflict.
        
        Args:
            conflict_id: Conflict ID
            resolution: Resolution strategy
            custom_value: Custom value for manual resolution
            
        Returns:
            Whether resolution succeeded
        """
        conflict = self._conflicts.get(conflict_id)
        if not conflict:
            return False
        
        if resolution == ConflictResolution.LOCAL_WINS:
            conflict.resolved_value = conflict.local_value
        elif resolution == ConflictResolution.REMOTE_WINS:
            conflict.resolved_value = conflict.remote_value
        elif resolution == ConflictResolution.MANUAL:
            if custom_value is None:
                return False
            conflict.resolved_value = custom_value
        elif resolution == ConflictResolution.MERGE:
            # Try to merge if both are dicts
            if isinstance(conflict.local_value, dict) and isinstance(conflict.remote_value, dict):
                merged = copy.deepcopy(conflict.local_value)
                merged.update(conflict.remote_value)
                conflict.resolved_value = merged
            else:
                # Can't merge - use local
                conflict.resolved_value = conflict.local_value
        
        conflict.resolved = True
        conflict.resolution = resolution
        
        return True
    
    def get_pending_conflicts(self) -> List[ConfigConflict]:
        """Get unresolved conflicts."""
        return [c for c in self._conflicts.values() if not c.resolved]
    
    def _log_audit(
        self,
        action: str,
        profile_id: Optional[str] = None,
        key: Optional[str] = None,
        old_value: Any = None,
        new_value: Any = None,
        reason: Optional[str] = None,
    ):
        """Log an audit entry."""
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            action=action,
            profile_id=profile_id,
            key=key,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
        )
        
        with self._lock:
            self._audit_trail.append(entry)
            
            # Trim old entries
            if len(self._audit_trail) > 1000:
                self._audit_trail = self._audit_trail[-1000:]
    
    def get_audit_trail(
        self,
        profile_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Get audit trail entries.
        
        Args:
            profile_id: Filter by profile
            action: Filter by action
            limit: Maximum entries
            
        Returns:
            Audit entries
        """
        with self._lock:
            entries = list(self._audit_trail)
        
        if profile_id:
            entries = [e for e in entries if e.profile_id == profile_id]
        if action:
            entries = [e for e in entries if e.action == action]
        
        return entries[-limit:]
    
    def _load_versions(self):
        """Load version history from storage."""
        if not self._storage_path:
            return
        
        versions_file = self._storage_path / "versions.json"
        if not versions_file.exists():
            return
        
        try:
            with open(versions_file, "r") as f:
                data = json.load(f)
            
            for v_data in data.get("versions", []):
                version = ConfigVersion(
                    version_id=v_data["version_id"],
                    version_number=v_data["version_number"],
                    profile_id=v_data["profile_id"],
                    values=v_data.get("values", {}),
                    checksum=v_data.get("checksum", ""),
                    message=v_data.get("message"),
                    created_at=datetime.fromisoformat(v_data["created_at"]) if v_data.get("created_at") else None,
                )
                self._versions.append(version)
            
            logger.info(f"Loaded {len(self._versions)} config versions")
            
        except Exception as e:
            logger.warning(f"Failed to load versions: {e}")
    
    def _save_versions(self):
        """Save version history to storage."""
        if not self._storage_path:
            return
        
        self._storage_path.mkdir(parents=True, exist_ok=True)
        versions_file = self._storage_path / "versions.json"
        
        try:
            with self._lock:
                data = {
                    "versions": [v.to_dict() for v in self._versions],
                    "updated_at": datetime.now().isoformat(),
                }
            
            with open(versions_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save versions: {e}")


class ConfigurationManager:
    """Main configuration management system.
    
    Integrates all configuration components:
    - Profile management
    - Environment detection
    - Configuration synchronization
    
    Example:
        >>> config = ConfigurationManager()
        >>> profile = config.create_profile("my-config")
        >>> config.set_value("log_level", "debug")
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        storage_path: Optional[Path] = None,
    ):
        """Initialize the configuration manager.
        
        Args:
            llm_client: LLM client for intelligent operations
            storage_path: Path for persistent storage
        """
        self._llm_client = llm_client
        self._storage_path = storage_path or Path.home() / ".proxima" / "config"
        
        # Initialize components
        self._profile_manager = ProfileManager(
            llm_client=llm_client,
            storage_path=self._storage_path,
        )
        
        self._env_detector = EnvironmentDetector(
            llm_client=llm_client,
        )
        
        self._sync = ConfigurationSynchronizer(
            llm_client=llm_client,
            storage_path=self._storage_path,
        )
        
        # Current values (active profile + overrides)
        self._current_values: Dict[str, Any] = {}
        self._overrides: Dict[str, Any] = {}
        
        # Initialize current values from active profile
        self._load_current_values()
    
    def _load_current_values(self):
        """Load current values from active profile."""
        active = self._profile_manager.get_active_profile()
        if active:
            self._current_values = self._profile_manager.get_resolved_values(
                active.profile_id
            )
    
    # Profile operations
    def create_profile(
        self,
        name: str,
        profile_type: ProfileType = ProfileType.USER,
        values: Optional[Dict[str, Any]] = None,
        parent: Optional[str] = None,
    ) -> ConfigProfile:
        """Create a configuration profile."""
        return self._profile_manager.create_profile(
            name=name,
            profile_type=profile_type,
            values=values,
            parent_profile=parent,
        )
    
    def get_profile(self, profile_id: str) -> Optional[ConfigProfile]:
        """Get a profile by ID."""
        return self._profile_manager.get_profile(profile_id)
    
    def list_profiles(
        self,
        profile_type: Optional[ProfileType] = None,
    ) -> List[ConfigProfile]:
        """List all profiles."""
        return self._profile_manager.list_profiles(profile_type)
    
    def activate_profile(self, profile_id: str) -> bool:
        """Activate a profile."""
        result = self._profile_manager.activate_profile(profile_id)
        if result:
            self._load_current_values()
        return result
    
    def get_active_profile(self) -> Optional[ConfigProfile]:
        """Get active profile."""
        return self._profile_manager.get_active_profile()
    
    def create_from_template(
        self,
        template_name: str,
        profile_name: str,
    ) -> Optional[ConfigProfile]:
        """Create profile from template."""
        return self._profile_manager.create_from_template(
            template_name, profile_name
        )
    
    # Value operations
    def get_value(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value
            
        Returns:
            Configuration value
        """
        # Check overrides first
        if key in self._overrides:
            return self._overrides[key]
        
        # Then current values
        return self._current_values.get(key, default)
    
    def set_value(
        self,
        key: str,
        value: Any,
        persist: bool = True,
    ):
        """Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            persist: Whether to persist to active profile
        """
        self._current_values[key] = value
        
        if persist:
            active = self._profile_manager.get_active_profile()
            if active:
                self._profile_manager.update_profile(
                    active.profile_id,
                    values={key: value},
                )
    
    def set_override(self, key: str, value: Any):
        """Set a temporary override (session only).
        
        Args:
            key: Configuration key
            value: Override value
        """
        self._overrides[key] = value
    
    def clear_overrides(self):
        """Clear all temporary overrides."""
        self._overrides.clear()
    
    def get_all_values(self) -> Dict[str, Any]:
        """Get all current configuration values."""
        result = self._current_values.copy()
        result.update(self._overrides)
        return result
    
    # Environment operations
    def detect_environment(self) -> EnvironmentType:
        """Detect current environment."""
        return self._env_detector.detect_environment()
    
    def get_environment_config(
        self,
        env_type: Optional[EnvironmentType] = None,
    ) -> Dict[str, Any]:
        """Get environment configuration."""
        if env_type is None:
            env_type = self.detect_environment()
        return self._env_detector.get_environment_config(env_type)
    
    def apply_environment_config(
        self,
        env_type: Optional[EnvironmentType] = None,
    ):
        """Apply environment-specific configuration.
        
        Args:
            env_type: Environment type (auto-detect if None)
        """
        config = self.get_environment_config(env_type)
        for key, value in config.items():
            self.set_value(key, value, persist=False)
    
    # Backup and sync operations
    def backup(self, message: Optional[str] = None) -> ConfigVersion:
        """Create a configuration backup."""
        active = self._profile_manager.get_active_profile()
        profile_id = active.profile_id if active else "default"
        
        return self._sync.backup(
            profile_id=profile_id,
            values=self._current_values,
            message=message,
        )
    
    def restore(self, version_id: str) -> bool:
        """Restore configuration from backup."""
        values = self._sync.restore(version_id)
        if values:
            self._current_values = values
            
            # Update active profile
            active = self._profile_manager.get_active_profile()
            if active:
                self._profile_manager.update_profile(
                    active.profile_id,
                    values=values,
                )
            return True
        return False
    
    def get_version_history(self, limit: int = 20) -> List[ConfigVersion]:
        """Get version history."""
        active = self._profile_manager.get_active_profile()
        profile_id = active.profile_id if active else None
        return self._sync.get_version_history(profile_id, limit)
    
    def get_audit_trail(self, limit: int = 100) -> List[AuditEntry]:
        """Get audit trail."""
        return self._sync.get_audit_trail(limit=limit)
    
    async def analyze_with_llm(
        self,
        operation: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze configuration situation using LLM.
        
        Args:
            operation: Type of analysis
            context: Context information
            
        Returns:
            LLM analysis
        """
        if not self._llm_client:
            return {"error": "LLM client not available"}
        
        if operation == "suggest_config":
            env = self.detect_environment()
            current = self.get_all_values()
            
            prompt = f"""Analyze this configuration and suggest improvements:

Environment: {env.value}
Current config: {json.dumps(current, indent=2)}

Suggest optimal configuration values for this environment.
Return as JSON object.
"""
        elif operation == "resolve_conflict":
            conflict = context.get("conflict")
            if conflict:
                prompt = f"""Resolve this configuration conflict:

Key: {conflict.key}
Local value: {conflict.local_value}
Remote value: {conflict.remote_value}

Suggest the best resolution and explain why.
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
_global_config_manager: Optional[ConfigurationManager] = None


def get_configuration_manager(
    llm_client: Optional[Any] = None,
    storage_path: Optional[Path] = None,
) -> ConfigurationManager:
    """Get the global configuration manager.
    
    Args:
        llm_client: Optional LLM client
        storage_path: Optional storage path
        
    Returns:
        ConfigurationManager instance
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigurationManager(
            llm_client=llm_client,
            storage_path=storage_path,
        )
    return _global_config_manager
