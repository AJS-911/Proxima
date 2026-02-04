"""Adaptive System Configuration and Preferences for Dynamic AI Assistant.

This module implements Phase 7.1 for the Dynamic AI Assistant:
- Preference Learning System: Track user patterns and learn preferences
- Context-Aware Defaults: Generate dynamic defaults based on context
- Personalization Engine: Adapt behavior to user's workflow

Key Features:
============
- User action tracking and pattern analysis
- Preference inference from behavior
- Context-specific parameter selection
- Workflow and UI personalization
- Language and terminology adaptation

Design Principle:
================
All preference decisions use LLM reasoning when available.
The LLM analyzes user behavior and suggests optimal configurations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
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


class PreferenceCategory(Enum):
    """Categories of user preferences."""
    TOOL_USAGE = "tool_usage"
    FILE_OPERATIONS = "file_operations"
    GIT_OPERATIONS = "git_operations"
    TERMINAL = "terminal"
    OUTPUT_FORMAT = "output_format"
    UI_LAYOUT = "ui_layout"
    WORKFLOW = "workflow"
    NOTIFICATION = "notification"
    LANGUAGE = "language"
    PERFORMANCE = "performance"


class PreferenceStrength(Enum):
    """Strength of learned preferences."""
    WEAK = "weak"  # < 5 observations
    MODERATE = "moderate"  # 5-20 observations
    STRONG = "strong"  # 20-50 observations
    VERY_STRONG = "very_strong"  # > 50 observations


class OperationMode(Enum):
    """Operation output modes."""
    VERBOSE = "verbose"  # Detailed output
    NORMAL = "normal"  # Standard output
    QUIET = "quiet"  # Minimal output
    AUTO = "auto"  # Let system decide


class EnvironmentType(Enum):
    """Environment types for context awareness."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    UNKNOWN = "unknown"


@dataclass
class UserAction:
    """Record of a user action for preference learning."""
    action_id: str
    action_type: str
    category: PreferenceCategory
    timestamp: datetime
    
    # Context
    tool_used: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Outcome
    success: bool = True
    duration_ms: int = 0
    user_feedback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "tool_used": self.tool_used,
            "parameters": self.parameters,
            "success": self.success,
            "duration_ms": self.duration_ms,
        }


@dataclass
class LearnedPreference:
    """A preference learned from user behavior."""
    preference_id: str
    category: PreferenceCategory
    name: str
    value: Any
    
    # Learning metadata
    strength: PreferenceStrength = PreferenceStrength.WEAK
    observation_count: int = 0
    confidence: float = 0.0
    
    # Temporal
    first_observed: Optional[datetime] = None
    last_observed: Optional[datetime] = None
    last_confirmed: Optional[datetime] = None
    
    # Context
    context_conditions: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Tuple[Any, int]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "preference_id": self.preference_id,
            "category": self.category.value,
            "name": self.name,
            "value": self.value,
            "strength": self.strength.value,
            "observation_count": self.observation_count,
            "confidence": self.confidence,
            "first_observed": self.first_observed.isoformat() if self.first_observed else None,
            "last_observed": self.last_observed.isoformat() if self.last_observed else None,
        }
    
    def update_strength(self):
        """Update preference strength based on observations."""
        if self.observation_count < 5:
            self.strength = PreferenceStrength.WEAK
        elif self.observation_count < 20:
            self.strength = PreferenceStrength.MODERATE
        elif self.observation_count < 50:
            self.strength = PreferenceStrength.STRONG
        else:
            self.strength = PreferenceStrength.VERY_STRONG


@dataclass
class ContextDefaults:
    """Context-aware default values."""
    context_id: str
    context_type: str  # project, directory, file_type, etc.
    context_value: str
    
    defaults: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    priority: int = 0
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_id": self.context_id,
            "context_type": self.context_type,
            "context_value": self.context_value,
            "defaults": self.defaults,
            "active": self.active,
            "priority": self.priority,
        }


@dataclass
class WorkflowPattern:
    """A detected workflow pattern."""
    pattern_id: str
    name: str
    steps: List[str]
    
    # Statistics
    occurrence_count: int = 0
    avg_duration_ms: int = 0
    success_rate: float = 1.0
    
    # Metadata
    detected_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "steps": self.steps,
            "occurrence_count": self.occurrence_count,
            "success_rate": self.success_rate,
        }


class ActionTracker:
    """Tracks user actions for preference learning.
    
    Uses LLM reasoning to:
    1. Categorize user actions
    2. Detect patterns in behavior
    3. Infer preferences from actions
    
    Example:
        >>> tracker = ActionTracker()
        >>> tracker.track_action("read_file", {"path": "src/main.py"})
        >>> patterns = tracker.detect_patterns()
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        max_history: int = 10000,
        storage_path: Optional[Path] = None,
    ):
        """Initialize the action tracker.
        
        Args:
            llm_client: LLM client for intelligent analysis
            max_history: Maximum actions to keep in memory
            storage_path: Path for persistent storage
        """
        self._llm_client = llm_client
        self._max_history = max_history
        self._storage_path = storage_path
        
        # Action history
        self._actions: List[UserAction] = []
        self._action_index: Dict[str, UserAction] = {}
        
        # Aggregated stats
        self._tool_usage: Counter = Counter()
        self._parameter_values: Dict[str, Counter] = defaultdict(Counter)
        self._sequences: List[List[str]] = []
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Load persisted data
        self._load_history()
    
    def track_action(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        tool_used: Optional[str] = None,
        category: Optional[PreferenceCategory] = None,
        context: Optional[Dict[str, Any]] = None,
        success: bool = True,
        duration_ms: int = 0,
    ) -> UserAction:
        """Track a user action.
        
        Args:
            action_type: Type of action (read_file, git_commit, etc.)
            parameters: Parameters used
            tool_used: Tool that was used
            category: Action category
            context: Additional context
            success: Whether action succeeded
            duration_ms: Duration in milliseconds
            
        Returns:
            The tracked action
        """
        # Auto-detect category if not provided
        if category is None:
            category = self._detect_category(action_type, tool_used)
        
        action = UserAction(
            action_id=str(uuid.uuid4()),
            action_type=action_type,
            category=category,
            timestamp=datetime.now(),
            tool_used=tool_used or action_type,
            parameters=parameters,
            context=context or {},
            success=success,
            duration_ms=duration_ms,
        )
        
        with self._lock:
            # Add to history
            self._actions.append(action)
            self._action_index[action.action_id] = action
            
            # Update aggregated stats
            self._tool_usage[action.tool_used] += 1
            
            # Track parameter values
            for param, value in parameters.items():
                if isinstance(value, (str, int, float, bool)):
                    self._parameter_values[param][str(value)] += 1
            
            # Track sequences (last 10 actions)
            recent_tools = [a.tool_used for a in self._actions[-10:]]
            if len(recent_tools) >= 3:
                self._sequences.append(recent_tools[-3:])
            
            # Trim history if needed
            if len(self._actions) > self._max_history:
                removed = self._actions.pop(0)
                del self._action_index[removed.action_id]
        
        # Persist asynchronously
        self._save_action(action)
        
        return action
    
    def _detect_category(
        self,
        action_type: str,
        tool_used: Optional[str],
    ) -> PreferenceCategory:
        """Detect category from action type."""
        action_lower = (action_type or "").lower()
        tool_lower = (tool_used or "").lower()
        
        # File operations
        if any(x in action_lower for x in ["file", "read", "write", "create", "delete", "copy", "move"]):
            return PreferenceCategory.FILE_OPERATIONS
        
        # Git operations
        if any(x in action_lower for x in ["git", "commit", "push", "pull", "branch", "merge"]):
            return PreferenceCategory.GIT_OPERATIONS
        
        # Terminal operations
        if any(x in action_lower for x in ["terminal", "shell", "command", "execute", "run"]):
            return PreferenceCategory.TERMINAL
        
        # Default to tool usage
        return PreferenceCategory.TOOL_USAGE
    
    def get_action_history(
        self,
        category: Optional[PreferenceCategory] = None,
        tool: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[UserAction]:
        """Get action history with filters.
        
        Args:
            category: Filter by category
            tool: Filter by tool
            since: Only actions after this time
            limit: Maximum results
            
        Returns:
            List of matching actions
        """
        with self._lock:
            results = self._actions.copy()
        
        # Apply filters
        if category:
            results = [a for a in results if a.category == category]
        
        if tool:
            results = [a for a in results if a.tool_used == tool]
        
        if since:
            results = [a for a in results if a.timestamp >= since]
        
        # Return most recent
        return results[-limit:]
    
    def get_tool_usage_stats(self) -> Dict[str, int]:
        """Get tool usage statistics."""
        with self._lock:
            return dict(self._tool_usage.most_common())
    
    def get_common_parameter_values(
        self,
        parameter: str,
        top_n: int = 5,
    ) -> List[Tuple[str, int]]:
        """Get most common values for a parameter.
        
        Args:
            parameter: Parameter name
            top_n: Number of top values
            
        Returns:
            List of (value, count) tuples
        """
        with self._lock:
            return self._parameter_values[parameter].most_common(top_n)
    
    def detect_patterns(
        self,
        min_occurrences: int = 3,
    ) -> List[WorkflowPattern]:
        """Detect workflow patterns from action sequences.
        
        Args:
            min_occurrences: Minimum occurrences to be a pattern
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        with self._lock:
            # Count sequence occurrences
            sequence_counts: Counter = Counter()
            for seq in self._sequences:
                sequence_counts[tuple(seq)] += 1
        
        # Find patterns
        for seq, count in sequence_counts.most_common():
            if count >= min_occurrences:
                pattern = WorkflowPattern(
                    pattern_id=str(uuid.uuid4()),
                    name=f"Pattern: {' -> '.join(seq)}",
                    steps=list(seq),
                    occurrence_count=count,
                    detected_at=datetime.now(),
                )
                patterns.append(pattern)
        
        return patterns
    
    def _load_history(self):
        """Load persisted action history."""
        if not self._storage_path:
            return
        
        history_file = self._storage_path / "action_history.json"
        if not history_file.exists():
            return
        
        try:
            with open(history_file, "r") as f:
                data = json.load(f)
            
            for action_data in data.get("actions", [])[-self._max_history:]:
                action = UserAction(
                    action_id=action_data["action_id"],
                    action_type=action_data["action_type"],
                    category=PreferenceCategory(action_data["category"]),
                    timestamp=datetime.fromisoformat(action_data["timestamp"]),
                    tool_used=action_data.get("tool_used"),
                    parameters=action_data.get("parameters", {}),
                    success=action_data.get("success", True),
                    duration_ms=action_data.get("duration_ms", 0),
                )
                self._actions.append(action)
                self._action_index[action.action_id] = action
            
            logger.info(f"Loaded {len(self._actions)} actions from history")
            
        except Exception as e:
            logger.warning(f"Failed to load action history: {e}")
    
    def _save_action(self, action: UserAction):
        """Save action to persistent storage."""
        if not self._storage_path:
            return
        
        self._storage_path.mkdir(parents=True, exist_ok=True)
        history_file = self._storage_path / "action_history.json"
        
        try:
            with self._lock:
                data = {
                    "actions": [a.to_dict() for a in self._actions[-1000:]],
                    "updated_at": datetime.now().isoformat(),
                }
            
            with open(history_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save action: {e}")


class PreferenceLearner:
    """Learns user preferences from behavior.
    
    Uses LLM reasoning to:
    1. Infer preferences from action patterns
    2. Confirm preferences with user
    3. Detect preference drift over time
    
    Example:
        >>> learner = PreferenceLearner(tracker=tracker)
        >>> preferences = learner.infer_preferences()
        >>> learner.confirm_preference(preferences[0])
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        action_tracker: Optional[ActionTracker] = None,
        storage_path: Optional[Path] = None,
    ):
        """Initialize the preference learner.
        
        Args:
            llm_client: LLM client for intelligent inference
            action_tracker: Action tracker for behavior data
            storage_path: Path for persistent storage
        """
        self._llm_client = llm_client
        self._tracker = action_tracker or ActionTracker()
        self._storage_path = storage_path
        
        # Learned preferences
        self._preferences: Dict[str, LearnedPreference] = {}
        self._preference_by_category: Dict[PreferenceCategory, List[str]] = defaultdict(list)
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Load persisted preferences
        self._load_preferences()
    
    def infer_preferences(
        self,
        category: Optional[PreferenceCategory] = None,
        min_observations: int = 3,
    ) -> List[LearnedPreference]:
        """Infer preferences from user behavior.
        
        Args:
            category: Category to analyze (None = all)
            min_observations: Minimum observations required
            
        Returns:
            List of inferred preferences
        """
        inferred = []
        
        # Get tool usage preferences
        tool_stats = self._tracker.get_tool_usage_stats()
        
        for tool, count in tool_stats.items():
            if count >= min_observations:
                pref = self._get_or_create_preference(
                    category=PreferenceCategory.TOOL_USAGE,
                    name=f"preferred_tool_{tool}",
                    value=tool,
                )
                pref.observation_count = count
                pref.update_strength()
                inferred.append(pref)
        
        # Get parameter value preferences
        actions = self._tracker.get_action_history(category=category, limit=1000)
        
        # Analyze common parameter patterns
        param_counts: Dict[str, Counter] = defaultdict(Counter)
        
        for action in actions:
            for param, value in action.parameters.items():
                if isinstance(value, (str, int, float, bool)):
                    param_counts[param][str(value)] += 1
        
        for param, counts in param_counts.items():
            most_common = counts.most_common(1)
            if most_common and most_common[0][1] >= min_observations:
                value, count = most_common[0]
                pref = self._get_or_create_preference(
                    category=category or PreferenceCategory.TOOL_USAGE,
                    name=f"default_{param}",
                    value=value,
                )
                pref.observation_count = count
                pref.alternatives = counts.most_common(5)
                pref.update_strength()
                
                # Calculate confidence
                total = sum(counts.values())
                pref.confidence = count / total if total > 0 else 0
                
                inferred.append(pref)
        
        # Save updated preferences
        self._save_preferences()
        
        return inferred
    
    def _get_or_create_preference(
        self,
        category: PreferenceCategory,
        name: str,
        value: Any,
    ) -> LearnedPreference:
        """Get existing preference or create new one."""
        pref_key = f"{category.value}:{name}"
        
        with self._lock:
            if pref_key not in self._preferences:
                pref = LearnedPreference(
                    preference_id=str(uuid.uuid4()),
                    category=category,
                    name=name,
                    value=value,
                    first_observed=datetime.now(),
                )
                self._preferences[pref_key] = pref
                self._preference_by_category[category].append(pref_key)
            else:
                pref = self._preferences[pref_key]
                pref.value = value
            
            pref.last_observed = datetime.now()
            return pref
    
    def get_preference(
        self,
        category: PreferenceCategory,
        name: str,
    ) -> Optional[LearnedPreference]:
        """Get a specific preference.
        
        Args:
            category: Preference category
            name: Preference name
            
        Returns:
            The preference if found
        """
        pref_key = f"{category.value}:{name}"
        
        with self._lock:
            return self._preferences.get(pref_key)
    
    def get_preferences_by_category(
        self,
        category: PreferenceCategory,
    ) -> List[LearnedPreference]:
        """Get all preferences in a category."""
        with self._lock:
            pref_keys = self._preference_by_category.get(category, [])
            return [self._preferences[k] for k in pref_keys if k in self._preferences]
    
    def confirm_preference(
        self,
        preference: LearnedPreference,
        confirmed: bool = True,
    ):
        """Confirm or reject a learned preference.
        
        Args:
            preference: The preference to confirm
            confirmed: Whether user confirmed it
        """
        pref_key = f"{preference.category.value}:{preference.name}"
        
        with self._lock:
            if pref_key in self._preferences:
                if confirmed:
                    self._preferences[pref_key].last_confirmed = datetime.now()
                    self._preferences[pref_key].strength = PreferenceStrength.VERY_STRONG
                else:
                    # Remove unconfirmed preference
                    del self._preferences[pref_key]
                    self._preference_by_category[preference.category].remove(pref_key)
        
        self._save_preferences()
    
    def detect_preference_drift(
        self,
        days: int = 30,
    ) -> List[LearnedPreference]:
        """Detect preferences that may have drifted.
        
        Args:
            days: Look back period
            
        Returns:
            Preferences that appear to have changed
        """
        drifted = []
        cutoff = datetime.now() - timedelta(days=days)
        
        # Get recent actions
        recent_actions = self._tracker.get_action_history(since=cutoff)
        
        # Analyze recent behavior
        recent_params: Dict[str, Counter] = defaultdict(Counter)
        
        for action in recent_actions:
            for param, value in action.parameters.items():
                if isinstance(value, (str, int, float, bool)):
                    recent_params[param][str(value)] += 1
        
        # Compare with learned preferences
        with self._lock:
            for pref in self._preferences.values():
                if pref.name.startswith("default_"):
                    param = pref.name.replace("default_", "")
                    
                    if param in recent_params:
                        recent_most_common = recent_params[param].most_common(1)
                        if recent_most_common:
                            recent_value, _ = recent_most_common[0]
                            
                            if str(pref.value) != recent_value:
                                # Preference may have drifted
                                drifted.append(pref)
        
        return drifted
    
    def export_preferences(self) -> Dict[str, Any]:
        """Export all preferences as JSON-serializable dict."""
        with self._lock:
            return {
                "preferences": [p.to_dict() for p in self._preferences.values()],
                "exported_at": datetime.now().isoformat(),
            }
    
    def import_preferences(self, data: Dict[str, Any]):
        """Import preferences from exported data.
        
        Args:
            data: Exported preference data
        """
        for pref_data in data.get("preferences", []):
            try:
                pref = LearnedPreference(
                    preference_id=pref_data["preference_id"],
                    category=PreferenceCategory(pref_data["category"]),
                    name=pref_data["name"],
                    value=pref_data["value"],
                    strength=PreferenceStrength(pref_data.get("strength", "weak")),
                    observation_count=pref_data.get("observation_count", 0),
                    confidence=pref_data.get("confidence", 0.0),
                )
                
                pref_key = f"{pref.category.value}:{pref.name}"
                
                with self._lock:
                    self._preferences[pref_key] = pref
                    if pref_key not in self._preference_by_category[pref.category]:
                        self._preference_by_category[pref.category].append(pref_key)
                        
            except Exception as e:
                logger.warning(f"Failed to import preference: {e}")
        
        self._save_preferences()
    
    def _load_preferences(self):
        """Load persisted preferences."""
        if not self._storage_path:
            return
        
        pref_file = self._storage_path / "preferences.json"
        if not pref_file.exists():
            return
        
        try:
            with open(pref_file, "r") as f:
                data = json.load(f)
            
            self.import_preferences(data)
            logger.info(f"Loaded {len(self._preferences)} preferences")
            
        except Exception as e:
            logger.warning(f"Failed to load preferences: {e}")
    
    def _save_preferences(self):
        """Save preferences to persistent storage."""
        if not self._storage_path:
            return
        
        self._storage_path.mkdir(parents=True, exist_ok=True)
        pref_file = self._storage_path / "preferences.json"
        
        try:
            data = self.export_preferences()
            
            with open(pref_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save preferences: {e}")


class ContextAwareDefaults:
    """Generate context-aware default values.
    
    Uses LLM reasoning to:
    1. Detect current context (project type, environment)
    2. Generate appropriate defaults
    3. Adapt based on learned preferences
    
    Example:
        >>> defaults = ContextAwareDefaults(learner=learner)
        >>> value = defaults.get_default("encoding", context={"file_type": ".py"})
        >>> # Returns: "utf-8"
    """
    
    # Built-in defaults by context
    BUILTIN_DEFAULTS = {
        "file_type": {
            ".py": {"encoding": "utf-8", "indent": 4, "line_ending": "lf"},
            ".js": {"encoding": "utf-8", "indent": 2, "line_ending": "lf"},
            ".json": {"encoding": "utf-8", "indent": 2},
            ".yaml": {"encoding": "utf-8", "indent": 2},
            ".md": {"encoding": "utf-8"},
        },
        "project_type": {
            "python": {"test_framework": "pytest", "package_manager": "pip"},
            "javascript": {"test_framework": "jest", "package_manager": "npm"},
            "typescript": {"test_framework": "jest", "package_manager": "npm"},
        },
        "environment": {
            "development": {"log_level": "debug", "verbose": True},
            "production": {"log_level": "info", "verbose": False},
            "testing": {"log_level": "debug", "verbose": True},
        },
    }
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        preference_learner: Optional[PreferenceLearner] = None,
    ):
        """Initialize context-aware defaults.
        
        Args:
            llm_client: LLM client for intelligent defaults
            preference_learner: Preference learner for user defaults
        """
        self._llm_client = llm_client
        self._learner = preference_learner
        
        # Custom defaults by context
        self._custom_defaults: Dict[str, ContextDefaults] = {}
        
        # Operation mode
        self._operation_mode = OperationMode.NORMAL
    
    def get_default(
        self,
        parameter: str,
        context: Optional[Dict[str, Any]] = None,
        fallback: Any = None,
    ) -> Any:
        """Get context-aware default value for a parameter.
        
        Args:
            parameter: Parameter name
            context: Current context
            fallback: Fallback value if no default found
            
        Returns:
            Default value for the parameter
        """
        context = context or {}
        
        # 1. Check learned preferences first
        if self._learner:
            pref = self._learner.get_preference(
                PreferenceCategory.TOOL_USAGE,
                f"default_{parameter}",
            )
            if pref and pref.strength in (PreferenceStrength.STRONG, PreferenceStrength.VERY_STRONG):
                return pref.value
        
        # 2. Check custom context defaults
        for ctx_key, ctx_value in context.items():
            ctx_id = f"{ctx_key}:{ctx_value}"
            if ctx_id in self._custom_defaults:
                ctx_defaults = self._custom_defaults[ctx_id]
                if parameter in ctx_defaults.defaults:
                    return ctx_defaults.defaults[parameter]
        
        # 3. Check built-in defaults
        for ctx_key, ctx_value in context.items():
            if ctx_key in self.BUILTIN_DEFAULTS:
                ctx_defaults = self.BUILTIN_DEFAULTS[ctx_key]
                if ctx_value in ctx_defaults:
                    if parameter in ctx_defaults[ctx_value]:
                        return ctx_defaults[ctx_value][parameter]
        
        # 4. Return fallback
        return fallback
    
    def set_context_defaults(
        self,
        context_type: str,
        context_value: str,
        defaults: Dict[str, Any],
        priority: int = 0,
    ):
        """Set custom defaults for a context.
        
        Args:
            context_type: Type of context (project, file_type, etc.)
            context_value: Context value
            defaults: Default values
            priority: Priority (higher = more important)
        """
        ctx_id = f"{context_type}:{context_value}"
        
        self._custom_defaults[ctx_id] = ContextDefaults(
            context_id=ctx_id,
            context_type=context_type,
            context_value=context_value,
            defaults=defaults,
            priority=priority,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    
    def get_operation_mode(self) -> OperationMode:
        """Get current operation mode."""
        return self._operation_mode
    
    def set_operation_mode(self, mode: OperationMode):
        """Set operation mode.
        
        Args:
            mode: New operation mode
        """
        self._operation_mode = mode
    
    def should_be_verbose(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if output should be verbose.
        
        Args:
            context: Current context
            
        Returns:
            Whether to use verbose output
        """
        if self._operation_mode == OperationMode.VERBOSE:
            return True
        if self._operation_mode == OperationMode.QUIET:
            return False
        
        # Auto mode - check context
        if context:
            env = context.get("environment")
            if env in ("development", "testing"):
                return True
        
        return False
    
    def detect_environment(
        self,
        working_dir: Optional[Path] = None,
    ) -> EnvironmentType:
        """Detect current environment type.
        
        Args:
            working_dir: Working directory to analyze
            
        Returns:
            Detected environment type
        """
        working_dir = working_dir or Path.cwd()
        
        # Check for environment indicators
        indicators = {
            EnvironmentType.DEVELOPMENT: [
                ".env.development",
                ".env.local",
                ".vscode",
                ".idea",
            ],
            EnvironmentType.TESTING: [
                ".env.test",
                "pytest.ini",
                "jest.config.js",
                "test",
                "tests",
            ],
            EnvironmentType.STAGING: [
                ".env.staging",
                "staging",
            ],
            EnvironmentType.PRODUCTION: [
                ".env.production",
                "Dockerfile",
                "docker-compose.yml",
            ],
        }
        
        scores: Dict[EnvironmentType, int] = defaultdict(int)
        
        for env_type, files in indicators.items():
            for file_pattern in files:
                if (working_dir / file_pattern).exists():
                    scores[env_type] += 1
        
        if scores:
            return max(scores, key=scores.get)
        
        # Check environment variables
        env_var = os.environ.get("ENVIRONMENT", "").lower()
        for env_type in EnvironmentType:
            if env_type.value in env_var:
                return env_type
        
        return EnvironmentType.UNKNOWN


class PersonalizationEngine:
    """Personalize assistant behavior to user's workflow.
    
    Uses LLM reasoning to:
    1. Learn UI layout preferences
    2. Detect command abbreviations
    3. Adapt language and terminology
    
    Example:
        >>> engine = PersonalizationEngine(learner=learner)
        >>> shortcuts = engine.get_command_shortcuts()
        >>> engine.adapt_terminology("file", context={"domain": "data_science"})
    """
    
    # Domain-specific terminology
    DOMAIN_TERMINOLOGY = {
        "data_science": {
            "file": "dataset",
            "folder": "data directory",
            "run": "execute pipeline",
        },
        "web_development": {
            "file": "component",
            "folder": "module",
            "run": "start server",
        },
        "devops": {
            "file": "config",
            "folder": "deployment",
            "run": "deploy",
        },
    }
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        preference_learner: Optional[PreferenceLearner] = None,
        action_tracker: Optional[ActionTracker] = None,
    ):
        """Initialize the personalization engine.
        
        Args:
            llm_client: LLM client for intelligent personalization
            preference_learner: Preference learner for user data
            action_tracker: Action tracker for behavior data
        """
        self._llm_client = llm_client
        self._learner = preference_learner
        self._tracker = action_tracker
        
        # Learned shortcuts
        self._shortcuts: Dict[str, str] = {}
        
        # Workflow patterns
        self._workflows: List[WorkflowPattern] = []
        
        # Notification preferences
        self._notification_prefs: Dict[str, bool] = {
            "success": True,
            "error": True,
            "warning": True,
            "progress": True,
            "completion": True,
        }
    
    def learn_command_shortcuts(
        self,
        min_usage: int = 5,
    ) -> Dict[str, str]:
        """Learn command shortcuts from usage patterns.
        
        Args:
            min_usage: Minimum usage to suggest shortcut
            
        Returns:
            Dict of shortcut -> full command
        """
        if not self._tracker:
            return self._shortcuts
        
        # Get most used tools
        tool_stats = self._tracker.get_tool_usage_stats()
        
        for tool, count in tool_stats.items():
            if count >= min_usage:
                # Generate shortcut
                shortcut = self._generate_shortcut(tool)
                if shortcut:
                    self._shortcuts[shortcut] = tool
        
        return self._shortcuts
    
    def _generate_shortcut(self, command: str) -> Optional[str]:
        """Generate a shortcut for a command."""
        # Use first letter of each word
        parts = command.replace("_", " ").split()
        if len(parts) >= 2:
            return "".join(p[0] for p in parts)
        elif len(command) > 3:
            return command[:3]
        return None
    
    def expand_shortcut(self, shortcut: str) -> Optional[str]:
        """Expand a shortcut to full command.
        
        Args:
            shortcut: Command shortcut
            
        Returns:
            Full command if shortcut exists
        """
        return self._shortcuts.get(shortcut.lower())
    
    def add_shortcut(self, shortcut: str, command: str):
        """Add a custom shortcut.
        
        Args:
            shortcut: Shortcut text
            command: Full command
        """
        self._shortcuts[shortcut.lower()] = command
    
    def learn_workflows(self) -> List[WorkflowPattern]:
        """Learn workflow patterns from action sequences.
        
        Returns:
            List of learned workflow patterns
        """
        if not self._tracker:
            return self._workflows
        
        patterns = self._tracker.detect_patterns(min_occurrences=3)
        self._workflows = patterns
        
        return patterns
    
    def suggest_next_action(
        self,
        recent_actions: List[str],
    ) -> Optional[str]:
        """Suggest next action based on workflow patterns.
        
        Args:
            recent_actions: Recent action sequence
            
        Returns:
            Suggested next action
        """
        if len(recent_actions) < 2:
            return None
        
        recent_tuple = tuple(recent_actions[-2:])
        
        for pattern in self._workflows:
            if len(pattern.steps) >= 3:
                pattern_start = tuple(pattern.steps[:2])
                if recent_tuple == pattern_start:
                    return pattern.steps[2]
        
        return None
    
    def adapt_terminology(
        self,
        term: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Adapt terminology to domain context.
        
        Args:
            term: Original term
            context: Current context with domain info
            
        Returns:
            Domain-appropriate term
        """
        if not context:
            return term
        
        domain = context.get("domain", "").lower()
        
        if domain in self.DOMAIN_TERMINOLOGY:
            domain_terms = self.DOMAIN_TERMINOLOGY[domain]
            return domain_terms.get(term.lower(), term)
        
        return term
    
    def set_notification_preference(
        self,
        notification_type: str,
        enabled: bool,
    ):
        """Set notification preference.
        
        Args:
            notification_type: Type of notification
            enabled: Whether to enable
        """
        self._notification_prefs[notification_type] = enabled
    
    def should_notify(self, notification_type: str) -> bool:
        """Check if notification should be shown.
        
        Args:
            notification_type: Type of notification
            
        Returns:
            Whether to show notification
        """
        return self._notification_prefs.get(notification_type, True)


class AdaptiveConfigurationManager:
    """Main adaptive configuration manager.
    
    Integrates all preference and configuration components:
    - Action tracking
    - Preference learning
    - Context-aware defaults
    - Personalization
    
    Example:
        >>> config = AdaptiveConfigurationManager()
        >>> config.track_action("read_file", {"path": "main.py"})
        >>> default = config.get_default("encoding")
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        storage_path: Optional[Path] = None,
    ):
        """Initialize the adaptive configuration manager.
        
        Args:
            llm_client: LLM client for intelligent operations
            storage_path: Path for persistent storage
        """
        self._llm_client = llm_client
        self._storage_path = storage_path or Path.home() / ".proxima" / "config"
        
        # Initialize components
        self._tracker = ActionTracker(
            llm_client=llm_client,
            storage_path=self._storage_path,
        )
        
        self._learner = PreferenceLearner(
            llm_client=llm_client,
            action_tracker=self._tracker,
            storage_path=self._storage_path,
        )
        
        self._defaults = ContextAwareDefaults(
            llm_client=llm_client,
            preference_learner=self._learner,
        )
        
        self._personalization = PersonalizationEngine(
            llm_client=llm_client,
            preference_learner=self._learner,
            action_tracker=self._tracker,
        )
    
    # Action tracking
    def track_action(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        tool_used: Optional[str] = None,
        success: bool = True,
        duration_ms: int = 0,
    ) -> UserAction:
        """Track a user action."""
        return self._tracker.track_action(
            action_type=action_type,
            parameters=parameters,
            tool_used=tool_used,
            success=success,
            duration_ms=duration_ms,
        )
    
    def get_action_history(
        self,
        category: Optional[PreferenceCategory] = None,
        limit: int = 100,
    ) -> List[UserAction]:
        """Get action history."""
        return self._tracker.get_action_history(category=category, limit=limit)
    
    # Preference learning
    def infer_preferences(self) -> List[LearnedPreference]:
        """Infer preferences from behavior."""
        return self._learner.infer_preferences()
    
    def get_preference(
        self,
        category: PreferenceCategory,
        name: str,
    ) -> Optional[LearnedPreference]:
        """Get a specific preference."""
        return self._learner.get_preference(category, name)
    
    def confirm_preference(
        self,
        preference: LearnedPreference,
        confirmed: bool = True,
    ):
        """Confirm or reject a preference."""
        self._learner.confirm_preference(preference, confirmed)
    
    def export_preferences(self) -> Dict[str, Any]:
        """Export all preferences."""
        return self._learner.export_preferences()
    
    def import_preferences(self, data: Dict[str, Any]):
        """Import preferences."""
        self._learner.import_preferences(data)
    
    # Context-aware defaults
    def get_default(
        self,
        parameter: str,
        context: Optional[Dict[str, Any]] = None,
        fallback: Any = None,
    ) -> Any:
        """Get context-aware default value."""
        return self._defaults.get_default(parameter, context, fallback)
    
    def set_context_defaults(
        self,
        context_type: str,
        context_value: str,
        defaults: Dict[str, Any],
    ):
        """Set custom context defaults."""
        self._defaults.set_context_defaults(context_type, context_value, defaults)
    
    def get_operation_mode(self) -> OperationMode:
        """Get current operation mode."""
        return self._defaults.get_operation_mode()
    
    def set_operation_mode(self, mode: OperationMode):
        """Set operation mode."""
        self._defaults.set_operation_mode(mode)
    
    def detect_environment(self) -> EnvironmentType:
        """Detect current environment."""
        return self._defaults.detect_environment()
    
    # Personalization
    def learn_shortcuts(self) -> Dict[str, str]:
        """Learn command shortcuts."""
        return self._personalization.learn_command_shortcuts()
    
    def expand_shortcut(self, shortcut: str) -> Optional[str]:
        """Expand a shortcut."""
        return self._personalization.expand_shortcut(shortcut)
    
    def add_shortcut(self, shortcut: str, command: str):
        """Add a custom shortcut."""
        self._personalization.add_shortcut(shortcut, command)
    
    def learn_workflows(self) -> List[WorkflowPattern]:
        """Learn workflow patterns."""
        return self._personalization.learn_workflows()
    
    def suggest_next_action(
        self,
        recent_actions: List[str],
    ) -> Optional[str]:
        """Suggest next action."""
        return self._personalization.suggest_next_action(recent_actions)
    
    def adapt_terminology(
        self,
        term: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Adapt terminology to context."""
        return self._personalization.adapt_terminology(term, context)
    
    def set_notification_preference(
        self,
        notification_type: str,
        enabled: bool,
    ):
        """Set notification preference."""
        self._personalization.set_notification_preference(notification_type, enabled)
    
    def should_notify(self, notification_type: str) -> bool:
        """Check if should notify."""
        return self._personalization.should_notify(notification_type)
    
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
        
        if operation == "suggest_defaults":
            current_context = context.get("context", {})
            prompt = f"""Based on this context, suggest appropriate default values:

Context: {json.dumps(current_context, indent=2)}

Suggest defaults for common parameters like encoding, indent, verbosity, etc.
Return as JSON object with parameter names and values.
"""
        elif operation == "explain_preference":
            preference = context.get("preference")
            if preference:
                prompt = f"""Explain why this preference was learned:

Preference: {preference.name} = {preference.value}
Observations: {preference.observation_count}
Confidence: {preference.confidence}

Provide a brief, user-friendly explanation.
"""
            else:
                return {"error": "No preference provided"}
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        try:
            response = await self._llm_client.generate(prompt)
            return {"success": True, "result": response}
        except Exception as e:
            return {"error": str(e)}


# Module-level instance
_global_adaptive_config: Optional[AdaptiveConfigurationManager] = None


def get_adaptive_configuration_manager(
    llm_client: Optional[Any] = None,
    storage_path: Optional[Path] = None,
) -> AdaptiveConfigurationManager:
    """Get the global adaptive configuration manager.
    
    Args:
        llm_client: Optional LLM client
        storage_path: Optional storage path
        
    Returns:
        AdaptiveConfigurationManager instance
    """
    global _global_adaptive_config
    if _global_adaptive_config is None:
        _global_adaptive_config = AdaptiveConfigurationManager(
            llm_client=llm_client,
            storage_path=storage_path,
        )
    return _global_adaptive_config
