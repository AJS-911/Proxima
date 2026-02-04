"""Intelligent Error Detection System for Dynamic AI Assistant.

This module implements Phase 8.1 for the Dynamic AI Assistant:
- Error Classification System
- Contextual Error Analysis
- Natural Language Error Explanation

Key Features:
============
- ML-based error categorization
- Error severity assessment
- Error pattern recognition and clustering
- Root cause analysis
- Contextual error capture
- LLM-generated error explanations

Design Principle:
================
All error analysis uses LLM reasoning when available.
The LLM analyzes errors and suggests causes/solutions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
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


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Error categories for classification."""
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    GIT = "git"
    GITHUB = "github"
    TERMINAL = "terminal"
    BUILD = "build"
    RUNTIME = "runtime"
    MEMORY = "memory"
    DISK = "disk"
    SYNTAX = "syntax"
    LOGIC = "logic"
    CONCURRENCY = "concurrency"
    UNKNOWN = "unknown"


class ErrorState(Enum):
    """Error lifecycle states."""
    DETECTED = "detected"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    RECOVERING = "recovering"
    RECOVERED = "recovered"
    FAILED = "failed"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"


@dataclass
class ErrorContext:
    """Context captured when an error occurs."""
    timestamp: datetime
    operation: str
    
    # System state
    working_directory: str = ""
    environment_vars: Dict[str, str] = field(default_factory=dict)
    python_version: str = ""
    platform: str = ""
    
    # Recent operations
    recent_operations: List[str] = field(default_factory=list)
    
    # Resource state
    memory_usage: Optional[float] = None
    disk_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    
    # Additional context
    user_input: Optional[str] = None
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "working_directory": self.working_directory,
            "python_version": self.python_version,
            "platform": self.platform,
            "recent_operations": self.recent_operations,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "cpu_usage": self.cpu_usage,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
        }


@dataclass
class ErrorPattern:
    """A recognized error pattern."""
    pattern_id: str
    pattern: str  # Regex pattern or keyword
    category: ErrorCategory
    severity: ErrorSeverity
    
    # Pattern metadata
    description: str = ""
    common_causes: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)
    
    # Statistics
    occurrence_count: int = 0
    last_seen: Optional[datetime] = None
    
    def matches(self, error_message: str) -> bool:
        """Check if error message matches this pattern."""
        try:
            return bool(re.search(self.pattern, error_message, re.IGNORECASE))
        except re.error:
            return self.pattern.lower() in error_message.lower()


@dataclass
class AnalyzedError:
    """An analyzed error with full context."""
    error_id: str
    original_error: Exception
    
    # Classification
    category: ErrorCategory
    severity: ErrorSeverity
    state: ErrorState
    
    # Error details
    error_type: str = ""
    error_message: str = ""
    stack_trace: str = ""
    
    # Context
    context: Optional[ErrorContext] = None
    
    # Analysis
    root_cause: Optional[str] = None
    impact: Optional[str] = None
    related_errors: List[str] = field(default_factory=list)
    
    # Explanation
    technical_explanation: Optional[str] = None
    user_explanation: Optional[str] = None
    suggested_actions: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: Optional[datetime] = None
    analyzed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "state": self.state.value,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "root_cause": self.root_cause,
            "impact": self.impact,
            "technical_explanation": self.technical_explanation,
            "user_explanation": self.user_explanation,
            "suggested_actions": self.suggested_actions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class ErrorCluster:
    """A cluster of related errors."""
    cluster_id: str
    name: str
    
    error_ids: List[str] = field(default_factory=list)
    common_category: Optional[ErrorCategory] = None
    common_pattern: Optional[str] = None
    
    # Statistics
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None
    total_occurrences: int = 0


class ErrorClassifier:
    """Classify errors using patterns and LLM reasoning.
    
    Uses LLM reasoning to:
    1. Categorize errors when patterns don't match
    2. Assess error severity
    3. Identify root causes
    
    Example:
        >>> classifier = ErrorClassifier()
        >>> result = classifier.classify(error)
        >>> print(result.category, result.severity)
    """
    
    # Built-in error patterns
    BUILTIN_PATTERNS = [
        # Filesystem errors
        ErrorPattern(
            pattern_id="fs_not_found",
            pattern=r"(FileNotFoundError|No such file|cannot find|does not exist)",
            category=ErrorCategory.FILESYSTEM,
            severity=ErrorSeverity.ERROR,
            description="File or directory not found",
            common_causes=["Path typo", "File deleted", "Wrong directory"],
            suggested_fixes=["Check path spelling", "Verify file exists", "Use absolute path"],
        ),
        ErrorPattern(
            pattern_id="fs_permission",
            pattern=r"(PermissionError|Permission denied|Access is denied)",
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.ERROR,
            description="Permission denied accessing file/directory",
            common_causes=["Insufficient privileges", "File locked", "Read-only filesystem"],
            suggested_fixes=["Run with elevated privileges", "Close programs using file", "Check permissions"],
        ),
        ErrorPattern(
            pattern_id="fs_disk_full",
            pattern=r"(No space left|disk is full|not enough space|ENOSPC)",
            category=ErrorCategory.DISK,
            severity=ErrorSeverity.CRITICAL,
            description="Disk space exhausted",
            common_causes=["Large files", "Log accumulation", "Temp files"],
            suggested_fixes=["Free disk space", "Delete unnecessary files", "Increase storage"],
        ),
        
        # Network errors
        ErrorPattern(
            pattern_id="net_connection",
            pattern=r"(ConnectionError|Connection refused|ConnectionRefusedError|ECONNREFUSED)",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            description="Network connection failed",
            common_causes=["Server down", "Wrong port", "Firewall blocking"],
            suggested_fixes=["Check server status", "Verify port number", "Check firewall rules"],
        ),
        ErrorPattern(
            pattern_id="net_timeout",
            pattern=r"(TimeoutError|timed out|deadline exceeded|ETIMEDOUT)",
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.WARNING,
            description="Operation timed out",
            common_causes=["Slow network", "Server overloaded", "Large data transfer"],
            suggested_fixes=["Increase timeout", "Retry operation", "Check network"],
        ),
        ErrorPattern(
            pattern_id="net_dns",
            pattern=r"(Name or service not known|getaddrinfo failed|DNS|ENOTFOUND)",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            description="DNS resolution failed",
            common_causes=["Invalid hostname", "DNS server issue", "No internet"],
            suggested_fixes=["Check hostname spelling", "Try IP address", "Check DNS settings"],
        ),
        
        # Authentication errors
        ErrorPattern(
            pattern_id="auth_invalid",
            pattern=r"(401|Unauthorized|authentication failed|invalid credentials|bad credentials)",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.ERROR,
            description="Authentication failed",
            common_causes=["Wrong credentials", "Expired token", "Account locked"],
            suggested_fixes=["Verify credentials", "Refresh token", "Check account status"],
        ),
        ErrorPattern(
            pattern_id="auth_forbidden",
            pattern=r"(403|Forbidden|access denied|not authorized)",
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.ERROR,
            description="Access forbidden",
            common_causes=["Missing permissions", "Resource restricted", "API limit"],
            suggested_fixes=["Request access", "Check API limits", "Verify permissions"],
        ),
        
        # Git errors
        ErrorPattern(
            pattern_id="git_not_repo",
            pattern=r"(not a git repository|fatal: not in a git directory)",
            category=ErrorCategory.GIT,
            severity=ErrorSeverity.ERROR,
            description="Not in a Git repository",
            common_causes=["Wrong directory", "Repository not initialized"],
            suggested_fixes=["Navigate to repo directory", "Run git init"],
        ),
        ErrorPattern(
            pattern_id="git_conflict",
            pattern=r"(CONFLICT|merge conflict|Automatic merge failed)",
            category=ErrorCategory.GIT,
            severity=ErrorSeverity.WARNING,
            description="Git merge conflict",
            common_causes=["Concurrent changes", "Diverged branches"],
            suggested_fixes=["Resolve conflicts manually", "Use merge tool"],
        ),
        ErrorPattern(
            pattern_id="git_detached",
            pattern=r"(detached HEAD|HEAD detached)",
            category=ErrorCategory.GIT,
            severity=ErrorSeverity.WARNING,
            description="Git HEAD is detached",
            common_causes=["Checkout specific commit", "Checkout tag"],
            suggested_fixes=["Checkout a branch", "Create new branch"],
        ),
        
        # Memory errors
        ErrorPattern(
            pattern_id="mem_out",
            pattern=r"(MemoryError|out of memory|cannot allocate|OOM)",
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.CRITICAL,
            description="Out of memory",
            common_causes=["Large data", "Memory leak", "Insufficient RAM"],
            suggested_fixes=["Reduce data size", "Increase memory", "Optimize code"],
        ),
        
        # Validation errors
        ErrorPattern(
            pattern_id="val_type",
            pattern=r"(TypeError|expected .+ got|type mismatch)",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            description="Type mismatch error",
            common_causes=["Wrong argument type", "None value", "Type conversion failed"],
            suggested_fixes=["Check parameter types", "Add type validation", "Handle None"],
        ),
        ErrorPattern(
            pattern_id="val_value",
            pattern=r"(ValueError|invalid value|out of range)",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            description="Invalid value error",
            common_causes=["Value out of range", "Invalid format", "Empty value"],
            suggested_fixes=["Validate input", "Check value bounds", "Provide default"],
        ),
        
        # Syntax errors
        ErrorPattern(
            pattern_id="syn_error",
            pattern=r"(SyntaxError|invalid syntax|unexpected token)",
            category=ErrorCategory.SYNTAX,
            severity=ErrorSeverity.ERROR,
            description="Syntax error in code",
            common_causes=["Typo", "Missing bracket", "Invalid character"],
            suggested_fixes=["Check syntax", "Use linter", "Review line"],
        ),
        
        # Import errors
        ErrorPattern(
            pattern_id="dep_import",
            pattern=r"(ImportError|ModuleNotFoundError|No module named)",
            category=ErrorCategory.DEPENDENCY,
            severity=ErrorSeverity.ERROR,
            description="Module import failed",
            common_causes=["Package not installed", "Wrong environment", "Circular import"],
            suggested_fixes=["Install package", "Activate venv", "Check imports"],
        ),
        
        # Runtime errors
        ErrorPattern(
            pattern_id="rt_attribute",
            pattern=r"(AttributeError|has no attribute|object has no)",
            category=ErrorCategory.RUNTIME,
            severity=ErrorSeverity.ERROR,
            description="Attribute access error",
            common_causes=["Typo in attribute", "Wrong object type", "None object"],
            suggested_fixes=["Check attribute name", "Verify object type", "Handle None"],
        ),
        ErrorPattern(
            pattern_id="rt_key",
            pattern=r"(KeyError|key not found|missing key)",
            category=ErrorCategory.RUNTIME,
            severity=ErrorSeverity.ERROR,
            description="Dictionary key error",
            common_causes=["Missing key", "Typo in key", "Key not set"],
            suggested_fixes=["Use .get() method", "Check key exists", "Set default"],
        ),
        ErrorPattern(
            pattern_id="rt_index",
            pattern=r"(IndexError|index out of range|list index)",
            category=ErrorCategory.RUNTIME,
            severity=ErrorSeverity.ERROR,
            description="Index out of range",
            common_causes=["Empty list", "Wrong index", "Off-by-one error"],
            suggested_fixes=["Check list length", "Validate index", "Use try-except"],
        ),
    ]
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        custom_patterns: Optional[List[ErrorPattern]] = None,
    ):
        """Initialize the error classifier.
        
        Args:
            llm_client: LLM client for intelligent classification
            custom_patterns: Additional error patterns
        """
        self._llm_client = llm_client
        
        # Combine built-in and custom patterns
        self._patterns = list(self.BUILTIN_PATTERNS)
        if custom_patterns:
            self._patterns.extend(custom_patterns)
        
        # Pattern statistics
        self._pattern_stats: Dict[str, int] = defaultdict(int)
    
    def classify(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
    ) -> Tuple[ErrorCategory, ErrorSeverity, Optional[ErrorPattern]]:
        """Classify an error.
        
        Args:
            error: The exception to classify
            context: Optional error context
            
        Returns:
            Tuple of (category, severity, matched_pattern)
        """
        error_message = str(error)
        error_type = type(error).__name__
        
        # Try pattern matching first
        for pattern in self._patterns:
            if pattern.matches(error_message) or pattern.matches(error_type):
                self._pattern_stats[pattern.pattern_id] += 1
                pattern.occurrence_count += 1
                pattern.last_seen = datetime.now()
                return pattern.category, pattern.severity, pattern
        
        # Fallback classification based on exception type
        category, severity = self._classify_by_type(error)
        
        return category, severity, None
    
    def _classify_by_type(
        self,
        error: Exception,
    ) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify by exception type."""
        type_map = {
            FileNotFoundError: (ErrorCategory.FILESYSTEM, ErrorSeverity.ERROR),
            PermissionError: (ErrorCategory.PERMISSION, ErrorSeverity.ERROR),
            ConnectionError: (ErrorCategory.NETWORK, ErrorSeverity.ERROR),
            TimeoutError: (ErrorCategory.TIMEOUT, ErrorSeverity.WARNING),
            MemoryError: (ErrorCategory.MEMORY, ErrorSeverity.CRITICAL),
            TypeError: (ErrorCategory.VALIDATION, ErrorSeverity.ERROR),
            ValueError: (ErrorCategory.VALIDATION, ErrorSeverity.ERROR),
            KeyError: (ErrorCategory.RUNTIME, ErrorSeverity.ERROR),
            IndexError: (ErrorCategory.RUNTIME, ErrorSeverity.ERROR),
            AttributeError: (ErrorCategory.RUNTIME, ErrorSeverity.ERROR),
            ImportError: (ErrorCategory.DEPENDENCY, ErrorSeverity.ERROR),
            SyntaxError: (ErrorCategory.SYNTAX, ErrorSeverity.ERROR),
            RuntimeError: (ErrorCategory.RUNTIME, ErrorSeverity.ERROR),
            OSError: (ErrorCategory.FILESYSTEM, ErrorSeverity.ERROR),
        }
        
        for exc_type, (category, severity) in type_map.items():
            if isinstance(error, exc_type):
                return category, severity
        
        return ErrorCategory.UNKNOWN, ErrorSeverity.ERROR
    
    async def classify_with_llm(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
    ) -> Tuple[ErrorCategory, ErrorSeverity, str]:
        """Classify error using LLM reasoning.
        
        Args:
            error: The exception to classify
            context: Optional error context
            
        Returns:
            Tuple of (category, severity, reasoning)
        """
        if not self._llm_client:
            category, severity, _ = self.classify(error, context)
            return category, severity, "LLM not available"
        
        prompt = f"""Analyze this error and classify it:

Error Type: {type(error).__name__}
Error Message: {str(error)}

Context:
- Operation: {context.operation if context else 'Unknown'}
- Working Directory: {context.working_directory if context else 'Unknown'}
- Tool: {context.tool_name if context else 'Unknown'}

Categories: {', '.join(c.value for c in ErrorCategory)}
Severities: {', '.join(s.value for s in ErrorSeverity)}

Return JSON with:
- category: one of the categories above
- severity: one of the severities above
- reasoning: brief explanation
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            
            # Parse response
            try:
                data = json.loads(response)
                category = ErrorCategory(data.get("category", "unknown"))
                severity = ErrorSeverity(data.get("severity", "error"))
                reasoning = data.get("reasoning", "")
                return category, severity, reasoning
            except (json.JSONDecodeError, ValueError):
                # Fallback to pattern matching
                category, severity, _ = self.classify(error, context)
                return category, severity, response
                
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            category, severity, _ = self.classify(error, context)
            return category, severity, f"LLM failed: {e}"
    
    def add_pattern(self, pattern: ErrorPattern):
        """Add a custom error pattern."""
        self._patterns.append(pattern)
    
    def get_pattern_stats(self) -> Dict[str, int]:
        """Get pattern matching statistics."""
        return dict(self._pattern_stats)


class ErrorContextCapture:
    """Capture context when errors occur.
    
    Captures system state, recent operations, and environment
    to provide comprehensive error context.
    
    Example:
        >>> capture = ErrorContextCapture()
        >>> context = capture.capture_context("file_read")
    """
    
    def __init__(
        self,
        max_recent_operations: int = 20,
    ):
        """Initialize the context capture.
        
        Args:
            max_recent_operations: Maximum recent operations to track
        """
        self._max_recent = max_recent_operations
        self._recent_operations: List[Tuple[datetime, str]] = []
        self._lock = threading.RLock()
    
    def record_operation(self, operation: str):
        """Record an operation for context.
        
        Args:
            operation: Operation description
        """
        with self._lock:
            self._recent_operations.append((datetime.now(), operation))
            
            # Trim old operations
            if len(self._recent_operations) > self._max_recent:
                self._recent_operations = self._recent_operations[-self._max_recent:]
    
    def capture_context(
        self,
        operation: str,
        user_input: Optional[str] = None,
        tool_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ErrorContext:
        """Capture current context.
        
        Args:
            operation: Current operation
            user_input: User's input
            tool_name: Tool being used
            parameters: Tool parameters
            
        Returns:
            Captured context
        """
        # Get recent operations
        with self._lock:
            recent = [op for _, op in self._recent_operations[-10:]]
        
        # Get resource usage (if available)
        memory_usage = None
        disk_usage = None
        cpu_usage = None
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            disk = psutil.disk_usage(os.getcwd())
            disk_usage = disk.percent
            
            cpu_usage = psutil.cpu_percent(interval=0.1)
        except ImportError:
            pass
        except Exception:
            pass
        
        # Capture environment (sanitized)
        safe_env_vars = {}
        for key in ["PATH", "HOME", "USER", "SHELL", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV"]:
            if key in os.environ:
                safe_env_vars[key] = os.environ[key]
        
        return ErrorContext(
            timestamp=datetime.now(),
            operation=operation,
            working_directory=os.getcwd(),
            environment_vars=safe_env_vars,
            python_version=sys.version,
            platform=sys.platform,
            recent_operations=recent,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            cpu_usage=cpu_usage,
            user_input=user_input,
            tool_name=tool_name,
            parameters=parameters or {},
        )


class ErrorAnalyzer:
    """Analyze errors for root cause and impact.
    
    Uses LLM reasoning to:
    1. Determine root cause
    2. Assess impact
    3. Find related errors
    
    Example:
        >>> analyzer = ErrorAnalyzer()
        >>> analysis = analyzer.analyze(error, context)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        classifier: Optional[ErrorClassifier] = None,
    ):
        """Initialize the error analyzer.
        
        Args:
            llm_client: LLM client for intelligent analysis
            classifier: Error classifier instance
        """
        self._llm_client = llm_client
        self._classifier = classifier or ErrorClassifier(llm_client=llm_client)
        
        # Error history for correlation
        self._error_history: List[AnalyzedError] = []
        self._error_clusters: Dict[str, ErrorCluster] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def analyze(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
    ) -> AnalyzedError:
        """Analyze an error.
        
        Args:
            error: The exception to analyze
            context: Error context
            
        Returns:
            Analyzed error
        """
        # Classify error
        category, severity, pattern = self._classifier.classify(error, context)
        
        # Create analyzed error
        analyzed = AnalyzedError(
            error_id=str(uuid.uuid4()),
            original_error=error,
            category=category,
            severity=severity,
            state=ErrorState.ANALYZED,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context,
            created_at=datetime.now(),
            analyzed_at=datetime.now(),
        )
        
        # Add suggestions from pattern
        if pattern:
            analyzed.suggested_actions = list(pattern.suggested_fixes)
            analyzed.root_cause = pattern.description
        
        # Find related errors
        analyzed.related_errors = self._find_related_errors(analyzed)
        
        # Store in history
        with self._lock:
            self._error_history.append(analyzed)
            
            # Trim old errors
            if len(self._error_history) > 1000:
                self._error_history = self._error_history[-1000:]
        
        # Update clusters
        self._update_clusters(analyzed)
        
        return analyzed
    
    async def analyze_with_llm(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
    ) -> AnalyzedError:
        """Analyze error with LLM reasoning.
        
        Args:
            error: The exception to analyze
            context: Error context
            
        Returns:
            Analyzed error with LLM insights
        """
        # Start with basic analysis
        analyzed = self.analyze(error, context)
        
        if not self._llm_client:
            return analyzed
        
        # Build analysis prompt
        prompt = f"""Analyze this error in detail:

Error Type: {analyzed.error_type}
Error Message: {analyzed.error_message}
Category: {analyzed.category.value}
Severity: {analyzed.severity.value}

Stack Trace:
{analyzed.stack_trace[:1000]}

Context:
- Operation: {context.operation if context else 'Unknown'}
- Working Directory: {context.working_directory if context else 'Unknown'}
- Recent Operations: {context.recent_operations if context else []}

Provide:
1. Root cause analysis
2. Impact assessment
3. Suggested actions (as list)
4. Technical explanation
5. Simple user-friendly explanation

Return as JSON.
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            
            try:
                data = json.loads(response)
                analyzed.root_cause = data.get("root_cause", analyzed.root_cause)
                analyzed.impact = data.get("impact")
                analyzed.suggested_actions = data.get("suggested_actions", analyzed.suggested_actions)
                analyzed.technical_explanation = data.get("technical_explanation")
                analyzed.user_explanation = data.get("user_explanation")
            except json.JSONDecodeError:
                # Use response as explanation
                analyzed.technical_explanation = response
                
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
        
        return analyzed
    
    def _find_related_errors(
        self,
        error: AnalyzedError,
    ) -> List[str]:
        """Find related errors in history."""
        related = []
        
        with self._lock:
            for hist_error in self._error_history[-50:]:
                if hist_error.error_id == error.error_id:
                    continue
                
                # Check for same category
                if hist_error.category == error.category:
                    related.append(hist_error.error_id)
                # Check for same error type
                elif hist_error.error_type == error.error_type:
                    related.append(hist_error.error_id)
                # Check for similar message
                elif self._message_similarity(hist_error.error_message, error.error_message) > 0.7:
                    related.append(hist_error.error_id)
        
        return related[:5]  # Limit to 5 related errors
    
    def _message_similarity(self, msg1: str, msg2: str) -> float:
        """Calculate simple similarity between messages."""
        words1 = set(msg1.lower().split())
        words2 = set(msg2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _update_clusters(self, error: AnalyzedError):
        """Update error clusters."""
        # Find or create cluster
        cluster_key = f"{error.category.value}_{error.error_type}"
        
        with self._lock:
            if cluster_key not in self._error_clusters:
                self._error_clusters[cluster_key] = ErrorCluster(
                    cluster_id=str(uuid.uuid4()),
                    name=cluster_key,
                    common_category=error.category,
                    first_occurrence=datetime.now(),
                )
            
            cluster = self._error_clusters[cluster_key]
            cluster.error_ids.append(error.error_id)
            cluster.last_occurrence = datetime.now()
            cluster.total_occurrences += 1
    
    def get_error_history(
        self,
        category: Optional[ErrorCategory] = None,
        limit: int = 50,
    ) -> List[AnalyzedError]:
        """Get error history.
        
        Args:
            category: Filter by category
            limit: Maximum results
            
        Returns:
            List of analyzed errors
        """
        with self._lock:
            errors = list(self._error_history)
        
        if category:
            errors = [e for e in errors if e.category == category]
        
        return errors[-limit:]
    
    def get_clusters(self) -> List[ErrorCluster]:
        """Get error clusters."""
        with self._lock:
            return list(self._error_clusters.values())


class ErrorExplainer:
    """Generate natural language error explanations.
    
    Uses LLM to generate:
    1. Technical explanations
    2. User-friendly explanations
    3. Cause breakdowns
    4. Learning resources
    
    Example:
        >>> explainer = ErrorExplainer()
        >>> explanation = explainer.explain(analyzed_error)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize the error explainer.
        
        Args:
            llm_client: LLM client for generating explanations
        """
        self._llm_client = llm_client
    
    def explain(
        self,
        error: AnalyzedError,
        technical: bool = True,
    ) -> str:
        """Generate error explanation.
        
        Args:
            error: Analyzed error
            technical: Whether to include technical details
            
        Returns:
            Explanation string
        """
        if technical:
            return self._technical_explanation(error)
        else:
            return self._user_explanation(error)
    
    def _technical_explanation(self, error: AnalyzedError) -> str:
        """Generate technical explanation."""
        parts = [
            f"**Error Type:** {error.error_type}",
            f"**Category:** {error.category.value}",
            f"**Severity:** {error.severity.value}",
            f"**Message:** {error.error_message}",
        ]
        
        if error.root_cause:
            parts.append(f"**Root Cause:** {error.root_cause}")
        
        if error.impact:
            parts.append(f"**Impact:** {error.impact}")
        
        if error.suggested_actions:
            parts.append("**Suggested Actions:**")
            for action in error.suggested_actions:
                parts.append(f"  - {action}")
        
        if error.technical_explanation:
            parts.append(f"\n**Details:** {error.technical_explanation}")
        
        return "\n".join(parts)
    
    def _user_explanation(self, error: AnalyzedError) -> str:
        """Generate user-friendly explanation."""
        if error.user_explanation:
            return error.user_explanation
        
        # Generate simple explanation
        explanations = {
            ErrorCategory.FILESYSTEM: "There was a problem accessing a file or folder.",
            ErrorCategory.NETWORK: "There was a network connection problem.",
            ErrorCategory.AUTHENTICATION: "There was a problem with your login credentials.",
            ErrorCategory.PERMISSION: "You don't have permission to perform this action.",
            ErrorCategory.TIMEOUT: "The operation took too long and was stopped.",
            ErrorCategory.MEMORY: "The system ran out of memory.",
            ErrorCategory.DISK: "The disk is full or there's a storage problem.",
            ErrorCategory.GIT: "There was a problem with Git operations.",
            ErrorCategory.DEPENDENCY: "A required package or module is missing.",
        }
        
        base = explanations.get(error.category, "An error occurred.")
        
        if error.suggested_actions:
            suggestion = error.suggested_actions[0]
            return f"{base}\n\nTry: {suggestion}"
        
        return base
    
    async def explain_with_llm(
        self,
        error: AnalyzedError,
        technical: bool = True,
    ) -> str:
        """Generate explanation using LLM.
        
        Args:
            error: Analyzed error
            technical: Whether to include technical details
            
        Returns:
            LLM-generated explanation
        """
        if not self._llm_client:
            return self.explain(error, technical)
        
        mode = "technical developer" if technical else "non-technical user"
        
        prompt = f"""Explain this error to a {mode}:

Error Type: {error.error_type}
Error Message: {error.error_message}
Category: {error.category.value}

{"Include technical details and stack trace information." if technical else "Keep it simple and avoid technical jargon."}

Provide:
1. What happened
2. Why it happened
3. How to fix it
"""
        
        try:
            return await self._llm_client.generate(prompt)
        except Exception as e:
            logger.warning(f"LLM explanation failed: {e}")
            return self.explain(error, technical)
    
    def get_learning_resources(
        self,
        error: AnalyzedError,
    ) -> List[str]:
        """Get learning resources for the error.
        
        Args:
            error: Analyzed error
            
        Returns:
            List of resource suggestions
        """
        resources = {
            ErrorCategory.GIT: [
                "Git documentation: https://git-scm.com/doc",
                "Pro Git book: https://git-scm.com/book",
            ],
            ErrorCategory.GITHUB: [
                "GitHub documentation: https://docs.github.com",
            ],
            ErrorCategory.FILESYSTEM: [
                "Python os module docs: https://docs.python.org/3/library/os.html",
                "Python pathlib docs: https://docs.python.org/3/library/pathlib.html",
            ],
            ErrorCategory.NETWORK: [
                "Python requests docs: https://docs.python-requests.org",
                "Python urllib docs: https://docs.python.org/3/library/urllib.html",
            ],
            ErrorCategory.DEPENDENCY: [
                "pip documentation: https://pip.pypa.io",
                "Python packaging guide: https://packaging.python.org",
            ],
        }
        
        return resources.get(error.category, [])


class IntelligentErrorDetector:
    """Main error detection system.
    
    Integrates all error detection components:
    - Classification
    - Context capture
    - Analysis
    - Explanation
    
    Example:
        >>> detector = IntelligentErrorDetector()
        >>> analyzed = detector.detect_and_analyze(error)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize the error detector.
        
        Args:
            llm_client: LLM client for intelligent detection
        """
        self._llm_client = llm_client
        
        # Initialize components
        self._classifier = ErrorClassifier(llm_client=llm_client)
        self._context_capture = ErrorContextCapture()
        self._analyzer = ErrorAnalyzer(
            llm_client=llm_client,
            classifier=self._classifier,
        )
        self._explainer = ErrorExplainer(llm_client=llm_client)
    
    def record_operation(self, operation: str):
        """Record an operation for context."""
        self._context_capture.record_operation(operation)
    
    def detect_and_analyze(
        self,
        error: Exception,
        operation: Optional[str] = None,
        user_input: Optional[str] = None,
        tool_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> AnalyzedError:
        """Detect and analyze an error.
        
        Args:
            error: The exception
            operation: Current operation
            user_input: User's input
            tool_name: Tool being used
            parameters: Tool parameters
            
        Returns:
            Analyzed error
        """
        # Capture context
        context = self._context_capture.capture_context(
            operation=operation or "unknown",
            user_input=user_input,
            tool_name=tool_name,
            parameters=parameters,
        )
        
        # Analyze error
        analyzed = self._analyzer.analyze(error, context)
        
        return analyzed
    
    async def detect_and_analyze_async(
        self,
        error: Exception,
        operation: Optional[str] = None,
        user_input: Optional[str] = None,
        tool_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> AnalyzedError:
        """Detect and analyze with LLM (async).
        
        Args:
            error: The exception
            operation: Current operation
            user_input: User's input
            tool_name: Tool being used
            parameters: Tool parameters
            
        Returns:
            Analyzed error with LLM insights
        """
        # Capture context
        context = self._context_capture.capture_context(
            operation=operation or "unknown",
            user_input=user_input,
            tool_name=tool_name,
            parameters=parameters,
        )
        
        # Analyze with LLM
        analyzed = await self._analyzer.analyze_with_llm(error, context)
        
        return analyzed
    
    def explain(
        self,
        error: AnalyzedError,
        technical: bool = True,
    ) -> str:
        """Generate error explanation.
        
        Args:
            error: Analyzed error
            technical: Whether to include technical details
            
        Returns:
            Explanation string
        """
        return self._explainer.explain(error, technical)
    
    async def explain_async(
        self,
        error: AnalyzedError,
        technical: bool = True,
    ) -> str:
        """Generate explanation with LLM (async).
        
        Args:
            error: Analyzed error
            technical: Whether to include technical details
            
        Returns:
            LLM-generated explanation
        """
        return await self._explainer.explain_with_llm(error, technical)
    
    def get_error_history(
        self,
        category: Optional[ErrorCategory] = None,
        limit: int = 50,
    ) -> List[AnalyzedError]:
        """Get error history."""
        return self._analyzer.get_error_history(category, limit)
    
    def get_clusters(self) -> List[ErrorCluster]:
        """Get error clusters."""
        return self._analyzer.get_clusters()
    
    def get_learning_resources(
        self,
        error: AnalyzedError,
    ) -> List[str]:
        """Get learning resources for error."""
        return self._explainer.get_learning_resources(error)


# Module-level instance
_global_error_detector: Optional[IntelligentErrorDetector] = None


def get_intelligent_error_detector(
    llm_client: Optional[Any] = None,
) -> IntelligentErrorDetector:
    """Get the global error detector.
    
    Args:
        llm_client: Optional LLM client
        
    Returns:
        IntelligentErrorDetector instance
    """
    global _global_error_detector
    if _global_error_detector is None:
        _global_error_detector = IntelligentErrorDetector(
            llm_client=llm_client,
        )
    return _global_error_detector
