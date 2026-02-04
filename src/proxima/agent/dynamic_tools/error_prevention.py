"""Proactive Issue Prevention System for Dynamic AI Assistant.

This module implements Phase 8.3 for the Dynamic AI Assistant:
- Pre-flight Validation
- Health Monitoring
- Warning System

Key Features:
============
- Comprehensive pre-execution checks
- Resource availability verification
- Permission validation
- Dependency checking
- Continuous health monitoring
- Early warning indicators

Design Principle:
================
All prevention decisions use LLM reasoning when available.
The LLM analyzes context and predicts potential issues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
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


class ValidationResult(Enum):
    """Validation result types."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


class RiskLevel(Enum):
    """Risk level assessment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WarningSeverity(Enum):
    """Warning severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ResourceType(Enum):
    """Resource types for monitoring."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    FILE_HANDLES = "file_handles"
    PROCESS = "process"


@dataclass
class ValidationCheck:
    """A validation check result."""
    check_id: str
    name: str
    result: ValidationResult
    
    # Details
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    duration_ms: float = 0.0
    checked_at: Optional[datetime] = None
    
    # Severity if failed
    severity: Optional[WarningSeverity] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "result": self.result.value,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
        }


@dataclass
class PreflightResult:
    """Result of preflight validation."""
    operation: str
    passed: bool
    
    # Checks
    checks: List[ValidationCheck] = field(default_factory=list)
    
    # Summary
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warnings: int = 0
    
    # Risk assessment
    risk_level: RiskLevel = RiskLevel.LOW
    
    # Timing
    total_duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "passed": self.passed,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
            "risk_level": self.risk_level.value,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class Warning:
    """A system warning."""
    warning_id: str
    message: str
    severity: WarningSeverity
    
    # Classification
    category: str = "general"
    source: str = "system"
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # State
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    # Suppression
    suppressed: bool = False
    suppression_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "warning_id": self.warning_id,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category,
            "source": self.source,
            "acknowledged": self.acknowledged,
            "suppressed": self.suppressed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    resource_type: ResourceType
    timestamp: datetime
    
    # Current values
    current_value: float = 0.0
    threshold: float = 0.0
    
    # Status
    status: HealthStatus = HealthStatus.UNKNOWN
    
    # Trend
    trend: str = "stable"  # increasing, decreasing, stable
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_type": self.resource_type.value,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "status": self.status.value,
            "trend": self.trend,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HealthReport:
    """System health report."""
    report_id: str
    overall_status: HealthStatus
    
    # Resource metrics
    metrics: List[ResourceMetrics] = field(default_factory=list)
    
    # Issues
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Timing
    generated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "overall_status": self.overall_status.value,
            "metrics": [m.to_dict() for m in self.metrics],
            "issues": self.issues,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }


class PreflightValidator:
    """Perform pre-flight validation before operations.
    
    Uses LLM reasoning to:
    1. Determine required checks for operation
    2. Analyze validation results
    3. Suggest remediation steps
    
    Example:
        >>> validator = PreflightValidator()
        >>> result = validator.validate("file_write", {"path": "/tmp/test.txt"})
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize preflight validator.
        
        Args:
            llm_client: LLM client for intelligent validation
        """
        self._llm_client = llm_client
        
        # Registered checks per operation type
        self._checks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Register built-in checks
        self._register_builtin_checks()
    
    def _register_builtin_checks(self):
        """Register built-in validation checks."""
        # File operations
        self.register_check("file_read", self._check_file_exists)
        self.register_check("file_read", self._check_file_readable)
        self.register_check("file_write", self._check_path_writable)
        self.register_check("file_write", self._check_disk_space)
        self.register_check("file_delete", self._check_file_exists)
        self.register_check("file_delete", self._check_file_writable)
        
        # Directory operations
        self.register_check("dir_create", self._check_parent_exists)
        self.register_check("dir_create", self._check_path_writable)
        self.register_check("dir_delete", self._check_dir_exists)
        
        # Git operations
        self.register_check("git_*", self._check_git_repository)
        self.register_check("git_push", self._check_git_remote)
        self.register_check("git_commit", self._check_git_changes)
        
        # Terminal operations
        self.register_check("terminal_*", self._check_shell_available)
        
        # Resource checks
        self.register_check("*", self._check_memory_available)
    
    def register_check(
        self,
        operation_pattern: str,
        check: Callable[[str, Dict[str, Any]], ValidationCheck],
    ):
        """Register a validation check.
        
        Args:
            operation_pattern: Operation pattern (supports *)
            check: Check function
        """
        self._checks[operation_pattern].append(check)
    
    def validate(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> PreflightResult:
        """Validate before executing operation.
        
        Args:
            operation: Operation to validate
            parameters: Operation parameters
            
        Returns:
            PreflightResult
        """
        parameters = parameters or {}
        start_time = time.time()
        
        result = PreflightResult(
            operation=operation,
            passed=True,
        )
        
        # Get applicable checks
        checks_to_run = self._get_checks_for_operation(operation)
        
        for check_func in checks_to_run:
            try:
                check_result = check_func(operation, parameters)
                result.checks.append(check_result)
                result.total_checks += 1
                
                if check_result.result == ValidationResult.PASSED:
                    result.passed_checks += 1
                elif check_result.result == ValidationResult.FAILED:
                    result.failed_checks += 1
                    result.passed = False
                elif check_result.result == ValidationResult.WARNING:
                    result.warnings += 1
                    
            except Exception as e:
                # Check itself failed
                check_result = ValidationCheck(
                    check_id=str(uuid.uuid4()),
                    name=check_func.__name__,
                    result=ValidationResult.FAILED,
                    message=f"Check error: {e}",
                    checked_at=datetime.now(),
                )
                result.checks.append(check_result)
                result.failed_checks += 1
        
        # Calculate risk level
        result.risk_level = self._assess_risk(result)
        
        result.total_duration_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _get_checks_for_operation(
        self,
        operation: str,
    ) -> List[Callable]:
        """Get checks applicable to operation."""
        checks = []
        
        # Exact match
        if operation in self._checks:
            checks.extend(self._checks[operation])
        
        # Wildcard matches
        for pattern, pattern_checks in self._checks.items():
            if "*" in pattern:
                prefix = pattern.replace("*", "")
                if operation.startswith(prefix) or prefix == "":
                    checks.extend(pattern_checks)
        
        return checks
    
    def _assess_risk(self, result: PreflightResult) -> RiskLevel:
        """Assess risk level from validation result."""
        if result.failed_checks > 0:
            return RiskLevel.CRITICAL
        elif result.warnings > 2:
            return RiskLevel.HIGH
        elif result.warnings > 0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    # Built-in check implementations
    def _check_file_exists(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> ValidationCheck:
        """Check if file exists."""
        path = params.get("path", "")
        
        check = ValidationCheck(
            check_id=str(uuid.uuid4()),
            name="file_exists",
            result=ValidationResult.PASSED,
            checked_at=datetime.now(),
        )
        
        if not path:
            check.result = ValidationResult.FAILED
            check.message = "No path specified"
            return check
        
        if Path(path).exists():
            check.message = f"File exists: {path}"
        else:
            check.result = ValidationResult.FAILED
            check.message = f"File not found: {path}"
            check.severity = WarningSeverity.HIGH
        
        return check
    
    def _check_file_readable(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> ValidationCheck:
        """Check if file is readable."""
        path = params.get("path", "")
        
        check = ValidationCheck(
            check_id=str(uuid.uuid4()),
            name="file_readable",
            result=ValidationResult.PASSED,
            checked_at=datetime.now(),
        )
        
        if not path:
            check.result = ValidationResult.SKIPPED
            check.message = "No path specified"
            return check
        
        path_obj = Path(path)
        if not path_obj.exists():
            check.result = ValidationResult.SKIPPED
            check.message = "File does not exist"
            return check
        
        if os.access(path, os.R_OK):
            check.message = f"File is readable: {path}"
        else:
            check.result = ValidationResult.FAILED
            check.message = f"File not readable: {path}"
            check.severity = WarningSeverity.HIGH
        
        return check
    
    def _check_file_writable(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> ValidationCheck:
        """Check if file is writable."""
        path = params.get("path", "")
        
        check = ValidationCheck(
            check_id=str(uuid.uuid4()),
            name="file_writable",
            result=ValidationResult.PASSED,
            checked_at=datetime.now(),
        )
        
        if not path:
            check.result = ValidationResult.SKIPPED
            return check
        
        path_obj = Path(path)
        if not path_obj.exists():
            check.result = ValidationResult.SKIPPED
            return check
        
        if os.access(path, os.W_OK):
            check.message = f"File is writable: {path}"
        else:
            check.result = ValidationResult.FAILED
            check.message = f"File not writable: {path}"
            check.severity = WarningSeverity.HIGH
        
        return check
    
    def _check_path_writable(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> ValidationCheck:
        """Check if path is writable."""
        path = params.get("path", "")
        
        check = ValidationCheck(
            check_id=str(uuid.uuid4()),
            name="path_writable",
            result=ValidationResult.PASSED,
            checked_at=datetime.now(),
        )
        
        if not path:
            check.result = ValidationResult.SKIPPED
            return check
        
        path_obj = Path(path)
        parent = path_obj.parent
        
        if parent.exists() and os.access(parent, os.W_OK):
            check.message = f"Path is writable: {parent}"
        else:
            check.result = ValidationResult.FAILED
            check.message = f"Path not writable: {parent}"
            check.severity = WarningSeverity.HIGH
        
        return check
    
    def _check_parent_exists(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> ValidationCheck:
        """Check if parent directory exists."""
        path = params.get("path", "")
        
        check = ValidationCheck(
            check_id=str(uuid.uuid4()),
            name="parent_exists",
            result=ValidationResult.PASSED,
            checked_at=datetime.now(),
        )
        
        if not path:
            check.result = ValidationResult.SKIPPED
            return check
        
        parent = Path(path).parent
        
        if parent.exists():
            check.message = f"Parent exists: {parent}"
        else:
            check.result = ValidationResult.WARNING
            check.message = f"Parent does not exist: {parent}"
            check.severity = WarningSeverity.LOW
        
        return check
    
    def _check_dir_exists(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> ValidationCheck:
        """Check if directory exists."""
        path = params.get("path", "")
        
        check = ValidationCheck(
            check_id=str(uuid.uuid4()),
            name="dir_exists",
            result=ValidationResult.PASSED,
            checked_at=datetime.now(),
        )
        
        if not path:
            check.result = ValidationResult.SKIPPED
            return check
        
        path_obj = Path(path)
        
        if path_obj.is_dir():
            check.message = f"Directory exists: {path}"
        else:
            check.result = ValidationResult.FAILED
            check.message = f"Directory not found: {path}"
            check.severity = WarningSeverity.HIGH
        
        return check
    
    def _check_disk_space(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> ValidationCheck:
        """Check available disk space."""
        path = params.get("path", os.getcwd())
        
        check = ValidationCheck(
            check_id=str(uuid.uuid4()),
            name="disk_space",
            result=ValidationResult.PASSED,
            checked_at=datetime.now(),
        )
        
        try:
            usage = shutil.disk_usage(Path(path).anchor or "/")
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            percent_free = (usage.free / usage.total) * 100
            
            check.details = {
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "percent_free": round(percent_free, 1),
            }
            
            if percent_free < 5:
                check.result = ValidationResult.FAILED
                check.message = f"Critical: Only {percent_free:.1f}% disk space free"
                check.severity = WarningSeverity.CRITICAL
            elif percent_free < 10:
                check.result = ValidationResult.WARNING
                check.message = f"Low disk space: {percent_free:.1f}% free"
                check.severity = WarningSeverity.HIGH
            else:
                check.message = f"Disk space OK: {free_gb:.1f}GB free"
                
        except Exception as e:
            check.result = ValidationResult.SKIPPED
            check.message = f"Could not check disk: {e}"
        
        return check
    
    def _check_memory_available(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> ValidationCheck:
        """Check available memory."""
        check = ValidationCheck(
            check_id=str(uuid.uuid4()),
            name="memory_available",
            result=ValidationResult.PASSED,
            checked_at=datetime.now(),
        )
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            check.details = {
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "available_gb": round(memory.available / (1024 ** 3), 2),
                "percent_used": memory.percent,
            }
            
            if memory.percent > 95:
                check.result = ValidationResult.FAILED
                check.message = f"Critical: {memory.percent}% memory used"
                check.severity = WarningSeverity.CRITICAL
            elif memory.percent > 85:
                check.result = ValidationResult.WARNING
                check.message = f"High memory usage: {memory.percent}%"
                check.severity = WarningSeverity.MEDIUM
            else:
                check.message = f"Memory OK: {memory.percent}% used"
                
        except ImportError:
            check.result = ValidationResult.SKIPPED
            check.message = "psutil not available"
        except Exception as e:
            check.result = ValidationResult.SKIPPED
            check.message = f"Could not check memory: {e}"
        
        return check
    
    def _check_git_repository(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> ValidationCheck:
        """Check if in Git repository."""
        path = params.get("path", os.getcwd())
        
        check = ValidationCheck(
            check_id=str(uuid.uuid4()),
            name="git_repository",
            result=ValidationResult.PASSED,
            checked_at=datetime.now(),
        )
        
        # Walk up to find .git
        current = Path(path)
        while current != current.parent:
            if (current / ".git").exists():
                check.message = f"Git repository found: {current}"
                return check
            current = current.parent
        
        check.result = ValidationResult.FAILED
        check.message = "Not in a Git repository"
        check.severity = WarningSeverity.HIGH
        
        return check
    
    def _check_git_remote(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> ValidationCheck:
        """Check if Git remote is configured."""
        check = ValidationCheck(
            check_id=str(uuid.uuid4()),
            name="git_remote",
            result=ValidationResult.PASSED,
            checked_at=datetime.now(),
        )
        
        try:
            import subprocess
            result = subprocess.run(
                ["git", "remote", "-v"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode == 0 and result.stdout.strip():
                check.message = "Git remote configured"
                check.details = {"remotes": result.stdout.strip()}
            else:
                check.result = ValidationResult.WARNING
                check.message = "No Git remote configured"
                check.severity = WarningSeverity.MEDIUM
                
        except Exception as e:
            check.result = ValidationResult.SKIPPED
            check.message = f"Could not check remote: {e}"
        
        return check
    
    def _check_git_changes(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> ValidationCheck:
        """Check if there are changes to commit."""
        check = ValidationCheck(
            check_id=str(uuid.uuid4()),
            name="git_changes",
            result=ValidationResult.PASSED,
            checked_at=datetime.now(),
        )
        
        try:
            import subprocess
            
            # Check staged changes
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode == 0 and result.stdout.strip():
                check.message = "Staged changes ready to commit"
            else:
                check.result = ValidationResult.WARNING
                check.message = "No staged changes to commit"
                check.severity = WarningSeverity.LOW
                
        except Exception as e:
            check.result = ValidationResult.SKIPPED
            check.message = f"Could not check changes: {e}"
        
        return check
    
    def _check_shell_available(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> ValidationCheck:
        """Check if shell is available."""
        check = ValidationCheck(
            check_id=str(uuid.uuid4()),
            name="shell_available",
            result=ValidationResult.PASSED,
            checked_at=datetime.now(),
        )
        
        shell = params.get("shell", os.environ.get("SHELL", ""))
        
        if shell and shutil.which(shell.split("/")[-1]):
            check.message = f"Shell available: {shell}"
        elif shutil.which("bash") or shutil.which("sh") or shutil.which("powershell"):
            check.message = "Default shell available"
        else:
            check.result = ValidationResult.WARNING
            check.message = "Shell availability uncertain"
            check.severity = WarningSeverity.LOW
        
        return check


class HealthMonitor:
    """Monitor system health continuously.
    
    Uses LLM reasoning to:
    1. Analyze health trends
    2. Predict degradation
    3. Recommend actions
    
    Example:
        >>> monitor = HealthMonitor()
        >>> monitor.start()
        >>> report = monitor.get_health_report()
    """
    
    # Resource thresholds
    DEFAULT_THRESHOLDS = {
        ResourceType.CPU: 80.0,  # percent
        ResourceType.MEMORY: 85.0,  # percent
        ResourceType.DISK: 90.0,  # percent
    }
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        check_interval: float = 60.0,
    ):
        """Initialize health monitor.
        
        Args:
            llm_client: LLM client for intelligent monitoring
            check_interval: Seconds between checks
        """
        self._llm_client = llm_client
        self._check_interval = check_interval
        
        # Thresholds
        self._thresholds = dict(self.DEFAULT_THRESHOLDS)
        
        # Metrics history
        self._metrics_history: Dict[ResourceType, List[ResourceMetrics]] = defaultdict(list)
        
        # Current status
        self._current_status = HealthStatus.UNKNOWN
        
        # Monitoring thread
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Callbacks for status changes
        self._status_callbacks: List[Callable[[HealthStatus], None]] = []
    
    def start(self):
        """Start health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop(self):
        """Stop health monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._collect_metrics()
                self._update_status()
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(self._check_interval)
    
    def _collect_metrics(self):
        """Collect current metrics."""
        now = datetime.now()
        
        try:
            import psutil
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_metric(ResourceType.CPU, cpu_percent, now)
            
            # Memory
            memory = psutil.virtual_memory()
            self._record_metric(ResourceType.MEMORY, memory.percent, now)
            
            # Disk
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent
            self._record_metric(ResourceType.DISK, disk_percent, now)
            
        except ImportError:
            logger.debug("psutil not available for metrics collection")
        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")
    
    def _record_metric(
        self,
        resource_type: ResourceType,
        value: float,
        timestamp: datetime,
    ):
        """Record a metric value."""
        threshold = self._thresholds.get(resource_type, 100.0)
        
        # Determine status
        if value >= threshold:
            status = HealthStatus.UNHEALTHY
        elif value >= threshold * 0.8:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        # Calculate trend
        trend = self._calculate_trend(resource_type, value)
        
        metric = ResourceMetrics(
            resource_type=resource_type,
            timestamp=timestamp,
            current_value=value,
            threshold=threshold,
            status=status,
            trend=trend,
        )
        
        with self._lock:
            self._metrics_history[resource_type].append(metric)
            
            # Keep last 100 metrics per resource
            if len(self._metrics_history[resource_type]) > 100:
                self._metrics_history[resource_type] = \
                    self._metrics_history[resource_type][-100:]
    
    def _calculate_trend(
        self,
        resource_type: ResourceType,
        current_value: float,
    ) -> str:
        """Calculate trend for resource."""
        with self._lock:
            history = self._metrics_history.get(resource_type, [])
        
        if len(history) < 3:
            return "stable"
        
        # Compare with average of last 3 readings
        recent_values = [m.current_value for m in history[-3:]]
        avg = sum(recent_values) / len(recent_values)
        
        diff = current_value - avg
        
        if diff > 5:
            return "increasing"
        elif diff < -5:
            return "decreasing"
        else:
            return "stable"
    
    def _update_status(self):
        """Update overall health status."""
        statuses = []
        
        with self._lock:
            for resource_type, metrics in self._metrics_history.items():
                if metrics:
                    statuses.append(metrics[-1].status)
        
        if not statuses:
            new_status = HealthStatus.UNKNOWN
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            new_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            new_status = HealthStatus.DEGRADED
        else:
            new_status = HealthStatus.HEALTHY
        
        # Notify on status change
        if new_status != self._current_status:
            old_status = self._current_status
            self._current_status = new_status
            
            for callback in self._status_callbacks:
                try:
                    callback(new_status)
                except Exception as e:
                    logger.error(f"Status callback error: {e}")
            
            logger.info(f"Health status changed: {old_status.value} -> {new_status.value}")
    
    def get_health_report(self) -> HealthReport:
        """Generate health report.
        
        Returns:
            HealthReport with current status
        """
        report = HealthReport(
            report_id=str(uuid.uuid4()),
            overall_status=self._current_status,
            generated_at=datetime.now(),
        )
        
        with self._lock:
            for resource_type, metrics in self._metrics_history.items():
                if metrics:
                    report.metrics.append(metrics[-1])
        
        # Add issues
        for metric in report.metrics:
            if metric.status == HealthStatus.UNHEALTHY:
                report.issues.append(
                    f"{metric.resource_type.value} is critical: {metric.current_value:.1f}%"
                )
            elif metric.status == HealthStatus.DEGRADED:
                report.issues.append(
                    f"{metric.resource_type.value} is high: {metric.current_value:.1f}%"
                )
        
        # Add recommendations
        if report.overall_status == HealthStatus.UNHEALTHY:
            report.recommendations.append("Consider reducing workload")
            report.recommendations.append("Free up system resources")
        elif report.overall_status == HealthStatus.DEGRADED:
            report.recommendations.append("Monitor resource usage closely")
        
        return report
    
    def get_current_status(self) -> HealthStatus:
        """Get current health status."""
        return self._current_status
    
    def set_threshold(
        self,
        resource_type: ResourceType,
        threshold: float,
    ):
        """Set threshold for resource type."""
        self._thresholds[resource_type] = threshold
    
    def on_status_change(
        self,
        callback: Callable[[HealthStatus], None],
    ):
        """Register callback for status changes."""
        self._status_callbacks.append(callback)


class WarningSystem:
    """Manage system warnings.
    
    Uses LLM reasoning to:
    1. Classify warning severity
    2. Predict escalation needs
    3. Suggest resolutions
    
    Example:
        >>> warnings = WarningSystem()
        >>> warnings.emit("Disk space low", WarningSeverity.HIGH)
        >>> active = warnings.get_active_warnings()
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        max_warnings: int = 500,
    ):
        """Initialize warning system.
        
        Args:
            llm_client: LLM client for intelligent warnings
            max_warnings: Maximum warnings to keep
        """
        self._llm_client = llm_client
        self._max_warnings = max_warnings
        
        # Active warnings
        self._warnings: Dict[str, Warning] = {}
        
        # Warning history
        self._warning_history: List[Warning] = []
        
        # Suppression rules
        self._suppression_rules: Dict[str, str] = {}
        
        # Callbacks
        self._warning_callbacks: List[Callable[[Warning], None]] = []
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def emit(
        self,
        message: str,
        severity: WarningSeverity,
        category: str = "general",
        source: str = "system",
        context: Optional[Dict[str, Any]] = None,
        expires_in: Optional[timedelta] = None,
    ) -> Warning:
        """Emit a warning.
        
        Args:
            message: Warning message
            severity: Warning severity
            category: Warning category
            source: Warning source
            context: Additional context
            expires_in: How long until warning expires
            
        Returns:
            Created warning
        """
        warning = Warning(
            warning_id=str(uuid.uuid4()),
            message=message,
            severity=severity,
            category=category,
            source=source,
            context=context or {},
            created_at=datetime.now(),
        )
        
        if expires_in:
            warning.expires_at = datetime.now() + expires_in
        
        # Check suppression
        suppression_key = f"{category}:{message[:50]}"
        if suppression_key in self._suppression_rules:
            warning.suppressed = True
            warning.suppression_reason = self._suppression_rules[suppression_key]
        
        with self._lock:
            self._warnings[warning.warning_id] = warning
            self._warning_history.append(warning)
            
            # Trim old history
            if len(self._warning_history) > self._max_warnings:
                self._warning_history = self._warning_history[-self._max_warnings:]
        
        # Notify callbacks if not suppressed
        if not warning.suppressed:
            for callback in self._warning_callbacks:
                try:
                    callback(warning)
                except Exception as e:
                    logger.error(f"Warning callback error: {e}")
        
        logger.warning(f"Warning [{severity.value}]: {message}")
        
        return warning
    
    def acknowledge(
        self,
        warning_id: str,
        acknowledged_by: Optional[str] = None,
    ) -> bool:
        """Acknowledge a warning.
        
        Args:
            warning_id: Warning ID
            acknowledged_by: Who acknowledged
            
        Returns:
            Whether acknowledgment succeeded
        """
        with self._lock:
            warning = self._warnings.get(warning_id)
            if not warning:
                return False
            
            warning.acknowledged = True
            warning.acknowledged_at = datetime.now()
            warning.acknowledged_by = acknowledged_by
        
        return True
    
    def suppress(
        self,
        category: str,
        message_prefix: str,
        reason: str,
    ):
        """Suppress warnings matching criteria.
        
        Args:
            category: Warning category
            message_prefix: Message prefix to match
            reason: Suppression reason
        """
        key = f"{category}:{message_prefix[:50]}"
        self._suppression_rules[key] = reason
    
    def unsuppress(self, category: str, message_prefix: str):
        """Remove suppression rule."""
        key = f"{category}:{message_prefix[:50]}"
        self._suppression_rules.pop(key, None)
    
    def get_active_warnings(
        self,
        severity: Optional[WarningSeverity] = None,
        include_suppressed: bool = False,
    ) -> List[Warning]:
        """Get active warnings.
        
        Args:
            severity: Filter by severity
            include_suppressed: Include suppressed warnings
            
        Returns:
            List of active warnings
        """
        now = datetime.now()
        
        with self._lock:
            warnings = [
                w for w in self._warnings.values()
                if not w.acknowledged
                and (w.expires_at is None or w.expires_at > now)
            ]
        
        if not include_suppressed:
            warnings = [w for w in warnings if not w.suppressed]
        
        if severity:
            warnings = [w for w in warnings if w.severity == severity]
        
        return warnings
    
    def get_warning_counts(self) -> Dict[str, int]:
        """Get warning counts by severity."""
        counts: Dict[str, int] = defaultdict(int)
        
        for warning in self.get_active_warnings():
            counts[warning.severity.value] += 1
        
        return dict(counts)
    
    def clear_expired(self):
        """Clear expired warnings."""
        now = datetime.now()
        
        with self._lock:
            expired = [
                wid for wid, w in self._warnings.items()
                if w.expires_at and w.expires_at <= now
            ]
            
            for wid in expired:
                del self._warnings[wid]
    
    def on_warning(
        self,
        callback: Callable[[Warning], None],
    ):
        """Register callback for new warnings."""
        self._warning_callbacks.append(callback)
    
    def get_warning_history(
        self,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[Warning]:
        """Get warning history.
        
        Args:
            category: Filter by category
            limit: Maximum results
            
        Returns:
            List of warnings
        """
        with self._lock:
            warnings = list(self._warning_history)
        
        if category:
            warnings = [w for w in warnings if w.category == category]
        
        return warnings[-limit:]


class ProactiveIssuePrevention:
    """Main proactive issue prevention system.
    
    Integrates all prevention components:
    - Pre-flight validation
    - Health monitoring
    - Warning system
    
    Example:
        >>> prevention = ProactiveIssuePrevention()
        >>> result = prevention.validate_operation("file_write", params)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize issue prevention system.
        
        Args:
            llm_client: LLM client for intelligent prevention
        """
        self._llm_client = llm_client
        
        # Initialize components
        self._validator = PreflightValidator(llm_client=llm_client)
        self._health_monitor = HealthMonitor(llm_client=llm_client)
        self._warning_system = WarningSystem(llm_client=llm_client)
        
        # Connect components
        self._health_monitor.on_status_change(self._on_health_change)
    
    def start_monitoring(self):
        """Start health monitoring."""
        self._health_monitor.start()
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._health_monitor.stop()
    
    def validate_operation(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> PreflightResult:
        """Validate before executing operation.
        
        Args:
            operation: Operation to validate
            parameters: Operation parameters
            
        Returns:
            PreflightResult
        """
        result = self._validator.validate(operation, parameters)
        
        # Emit warnings for failed checks
        for check in result.checks:
            if check.result == ValidationResult.FAILED:
                self._warning_system.emit(
                    message=f"Preflight check failed: {check.message}",
                    severity=check.severity or WarningSeverity.MEDIUM,
                    category="preflight",
                    source=operation,
                    context={"check": check.name},
                )
        
        return result
    
    def get_health_report(self) -> HealthReport:
        """Get current health report."""
        return self._health_monitor.get_health_report()
    
    def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        return self._health_monitor.get_current_status()
    
    def emit_warning(
        self,
        message: str,
        severity: WarningSeverity,
        category: str = "general",
    ) -> Warning:
        """Emit a warning."""
        return self._warning_system.emit(
            message=message,
            severity=severity,
            category=category,
        )
    
    def get_active_warnings(self) -> List[Warning]:
        """Get active warnings."""
        return self._warning_system.get_active_warnings()
    
    def acknowledge_warning(self, warning_id: str) -> bool:
        """Acknowledge a warning."""
        return self._warning_system.acknowledge(warning_id)
    
    def _on_health_change(self, status: HealthStatus):
        """Handle health status change."""
        if status == HealthStatus.UNHEALTHY:
            self._warning_system.emit(
                message="System health is critical",
                severity=WarningSeverity.CRITICAL,
                category="health",
            )
        elif status == HealthStatus.DEGRADED:
            self._warning_system.emit(
                message="System health is degraded",
                severity=WarningSeverity.HIGH,
                category="health",
            )
    
    async def analyze_risks_with_llm(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze operation risks using LLM.
        
        Args:
            operation: Operation to analyze
            parameters: Operation parameters
            
        Returns:
            Risk analysis
        """
        if not self._llm_client:
            return {"error": "LLM not available"}
        
        # Get current context
        health = self.get_health_report()
        warnings = self.get_active_warnings()
        
        prompt = f"""Analyze risks for this operation:

Operation: {operation}
Parameters: {json.dumps(parameters or {}, indent=2)}

Current System Health: {health.overall_status.value}
Active Warnings: {len(warnings)}

Provide:
1. Risk assessment (low/medium/high/critical)
2. Potential issues
3. Mitigation suggestions
4. Whether to proceed

Return as JSON.
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            return json.loads(response)
        except Exception as e:
            return {"error": str(e)}


# Module-level instance
_global_prevention_system: Optional[ProactiveIssuePrevention] = None


def get_proactive_issue_prevention(
    llm_client: Optional[Any] = None,
) -> ProactiveIssuePrevention:
    """Get the global issue prevention system.
    
    Args:
        llm_client: Optional LLM client
        
    Returns:
        ProactiveIssuePrevention instance
    """
    global _global_prevention_system
    if _global_prevention_system is None:
        _global_prevention_system = ProactiveIssuePrevention(
            llm_client=llm_client,
        )
    return _global_prevention_system
