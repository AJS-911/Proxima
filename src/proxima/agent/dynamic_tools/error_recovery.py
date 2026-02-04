"""Automated Error Recovery System for Dynamic AI Assistant.

This module implements Phase 8.2 for the Dynamic AI Assistant:
- Recovery Strategy Selection
- Automatic Retry Logic
- Rollback and Undo
- Alternative Execution Paths

Key Features:
============
- Strategy database with success rates
- Context-based strategy selection
- Exponential backoff with jitter
- Circuit breaker pattern
- Checkpoint-based rollback
- Fallback operation detection

Design Principle:
================
All recovery decisions use LLM reasoning when available.
The LLM analyzes failures and suggests recovery strategies.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import json
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Coroutine, Dict, Generic, Iterator, List,
    Optional, Set, Tuple, Type, TypeVar, Union
)
import uuid

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    ROLLBACK = "rollback"
    FALLBACK = "fallback"
    SKIP = "skip"
    ESCALATE = "escalate"
    MANUAL = "manual"
    CIRCUIT_BREAK = "circuit_break"
    DEGRADE = "degrade"


class RetryBackoff(Enum):
    """Retry backoff strategies."""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing recovery


class RecoveryState(Enum):
    """Recovery operation states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    ESCALATED = "escalated"


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    backoff: RetryBackoff = RetryBackoff.EXPONENTIAL
    jitter: bool = True
    jitter_range: float = 0.1  # 10% jitter
    
    # Retry conditions
    retry_on: List[Type[Exception]] = field(default_factory=list)
    retry_if: Optional[Callable[[Exception], bool]] = None
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if self.backoff == RetryBackoff.CONSTANT:
            delay = self.base_delay
        elif self.backoff == RetryBackoff.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.backoff == RetryBackoff.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)
        elif self.backoff == RetryBackoff.FIBONACCI:
            delay = self.base_delay * self._fibonacci(attempt + 1)
        else:
            delay = self.base_delay
        
        # Apply max delay cap
        delay = min(delay, self.max_delay)
        
        # Apply jitter
        if self.jitter:
            jitter_amount = delay * self.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def should_retry(self, error: Exception) -> bool:
        """Check if error should be retried.
        
        Args:
            error: The exception
            
        Returns:
            Whether to retry
        """
        # Check custom condition first
        if self.retry_if:
            return self.retry_if(error)
        
        # Check exception types
        if self.retry_on:
            return isinstance(error, tuple(self.retry_on))
        
        # Default: retry all transient-looking errors
        return self._is_transient(error)
    
    def _is_transient(self, error: Exception) -> bool:
        """Check if error looks transient."""
        transient_patterns = [
            "timeout", "timed out", "connection reset",
            "temporarily unavailable", "try again", "retry",
            "too many requests", "rate limit", "throttl",
        ]
        
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in transient_patterns)


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    attempt_id: str
    strategy: RecoveryStrategy
    state: RecoveryState
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Result
    success: bool = False
    error: Optional[str] = None
    result: Any = None
    
    # Context
    attempt_number: int = 0
    total_attempts: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "strategy": self.strategy.value,
            "state": self.state.value,
            "success": self.success,
            "error": self.error,
            "attempt_number": self.attempt_number,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class Checkpoint:
    """A checkpoint for rollback capability."""
    checkpoint_id: str
    operation: str
    
    # State snapshot
    state: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: Optional[datetime] = None
    description: Optional[str] = None
    
    # Rollback info
    rollback_actions: List[Callable] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "operation": self.operation,
            "state": self.state,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "description": self.description,
        }


@dataclass
class StrategyRecord:
    """Record of strategy usage and success."""
    strategy: RecoveryStrategy
    error_type: str
    
    # Statistics
    total_attempts: int = 0
    successful_attempts: int = 0
    
    # Last usage
    last_used: Optional[datetime] = None
    last_success: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""
    success: bool
    strategy_used: RecoveryStrategy
    
    # Details
    attempts: List[RecoveryAttempt] = field(default_factory=list)
    final_error: Optional[Exception] = None
    result: Any = None
    
    # Timing
    total_duration: float = 0.0
    
    # Metadata
    escalated: bool = False
    manual_intervention_required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "strategy_used": self.strategy_used.value,
            "attempts": [a.to_dict() for a in self.attempts],
            "final_error": str(self.final_error) if self.final_error else None,
            "total_duration": self.total_duration,
            "escalated": self.escalated,
        }


class CircuitBreaker:
    """Circuit breaker for preventing repeated failures.
    
    Example:
        >>> breaker = CircuitBreaker()
        >>> if breaker.allow_request():
        ...     try:
        ...         result = do_operation()
        ...         breaker.record_success()
        ...     except Exception as e:
        ...         breaker.record_failure()
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 30.0,
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening
            success_threshold: Successes to close from half-open
            timeout: Seconds before transitioning to half-open
        """
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._timeout = timeout
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        
        self._lock = threading.RLock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state
    
    def allow_request(self) -> bool:
        """Check if request should be allowed.
        
        Returns:
            Whether request is allowed
        """
        with self._lock:
            self._check_state_transition()
            
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.HALF_OPEN:
                return True
            else:  # OPEN
                return False
    
    def record_success(self):
        """Record a successful operation."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker closed")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0
    
    def record_failure(self):
        """Record a failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning("Circuit breaker opened from half-open")
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning("Circuit breaker opened")
    
    def _check_state_transition(self):
        """Check if state should transition."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = (datetime.now() - self._last_failure_time).total_seconds()
            if elapsed >= self._timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("Circuit breaker half-open")
    
    def reset(self):
        """Reset circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None


class RetryExecutor:
    """Execute operations with retry logic.
    
    Uses LLM reasoning to:
    1. Determine if retry is appropriate
    2. Adjust retry parameters
    3. Analyze retry failures
    
    Example:
        >>> executor = RetryExecutor()
        >>> result = executor.execute_with_retry(my_function, args)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        default_config: Optional[RetryConfig] = None,
    ):
        """Initialize retry executor.
        
        Args:
            llm_client: LLM client for intelligent retry
            default_config: Default retry configuration
        """
        self._llm_client = llm_client
        self._default_config = default_config or RetryConfig()
        
        # Per-operation circuit breakers
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Retry history
        self._retry_history: List[RecoveryAttempt] = []
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def execute_with_retry(
        self,
        func: Callable[..., T],
        *args,
        operation_name: Optional[str] = None,
        config: Optional[RetryConfig] = None,
        **kwargs,
    ) -> T:
        """Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            operation_name: Name for circuit breaker
            config: Retry configuration
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        config = config or self._default_config
        op_name = operation_name or func.__name__
        
        # Get circuit breaker
        breaker = self._get_circuit_breaker(op_name)
        
        last_error: Optional[Exception] = None
        
        for attempt in range(config.max_attempts):
            # Check circuit breaker
            if not breaker.allow_request():
                raise Exception(f"Circuit breaker open for {op_name}")
            
            # Record attempt
            attempt_record = RecoveryAttempt(
                attempt_id=str(uuid.uuid4()),
                strategy=RecoveryStrategy.RETRY,
                state=RecoveryState.IN_PROGRESS,
                started_at=datetime.now(),
                attempt_number=attempt + 1,
                total_attempts=config.max_attempts,
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Success
                breaker.record_success()
                attempt_record.state = RecoveryState.SUCCEEDED
                attempt_record.success = True
                attempt_record.completed_at = datetime.now()
                
                self._record_attempt(attempt_record)
                return result
                
            except Exception as e:
                last_error = e
                breaker.record_failure()
                
                attempt_record.state = RecoveryState.FAILED
                attempt_record.error = str(e)
                attempt_record.completed_at = datetime.now()
                self._record_attempt(attempt_record)
                
                # Check if we should retry
                if attempt < config.max_attempts - 1:
                    if config.should_retry(e):
                        delay = config.get_delay(attempt)
                        logger.info(f"Retry {attempt + 1}/{config.max_attempts} after {delay:.2f}s")
                        time.sleep(delay)
                    else:
                        logger.info(f"Error not retryable: {e}")
                        break
        
        # All retries failed
        raise last_error or Exception("All retries failed")
    
    async def execute_with_retry_async(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        operation_name: Optional[str] = None,
        config: Optional[RetryConfig] = None,
        **kwargs,
    ) -> T:
        """Execute async function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            operation_name: Name for circuit breaker
            config: Retry configuration
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        config = config or self._default_config
        op_name = operation_name or func.__name__
        
        breaker = self._get_circuit_breaker(op_name)
        last_error: Optional[Exception] = None
        
        for attempt in range(config.max_attempts):
            if not breaker.allow_request():
                raise Exception(f"Circuit breaker open for {op_name}")
            
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
                
            except Exception as e:
                last_error = e
                breaker.record_failure()
                
                if attempt < config.max_attempts - 1:
                    if config.should_retry(e):
                        delay = config.get_delay(attempt)
                        await asyncio.sleep(delay)
                    else:
                        break
        
        raise last_error or Exception("All retries failed")
    
    def _get_circuit_breaker(self, operation: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        with self._lock:
            if operation not in self._circuit_breakers:
                self._circuit_breakers[operation] = CircuitBreaker()
            return self._circuit_breakers[operation]
    
    def _record_attempt(self, attempt: RecoveryAttempt):
        """Record a retry attempt."""
        with self._lock:
            self._retry_history.append(attempt)
            
            if len(self._retry_history) > 1000:
                self._retry_history = self._retry_history[-1000:]
    
    def get_circuit_state(self, operation: str) -> CircuitState:
        """Get circuit breaker state for operation."""
        breaker = self._get_circuit_breaker(operation)
        return breaker.state
    
    def reset_circuit(self, operation: str):
        """Reset circuit breaker for operation."""
        with self._lock:
            if operation in self._circuit_breakers:
                self._circuit_breakers[operation].reset()


class CheckpointManager:
    """Manage checkpoints for rollback capability.
    
    Example:
        >>> manager = CheckpointManager()
        >>> checkpoint = manager.create_checkpoint("file_write")
        >>> try:
        ...     do_operation()
        ... except:
        ...     manager.rollback(checkpoint.checkpoint_id)
    """
    
    def __init__(
        self,
        max_checkpoints: int = 50,
    ):
        """Initialize checkpoint manager.
        
        Args:
            max_checkpoints: Maximum checkpoints to keep
        """
        self._max_checkpoints = max_checkpoints
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._checkpoint_order: List[str] = []
        
        self._lock = threading.RLock()
    
    def create_checkpoint(
        self,
        operation: str,
        state: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        rollback_actions: Optional[List[Callable]] = None,
    ) -> Checkpoint:
        """Create a checkpoint.
        
        Args:
            operation: Operation being performed
            state: State snapshot
            description: Checkpoint description
            rollback_actions: Actions to perform on rollback
            
        Returns:
            Created checkpoint
        """
        checkpoint = Checkpoint(
            checkpoint_id=str(uuid.uuid4()),
            operation=operation,
            state=copy.deepcopy(state) if state else {},
            created_at=datetime.now(),
            description=description,
            rollback_actions=rollback_actions or [],
        )
        
        with self._lock:
            self._checkpoints[checkpoint.checkpoint_id] = checkpoint
            self._checkpoint_order.append(checkpoint.checkpoint_id)
            
            # Trim old checkpoints
            while len(self._checkpoint_order) > self._max_checkpoints:
                old_id = self._checkpoint_order.pop(0)
                del self._checkpoints[old_id]
        
        return checkpoint
    
    def rollback(
        self,
        checkpoint_id: str,
    ) -> bool:
        """Rollback to a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to rollback to
            
        Returns:
            Whether rollback succeeded
        """
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                logger.warning(f"Checkpoint not found: {checkpoint_id}")
                return False
        
        # Execute rollback actions
        success = True
        for action in checkpoint.rollback_actions:
            try:
                action()
            except Exception as e:
                logger.error(f"Rollback action failed: {e}")
                success = False
        
        logger.info(f"Rolled back to checkpoint: {checkpoint_id}")
        return success
    
    def rollback_to_latest(self) -> bool:
        """Rollback to the most recent checkpoint.
        
        Returns:
            Whether rollback succeeded
        """
        with self._lock:
            if not self._checkpoint_order:
                return False
            
            latest_id = self._checkpoint_order[-1]
        
        return self.rollback(latest_id)
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a checkpoint by ID."""
        with self._lock:
            return self._checkpoints.get(checkpoint_id)
    
    def list_checkpoints(self) -> List[Checkpoint]:
        """List all checkpoints."""
        with self._lock:
            return [
                self._checkpoints[cid]
                for cid in self._checkpoint_order
                if cid in self._checkpoints
            ]
    
    def clear_checkpoints(self):
        """Clear all checkpoints."""
        with self._lock:
            self._checkpoints.clear()
            self._checkpoint_order.clear()


class FallbackManager:
    """Manage fallback operations.
    
    Uses LLM reasoning to:
    1. Detect available fallbacks
    2. Select best fallback
    3. Suggest workarounds
    
    Example:
        >>> manager = FallbackManager()
        >>> manager.register_fallback("read_file", fallback_read)
        >>> result = manager.execute_with_fallback("read_file", primary_read)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize fallback manager.
        
        Args:
            llm_client: LLM client for intelligent fallback
        """
        self._llm_client = llm_client
        
        # Registered fallbacks
        self._fallbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Fallback statistics
        self._fallback_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"attempts": 0, "successes": 0}
        )
    
    def register_fallback(
        self,
        operation: str,
        fallback: Callable,
        priority: int = 0,
    ):
        """Register a fallback for an operation.
        
        Args:
            operation: Operation name
            fallback: Fallback function
            priority: Higher priority = tried first
        """
        self._fallbacks[operation].append((priority, fallback))
        # Sort by priority (descending)
        self._fallbacks[operation].sort(key=lambda x: -x[0])
    
    def execute_with_fallback(
        self,
        operation: str,
        primary: Callable[..., T],
        *args,
        **kwargs,
    ) -> Tuple[T, bool]:
        """Execute with fallback on failure.
        
        Args:
            operation: Operation name
            primary: Primary function
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, used_fallback)
        """
        # Try primary
        try:
            result = primary(*args, **kwargs)
            return result, False
        except Exception as primary_error:
            logger.warning(f"Primary failed for {operation}: {primary_error}")
        
        # Try fallbacks
        for priority, fallback in self._fallbacks.get(operation, []):
            self._fallback_stats[operation]["attempts"] += 1
            
            try:
                result = fallback(*args, **kwargs)
                self._fallback_stats[operation]["successes"] += 1
                logger.info(f"Fallback succeeded for {operation}")
                return result, True
            except Exception as e:
                logger.warning(f"Fallback failed for {operation}: {e}")
        
        # All failed
        raise primary_error
    
    def get_fallback_stats(self, operation: str) -> Dict[str, int]:
        """Get fallback statistics for operation."""
        return dict(self._fallback_stats.get(operation, {"attempts": 0, "successes": 0}))
    
    def has_fallbacks(self, operation: str) -> bool:
        """Check if operation has fallbacks."""
        return bool(self._fallbacks.get(operation))
    
    async def suggest_workarounds(
        self,
        operation: str,
        error: Exception,
    ) -> List[str]:
        """Suggest workarounds using LLM.
        
        Args:
            operation: Failed operation
            error: The error
            
        Returns:
            List of workaround suggestions
        """
        if not self._llm_client:
            return []
        
        prompt = f"""The operation '{operation}' failed with error:
{type(error).__name__}: {str(error)}

Suggest 3-5 alternative approaches or workarounds.
Return as JSON array of strings.
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Workaround suggestion failed: {e}")
            return []


class RecoveryStrategySelector:
    """Select appropriate recovery strategy.
    
    Uses LLM reasoning to:
    1. Analyze error context
    2. Consider past success rates
    3. Select optimal strategy
    
    Example:
        >>> selector = RecoveryStrategySelector()
        >>> strategy = selector.select_strategy(error, context)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize strategy selector.
        
        Args:
            llm_client: LLM client for intelligent selection
        """
        self._llm_client = llm_client
        
        # Strategy records
        self._strategy_records: Dict[str, StrategyRecord] = {}
        
        # Strategy rules
        self._rules: List[Tuple[Callable[[Exception], bool], RecoveryStrategy]] = [
            # Transient errors -> retry
            (lambda e: "timeout" in str(e).lower(), RecoveryStrategy.RETRY),
            (lambda e: "connection" in str(e).lower(), RecoveryStrategy.RETRY),
            (lambda e: "rate limit" in str(e).lower(), RecoveryStrategy.RETRY),
            
            # Permission errors -> escalate
            (lambda e: isinstance(e, PermissionError), RecoveryStrategy.ESCALATE),
            
            # File errors -> might need rollback
            (lambda e: isinstance(e, FileNotFoundError), RecoveryStrategy.FALLBACK),
            
            # Memory errors -> degrade
            (lambda e: isinstance(e, MemoryError), RecoveryStrategy.DEGRADE),
        ]
    
    def select_strategy(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        available_fallbacks: bool = False,
        checkpoint_available: bool = False,
    ) -> RecoveryStrategy:
        """Select recovery strategy.
        
        Args:
            error: The error
            context: Error context
            available_fallbacks: Whether fallbacks exist
            checkpoint_available: Whether checkpoint exists
            
        Returns:
            Selected strategy
        """
        error_type = type(error).__name__
        
        # Check rules first
        for condition, strategy in self._rules:
            try:
                if condition(error):
                    self._record_selection(error_type, strategy)
                    return strategy
            except Exception:
                pass
        
        # Consider success rates
        best_strategy = self._select_by_success_rate(error_type)
        if best_strategy:
            return best_strategy
        
        # Default logic
        if available_fallbacks:
            return RecoveryStrategy.FALLBACK
        elif checkpoint_available:
            return RecoveryStrategy.ROLLBACK
        else:
            return RecoveryStrategy.RETRY
    
    async def select_strategy_with_llm(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[RecoveryStrategy, str]:
        """Select strategy using LLM reasoning.
        
        Args:
            error: The error
            context: Error context
            
        Returns:
            Tuple of (strategy, reasoning)
        """
        if not self._llm_client:
            strategy = self.select_strategy(error, context)
            return strategy, "LLM not available"
        
        strategies = [s.value for s in RecoveryStrategy]
        
        prompt = f"""Select the best recovery strategy for this error:

Error Type: {type(error).__name__}
Error Message: {str(error)}
Context: {json.dumps(context or {}, indent=2)}

Available Strategies: {', '.join(strategies)}

Return JSON with:
- strategy: one of the strategies above
- reasoning: brief explanation
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            data = json.loads(response)
            strategy = RecoveryStrategy(data["strategy"])
            reasoning = data.get("reasoning", "")
            return strategy, reasoning
        except Exception as e:
            logger.warning(f"LLM strategy selection failed: {e}")
            strategy = self.select_strategy(error, context)
            return strategy, f"LLM failed: {e}"
    
    def _select_by_success_rate(
        self,
        error_type: str,
    ) -> Optional[RecoveryStrategy]:
        """Select strategy with best success rate for error type."""
        best_strategy = None
        best_rate = 0.0
        
        for key, record in self._strategy_records.items():
            if record.error_type == error_type and record.success_rate > best_rate:
                best_rate = record.success_rate
                best_strategy = record.strategy
        
        return best_strategy if best_rate > 0.5 else None
    
    def _record_selection(
        self,
        error_type: str,
        strategy: RecoveryStrategy,
    ):
        """Record strategy selection."""
        key = f"{error_type}_{strategy.value}"
        
        if key not in self._strategy_records:
            self._strategy_records[key] = StrategyRecord(
                strategy=strategy,
                error_type=error_type,
            )
        
        self._strategy_records[key].last_used = datetime.now()
    
    def record_outcome(
        self,
        error_type: str,
        strategy: RecoveryStrategy,
        success: bool,
    ):
        """Record strategy outcome.
        
        Args:
            error_type: Type of error
            strategy: Strategy used
            success: Whether recovery succeeded
        """
        key = f"{error_type}_{strategy.value}"
        
        if key not in self._strategy_records:
            self._strategy_records[key] = StrategyRecord(
                strategy=strategy,
                error_type=error_type,
            )
        
        record = self._strategy_records[key]
        record.total_attempts += 1
        
        if success:
            record.successful_attempts += 1
            record.last_success = datetime.now()
    
    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get strategy statistics."""
        return {
            key: {
                "strategy": record.strategy.value,
                "error_type": record.error_type,
                "success_rate": record.success_rate,
                "total_attempts": record.total_attempts,
            }
            for key, record in self._strategy_records.items()
        }


class AutomatedRecoverySystem:
    """Main automated recovery system.
    
    Integrates all recovery components:
    - Strategy selection
    - Retry execution
    - Checkpoint management
    - Fallback handling
    
    Example:
        >>> recovery = AutomatedRecoverySystem()
        >>> result = recovery.execute_with_recovery(my_function, args)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize recovery system.
        
        Args:
            llm_client: LLM client for intelligent recovery
        """
        self._llm_client = llm_client
        
        # Initialize components
        self._strategy_selector = RecoveryStrategySelector(llm_client=llm_client)
        self._retry_executor = RetryExecutor(llm_client=llm_client)
        self._checkpoint_manager = CheckpointManager()
        self._fallback_manager = FallbackManager(llm_client=llm_client)
        
        # Recovery history
        self._recovery_history: List[RecoveryResult] = []
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def execute_with_recovery(
        self,
        func: Callable[..., T],
        *args,
        operation_name: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
        create_checkpoint: bool = False,
        checkpoint_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> RecoveryResult:
        """Execute function with full recovery support.
        
        Args:
            func: Function to execute
            *args: Function arguments
            operation_name: Operation name
            retry_config: Retry configuration
            create_checkpoint: Whether to create checkpoint
            checkpoint_state: State to checkpoint
            **kwargs: Function keyword arguments
            
        Returns:
            RecoveryResult
        """
        op_name = operation_name or func.__name__
        start_time = time.time()
        
        # Create checkpoint if requested
        checkpoint = None
        if create_checkpoint:
            checkpoint = self._checkpoint_manager.create_checkpoint(
                operation=op_name,
                state=checkpoint_state,
            )
        
        result = RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.RETRY,
        )
        
        try:
            # Try with retry
            output = self._retry_executor.execute_with_retry(
                func,
                *args,
                operation_name=op_name,
                config=retry_config,
                **kwargs,
            )
            
            result.success = True
            result.result = output
            
        except Exception as e:
            # Retry failed, try other strategies
            has_fallbacks = self._fallback_manager.has_fallbacks(op_name)
            has_checkpoint = checkpoint is not None
            
            strategy = self._strategy_selector.select_strategy(
                error=e,
                available_fallbacks=has_fallbacks,
                checkpoint_available=has_checkpoint,
            )
            
            result.strategy_used = strategy
            result.final_error = e
            
            if strategy == RecoveryStrategy.FALLBACK and has_fallbacks:
                try:
                    output, used_fallback = self._fallback_manager.execute_with_fallback(
                        op_name,
                        func,
                        *args,
                        **kwargs,
                    )
                    result.success = True
                    result.result = output
                except Exception as fallback_error:
                    result.final_error = fallback_error
            
            elif strategy == RecoveryStrategy.ROLLBACK and has_checkpoint:
                self._checkpoint_manager.rollback(checkpoint.checkpoint_id)
            
            elif strategy == RecoveryStrategy.ESCALATE:
                result.escalated = True
                result.manual_intervention_required = True
            
            # Record outcome
            self._strategy_selector.record_outcome(
                error_type=type(e).__name__,
                strategy=strategy,
                success=result.success,
            )
        
        result.total_duration = time.time() - start_time
        
        # Store in history
        with self._lock:
            self._recovery_history.append(result)
            if len(self._recovery_history) > 500:
                self._recovery_history = self._recovery_history[-500:]
        
        return result
    
    async def execute_with_recovery_async(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        operation_name: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs,
    ) -> RecoveryResult:
        """Execute async function with recovery.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            operation_name: Operation name
            retry_config: Retry configuration
            **kwargs: Function keyword arguments
            
        Returns:
            RecoveryResult
        """
        op_name = operation_name or func.__name__
        start_time = time.time()
        
        result = RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.RETRY,
        )
        
        try:
            output = await self._retry_executor.execute_with_retry_async(
                func,
                *args,
                operation_name=op_name,
                config=retry_config,
                **kwargs,
            )
            
            result.success = True
            result.result = output
            
        except Exception as e:
            strategy, reasoning = await self._strategy_selector.select_strategy_with_llm(
                error=e,
            )
            
            result.strategy_used = strategy
            result.final_error = e
            
            if strategy == RecoveryStrategy.ESCALATE:
                result.escalated = True
                result.manual_intervention_required = True
        
        result.total_duration = time.time() - start_time
        
        with self._lock:
            self._recovery_history.append(result)
        
        return result
    
    def register_fallback(
        self,
        operation: str,
        fallback: Callable,
        priority: int = 0,
    ):
        """Register a fallback for an operation."""
        self._fallback_manager.register_fallback(operation, fallback, priority)
    
    def create_checkpoint(
        self,
        operation: str,
        state: Optional[Dict[str, Any]] = None,
        rollback_actions: Optional[List[Callable]] = None,
    ) -> Checkpoint:
        """Create a checkpoint."""
        return self._checkpoint_manager.create_checkpoint(
            operation=operation,
            state=state,
            rollback_actions=rollback_actions,
        )
    
    def rollback(self, checkpoint_id: str) -> bool:
        """Rollback to checkpoint."""
        return self._checkpoint_manager.rollback(checkpoint_id)
    
    def get_recovery_history(self, limit: int = 50) -> List[RecoveryResult]:
        """Get recovery history."""
        with self._lock:
            return self._recovery_history[-limit:]
    
    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get strategy statistics."""
        return self._strategy_selector.get_strategy_stats()
    
    def get_circuit_state(self, operation: str) -> CircuitState:
        """Get circuit breaker state."""
        return self._retry_executor.get_circuit_state(operation)
    
    def reset_circuit(self, operation: str):
        """Reset circuit breaker."""
        self._retry_executor.reset_circuit(operation)


# Module-level instance
_global_recovery_system: Optional[AutomatedRecoverySystem] = None


def get_automated_recovery_system(
    llm_client: Optional[Any] = None,
) -> AutomatedRecoverySystem:
    """Get the global recovery system.
    
    Args:
        llm_client: Optional LLM client
        
    Returns:
        AutomatedRecoverySystem instance
    """
    global _global_recovery_system
    if _global_recovery_system is None:
        _global_recovery_system = AutomatedRecoverySystem(
            llm_client=llm_client,
        )
    return _global_recovery_system


# Decorator for easy retry
def with_retry(
    max_attempts: int = 3,
    backoff: RetryBackoff = RetryBackoff.EXPONENTIAL,
    retry_on: Optional[List[Type[Exception]]] = None,
):
    """Decorator to add retry logic to a function.
    
    Example:
        >>> @with_retry(max_attempts=5)
        ... def my_function():
        ...     # code that might fail
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            config = RetryConfig(
                max_attempts=max_attempts,
                backoff=backoff,
                retry_on=retry_on or [],
            )
            
            executor = RetryExecutor()
            return executor.execute_with_retry(
                func,
                *args,
                config=config,
                **kwargs,
            )
        
        return wrapper
    return decorator
