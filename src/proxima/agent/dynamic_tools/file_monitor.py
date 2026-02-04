"""File Monitoring and Synchronization for Dynamic AI Assistant.

This module implements Phase 5.2 for the Dynamic AI Assistant:
- File System Watching: Real-time file monitoring with watchdog
- Change Detection and Analysis: Diff generation, categorization, impact analysis
- Synchronization Logic: Bidirectional sync, merge strategies, conflict resolution

Key Features:
============
- Real-time file system event monitoring
- Intelligent change detection and analysis
- File synchronization with conflict resolution
- LLM-driven change analysis and suggestions

Design Principle:
================
All monitoring and sync decisions use LLM reasoning when available.
The LLM analyzes changes and suggests appropriate actions.
"""

from __future__ import annotations

import asyncio
import difflib
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
from queue import Empty, Queue
from typing import (
    Any, Callable, Dict, Generator, Iterator, List, 
    Optional, Pattern, Set, Tuple, Union
)
import uuid

logger = logging.getLogger(__name__)


# Try to import watchdog
try:
    from watchdog.observers import Observer
    from watchdog.events import (
        FileSystemEventHandler,
        FileSystemEvent,
        FileCreatedEvent,
        FileModifiedEvent,
        FileDeletedEvent,
        FileMovedEvent,
        DirCreatedEvent,
        DirModifiedEvent,
        DirDeletedEvent,
        DirMovedEvent,
    )
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object


class EventType(Enum):
    """File system event types."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"
    RENAMED = "renamed"


class ChangeCategory(Enum):
    """Change categories for analysis."""
    ADDITION = "addition"
    DELETION = "deletion"
    MODIFICATION = "modification"
    REFACTORING = "refactoring"
    FORMATTING = "formatting"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class SyncDirection(Enum):
    """Synchronization directions."""
    SOURCE_TO_TARGET = "source_to_target"
    TARGET_TO_SOURCE = "target_to_source"
    BIDIRECTIONAL = "bidirectional"


class MergeStrategy(Enum):
    """Merge strategies for conflict resolution."""
    NEWER_WINS = "newer_wins"
    LARGER_WINS = "larger_wins"
    SOURCE_WINS = "source_wins"
    TARGET_WINS = "target_wins"
    MANUAL = "manual"
    MERGE = "merge"


@dataclass
class FileEvent:
    """A file system event."""
    event_id: str
    event_type: EventType
    path: Path
    timestamp: datetime
    
    # For move/rename events
    old_path: Optional[Path] = None
    
    # File metadata
    is_directory: bool = False
    file_size: Optional[int] = None
    
    # Processing state
    is_processed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "path": str(self.path),
            "timestamp": self.timestamp.isoformat(),
            "old_path": str(self.old_path) if self.old_path else None,
            "is_directory": self.is_directory,
            "file_size": self.file_size,
            "is_processed": self.is_processed,
        }


@dataclass
class FileChange:
    """A detected file change with diff information."""
    change_id: str
    path: Path
    category: ChangeCategory
    timestamp: datetime
    
    # Diff information
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    diff_lines: Optional[List[str]] = None
    
    # Statistics
    lines_added: int = 0
    lines_removed: int = 0
    lines_modified: int = 0
    
    # Analysis
    summary: Optional[str] = None
    impact_level: str = "low"  # low, medium, high
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_id": self.change_id,
            "path": str(self.path),
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "lines_modified": self.lines_modified,
            "summary": self.summary,
            "impact_level": self.impact_level,
        }


@dataclass
class SyncConflict:
    """A synchronization conflict."""
    conflict_id: str
    path: Path
    source_path: Path
    target_path: Path
    
    # Versions
    source_modified: datetime
    target_modified: datetime
    source_size: int
    target_size: int
    
    # Content
    source_content: Optional[str] = None
    target_content: Optional[str] = None
    
    # Resolution
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "path": str(self.path),
            "source_modified": self.source_modified.isoformat(),
            "target_modified": self.target_modified.isoformat(),
            "source_size": self.source_size,
            "target_size": self.target_size,
            "resolution": self.resolution,
        }


@dataclass
class SyncState:
    """Synchronization state between two directories."""
    source_path: Path
    target_path: Path
    
    # File states (path -> checksum)
    source_files: Dict[str, str] = field(default_factory=dict)
    target_files: Dict[str, str] = field(default_factory=dict)
    
    # Sync metadata
    last_sync: Optional[datetime] = None
    sync_count: int = 0
    
    # Pending operations
    pending_copies: List[Tuple[Path, Path]] = field(default_factory=list)
    pending_deletes: List[Path] = field(default_factory=list)
    conflicts: List[SyncConflict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "target_path": str(self.target_path),
            "source_file_count": len(self.source_files),
            "target_file_count": len(self.target_files),
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "sync_count": self.sync_count,
            "pending_copies": len(self.pending_copies),
            "pending_deletes": len(self.pending_deletes),
            "conflicts": len(self.conflicts),
        }


class EventHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """File system event handler using watchdog."""
    
    def __init__(
        self,
        event_queue: Queue,
        patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ):
        """Initialize the event handler.
        
        Args:
            event_queue: Queue to receive events
            patterns: Glob patterns to include
            ignore_patterns: Glob patterns to ignore
        """
        if WATCHDOG_AVAILABLE:
            super().__init__()
        
        self._queue = event_queue
        self._patterns = patterns or ["*"]
        self._ignore_patterns = ignore_patterns or [
            "*.pyc", "__pycache__", ".git", ".svn", ".hg",
            "node_modules", "*.tmp", "*.bak", "*.swp",
        ]
    
    def _should_process(self, path: str) -> bool:
        """Check if path should be processed."""
        path_obj = Path(path)
        name = path_obj.name
        
        # Check ignore patterns
        for pattern in self._ignore_patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return False
            elif pattern in str(path_obj):
                return False
        
        # Check include patterns
        for pattern in self._patterns:
            if pattern == "*":
                return True
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern in name:
                return True
        
        return False
    
    def _create_event(
        self,
        event_type: EventType,
        path: str,
        is_directory: bool,
        old_path: Optional[str] = None,
    ) -> FileEvent:
        """Create a FileEvent from a watchdog event."""
        path_obj = Path(path)
        
        file_size = None
        if not is_directory and path_obj.exists():
            try:
                file_size = path_obj.stat().st_size
            except Exception:
                pass
        
        return FileEvent(
            event_id=str(uuid.uuid4())[:8],
            event_type=event_type,
            path=path_obj,
            timestamp=datetime.now(),
            old_path=Path(old_path) if old_path else None,
            is_directory=is_directory,
            file_size=file_size,
        )
    
    def on_created(self, event):
        """Handle created event."""
        if self._should_process(event.src_path):
            file_event = self._create_event(
                EventType.CREATED,
                event.src_path,
                event.is_directory,
            )
            self._queue.put(file_event)
    
    def on_modified(self, event):
        """Handle modified event."""
        if self._should_process(event.src_path):
            file_event = self._create_event(
                EventType.MODIFIED,
                event.src_path,
                event.is_directory,
            )
            self._queue.put(file_event)
    
    def on_deleted(self, event):
        """Handle deleted event."""
        if self._should_process(event.src_path):
            file_event = self._create_event(
                EventType.DELETED,
                event.src_path,
                event.is_directory,
            )
            self._queue.put(file_event)
    
    def on_moved(self, event):
        """Handle moved event."""
        if self._should_process(event.dest_path):
            file_event = self._create_event(
                EventType.MOVED,
                event.dest_path,
                event.is_directory,
                old_path=event.src_path,
            )
            self._queue.put(file_event)


class FileSystemWatcher:
    """File system watcher with event aggregation.
    
    Uses LLM reasoning to:
    1. Filter and prioritize events based on importance
    2. Aggregate related events to reduce noise
    3. Suggest actions based on detected changes
    
    Example:
        >>> watcher = FileSystemWatcher(llm_client=client)
        >>> watcher.start(Path("./src"))
        >>> for event in watcher.get_events():
        ...     print(event)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        aggregation_window: float = 0.5,
    ):
        """Initialize the file system watcher.
        
        Args:
            llm_client: LLM client for event analysis
            patterns: Glob patterns to include
            ignore_patterns: Glob patterns to ignore
            aggregation_window: Seconds to aggregate events
        """
        self._llm_client = llm_client
        self._patterns = patterns
        self._ignore_patterns = ignore_patterns
        self._aggregation_window = aggregation_window
        
        # Event handling
        self._event_queue: Queue = Queue()
        self._aggregated_events: List[FileEvent] = []
        self._event_callbacks: List[Callable[[FileEvent], None]] = []
        
        # Watchdog
        self._observer: Optional[Observer] = None
        self._watched_paths: Set[Path] = set()
        
        # State
        self._is_running = False
        self._aggregator_thread: Optional[threading.Thread] = None
        
        # Event history
        self._event_history: List[FileEvent] = []
        self._max_history = 1000
    
    def start(
        self,
        path: Path,
        recursive: bool = True,
    ) -> bool:
        """Start watching a path.
        
        Args:
            path: Path to watch
            recursive: Watch subdirectories
            
        Returns:
            True if started successfully
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available, using polling fallback")
            return self._start_polling(path, recursive)
        
        path = Path(path)
        
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        if path in self._watched_paths:
            return True
        
        # Create handler
        handler = EventHandler(
            self._event_queue,
            self._patterns,
            self._ignore_patterns,
        )
        
        # Create and start observer
        if self._observer is None:
            self._observer = Observer()
        
        self._observer.schedule(handler, str(path), recursive=recursive)
        
        if not self._is_running:
            self._observer.start()
            self._is_running = True
            
            # Start event aggregator
            self._aggregator_thread = threading.Thread(
                target=self._aggregate_events,
                daemon=True,
            )
            self._aggregator_thread.start()
        
        self._watched_paths.add(path)
        logger.info(f"Started watching: {path}")
        
        return True
    
    def _start_polling(
        self,
        path: Path,
        recursive: bool,
    ) -> bool:
        """Start polling-based watching (fallback when watchdog not available)."""
        self._watched_paths.add(path)
        self._is_running = True
        
        # Start polling thread
        self._aggregator_thread = threading.Thread(
            target=self._poll_changes,
            args=(path, recursive),
            daemon=True,
        )
        self._aggregator_thread.start()
        
        return True
    
    def _poll_changes(self, path: Path, recursive: bool):
        """Poll for file changes (fallback method)."""
        last_state: Dict[str, Tuple[float, int]] = {}
        
        def get_state(p: Path) -> Dict[str, Tuple[float, int]]:
            state = {}
            try:
                if recursive:
                    for root, dirs, files in os.walk(p):
                        # Skip ignored directories
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                        
                        for f in files:
                            fp = Path(root) / f
                            try:
                                st = fp.stat()
                                state[str(fp)] = (st.st_mtime, st.st_size)
                            except Exception:
                                pass
                else:
                    for fp in p.iterdir():
                        if fp.is_file():
                            st = fp.stat()
                            state[str(fp)] = (st.st_mtime, st.st_size)
            except Exception:
                pass
            return state
        
        last_state = get_state(path)
        
        while self._is_running:
            time.sleep(1.0)  # Poll interval
            
            current_state = get_state(path)
            
            # Find changes
            for fp, (mtime, size) in current_state.items():
                if fp not in last_state:
                    # New file
                    event = FileEvent(
                        event_id=str(uuid.uuid4())[:8],
                        event_type=EventType.CREATED,
                        path=Path(fp),
                        timestamp=datetime.now(),
                        file_size=size,
                    )
                    self._event_queue.put(event)
                elif last_state[fp] != (mtime, size):
                    # Modified file
                    event = FileEvent(
                        event_id=str(uuid.uuid4())[:8],
                        event_type=EventType.MODIFIED,
                        path=Path(fp),
                        timestamp=datetime.now(),
                        file_size=size,
                    )
                    self._event_queue.put(event)
            
            for fp in last_state:
                if fp not in current_state:
                    # Deleted file
                    event = FileEvent(
                        event_id=str(uuid.uuid4())[:8],
                        event_type=EventType.DELETED,
                        path=Path(fp),
                        timestamp=datetime.now(),
                    )
                    self._event_queue.put(event)
            
            last_state = current_state
    
    def stop(self):
        """Stop watching all paths."""
        self._is_running = False
        
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        
        self._watched_paths.clear()
        logger.info("Stopped file system watcher")
    
    def _aggregate_events(self):
        """Aggregate events from the queue."""
        pending: Dict[str, List[FileEvent]] = defaultdict(list)
        last_flush = time.time()
        
        while self._is_running:
            # Get events from queue
            try:
                event = self._event_queue.get(timeout=0.1)
                pending[str(event.path)].append(event)
            except Empty:
                pass
            
            # Flush aggregated events
            current_time = time.time()
            if current_time - last_flush >= self._aggregation_window:
                for path, events in pending.items():
                    if events:
                        # Aggregate to single event
                        aggregated = self._aggregate_path_events(events)
                        if aggregated:
                            self._process_event(aggregated)
                
                pending.clear()
                last_flush = current_time
    
    def _aggregate_path_events(
        self,
        events: List[FileEvent],
    ) -> Optional[FileEvent]:
        """Aggregate multiple events for the same path."""
        if not events:
            return None
        
        # If only one event, return it
        if len(events) == 1:
            return events[0]
        
        # Get event types
        types = [e.event_type for e in events]
        
        # Determine final event type
        if EventType.DELETED in types:
            if EventType.CREATED in types:
                # Created then deleted - no net change
                return None
            return events[-1]  # Return delete event
        
        if EventType.CREATED in types:
            return events[0]  # Return create event
        
        if EventType.MOVED in types:
            return events[-1]  # Return move event
        
        # Multiple modifications - return latest
        return events[-1]
    
    def _process_event(self, event: FileEvent):
        """Process a single event."""
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
        
        # Call callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def add_callback(self, callback: Callable[[FileEvent], None]):
        """Add an event callback.
        
        Args:
            callback: Function to call for each event
        """
        self._event_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[FileEvent], None]):
        """Remove an event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)
    
    def get_events(
        self,
        timeout: Optional[float] = None,
    ) -> Iterator[FileEvent]:
        """Get events as an iterator.
        
        Args:
            timeout: Optional timeout in seconds
            
        Yields:
            File events
        """
        start_time = time.time()
        
        while self._is_running:
            if timeout and time.time() - start_time > timeout:
                break
            
            try:
                event = self._event_queue.get(timeout=0.1)
                yield event
            except Empty:
                continue
    
    def get_recent_events(
        self,
        limit: int = 50,
        event_types: Optional[List[EventType]] = None,
    ) -> List[FileEvent]:
        """Get recent events from history.
        
        Args:
            limit: Maximum events to return
            event_types: Filter by event types
            
        Returns:
            List of recent events
        """
        events = self._event_history[-limit:]
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        return events


class ChangeDetector:
    """Change detection and analysis.
    
    Uses LLM reasoning to:
    1. Categorize changes (addition, deletion, refactoring, etc.)
    2. Summarize changes in natural language
    3. Assess impact of changes
    
    Example:
        >>> detector = ChangeDetector(llm_client=client)
        >>> change = detector.analyze_change(old_content, new_content, path)
        >>> print(change.summary)
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        """Initialize the change detector.
        
        Args:
            llm_client: LLM client for analysis
        """
        self._llm_client = llm_client
        
        # Content cache for diff generation
        self._content_cache: Dict[str, Tuple[str, datetime]] = {}
        self._cache_ttl = timedelta(minutes=30)
    
    def analyze_change(
        self,
        old_content: Optional[str],
        new_content: Optional[str],
        path: Path,
    ) -> FileChange:
        """Analyze a file change.
        
        Args:
            old_content: Previous file content
            new_content: New file content
            path: File path
            
        Returns:
            Analyzed file change
        """
        change = FileChange(
            change_id=str(uuid.uuid4())[:8],
            path=path,
            category=ChangeCategory.UNKNOWN,
            timestamp=datetime.now(),
            old_content=old_content,
            new_content=new_content,
        )
        
        if old_content is None and new_content is not None:
            change.category = ChangeCategory.ADDITION
            change.lines_added = new_content.count('\n') + 1
        elif old_content is not None and new_content is None:
            change.category = ChangeCategory.DELETION
            change.lines_removed = old_content.count('\n') + 1
        elif old_content is not None and new_content is not None:
            # Generate diff
            diff = self._generate_diff(old_content, new_content)
            change.diff_lines = diff
            
            # Count changes
            for line in diff:
                if line.startswith('+') and not line.startswith('+++'):
                    change.lines_added += 1
                elif line.startswith('-') and not line.startswith('---'):
                    change.lines_removed += 1
            
            change.lines_modified = min(change.lines_added, change.lines_removed)
            
            # Categorize change
            change.category = self._categorize_change(
                old_content, new_content, diff, path
            )
            
            # Assess impact
            change.impact_level = self._assess_impact(change)
        
        # Generate summary
        change.summary = self._generate_summary(change)
        
        return change
    
    def _generate_diff(
        self,
        old_content: str,
        new_content: str,
    ) -> List[str]:
        """Generate unified diff between contents."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile='old', tofile='new',
            lineterm='',
        ))
        
        return diff
    
    def _categorize_change(
        self,
        old_content: str,
        new_content: str,
        diff: List[str],
        path: Path,
    ) -> ChangeCategory:
        """Categorize the type of change."""
        # Check for formatting-only changes
        if self._is_formatting_only(old_content, new_content):
            return ChangeCategory.FORMATTING
        
        # Check for documentation changes
        if self._is_documentation_change(diff, path):
            return ChangeCategory.DOCUMENTATION
        
        # Check for configuration changes
        if self._is_config_file(path):
            return ChangeCategory.CONFIGURATION
        
        # Check for refactoring (moved code)
        if self._is_refactoring(old_content, new_content, diff):
            return ChangeCategory.REFACTORING
        
        # Default to modification
        return ChangeCategory.MODIFICATION
    
    def _is_formatting_only(self, old: str, new: str) -> bool:
        """Check if change is only whitespace/formatting."""
        # Remove all whitespace and compare
        old_no_ws = re.sub(r'\s+', '', old)
        new_no_ws = re.sub(r'\s+', '', new)
        return old_no_ws == new_no_ws
    
    def _is_documentation_change(
        self,
        diff: List[str],
        path: Path,
    ) -> bool:
        """Check if change is primarily documentation."""
        # Check file extension
        doc_extensions = {'.md', '.rst', '.txt', '.doc'}
        if path.suffix.lower() in doc_extensions:
            return True
        
        # Check if changes are in comments/docstrings
        comment_patterns = [
            r'^[+-]\s*#',  # Python comments
            r'^[+-]\s*//',  # C-style comments
            r'^[+-]\s*\*',  # Multi-line comment continuation
            r'^[+-]\s*"""',  # Python docstrings
            r'^[+-]\s*\'\'\'',  # Python docstrings
        ]
        
        comment_lines = 0
        total_changed = 0
        
        for line in diff:
            if line.startswith('+') or line.startswith('-'):
                if line.startswith('+++') or line.startswith('---'):
                    continue
                total_changed += 1
                for pattern in comment_patterns:
                    if re.match(pattern, line):
                        comment_lines += 1
                        break
        
        return total_changed > 0 and comment_lines / total_changed > 0.8
    
    def _is_config_file(self, path: Path) -> bool:
        """Check if file is a configuration file."""
        config_extensions = {
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            '.conf', '.xml', '.properties', '.env',
        }
        config_names = {
            'config', 'settings', '.gitignore', '.dockerignore',
            'Makefile', 'Dockerfile', 'docker-compose',
        }
        
        return (
            path.suffix.lower() in config_extensions or
            path.stem.lower() in config_names or
            any(name in path.name.lower() for name in ['config', 'settings'])
        )
    
    def _is_refactoring(
        self,
        old: str,
        new: str,
        diff: List[str],
    ) -> bool:
        """Check if change is a refactoring (code moved but not changed)."""
        # Extract added and removed lines (without + and - prefix)
        added = []
        removed = []
        
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                added.append(line[1:].strip())
            elif line.startswith('-') and not line.startswith('---'):
                removed.append(line[1:].strip())
        
        if not added or not removed:
            return False
        
        # Check if significant overlap (code moved)
        added_set = set(added)
        removed_set = set(removed)
        
        # Remove empty lines
        added_set.discard('')
        removed_set.discard('')
        
        if not added_set or not removed_set:
            return False
        
        overlap = added_set & removed_set
        overlap_ratio = len(overlap) / max(len(added_set), len(removed_set))
        
        return overlap_ratio > 0.5
    
    def _assess_impact(self, change: FileChange) -> str:
        """Assess the impact level of a change."""
        # High impact indicators
        high_impact_indicators = [
            change.lines_added > 100,
            change.lines_removed > 100,
            change.category == ChangeCategory.REFACTORING,
        ]
        
        if any(high_impact_indicators):
            return "high"
        
        # Medium impact indicators
        medium_impact_indicators = [
            change.lines_added > 20,
            change.lines_removed > 20,
            change.category == ChangeCategory.CONFIGURATION,
        ]
        
        if any(medium_impact_indicators):
            return "medium"
        
        return "low"
    
    def _generate_summary(self, change: FileChange) -> str:
        """Generate a human-readable summary of the change."""
        parts = []
        
        # Category
        if change.category == ChangeCategory.ADDITION:
            parts.append(f"Added new file with {change.lines_added} lines")
        elif change.category == ChangeCategory.DELETION:
            parts.append(f"Deleted file with {change.lines_removed} lines")
        elif change.category == ChangeCategory.FORMATTING:
            parts.append("Formatting changes only")
        elif change.category == ChangeCategory.DOCUMENTATION:
            parts.append("Documentation update")
        elif change.category == ChangeCategory.CONFIGURATION:
            parts.append("Configuration change")
        elif change.category == ChangeCategory.REFACTORING:
            parts.append("Code refactoring")
        else:
            parts.append(f"Modified: +{change.lines_added}, -{change.lines_removed} lines")
        
        # Impact
        parts.append(f"Impact: {change.impact_level}")
        
        return " | ".join(parts)
    
    def cache_content(self, path: Path, content: str):
        """Cache file content for later diff generation.
        
        Args:
            path: File path
            content: File content
        """
        key = str(path)
        self._content_cache[key] = (content, datetime.now())
        
        # Clean old cache entries
        now = datetime.now()
        expired = [
            k for k, (_, ts) in self._content_cache.items()
            if now - ts > self._cache_ttl
        ]
        for k in expired:
            del self._content_cache[k]
    
    def get_cached_content(self, path: Path) -> Optional[str]:
        """Get cached file content.
        
        Args:
            path: File path
            
        Returns:
            Cached content or None
        """
        key = str(path)
        if key in self._content_cache:
            content, ts = self._content_cache[key]
            if datetime.now() - ts < self._cache_ttl:
                return content
        return None
    
    async def analyze_with_llm(
        self,
        change: FileChange,
    ) -> Dict[str, Any]:
        """Analyze change using LLM for detailed insights.
        
        Args:
            change: The file change to analyze
            
        Returns:
            LLM analysis results
        """
        if not self._llm_client:
            return {"error": "LLM client not available"}
        
        # Build prompt
        diff_preview = ""
        if change.diff_lines:
            diff_preview = "\n".join(change.diff_lines[:50])
        
        prompt = f"""Analyze this code change:

File: {change.path}
Category: {change.category.value}
Lines Added: {change.lines_added}
Lines Removed: {change.lines_removed}
Impact: {change.impact_level}

Diff:
```
{diff_preview}
```

Provide:
1. A brief description of what changed
2. Potential risks or concerns
3. Suggestions for the developer
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            return {
                "change": change.to_dict(),
                "analysis": response,
            }
        except Exception as e:
            return {"error": str(e)}


class FileSynchronizer:
    """File synchronization with conflict resolution.
    
    Uses LLM reasoning to:
    1. Determine optimal sync strategy
    2. Resolve conflicts intelligently
    3. Suggest merge operations
    
    Example:
        >>> sync = FileSynchronizer(llm_client=client)
        >>> state = sync.analyze(source, target)
        >>> sync.execute(state, strategy=MergeStrategy.NEWER_WINS)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        backup_dir: Optional[Path] = None,
    ):
        """Initialize the file synchronizer.
        
        Args:
            llm_client: LLM client for conflict resolution
            backup_dir: Directory for backups
        """
        self._llm_client = llm_client
        self._backup_dir = backup_dir or Path("~/.proxima/sync_backups").expanduser()
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Sync states
        self._states: Dict[str, SyncState] = {}
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate file checksum."""
        hash_func = hashlib.md5()
        
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception:
            return ""
    
    def _scan_directory(
        self,
        path: Path,
        ignore_patterns: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Scan directory and return file checksums.
        
        Args:
            path: Directory path
            ignore_patterns: Patterns to ignore
            
        Returns:
            Dict mapping relative paths to checksums
        """
        ignore_patterns = ignore_patterns or ['.git', '__pycache__', '*.pyc']
        files = {}
        
        for root, dirs, filenames in os.walk(path):
            # Filter ignored directories
            dirs[:] = [
                d for d in dirs
                if not any(
                    d == p or d.endswith(p.lstrip('*'))
                    for p in ignore_patterns
                )
            ]
            
            for filename in filenames:
                # Check ignore patterns
                skip = False
                for pattern in ignore_patterns:
                    if pattern.startswith('*'):
                        if filename.endswith(pattern[1:]):
                            skip = True
                            break
                    elif filename == pattern:
                        skip = True
                        break
                
                if skip:
                    continue
                
                filepath = Path(root) / filename
                rel_path = str(filepath.relative_to(path))
                files[rel_path] = self._calculate_checksum(filepath)
        
        return files
    
    def analyze(
        self,
        source: Path,
        target: Path,
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
    ) -> SyncState:
        """Analyze two directories for synchronization.
        
        Args:
            source: Source directory
            target: Target directory
            direction: Sync direction
            
        Returns:
            Sync state with pending operations
        """
        source = Path(source)
        target = Path(target)
        
        # Create or get existing state
        state_key = f"{source}:{target}"
        state = self._states.get(state_key)
        
        if state is None:
            state = SyncState(source_path=source, target_path=target)
            self._states[state_key] = state
        
        # Scan directories
        state.source_files = self._scan_directory(source)
        state.target_files = self._scan_directory(target)
        
        # Clear pending operations
        state.pending_copies = []
        state.pending_deletes = []
        state.conflicts = []
        
        # Find differences
        source_only = set(state.source_files.keys()) - set(state.target_files.keys())
        target_only = set(state.target_files.keys()) - set(state.source_files.keys())
        common = set(state.source_files.keys()) & set(state.target_files.keys())
        
        # Handle source-only files
        if direction in (SyncDirection.SOURCE_TO_TARGET, SyncDirection.BIDIRECTIONAL):
            for rel_path in source_only:
                state.pending_copies.append((
                    source / rel_path,
                    target / rel_path,
                ))
        
        # Handle target-only files
        if direction == SyncDirection.TARGET_TO_SOURCE:
            for rel_path in target_only:
                state.pending_copies.append((
                    target / rel_path,
                    source / rel_path,
                ))
        elif direction == SyncDirection.BIDIRECTIONAL:
            for rel_path in target_only:
                state.pending_copies.append((
                    target / rel_path,
                    source / rel_path,
                ))
        
        # Handle common files with different checksums
        for rel_path in common:
            if state.source_files[rel_path] != state.target_files[rel_path]:
                # Potential conflict
                source_file = source / rel_path
                target_file = target / rel_path
                
                source_stat = source_file.stat()
                target_stat = target_file.stat()
                
                conflict = SyncConflict(
                    conflict_id=str(uuid.uuid4())[:8],
                    path=Path(rel_path),
                    source_path=source_file,
                    target_path=target_file,
                    source_modified=datetime.fromtimestamp(source_stat.st_mtime),
                    target_modified=datetime.fromtimestamp(target_stat.st_mtime),
                    source_size=source_stat.st_size,
                    target_size=target_stat.st_size,
                )
                
                state.conflicts.append(conflict)
        
        return state
    
    def resolve_conflict(
        self,
        conflict: SyncConflict,
        strategy: MergeStrategy,
    ) -> Tuple[Path, Path]:
        """Resolve a sync conflict.
        
        Args:
            conflict: The conflict to resolve
            strategy: Resolution strategy
            
        Returns:
            Tuple of (source_path, target_path) for copy operation
        """
        if strategy == MergeStrategy.NEWER_WINS:
            if conflict.source_modified > conflict.target_modified:
                return (conflict.source_path, conflict.target_path)
            else:
                return (conflict.target_path, conflict.source_path)
        
        elif strategy == MergeStrategy.LARGER_WINS:
            if conflict.source_size > conflict.target_size:
                return (conflict.source_path, conflict.target_path)
            else:
                return (conflict.target_path, conflict.source_path)
        
        elif strategy == MergeStrategy.SOURCE_WINS:
            return (conflict.source_path, conflict.target_path)
        
        elif strategy == MergeStrategy.TARGET_WINS:
            return (conflict.target_path, conflict.source_path)
        
        else:
            # Default: newer wins
            if conflict.source_modified > conflict.target_modified:
                return (conflict.source_path, conflict.target_path)
            else:
                return (conflict.target_path, conflict.source_path)
    
    def execute(
        self,
        state: SyncState,
        strategy: MergeStrategy = MergeStrategy.NEWER_WINS,
        dry_run: bool = False,
        create_backups: bool = True,
    ) -> Dict[str, Any]:
        """Execute synchronization.
        
        Args:
            state: Sync state to execute
            strategy: Merge strategy for conflicts
            dry_run: If True, don't actually perform operations
            create_backups: Create backups before overwriting
            
        Returns:
            Execution results
        """
        results = {
            "copied": [],
            "deleted": [],
            "conflicts_resolved": [],
            "errors": [],
        }
        
        # Execute pending copies
        for source, target in state.pending_copies:
            if dry_run:
                results["copied"].append({
                    "source": str(source),
                    "target": str(target),
                    "dry_run": True,
                })
                continue
            
            try:
                # Create backup if target exists
                if create_backups and target.exists():
                    self._create_backup(target)
                
                # Create parent directories
                target.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source, target)
                
                results["copied"].append({
                    "source": str(source),
                    "target": str(target),
                })
            except Exception as e:
                results["errors"].append({
                    "operation": "copy",
                    "source": str(source),
                    "target": str(target),
                    "error": str(e),
                })
        
        # Resolve and execute conflicts
        for conflict in state.conflicts:
            try:
                source, target = self.resolve_conflict(conflict, strategy)
                
                if dry_run:
                    results["conflicts_resolved"].append({
                        "conflict_id": conflict.conflict_id,
                        "source": str(source),
                        "target": str(target),
                        "dry_run": True,
                    })
                    continue
                
                # Create backup
                if create_backups and target.exists():
                    self._create_backup(target)
                
                # Copy
                shutil.copy2(source, target)
                
                conflict.resolution = f"Copied from {source}"
                conflict.resolved_at = datetime.now()
                
                results["conflicts_resolved"].append({
                    "conflict_id": conflict.conflict_id,
                    "source": str(source),
                    "target": str(target),
                })
            except Exception as e:
                results["errors"].append({
                    "operation": "resolve_conflict",
                    "conflict_id": conflict.conflict_id,
                    "error": str(e),
                })
        
        # Update state
        if not dry_run:
            state.last_sync = datetime.now()
            state.sync_count += 1
        
        return results
    
    def _create_backup(self, path: Path):
        """Create a backup of a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path.name}.{timestamp}.bak"
        backup_path = self._backup_dir / backup_name
        
        shutil.copy2(path, backup_path)
    
    def verify(self, state: SyncState) -> Dict[str, Any]:
        """Verify synchronization was successful.
        
        Args:
            state: Sync state to verify
            
        Returns:
            Verification results
        """
        # Rescan directories
        source_files = self._scan_directory(state.source_path)
        target_files = self._scan_directory(state.target_path)
        
        # Check for differences
        source_only = set(source_files.keys()) - set(target_files.keys())
        target_only = set(target_files.keys()) - set(source_files.keys())
        
        different = []
        for path in set(source_files.keys()) & set(target_files.keys()):
            if source_files[path] != target_files[path]:
                different.append(path)
        
        return {
            "synchronized": len(source_only) == 0 and len(different) == 0,
            "source_only": list(source_only),
            "target_only": list(target_only),
            "different": different,
            "source_file_count": len(source_files),
            "target_file_count": len(target_files),
        }
    
    def schedule_sync(
        self,
        source: Path,
        target: Path,
        interval_seconds: int = 300,
        strategy: MergeStrategy = MergeStrategy.NEWER_WINS,
    ) -> str:
        """Schedule periodic synchronization.
        
        Args:
            source: Source directory
            target: Target directory
            interval_seconds: Sync interval
            strategy: Merge strategy
            
        Returns:
            Schedule ID
        """
        schedule_id = str(uuid.uuid4())[:8]
        
        def sync_task():
            while True:
                try:
                    state = self.analyze(source, target)
                    self.execute(state, strategy)
                except Exception as e:
                    logger.error(f"Scheduled sync error: {e}")
                
                time.sleep(interval_seconds)
        
        thread = threading.Thread(target=sync_task, daemon=True)
        thread.start()
        
        return schedule_id


class FileMonitorManager:
    """Main file monitoring and synchronization manager.
    
    Integrates all file monitoring components:
    - File system watching
    - Change detection and analysis
    - File synchronization
    
    Example:
        >>> monitor = FileMonitorManager(llm_client=client)
        >>> monitor.start_watching(Path("./src"))
        >>> changes = monitor.get_changes()
        >>> sync_result = monitor.sync(source, target)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        workspace_path: Optional[Path] = None,
    ):
        """Initialize file monitor manager.
        
        Args:
            llm_client: LLM client for intelligent analysis
            workspace_path: Default workspace path
        """
        self._llm_client = llm_client
        self._workspace_path = workspace_path or Path.cwd()
        
        # Initialize components
        self._watcher = FileSystemWatcher(llm_client=llm_client)
        self._change_detector = ChangeDetector(llm_client=llm_client)
        self._synchronizer = FileSynchronizer(llm_client=llm_client)
        
        # Change tracking
        self._recent_changes: List[FileChange] = []
        self._max_changes = 500
        
        # Setup event callback
        self._watcher.add_callback(self._on_file_event)
    
    def _on_file_event(self, event: FileEvent):
        """Handle file events from watcher."""
        # Get cached content for diff
        old_content = self._change_detector.get_cached_content(event.path)
        new_content = None
        
        if event.event_type in (EventType.CREATED, EventType.MODIFIED):
            if event.path.exists() and event.path.is_file():
                try:
                    new_content = event.path.read_text(errors='ignore')
                    # Cache new content
                    self._change_detector.cache_content(event.path, new_content)
                except Exception:
                    pass
        
        # Analyze change
        change = self._change_detector.analyze_change(
            old_content, new_content, event.path
        )
        
        # Store change
        self._recent_changes.append(change)
        if len(self._recent_changes) > self._max_changes:
            self._recent_changes = self._recent_changes[-self._max_changes:]
    
    # Watching
    def start_watching(
        self,
        path: Optional[Path] = None,
        recursive: bool = True,
        patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ) -> bool:
        """Start watching a path for changes.
        
        Args:
            path: Path to watch
            recursive: Watch subdirectories
            patterns: Patterns to include
            ignore_patterns: Patterns to ignore
            
        Returns:
            True if started successfully
        """
        path = path or self._workspace_path
        
        if patterns:
            self._watcher._patterns = patterns
        if ignore_patterns:
            self._watcher._ignore_patterns = ignore_patterns
        
        return self._watcher.start(path, recursive)
    
    def stop_watching(self):
        """Stop watching all paths."""
        self._watcher.stop()
    
    def get_events(
        self,
        limit: int = 50,
        event_types: Optional[List[EventType]] = None,
    ) -> List[FileEvent]:
        """Get recent file events.
        
        Args:
            limit: Maximum events
            event_types: Filter by types
            
        Returns:
            List of file events
        """
        return self._watcher.get_recent_events(limit, event_types)
    
    # Change Detection
    def get_changes(
        self,
        limit: int = 50,
        categories: Optional[List[ChangeCategory]] = None,
    ) -> List[FileChange]:
        """Get recent file changes.
        
        Args:
            limit: Maximum changes
            categories: Filter by categories
            
        Returns:
            List of file changes
        """
        changes = self._recent_changes[-limit:]
        
        if categories:
            changes = [c for c in changes if c.category in categories]
        
        return changes
    
    def analyze_change(
        self,
        old_content: Optional[str],
        new_content: Optional[str],
        path: Path,
    ) -> FileChange:
        """Analyze a file change manually.
        
        Args:
            old_content: Previous content
            new_content: New content
            path: File path
            
        Returns:
            Analyzed file change
        """
        return self._change_detector.analyze_change(old_content, new_content, path)
    
    def generate_diff(
        self,
        old_content: str,
        new_content: str,
    ) -> List[str]:
        """Generate diff between two contents.
        
        Args:
            old_content: Previous content
            new_content: New content
            
        Returns:
            Diff lines
        """
        return self._change_detector._generate_diff(old_content, new_content)
    
    # Synchronization
    def analyze_sync(
        self,
        source: Path,
        target: Path,
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
    ) -> SyncState:
        """Analyze directories for synchronization.
        
        Args:
            source: Source directory
            target: Target directory
            direction: Sync direction
            
        Returns:
            Sync state
        """
        return self._synchronizer.analyze(source, target, direction)
    
    def execute_sync(
        self,
        state: SyncState,
        strategy: MergeStrategy = MergeStrategy.NEWER_WINS,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute synchronization.
        
        Args:
            state: Sync state
            strategy: Merge strategy
            dry_run: Simulate only
            
        Returns:
            Execution results
        """
        return self._synchronizer.execute(state, strategy, dry_run)
    
    def sync(
        self,
        source: Path,
        target: Path,
        strategy: MergeStrategy = MergeStrategy.NEWER_WINS,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Analyze and synchronize in one step.
        
        Args:
            source: Source directory
            target: Target directory
            strategy: Merge strategy
            dry_run: Simulate only
            
        Returns:
            Sync results
        """
        state = self.analyze_sync(source, target)
        return self.execute_sync(state, strategy, dry_run)
    
    def verify_sync(self, state: SyncState) -> Dict[str, Any]:
        """Verify synchronization was successful.
        
        Args:
            state: Sync state to verify
            
        Returns:
            Verification results
        """
        return self._synchronizer.verify(state)
    
    async def analyze_changes_with_llm(
        self,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Analyze recent changes using LLM.
        
        Args:
            limit: Number of changes to analyze
            
        Returns:
            LLM analysis results
        """
        if not self._llm_client:
            return {"error": "LLM client not available"}
        
        changes = self.get_changes(limit)
        
        if not changes:
            return {"message": "No recent changes to analyze"}
        
        # Build summary of changes
        change_summary = []
        for change in changes:
            change_summary.append(
                f"- {change.path.name}: {change.category.value} "
                f"(+{change.lines_added}, -{change.lines_removed})"
            )
        
        prompt = f"""Analyze these recent file changes:

{chr(10).join(change_summary)}

Provide:
1. A summary of the overall development activity
2. Any patterns or concerns you notice
3. Suggestions for the developer
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            return {
                "change_count": len(changes),
                "analysis": response,
            }
        except Exception as e:
            return {"error": str(e)}


# Module-level instances
_global_file_monitor: Optional[FileMonitorManager] = None


def get_file_monitor_manager(
    llm_client: Optional[Any] = None,
    workspace_path: Optional[Path] = None,
) -> FileMonitorManager:
    """Get the global file monitor manager.
    
    Args:
        llm_client: Optional LLM client
        workspace_path: Optional workspace path
        
    Returns:
        FileMonitorManager instance
    """
    global _global_file_monitor
    if _global_file_monitor is None:
        _global_file_monitor = FileMonitorManager(
            llm_client=llm_client,
            workspace_path=workspace_path,
        )
    return _global_file_monitor
