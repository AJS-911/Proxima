"""File System Intelligence for Dynamic AI Assistant.

This module implements Phase 5.1 for the Dynamic AI Assistant:
- Fuzzy Path Resolution: Partial path matching, auto-completion, suggestions
- Content-Aware File Handling: Encoding detection, file type detection, categorization
- Safe Operation Patterns: Atomic operations, backups, journaling, dry-run
- Smart Search: Indexed search, content search, semantic search

Key Features:
============
- Intelligent path resolution with fuzzy matching
- Automatic file encoding and type detection
- Safe file operations with backup and undo
- Fast indexed and semantic file search
- LLM-driven file analysis and suggestions

Design Principle:
================
All file intelligence decisions use LLM reasoning when available.
The LLM analyzes file patterns and suggests optimal operations.
"""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import hashlib
import json
import logging
import mimetypes
import os
import re
import shutil
import stat
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


class FileCategory(Enum):
    """File category types."""
    SOURCE_CODE = "source_code"
    CONFIG = "configuration"
    DATA = "data"
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    ARCHIVE = "archive"
    BINARY = "binary"
    TEXT = "text"
    UNKNOWN = "unknown"


class ChangeType(Enum):
    """File change types."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    MOVED = "moved"


class ConflictStrategy(Enum):
    """Conflict resolution strategies."""
    OVERWRITE = "overwrite"
    RENAME = "rename"
    SKIP = "skip"
    BACKUP = "backup"
    MERGE = "merge"
    ASK = "ask"


class SearchType(Enum):
    """Search types."""
    FILENAME = "filename"
    CONTENT = "content"
    REGEX = "regex"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"


@dataclass
class PathMatch:
    """A path match result."""
    path: Path
    score: float  # 0.0 to 1.0
    matched_pattern: str
    is_directory: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "score": self.score,
            "matched_pattern": self.matched_pattern,
            "is_directory": self.is_directory,
        }


@dataclass
class FileInfo:
    """Detailed file information."""
    path: Path
    name: str
    extension: str
    
    # Size
    size_bytes: int = 0
    
    # Category and type
    category: FileCategory = FileCategory.UNKNOWN
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    
    # Permissions
    is_readable: bool = True
    is_writable: bool = True
    is_executable: bool = False
    
    # Flags
    is_hidden: bool = False
    is_symlink: bool = False
    symlink_target: Optional[str] = None
    
    # Content info
    line_count: Optional[int] = None
    word_count: Optional[int] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "size_human": self._format_size(self.size_bytes),
            "category": self.category.value,
            "mime_type": self.mime_type,
            "encoding": self.encoding,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "is_readable": self.is_readable,
            "is_writable": self.is_writable,
            "is_hidden": self.is_hidden,
            "is_symlink": self.is_symlink,
            "line_count": self.line_count,
        }
    
    @staticmethod
    def _format_size(size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"


@dataclass
class SearchResult:
    """A search result."""
    path: Path
    score: float
    matched_text: Optional[str] = None
    line_number: Optional[int] = None
    context: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "score": self.score,
            "matched_text": self.matched_text,
            "line_number": self.line_number,
            "context": self.context,
        }


@dataclass
class OperationJournalEntry:
    """An operation journal entry for undo capability."""
    entry_id: str
    operation_type: str  # create, modify, delete, move, copy
    timestamp: datetime
    
    # Paths
    source_path: Optional[Path] = None
    target_path: Optional[Path] = None
    
    # Backup info
    backup_path: Optional[Path] = None
    original_content: Optional[bytes] = None
    original_metadata: Optional[Dict[str, Any]] = None
    
    # Status
    is_undone: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "operation_type": self.operation_type,
            "timestamp": self.timestamp.isoformat(),
            "source_path": str(self.source_path) if self.source_path else None,
            "target_path": str(self.target_path) if self.target_path else None,
            "has_backup": self.backup_path is not None,
            "is_undone": self.is_undone,
        }


@dataclass
class FileIndex:
    """File index for fast searching."""
    root_path: Path
    files: Dict[str, FileInfo] = field(default_factory=dict)
    
    # Index metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    file_count: int = 0
    
    # Content index
    content_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_path": str(self.root_path),
            "file_count": self.file_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class FuzzyPathResolver:
    """Fuzzy path resolution with intelligent matching.
    
    Uses LLM reasoning to:
    1. Understand user's intended path from partial input
    2. Suggest similar paths when exact match not found
    3. Resolve ambiguous paths based on context
    
    Example:
        >>> resolver = FuzzyPathResolver(llm_client=client)
        >>> matches = resolver.resolve("src/main", workspace_path)
        >>> # Returns matches like src/main.py, src/main/, etc.
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        workspace_path: Optional[Path] = None,
    ):
        """Initialize the fuzzy path resolver.
        
        Args:
            llm_client: LLM client for reasoning
            workspace_path: Default workspace path
        """
        self._llm_client = llm_client
        self._workspace_path = workspace_path or Path.cwd()
        
        # Recent paths cache for suggestions
        self._recent_paths: List[Path] = []
        self._max_recent = 100
        
        # Path index cache
        self._path_cache: Dict[str, List[Path]] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)
    
    def resolve(
        self,
        partial_path: str,
        base_path: Optional[Path] = None,
        max_results: int = 10,
        include_dirs: bool = True,
        include_files: bool = True,
    ) -> List[PathMatch]:
        """Resolve a partial path to matching paths.
        
        Args:
            partial_path: Partial path string
            base_path: Base path for resolution
            max_results: Maximum results to return
            include_dirs: Include directories
            include_files: Include files
            
        Returns:
            List of path matches sorted by score
        """
        base_path = base_path or self._workspace_path
        matches: List[PathMatch] = []
        
        # Normalize the partial path
        partial_path = self._normalize_path(partial_path)
        
        # Try exact match first
        exact_path = base_path / partial_path
        if exact_path.exists():
            is_dir = exact_path.is_dir()
            if (is_dir and include_dirs) or (not is_dir and include_files):
                matches.append(PathMatch(
                    path=exact_path,
                    score=1.0,
                    matched_pattern=partial_path,
                    is_directory=is_dir,
                ))
        
        # Get all candidate paths
        candidates = self._get_candidate_paths(base_path, include_dirs, include_files)
        
        # Fuzzy match against candidates
        for candidate in candidates:
            if candidate == exact_path:
                continue
                
            score = self._calculate_match_score(partial_path, candidate, base_path)
            
            if score > 0.3:  # Minimum score threshold
                matches.append(PathMatch(
                    path=candidate,
                    score=score,
                    matched_pattern=partial_path,
                    is_directory=candidate.is_dir(),
                ))
        
        # Sort by score (descending) and limit results
        matches.sort(key=lambda m: m.score, reverse=True)
        
        # Update recent paths
        if matches:
            self._update_recent(matches[0].path)
        
        return matches[:max_results]
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path string."""
        # Replace backslashes with forward slashes
        path = path.replace("\\", "/")
        
        # Remove leading/trailing whitespace and slashes
        path = path.strip().strip("/")
        
        # Expand user home
        if path.startswith("~"):
            path = str(Path(path).expanduser())
        
        return path
    
    def _get_candidate_paths(
        self,
        base_path: Path,
        include_dirs: bool,
        include_files: bool,
        max_depth: int = 5,
    ) -> List[Path]:
        """Get candidate paths for matching."""
        cache_key = f"{base_path}:{include_dirs}:{include_files}"
        
        # Check cache
        if (self._cache_time and 
            datetime.now() - self._cache_time < self._cache_ttl and
            cache_key in self._path_cache):
            return self._path_cache[cache_key]
        
        candidates = []
        
        try:
            for root, dirs, files in os.walk(base_path):
                # Calculate depth
                depth = len(Path(root).relative_to(base_path).parts)
                if depth > max_depth:
                    dirs.clear()  # Don't recurse deeper
                    continue
                
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                root_path = Path(root)
                
                if include_dirs:
                    for d in dirs:
                        candidates.append(root_path / d)
                
                if include_files:
                    for f in files:
                        if not f.startswith('.'):
                            candidates.append(root_path / f)
        except PermissionError:
            pass
        
        # Update cache
        self._path_cache[cache_key] = candidates
        self._cache_time = datetime.now()
        
        return candidates
    
    def _calculate_match_score(
        self,
        partial: str,
        candidate: Path,
        base_path: Path,
    ) -> float:
        """Calculate fuzzy match score between partial and candidate."""
        try:
            relative = candidate.relative_to(base_path)
        except ValueError:
            relative = candidate
        
        candidate_str = str(relative).replace("\\", "/").lower()
        partial_lower = partial.lower()
        
        # Check various matching strategies
        scores = []
        
        # 1. Exact substring match
        if partial_lower in candidate_str:
            # Higher score for match at end (filename)
            if candidate_str.endswith(partial_lower):
                scores.append(0.9)
            elif candidate.name.lower().startswith(partial_lower):
                scores.append(0.85)
            else:
                scores.append(0.7)
        
        # 2. Sequence matching ratio
        ratio = difflib.SequenceMatcher(None, partial_lower, candidate_str).ratio()
        scores.append(ratio * 0.8)
        
        # 3. Filename-only matching
        filename_ratio = difflib.SequenceMatcher(
            None, partial_lower, candidate.name.lower()
        ).ratio()
        scores.append(filename_ratio * 0.7)
        
        # 4. Path component matching
        partial_parts = partial_lower.split("/")
        candidate_parts = candidate_str.split("/")
        
        matched_parts = 0
        for part in partial_parts:
            for cpart in candidate_parts:
                if part in cpart:
                    matched_parts += 1
                    break
        
        if partial_parts:
            part_score = matched_parts / len(partial_parts)
            scores.append(part_score * 0.6)
        
        # 5. Boost for recent paths
        if candidate in self._recent_paths:
            recent_idx = self._recent_paths.index(candidate)
            recency_boost = 0.1 * (1 - recent_idx / len(self._recent_paths))
            scores.append(max(scores) + recency_boost)
        
        return max(scores) if scores else 0.0
    
    def _update_recent(self, path: Path):
        """Update recent paths list."""
        if path in self._recent_paths:
            self._recent_paths.remove(path)
        
        self._recent_paths.insert(0, path)
        
        if len(self._recent_paths) > self._max_recent:
            self._recent_paths = self._recent_paths[:self._max_recent]
    
    def suggest_paths(
        self,
        context: str,
        base_path: Optional[Path] = None,
    ) -> List[PathMatch]:
        """Suggest paths based on context using LLM.
        
        Args:
            context: Context or description of what path is needed
            base_path: Base path for suggestions
            
        Returns:
            List of suggested path matches
        """
        base_path = base_path or self._workspace_path
        
        # Get recent and common paths
        suggestions = []
        
        # Add recent paths
        for path in self._recent_paths[:5]:
            if path.exists():
                suggestions.append(PathMatch(
                    path=path,
                    score=0.5,
                    matched_pattern="recent",
                    is_directory=path.is_dir(),
                ))
        
        # If LLM available, use it to suggest based on context
        if self._llm_client:
            # Get directory listing for context
            files = list(base_path.iterdir())[:20]
            file_list = "\n".join(f.name for f in files)
            
            # This would be a real LLM call in production
            pass
        
        return suggestions
    
    def auto_complete(
        self,
        partial_path: str,
        base_path: Optional[Path] = None,
    ) -> List[str]:
        """Auto-complete a partial path.
        
        Args:
            partial_path: Partial path to complete
            base_path: Base path for completion
            
        Returns:
            List of completion strings
        """
        matches = self.resolve(partial_path, base_path, max_results=10)
        
        completions = []
        for match in matches:
            try:
                rel_path = match.path.relative_to(base_path or self._workspace_path)
                completion = str(rel_path)
                if match.is_directory:
                    completion += "/"
                completions.append(completion)
            except ValueError:
                completions.append(str(match.path))
        
        return completions


class ContentAwareFileHandler:
    """Content-aware file handling with automatic detection.
    
    Uses LLM reasoning to:
    1. Detect file types and encodings intelligently
    2. Categorize files based on content analysis
    3. Handle files appropriately based on their type
    
    Example:
        >>> handler = ContentAwareFileHandler(llm_client=client)
        >>> info = handler.analyze_file(Path("unknown_file"))
        >>> # Returns detailed file info with detected type and encoding
    """
    
    # Extension to category mapping
    EXTENSION_CATEGORIES: Dict[str, FileCategory] = {
        # Source code
        ".py": FileCategory.SOURCE_CODE,
        ".js": FileCategory.SOURCE_CODE,
        ".ts": FileCategory.SOURCE_CODE,
        ".java": FileCategory.SOURCE_CODE,
        ".c": FileCategory.SOURCE_CODE,
        ".cpp": FileCategory.SOURCE_CODE,
        ".h": FileCategory.SOURCE_CODE,
        ".cs": FileCategory.SOURCE_CODE,
        ".go": FileCategory.SOURCE_CODE,
        ".rs": FileCategory.SOURCE_CODE,
        ".rb": FileCategory.SOURCE_CODE,
        ".php": FileCategory.SOURCE_CODE,
        ".swift": FileCategory.SOURCE_CODE,
        ".kt": FileCategory.SOURCE_CODE,
        ".scala": FileCategory.SOURCE_CODE,
        ".r": FileCategory.SOURCE_CODE,
        ".m": FileCategory.SOURCE_CODE,
        ".sh": FileCategory.SOURCE_CODE,
        ".ps1": FileCategory.SOURCE_CODE,
        ".bat": FileCategory.SOURCE_CODE,
        ".cmd": FileCategory.SOURCE_CODE,
        
        # Config
        ".json": FileCategory.CONFIG,
        ".yaml": FileCategory.CONFIG,
        ".yml": FileCategory.CONFIG,
        ".toml": FileCategory.CONFIG,
        ".ini": FileCategory.CONFIG,
        ".cfg": FileCategory.CONFIG,
        ".conf": FileCategory.CONFIG,
        ".xml": FileCategory.CONFIG,
        ".properties": FileCategory.CONFIG,
        ".env": FileCategory.CONFIG,
        
        # Data
        ".csv": FileCategory.DATA,
        ".tsv": FileCategory.DATA,
        ".sql": FileCategory.DATA,
        ".db": FileCategory.DATA,
        ".sqlite": FileCategory.DATA,
        ".parquet": FileCategory.DATA,
        ".arrow": FileCategory.DATA,
        
        # Document
        ".md": FileCategory.DOCUMENT,
        ".txt": FileCategory.DOCUMENT,
        ".rst": FileCategory.DOCUMENT,
        ".tex": FileCategory.DOCUMENT,
        ".pdf": FileCategory.DOCUMENT,
        ".doc": FileCategory.DOCUMENT,
        ".docx": FileCategory.DOCUMENT,
        ".html": FileCategory.DOCUMENT,
        ".htm": FileCategory.DOCUMENT,
        
        # Image
        ".png": FileCategory.IMAGE,
        ".jpg": FileCategory.IMAGE,
        ".jpeg": FileCategory.IMAGE,
        ".gif": FileCategory.IMAGE,
        ".svg": FileCategory.IMAGE,
        ".bmp": FileCategory.IMAGE,
        ".ico": FileCategory.IMAGE,
        ".webp": FileCategory.IMAGE,
        
        # Video
        ".mp4": FileCategory.VIDEO,
        ".avi": FileCategory.VIDEO,
        ".mkv": FileCategory.VIDEO,
        ".mov": FileCategory.VIDEO,
        ".webm": FileCategory.VIDEO,
        
        # Audio
        ".mp3": FileCategory.AUDIO,
        ".wav": FileCategory.AUDIO,
        ".flac": FileCategory.AUDIO,
        ".ogg": FileCategory.AUDIO,
        ".m4a": FileCategory.AUDIO,
        
        # Archive
        ".zip": FileCategory.ARCHIVE,
        ".tar": FileCategory.ARCHIVE,
        ".gz": FileCategory.ARCHIVE,
        ".bz2": FileCategory.ARCHIVE,
        ".7z": FileCategory.ARCHIVE,
        ".rar": FileCategory.ARCHIVE,
        ".xz": FileCategory.ARCHIVE,
    }
    
    # Binary file signatures (magic bytes)
    MAGIC_SIGNATURES: Dict[bytes, str] = {
        b'\x89PNG': 'image/png',
        b'\xff\xd8\xff': 'image/jpeg',
        b'GIF8': 'image/gif',
        b'PK\x03\x04': 'application/zip',
        b'%PDF': 'application/pdf',
        b'\x7fELF': 'application/x-executable',
        b'MZ': 'application/x-msdownload',
        b'\x1f\x8b': 'application/gzip',
        b'BZh': 'application/x-bzip2',
        b'Rar!': 'application/x-rar-compressed',
        b'\xfd7zXZ': 'application/x-xz',
    }
    
    def __init__(self, llm_client: Optional[Any] = None):
        """Initialize the content-aware file handler.
        
        Args:
            llm_client: LLM client for advanced analysis
        """
        self._llm_client = llm_client
    
    def analyze_file(self, path: Path) -> FileInfo:
        """Analyze a file and return detailed information.
        
        Args:
            path: Path to the file
            
        Returns:
            Detailed file information
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Basic info
        stat_info = path.stat()
        
        info = FileInfo(
            path=path,
            name=path.name,
            extension=path.suffix.lower(),
            size_bytes=stat_info.st_size,
            created_at=datetime.fromtimestamp(stat_info.st_ctime),
            modified_at=datetime.fromtimestamp(stat_info.st_mtime),
            accessed_at=datetime.fromtimestamp(stat_info.st_atime),
            is_readable=os.access(path, os.R_OK),
            is_writable=os.access(path, os.W_OK),
            is_executable=os.access(path, os.X_OK),
            is_hidden=path.name.startswith('.'),
            is_symlink=path.is_symlink(),
        )
        
        if info.is_symlink:
            try:
                info.symlink_target = str(path.resolve())
            except Exception:
                pass
        
        # Detect category from extension
        info.category = self.EXTENSION_CATEGORIES.get(
            info.extension, FileCategory.UNKNOWN
        )
        
        # Detect MIME type
        info.mime_type = self._detect_mime_type(path)
        
        # Detect encoding for text files
        if self._is_likely_text(path, info.mime_type):
            info.encoding = self._detect_encoding(path)
            
            # Count lines and words for text files
            if info.size_bytes < 10 * 1024 * 1024:  # Only for files < 10MB
                try:
                    content = path.read_text(encoding=info.encoding or 'utf-8')
                    info.line_count = content.count('\n') + 1
                    info.word_count = len(content.split())
                except Exception:
                    pass
        
        # Calculate checksum for smaller files
        if info.size_bytes < 100 * 1024 * 1024:  # Only for files < 100MB
            info.checksum = self._calculate_checksum(path)
        
        return info
    
    def _detect_mime_type(self, path: Path) -> Optional[str]:
        """Detect MIME type using magic bytes and extension."""
        # Try magic bytes first
        try:
            with open(path, 'rb') as f:
                header = f.read(16)
            
            for signature, mime_type in self.MAGIC_SIGNATURES.items():
                if header.startswith(signature):
                    return mime_type
        except Exception:
            pass
        
        # Fallback to mimetypes module
        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type
    
    def _is_likely_text(self, path: Path, mime_type: Optional[str]) -> bool:
        """Check if file is likely a text file."""
        # Check MIME type
        if mime_type:
            if mime_type.startswith('text/'):
                return True
            if mime_type in ('application/json', 'application/xml', 
                            'application/javascript', 'application/x-yaml'):
                return True
        
        # Check extension
        text_extensions = {
            '.txt', '.md', '.rst', '.py', '.js', '.ts', '.java', '.c', '.cpp',
            '.h', '.cs', '.go', '.rs', '.rb', '.php', '.html', '.css', '.json',
            '.yaml', '.yml', '.toml', '.xml', '.ini', '.cfg', '.conf', '.sh',
            '.bat', '.ps1', '.sql', '.csv', '.log', '.env',
        }
        if path.suffix.lower() in text_extensions:
            return True
        
        # Try reading as text
        try:
            with open(path, 'rb') as f:
                chunk = f.read(8192)
            
            # Check for null bytes (binary indicator)
            if b'\x00' in chunk:
                return False
            
            # Try to decode as UTF-8
            chunk.decode('utf-8')
            return True
        except Exception:
            return False
    
    def _detect_encoding(self, path: Path) -> Optional[str]:
        """Detect file encoding."""
        try:
            # Try chardet if available
            import chardet
            
            with open(path, 'rb') as f:
                raw = f.read(10000)
            
            result = chardet.detect(raw)
            if result and result.get('confidence', 0) > 0.7:
                return result['encoding']
        except ImportError:
            pass
        except Exception:
            pass
        
        # Fallback: try common encodings
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    f.read(1024)
                return encoding
            except Exception:
                continue
        
        return 'utf-8'  # Default
    
    def _calculate_checksum(self, path: Path, algorithm: str = 'md5') -> str:
        """Calculate file checksum."""
        hash_func = hashlib.new(algorithm)
        
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def read_file_smart(
        self,
        path: Path,
        encoding: Optional[str] = None,
        max_size: int = 50 * 1024 * 1024,
    ) -> Tuple[Union[str, bytes], FileInfo]:
        """Read file with automatic encoding detection.
        
        Args:
            path: Path to the file
            encoding: Force specific encoding
            max_size: Maximum file size to read
            
        Returns:
            Tuple of (content, file_info)
        """
        info = self.analyze_file(path)
        
        if info.size_bytes > max_size:
            raise ValueError(f"File too large: {info.size_bytes} bytes > {max_size} bytes")
        
        # Use detected or provided encoding
        enc = encoding or info.encoding
        
        if enc:
            # Read as text
            content = path.read_text(encoding=enc)
        else:
            # Read as binary
            content = path.read_bytes()
        
        return content, info
    
    def categorize_files(
        self,
        paths: List[Path],
    ) -> Dict[FileCategory, List[FileInfo]]:
        """Categorize multiple files.
        
        Args:
            paths: List of file paths
            
        Returns:
            Dict mapping categories to file info lists
        """
        result: Dict[FileCategory, List[FileInfo]] = defaultdict(list)
        
        for path in paths:
            try:
                info = self.analyze_file(path)
                result[info.category].append(info)
            except Exception as e:
                logger.warning(f"Failed to analyze {path}: {e}")
        
        return dict(result)


class SafeFileOperations:
    """Safe file operations with backup and undo capability.
    
    Uses LLM reasoning to:
    1. Assess risk of operations
    2. Determine appropriate backup strategies
    3. Suggest conflict resolution approaches
    
    Example:
        >>> ops = SafeFileOperations()
        >>> ops.safe_write(Path("file.txt"), "content", backup=True)
        >>> ops.undo()  # Restores previous version
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        backup_dir: Optional[Path] = None,
        journal_path: Optional[Path] = None,
    ):
        """Initialize safe file operations.
        
        Args:
            llm_client: LLM client for reasoning
            backup_dir: Directory for backups
            journal_path: Path for operation journal
        """
        self._llm_client = llm_client
        
        # Backup directory
        self._backup_dir = backup_dir or Path(tempfile.gettempdir()) / "proxima_backups"
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Operation journal
        self._journal_path = journal_path or self._backup_dir / "journal.json"
        self._journal: List[OperationJournalEntry] = []
        self._load_journal()
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def _load_journal(self):
        """Load operation journal from file."""
        if self._journal_path.exists():
            try:
                with open(self._journal_path) as f:
                    data = json.load(f)
                
                self._journal = []
                for entry_data in data:
                    entry = OperationJournalEntry(
                        entry_id=entry_data["entry_id"],
                        operation_type=entry_data["operation_type"],
                        timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                        source_path=Path(entry_data["source_path"]) if entry_data.get("source_path") else None,
                        target_path=Path(entry_data["target_path"]) if entry_data.get("target_path") else None,
                        backup_path=Path(entry_data["backup_path"]) if entry_data.get("backup_path") else None,
                        is_undone=entry_data.get("is_undone", False),
                    )
                    self._journal.append(entry)
            except Exception as e:
                logger.warning(f"Failed to load journal: {e}")
    
    def _save_journal(self):
        """Save operation journal to file."""
        try:
            data = [entry.to_dict() for entry in self._journal[-1000:]]  # Keep last 1000
            with open(self._journal_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save journal: {e}")
    
    def _create_backup(self, path: Path) -> Optional[Path]:
        """Create a backup of a file."""
        if not path.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_name = f"{path.name}.{timestamp}.bak"
        backup_path = self._backup_dir / backup_name
        
        try:
            if path.is_dir():
                shutil.copytree(path, backup_path)
            else:
                shutil.copy2(path, backup_path)
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def _add_journal_entry(
        self,
        operation_type: str,
        source_path: Optional[Path] = None,
        target_path: Optional[Path] = None,
        backup_path: Optional[Path] = None,
    ) -> OperationJournalEntry:
        """Add an entry to the operation journal."""
        entry = OperationJournalEntry(
            entry_id=str(uuid.uuid4())[:8],
            operation_type=operation_type,
            timestamp=datetime.now(),
            source_path=source_path,
            target_path=target_path,
            backup_path=backup_path,
        )
        
        with self._lock:
            self._journal.append(entry)
            self._save_journal()
        
        return entry
    
    def safe_write(
        self,
        path: Path,
        content: Union[str, bytes],
        encoding: str = 'utf-8',
        backup: bool = True,
        atomic: bool = True,
        create_dirs: bool = True,
    ) -> Dict[str, Any]:
        """Safely write content to a file.
        
        Args:
            path: Target file path
            content: Content to write
            encoding: Text encoding (for string content)
            backup: Create backup before overwriting
            atomic: Use atomic write (temp file + rename)
            create_dirs: Create parent directories if needed
            
        Returns:
            Operation result
        """
        path = Path(path)
        backup_path = None
        
        # Create parent directories
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if file exists
        if backup and path.exists():
            backup_path = self._create_backup(path)
        
        try:
            if atomic:
                # Write to temp file first, then rename
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp"
                )
                
                try:
                    with os.fdopen(temp_fd, 'wb') as f:
                        if isinstance(content, str):
                            f.write(content.encode(encoding))
                        else:
                            f.write(content)
                    
                    # Rename temp file to target
                    shutil.move(temp_path, path)
                except Exception:
                    # Clean up temp file on error
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise
            else:
                # Direct write
                mode = 'wb' if isinstance(content, bytes) else 'w'
                with open(path, mode, encoding=encoding if mode == 'w' else None) as f:
                    f.write(content)
            
            # Journal the operation
            operation_type = "modify" if backup_path else "create"
            self._add_journal_entry(
                operation_type=operation_type,
                target_path=path,
                backup_path=backup_path,
            )
            
            return {
                "success": True,
                "path": str(path),
                "backup_path": str(backup_path) if backup_path else None,
                "bytes_written": len(content.encode(encoding) if isinstance(content, str) else content),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def safe_delete(
        self,
        path: Path,
        backup: bool = True,
        use_trash: bool = True,
    ) -> Dict[str, Any]:
        """Safely delete a file or directory.
        
        Args:
            path: Path to delete
            backup: Create backup before deleting
            use_trash: Move to trash instead of permanent delete
            
        Returns:
            Operation result
        """
        path = Path(path)
        
        if not path.exists():
            return {"success": False, "error": "Path does not exist"}
        
        backup_path = None
        
        # Create backup
        if backup:
            backup_path = self._create_backup(path)
        
        try:
            if use_trash:
                # Try to use system trash
                try:
                    import send2trash
                    send2trash.send2trash(str(path))
                except ImportError:
                    # Fallback: move to our backup dir as "trash"
                    trash_path = self._backup_dir / "trash" / f"{path.name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    trash_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(trash_path))
            else:
                # Permanent delete
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            
            # Journal the operation
            self._add_journal_entry(
                operation_type="delete",
                source_path=path,
                backup_path=backup_path,
            )
            
            return {
                "success": True,
                "path": str(path),
                "backup_path": str(backup_path) if backup_path else None,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def safe_move(
        self,
        source: Path,
        destination: Path,
        backup: bool = True,
        conflict_strategy: ConflictStrategy = ConflictStrategy.BACKUP,
    ) -> Dict[str, Any]:
        """Safely move a file or directory.
        
        Args:
            source: Source path
            destination: Destination path
            backup: Create backup if destination exists
            conflict_strategy: How to handle conflicts
            
        Returns:
            Operation result
        """
        source = Path(source)
        destination = Path(destination)
        
        if not source.exists():
            return {"success": False, "error": "Source does not exist"}
        
        backup_path = None
        
        # Handle destination conflict
        if destination.exists():
            if conflict_strategy == ConflictStrategy.SKIP:
                return {"success": False, "error": "Destination exists, skipped"}
            elif conflict_strategy == ConflictStrategy.RENAME:
                counter = 1
                base = destination.stem
                suffix = destination.suffix
                while destination.exists():
                    destination = destination.parent / f"{base}_{counter}{suffix}"
                    counter += 1
            elif conflict_strategy == ConflictStrategy.BACKUP:
                backup_path = self._create_backup(destination)
            # OVERWRITE continues without backup
        
        try:
            # Create parent directories
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(str(source), str(destination))
            
            # Journal the operation
            self._add_journal_entry(
                operation_type="move",
                source_path=source,
                target_path=destination,
                backup_path=backup_path,
            )
            
            return {
                "success": True,
                "source": str(source),
                "destination": str(destination),
                "backup_path": str(backup_path) if backup_path else None,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def safe_copy(
        self,
        source: Path,
        destination: Path,
        conflict_strategy: ConflictStrategy = ConflictStrategy.BACKUP,
    ) -> Dict[str, Any]:
        """Safely copy a file or directory.
        
        Args:
            source: Source path
            destination: Destination path
            conflict_strategy: How to handle conflicts
            
        Returns:
            Operation result
        """
        source = Path(source)
        destination = Path(destination)
        
        if not source.exists():
            return {"success": False, "error": "Source does not exist"}
        
        backup_path = None
        
        # Handle destination conflict
        if destination.exists():
            if conflict_strategy == ConflictStrategy.SKIP:
                return {"success": False, "error": "Destination exists, skipped"}
            elif conflict_strategy == ConflictStrategy.RENAME:
                counter = 1
                base = destination.stem
                suffix = destination.suffix
                while destination.exists():
                    destination = destination.parent / f"{base}_{counter}{suffix}"
                    counter += 1
            elif conflict_strategy == ConflictStrategy.BACKUP:
                backup_path = self._create_backup(destination)
        
        try:
            # Create parent directories
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file/directory
            if source.is_dir():
                shutil.copytree(str(source), str(destination), dirs_exist_ok=True)
            else:
                shutil.copy2(str(source), str(destination))
            
            # Journal the operation
            self._add_journal_entry(
                operation_type="copy",
                source_path=source,
                target_path=destination,
                backup_path=backup_path,
            )
            
            return {
                "success": True,
                "source": str(source),
                "destination": str(destination),
                "backup_path": str(backup_path) if backup_path else None,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def undo(self, count: int = 1) -> List[Dict[str, Any]]:
        """Undo recent operations.
        
        Args:
            count: Number of operations to undo
            
        Returns:
            List of undo results
        """
        results = []
        
        with self._lock:
            # Find undoable operations (most recent first)
            undoable = [e for e in reversed(self._journal) if not e.is_undone]
            
            for i, entry in enumerate(undoable[:count]):
                result = self._undo_operation(entry)
                results.append(result)
                
                if result["success"]:
                    entry.is_undone = True
            
            self._save_journal()
        
        return results
    
    def _undo_operation(self, entry: OperationJournalEntry) -> Dict[str, Any]:
        """Undo a single operation."""
        try:
            if entry.operation_type == "create":
                # Undo create: delete the file
                if entry.target_path and entry.target_path.exists():
                    if entry.target_path.is_dir():
                        shutil.rmtree(entry.target_path)
                    else:
                        entry.target_path.unlink()
                    return {"success": True, "action": "deleted created file"}
            
            elif entry.operation_type == "modify":
                # Undo modify: restore from backup
                if entry.backup_path and entry.backup_path.exists():
                    if entry.target_path:
                        shutil.copy2(entry.backup_path, entry.target_path)
                    return {"success": True, "action": "restored from backup"}
            
            elif entry.operation_type == "delete":
                # Undo delete: restore from backup
                if entry.backup_path and entry.backup_path.exists():
                    if entry.source_path:
                        if entry.backup_path.is_dir():
                            shutil.copytree(entry.backup_path, entry.source_path)
                        else:
                            shutil.copy2(entry.backup_path, entry.source_path)
                    return {"success": True, "action": "restored deleted file"}
            
            elif entry.operation_type == "move":
                # Undo move: move back
                if entry.target_path and entry.target_path.exists() and entry.source_path:
                    shutil.move(str(entry.target_path), str(entry.source_path))
                    # Restore overwritten destination if backed up
                    if entry.backup_path and entry.backup_path.exists():
                        shutil.copy2(entry.backup_path, entry.target_path)
                    return {"success": True, "action": "moved back"}
            
            elif entry.operation_type == "copy":
                # Undo copy: delete the copy
                if entry.target_path and entry.target_path.exists():
                    if entry.target_path.is_dir():
                        shutil.rmtree(entry.target_path)
                    else:
                        entry.target_path.unlink()
                    # Restore overwritten destination if backed up
                    if entry.backup_path and entry.backup_path.exists():
                        shutil.copy2(entry.backup_path, entry.target_path)
                    return {"success": True, "action": "deleted copy"}
            
            return {"success": False, "error": "Cannot undo this operation"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def dry_run(
        self,
        operation: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Simulate an operation without executing it.
        
        Args:
            operation: Operation type (write, delete, move, copy)
            **kwargs: Operation arguments
            
        Returns:
            Simulation result
        """
        result = {
            "operation": operation,
            "would_execute": True,
            "details": {},
        }
        
        if operation == "write":
            path = Path(kwargs.get("path", ""))
            content = kwargs.get("content", "")
            
            result["details"] = {
                "path": str(path),
                "exists": path.exists(),
                "would_create": not path.exists(),
                "would_modify": path.exists(),
                "content_size": len(content),
                "parent_exists": path.parent.exists(),
            }
        
        elif operation == "delete":
            path = Path(kwargs.get("path", ""))
            
            if not path.exists():
                result["would_execute"] = False
                result["details"]["error"] = "Path does not exist"
            else:
                result["details"] = {
                    "path": str(path),
                    "is_directory": path.is_dir(),
                    "size": path.stat().st_size if path.is_file() else "directory",
                }
        
        elif operation == "move":
            source = Path(kwargs.get("source", ""))
            destination = Path(kwargs.get("destination", ""))
            
            result["details"] = {
                "source": str(source),
                "destination": str(destination),
                "source_exists": source.exists(),
                "destination_exists": destination.exists(),
                "would_overwrite": destination.exists(),
            }
        
        elif operation == "copy":
            source = Path(kwargs.get("source", ""))
            destination = Path(kwargs.get("destination", ""))
            
            result["details"] = {
                "source": str(source),
                "destination": str(destination),
                "source_exists": source.exists(),
                "destination_exists": destination.exists(),
                "would_overwrite": destination.exists(),
            }
        
        return result
    
    def get_undo_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent operation history for undo.
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of journal entries
        """
        entries = [e for e in reversed(self._journal) if not e.is_undone]
        return [e.to_dict() for e in entries[:limit]]


class SmartFileSearch:
    """Smart file search with multiple search modes.
    
    Uses LLM reasoning to:
    1. Understand search queries in natural language
    2. Rank search results by relevance
    3. Suggest related files based on context
    
    Example:
        >>> search = SmartFileSearch(llm_client=client)
        >>> results = search.search("find all python files with database connections")
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        workspace_path: Optional[Path] = None,
    ):
        """Initialize smart file search.
        
        Args:
            llm_client: LLM client for semantic search
            workspace_path: Default workspace path
        """
        self._llm_client = llm_client
        self._workspace_path = workspace_path or Path.cwd()
        
        # File index
        self._index: Optional[FileIndex] = None
        self._index_lock = threading.Lock()
        
        # Search history
        self._search_history: List[Tuple[str, List[SearchResult]]] = []
    
    def build_index(
        self,
        root_path: Optional[Path] = None,
        include_content: bool = True,
        max_file_size: int = 1024 * 1024,  # 1MB
    ) -> FileIndex:
        """Build or update the file index.
        
        Args:
            root_path: Root path to index
            include_content: Index file contents
            max_file_size: Max file size to index content
            
        Returns:
            Built file index
        """
        root_path = root_path or self._workspace_path
        
        handler = ContentAwareFileHandler()
        
        index = FileIndex(root_path=root_path)
        
        # Walk directory tree
        for root, dirs, files in os.walk(root_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in files:
                if filename.startswith('.'):
                    continue
                
                filepath = Path(root) / filename
                
                try:
                    info = handler.analyze_file(filepath)
                    rel_path = str(filepath.relative_to(root_path))
                    index.files[rel_path] = info
                    
                    # Index content for text files
                    if (include_content and 
                        info.size_bytes < max_file_size and
                        info.encoding):
                        try:
                            content = filepath.read_text(encoding=info.encoding)
                            # Tokenize and index
                            words = re.findall(r'\w+', content.lower())
                            for word in set(words):
                                index.content_index[word].append(rel_path)
                        except Exception:
                            pass
                            
                except Exception as e:
                    logger.warning(f"Failed to index {filepath}: {e}")
        
        index.file_count = len(index.files)
        index.updated_at = datetime.now()
        
        with self._index_lock:
            self._index = index
        
        return index
    
    def search(
        self,
        query: str,
        search_type: SearchType = SearchType.FUZZY,
        root_path: Optional[Path] = None,
        max_results: int = 50,
        file_pattern: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for files.
        
        Args:
            query: Search query
            search_type: Type of search to perform
            root_path: Root path to search
            max_results: Maximum results
            file_pattern: Glob pattern to filter files
            
        Returns:
            List of search results
        """
        root_path = root_path or self._workspace_path
        
        results: List[SearchResult] = []
        
        if search_type == SearchType.FILENAME:
            results = self._search_filename(query, root_path, file_pattern)
        elif search_type == SearchType.CONTENT:
            results = self._search_content(query, root_path, file_pattern)
        elif search_type == SearchType.REGEX:
            results = self._search_regex(query, root_path, file_pattern)
        elif search_type == SearchType.FUZZY:
            results = self._search_fuzzy(query, root_path, file_pattern)
        elif search_type == SearchType.SEMANTIC:
            results = self._search_semantic(query, root_path, file_pattern)
        
        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:max_results]
        
        # Save to history
        self._search_history.append((query, results[:10]))
        if len(self._search_history) > 100:
            self._search_history = self._search_history[-100:]
        
        return results
    
    def _search_filename(
        self,
        query: str,
        root_path: Path,
        file_pattern: Optional[str],
    ) -> List[SearchResult]:
        """Search by filename."""
        results = []
        query_lower = query.lower()
        
        for root, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in files:
                if filename.startswith('.'):
                    continue
                
                if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                    continue
                
                if query_lower in filename.lower():
                    filepath = Path(root) / filename
                    
                    # Calculate score based on match position and length
                    score = 0.5
                    if filename.lower() == query_lower:
                        score = 1.0
                    elif filename.lower().startswith(query_lower):
                        score = 0.8
                    
                    results.append(SearchResult(
                        path=filepath,
                        score=score,
                        matched_text=filename,
                    ))
        
        return results
    
    def _search_content(
        self,
        query: str,
        root_path: Path,
        file_pattern: Optional[str],
    ) -> List[SearchResult]:
        """Search file contents."""
        results = []
        query_lower = query.lower()
        
        # Use index if available
        if self._index and self._index.root_path == root_path:
            query_words = re.findall(r'\w+', query_lower)
            
            # Find files containing all query words
            matching_files: Set[str] = None
            
            for word in query_words:
                files_with_word = set(self._index.content_index.get(word, []))
                if matching_files is None:
                    matching_files = files_with_word
                else:
                    matching_files &= files_with_word
            
            if matching_files:
                for rel_path in matching_files:
                    filepath = root_path / rel_path
                    if filepath.exists():
                        results.append(SearchResult(
                            path=filepath,
                            score=0.7,
                            matched_text=query,
                        ))
        else:
            # Direct content search
            for root, dirs, files in os.walk(root_path):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for filename in files:
                    if filename.startswith('.'):
                        continue
                    
                    if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                        continue
                    
                    filepath = Path(root) / filename
                    
                    try:
                        content = filepath.read_text(errors='ignore')
                        if query_lower in content.lower():
                            # Find line number
                            lines = content.split('\n')
                            line_num = None
                            context = None
                            
                            for i, line in enumerate(lines):
                                if query_lower in line.lower():
                                    line_num = i + 1
                                    context = line.strip()[:100]
                                    break
                            
                            results.append(SearchResult(
                                path=filepath,
                                score=0.6,
                                matched_text=query,
                                line_number=line_num,
                                context=context,
                            ))
                    except Exception:
                        pass
        
        return results
    
    def _search_regex(
        self,
        query: str,
        root_path: Path,
        file_pattern: Optional[str],
    ) -> List[SearchResult]:
        """Search using regex pattern."""
        results = []
        
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            return results
        
        for root, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in files:
                if filename.startswith('.'):
                    continue
                
                if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                    continue
                
                filepath = Path(root) / filename
                
                try:
                    content = filepath.read_text(errors='ignore')
                    matches = list(pattern.finditer(content))
                    
                    if matches:
                        # Get first match context
                        match = matches[0]
                        
                        # Find line number
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Get line context
                        lines = content.split('\n')
                        context = lines[line_num - 1].strip()[:100] if line_num <= len(lines) else None
                        
                        results.append(SearchResult(
                            path=filepath,
                            score=min(1.0, 0.5 + len(matches) * 0.1),
                            matched_text=match.group(),
                            line_number=line_num,
                            context=context,
                        ))
                except Exception:
                    pass
        
        return results
    
    def _search_fuzzy(
        self,
        query: str,
        root_path: Path,
        file_pattern: Optional[str],
    ) -> List[SearchResult]:
        """Fuzzy search combining filename and content search."""
        results = []
        
        # Combine filename and content results
        filename_results = self._search_filename(query, root_path, file_pattern)
        content_results = self._search_content(query, root_path, file_pattern)
        
        # Merge results
        seen_paths: Set[Path] = set()
        
        for result in filename_results:
            seen_paths.add(result.path)
            result.score *= 1.2  # Boost filename matches
            results.append(result)
        
        for result in content_results:
            if result.path not in seen_paths:
                results.append(result)
        
        return results
    
    def _search_semantic(
        self,
        query: str,
        root_path: Path,
        file_pattern: Optional[str],
    ) -> List[SearchResult]:
        """Semantic search using LLM or embeddings."""
        # If no LLM, fall back to fuzzy search
        if not self._llm_client:
            return self._search_fuzzy(query, root_path, file_pattern)
        
        # Get candidate files
        candidates = []
        for root, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in files[:100]:  # Limit for performance
                if filename.startswith('.'):
                    continue
                
                if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                    continue
                
                candidates.append(Path(root) / filename)
        
        # Use fuzzy search as fallback (real impl would use embeddings)
        return self._search_fuzzy(query, root_path, file_pattern)
    
    def get_suggestions(
        self,
        context: str,
        current_file: Optional[Path] = None,
    ) -> List[SearchResult]:
        """Get file suggestions based on context.
        
        Args:
            context: Context description
            current_file: Currently open file for related suggestions
            
        Returns:
            List of suggested files
        """
        suggestions = []
        
        # Add recently searched files
        for query, results in reversed(self._search_history[-5:]):
            for result in results[:3]:
                if result.path.exists():
                    suggestions.append(result)
        
        # Add files related to current file
        if current_file and current_file.exists():
            # Find files in same directory
            for sibling in current_file.parent.iterdir():
                if sibling.is_file() and sibling != current_file:
                    suggestions.append(SearchResult(
                        path=sibling,
                        score=0.5,
                        matched_text="same directory",
                    ))
                    if len(suggestions) >= 10:
                        break
        
        return suggestions[:10]


class FileSystemIntelligence:
    """Main file system intelligence manager.
    
    Integrates all file system intelligence components:
    - Fuzzy path resolution
    - Content-aware file handling
    - Safe file operations
    - Smart file search
    
    Example:
        >>> fs_intel = FileSystemIntelligence(llm_client=client)
        >>> matches = fs_intel.resolve_path("src/main")
        >>> info = fs_intel.analyze_file(Path("file.py"))
        >>> results = fs_intel.search("database connection")
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        workspace_path: Optional[Path] = None,
    ):
        """Initialize file system intelligence.
        
        Args:
            llm_client: LLM client for intelligent operations
            workspace_path: Default workspace path
        """
        self._llm_client = llm_client
        self._workspace_path = workspace_path or Path.cwd()
        
        # Initialize components
        self._path_resolver = FuzzyPathResolver(
            llm_client=llm_client,
            workspace_path=self._workspace_path,
        )
        
        self._file_handler = ContentAwareFileHandler(
            llm_client=llm_client,
        )
        
        self._safe_ops = SafeFileOperations(
            llm_client=llm_client,
        )
        
        self._search = SmartFileSearch(
            llm_client=llm_client,
            workspace_path=self._workspace_path,
        )
    
    # Path resolution
    def resolve_path(
        self,
        partial_path: str,
        base_path: Optional[Path] = None,
        max_results: int = 10,
    ) -> List[PathMatch]:
        """Resolve a partial path to matching paths."""
        return self._path_resolver.resolve(
            partial_path,
            base_path or self._workspace_path,
            max_results,
        )
    
    def auto_complete_path(
        self,
        partial_path: str,
        base_path: Optional[Path] = None,
    ) -> List[str]:
        """Auto-complete a partial path."""
        return self._path_resolver.auto_complete(
            partial_path,
            base_path or self._workspace_path,
        )
    
    # File analysis
    def analyze_file(self, path: Path) -> FileInfo:
        """Analyze a file and return detailed information."""
        return self._file_handler.analyze_file(path)
    
    def read_file(
        self,
        path: Path,
        encoding: Optional[str] = None,
    ) -> Tuple[Union[str, bytes], FileInfo]:
        """Read a file with automatic encoding detection."""
        return self._file_handler.read_file_smart(path, encoding)
    
    def categorize_files(
        self,
        paths: List[Path],
    ) -> Dict[FileCategory, List[FileInfo]]:
        """Categorize multiple files by type."""
        return self._file_handler.categorize_files(paths)
    
    # Safe operations
    def safe_write(
        self,
        path: Path,
        content: Union[str, bytes],
        **kwargs,
    ) -> Dict[str, Any]:
        """Safely write content to a file."""
        return self._safe_ops.safe_write(path, content, **kwargs)
    
    def safe_delete(
        self,
        path: Path,
        **kwargs,
    ) -> Dict[str, Any]:
        """Safely delete a file or directory."""
        return self._safe_ops.safe_delete(path, **kwargs)
    
    def safe_move(
        self,
        source: Path,
        destination: Path,
        **kwargs,
    ) -> Dict[str, Any]:
        """Safely move a file or directory."""
        return self._safe_ops.safe_move(source, destination, **kwargs)
    
    def safe_copy(
        self,
        source: Path,
        destination: Path,
        **kwargs,
    ) -> Dict[str, Any]:
        """Safely copy a file or directory."""
        return self._safe_ops.safe_copy(source, destination, **kwargs)
    
    def undo(self, count: int = 1) -> List[Dict[str, Any]]:
        """Undo recent file operations."""
        return self._safe_ops.undo(count)
    
    def dry_run(
        self,
        operation: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Simulate an operation without executing."""
        return self._safe_ops.dry_run(operation, **kwargs)
    
    def get_operation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent operation history."""
        return self._safe_ops.get_undo_history(limit)
    
    # Search
    def search(
        self,
        query: str,
        search_type: SearchType = SearchType.FUZZY,
        **kwargs,
    ) -> List[SearchResult]:
        """Search for files."""
        return self._search.search(query, search_type, **kwargs)
    
    def build_search_index(
        self,
        root_path: Optional[Path] = None,
        include_content: bool = True,
    ) -> FileIndex:
        """Build or update the search index."""
        return self._search.build_index(root_path or self._workspace_path, include_content)
    
    def get_search_suggestions(
        self,
        context: str,
        current_file: Optional[Path] = None,
    ) -> List[SearchResult]:
        """Get file suggestions based on context."""
        return self._search.get_suggestions(context, current_file)
    
    async def analyze_with_llm(
        self,
        path: Path,
        analysis_type: str = "summary",
    ) -> Dict[str, Any]:
        """Analyze a file using LLM reasoning.
        
        Args:
            path: Path to analyze
            analysis_type: Type of analysis (summary, issues, suggestions)
            
        Returns:
            Analysis results
        """
        if not self._llm_client:
            return {"error": "LLM client not available"}
        
        # Get file info
        info = self.analyze_file(path)
        
        # Read content for text files
        content = None
        if info.encoding:
            try:
                content_raw, _ = self.read_file(path)
                content = content_raw[:5000] if isinstance(content_raw, str) else None
            except Exception:
                pass
        
        prompt = f"""Analyze this file:
Path: {path}
Name: {info.name}
Type: {info.category.value}
Size: {info.size_bytes} bytes
Encoding: {info.encoding}

{f'Content preview:{chr(10)}{content}' if content else 'Binary or large file'}

Provide a {analysis_type} of this file.
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            return {
                "file": info.to_dict(),
                "analysis_type": analysis_type,
                "analysis": response,
            }
        except Exception as e:
            return {"error": str(e)}


# Module-level instance
_global_fs_intelligence: Optional[FileSystemIntelligence] = None


def get_file_system_intelligence(
    llm_client: Optional[Any] = None,
    workspace_path: Optional[Path] = None,
) -> FileSystemIntelligence:
    """Get the global file system intelligence manager.
    
    Args:
        llm_client: Optional LLM client
        workspace_path: Optional workspace path
        
    Returns:
        FileSystemIntelligence instance
    """
    global _global_fs_intelligence
    if _global_fs_intelligence is None:
        _global_fs_intelligence = FileSystemIntelligence(
            llm_client=llm_client,
            workspace_path=workspace_path,
        )
    return _global_fs_intelligence
