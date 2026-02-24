"""File System Tools for Dynamic Tool System.

This module provides file system operation tools that can be
dynamically discovered and used by the LLM.

Tools included:
- ReadFile: Read file contents
- WriteFile: Write/create files
- ListDirectory: List directory contents
- CreateDirectory: Create directories
- DeleteFile: Delete files
- MoveFile: Move/rename files
- SearchFiles: Search for files by pattern
- FileInfo: Get file metadata
"""

from __future__ import annotations

import os
import shutil
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import fnmatch
import re

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
from ..execution_context import ExecutionContext
from ..tool_registry import register_tool


@register_tool
class ReadFileTool(BaseTool):
    """Read the contents of a file."""
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. Supports text files with various encodings. "
            "Can read specific line ranges for large files."
        )
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILE_SYSTEM
    
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
                description="Path to the file to read",
                param_type=ParameterType.PATH,
                required=True,
            ),
            ToolParameter(
                name="start_line",
                description="Starting line number (1-based). If not specified, reads from beginning.",
                param_type=ParameterType.INTEGER,
                required=False,
            ),
            ToolParameter(
                name="end_line",
                description="Ending line number (1-based, inclusive). If not specified, reads to end.",
                param_type=ParameterType.INTEGER,
                required=False,
            ),
            ToolParameter(
                name="encoding",
                description="File encoding (default: utf-8)",
                param_type=ParameterType.STRING,
                required=False,
                default="utf-8",
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["file", "read", "view", "content", "text", "open"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "Read the contents of config.py",
            "Show me what's in the README.md file",
            "Read lines 10-50 of main.py",
            "Open and display the package.json file",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        path = Path(parameters["path"])
        
        # Make path absolute if relative
        if not path.is_absolute():
            path = Path(context.current_directory) / path
        
        if not path.exists():
            return ToolResult(
                success=False,
                error=f"File not found: {path}",
            )
        
        if not path.is_file():
            return ToolResult(
                success=False,
                error=f"Path is not a file: {path}",
            )
        
        encoding = parameters.get("encoding", "utf-8")
        start_line = parameters.get("start_line")
        end_line = parameters.get("end_line")
        
        try:
            with open(path, "r", encoding=encoding) as f:
                if start_line or end_line:
                    lines = f.readlines()
                    total_lines = len(lines)
                    
                    start = (start_line or 1) - 1  # Convert to 0-based
                    end = end_line or total_lines
                    
                    content = "".join(lines[start:end])
                    line_info = f"Lines {start + 1}-{min(end, total_lines)} of {total_lines}"
                else:
                    content = f.read()
                    total_lines = content.count("\n") + 1
                    line_info = f"{total_lines} lines"
            
            # Track file access
            context.add_recent_file(str(path))
            
            return ToolResult(
                success=True,
                result=content,
                message=f"Read {path.name} ({line_info})",
                metaresult={
                    "path": str(path),
                    "total_lines": total_lines if 'total_lines' in dir() else None,
                    "encoding": encoding,
                },
            )
        
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                error=f"Cannot decode file with encoding '{encoding}'. Try a different encoding.",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error reading file: {str(e)}",
            )


@register_tool
class WriteFileTool(BaseTool):
    """Write content to a file."""
    
    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return (
            "Write content to a file. Can create new files or overwrite existing ones. "
            "Creates parent directories if they don't exist."
        )
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILE_SYSTEM
    
    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_WRITE
    
    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.MEDIUM
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                description="Path to the file to write",
                param_type=ParameterType.PATH,
                required=True,
            ),
            ToolParameter(
                name="content",
                description="Content to write to the file",
                param_type=ParameterType.STRING,
                required=True,
            ),
            ToolParameter(
                name="encoding",
                description="File encoding (default: utf-8)",
                param_type=ParameterType.STRING,
                required=False,
                default="utf-8",
            ),
            ToolParameter(
                name="create_parents",
                description="Create parent directories if they don't exist (default: true)",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["file", "write", "create", "save", "modify", "edit"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "Create a new file called hello.py with print('hello')",
            "Write the configuration to config.json",
            "Save this content to output.txt",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        path = Path(parameters["path"])
        content = parameters["content"]
        encoding = parameters.get("encoding", "utf-8")
        create_parents = parameters.get("create_parents", True)
        
        # Make path absolute if relative
        if not path.is_absolute():
            path = Path(context.current_directory) / path
        
        try:
            # Create parent directories if needed
            if create_parents:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            existed = path.exists()
            
            with open(path, "w", encoding=encoding) as f:
                f.write(content)
            
            context.add_recent_file(str(path))
            
            action = "Updated" if existed else "Created"
            return ToolResult(
                success=True,
                message=f"{action} {path.name} ({len(content)} characters)",
                result={"path": str(path), "action": action.lower()},
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error writing file: {str(e)}",
            )


@register_tool
class ListDirectoryTool(BaseTool):
    """List contents of a directory."""
    
    @property
    def name(self) -> str:
        return "list_directory"
    
    @property
    def description(self) -> str:
        return (
            "List the contents of a directory. Shows files and subdirectories "
            "with optional filtering and detailed information."
        )
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILE_SYSTEM
    
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
                description="Path to the directory (default: current directory)",
                param_type=ParameterType.PATH,
                required=False,
                default=".",
            ),
            ToolParameter(
                name="pattern",
                description="Glob pattern to filter files (e.g., '*.py')",
                param_type=ParameterType.STRING,
                required=False,
            ),
            ToolParameter(
                name="recursive",
                description="List subdirectories recursively",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=False,
            ),
            ToolParameter(
                name="show_hidden",
                description="Include hidden files (starting with .)",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=False,
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["directory", "list", "ls", "dir", "files", "folders", "browse"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "List files in the current directory",
            "Show all Python files in src/",
            "List all files recursively in the project",
            "What files are in the tests folder?",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        path = Path(parameters.get("path", "."))
        pattern = parameters.get("pattern")
        recursive = parameters.get("recursive", False)
        show_hidden = parameters.get("show_hidden", False)
        
        # Make path absolute
        if not path.is_absolute():
            path = Path(context.current_directory) / path
        
        if not path.exists():
            return ToolResult(
                success=False,
                error=f"Directory not found: {path}",
            )
        
        if not path.is_dir():
            return ToolResult(
                success=False,
                error=f"Path is not a directory: {path}",
            )
        
        try:
            entries = []
            
            if recursive:
                items = path.rglob(pattern or "*")
            else:
                items = path.glob(pattern or "*")
            
            for item in items:
                # Skip hidden files unless requested
                if not show_hidden and item.name.startswith("."):
                    continue
                
                entry = {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "path": str(item.relative_to(path)),
                }
                
                if item.is_file():
                    try:
                        stat_info = item.stat()
                        entry["size"] = stat_info.st_size
                        entry["modified"] = datetime.fromtimestamp(
                            stat_info.st_mtime
                        ).isoformat()
                    except:
                        pass
                
                entries.append(entry)
            
            # Sort: directories first, then alphabetically
            entries.sort(key=lambda e: (0 if e["type"] == "directory" else 1, e["name"].lower()))
            
            return ToolResult(
                success=True,
                result=entries,
                message=f"Listed {len(entries)} items in {path.name or path}",
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error listing directory: {str(e)}",
            )


@register_tool
class SearchFilesTool(BaseTool):
    """Search for files by name or content."""
    
    @property
    def name(self) -> str:
        return "search_files"
    
    @property
    def description(self) -> str:
        return (
            "Search for files by name pattern or content. Can search recursively "
            "through directories and optionally search within file contents."
        )
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILE_SYSTEM
    
    @property
    def sub_category(self) -> Optional[ToolCategory]:
        return ToolCategory.ANALYSIS
    
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
                description="Directory to search in (default: current directory)",
                param_type=ParameterType.PATH,
                required=False,
                default=".",
            ),
            ToolParameter(
                name="name_pattern",
                description="Glob pattern for file names (e.g., '*.py', 'test_*.py')",
                param_type=ParameterType.STRING,
                required=False,
            ),
            ToolParameter(
                name="content_pattern",
                description="Text or regex pattern to search for in file contents",
                param_type=ParameterType.STRING,
                required=False,
            ),
            ToolParameter(
                name="use_regex",
                description="Treat content_pattern as regex",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=False,
            ),
            ToolParameter(
                name="max_results",
                description="Maximum number of results to return",
                param_type=ParameterType.INTEGER,
                required=False,
                default=50,
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["search", "find", "grep", "locate", "files", "content"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "Find all Python files in the project",
            "Search for files containing 'TODO'",
            "Find all test files",
            "Search for the function 'process_data' in the codebase",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        path = Path(parameters.get("path", "."))
        name_pattern = parameters.get("name_pattern", "*")
        content_pattern = parameters.get("content_pattern")
        use_regex = parameters.get("use_regex", False)
        max_results = parameters.get("max_results", 50)
        
        if not path.is_absolute():
            path = Path(context.current_directory) / path
        
        if not path.exists():
            return ToolResult(
                success=False,
                error=f"Directory not found: {path}",
            )
        
        try:
            results = []
            
            for file_path in path.rglob(name_pattern):
                if len(results) >= max_results:
                    break
                
                if not file_path.is_file():
                    continue
                
                result_entry = {
                    "path": str(file_path.relative_to(path)),
                    "full_path": str(file_path),
                }
                
                # Search content if pattern provided
                if content_pattern:
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        
                        if use_regex:
                            matches = re.findall(content_pattern, content)
                            if matches:
                                result_entry["matches"] = len(matches)
                                results.append(result_entry)
                        else:
                            if content_pattern.lower() in content.lower():
                                # Find line numbers
                                lines = content.split("\n")
                                match_lines = []
                                for i, line in enumerate(lines, 1):
                                    if content_pattern.lower() in line.lower():
                                        match_lines.append(i)
                                result_entry["match_lines"] = match_lines[:10]
                                results.append(result_entry)
                    except:
                        pass  # Skip files that can't be read
                else:
                    results.append(result_entry)
            
            return ToolResult(
                success=True,
                result=results,
                message=f"Found {len(results)} matching files",
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Search failed: {str(e)}",
            )


@register_tool
class DeleteFileTool(BaseTool):
    """Delete a file or directory."""
    
    @property
    def name(self) -> str:
        return "delete_file"
    
    @property
    def description(self) -> str:
        return (
            "Delete a file or directory. For directories, can optionally delete recursively. "
            "This operation cannot be undone."
        )
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILE_SYSTEM
    
    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_WRITE
    
    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                description="Path to the file or directory to delete",
                param_type=ParameterType.PATH,
                required=True,
            ),
            ToolParameter(
                name="recursive",
                description="Delete directories recursively (required for non-empty directories)",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=False,
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["delete", "remove", "rm", "unlink", "file", "directory"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "Delete the temp.txt file",
            "Remove the old_backup directory",
            "Delete the build folder and all its contents",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        path = Path(parameters["path"])
        recursive = parameters.get("recursive", False)
        
        if not path.is_absolute():
            path = Path(context.current_directory) / path
        
        if not path.exists():
            return ToolResult(
                success=False,
                error=f"Path not found: {path}",
            )
        
        try:
            if path.is_file():
                path.unlink()
                return ToolResult(
                    success=True,
                    message=f"Deleted file: {path.name}",
                )
            elif path.is_dir():
                if recursive:
                    shutil.rmtree(path)
                    return ToolResult(
                        success=True,
                        message=f"Deleted directory: {path.name}",
                    )
                else:
                    path.rmdir()  # Only works for empty directories
                    return ToolResult(
                        success=True,
                        message=f"Deleted empty directory: {path.name}",
                    )
        except OSError as e:
            if "not empty" in str(e).lower():
                return ToolResult(
                    success=False,
                    error="Directory is not empty. Use recursive=true to delete.",
                )
            return ToolResult(
                success=False,
                error=f"Error deleting: {str(e)}",
            )


@register_tool
class MoveFileTool(BaseTool):
    """Move or rename a file or directory."""
    
    @property
    def name(self) -> str:
        return "move_file"
    
    @property
    def description(self) -> str:
        return "Move or rename a file or directory to a new location."
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILE_SYSTEM
    
    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_WRITE
    
    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.MEDIUM
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="source",
                description="Path to the file or directory to move",
                param_type=ParameterType.PATH,
                required=True,
            ),
            ToolParameter(
                name="destination",
                description="Destination path",
                param_type=ParameterType.PATH,
                required=True,
            ),
            ToolParameter(
                name="overwrite",
                description="Overwrite destination if it exists",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=False,
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["move", "rename", "mv", "relocate", "file"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "Rename old_name.py to new_name.py",
            "Move config.json to the backup folder",
            "Rename the src folder to source",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        source = Path(parameters["source"])
        destination = Path(parameters["destination"])
        overwrite = parameters.get("overwrite", False)
        
        if not source.is_absolute():
            source = Path(context.current_directory) / source
        if not destination.is_absolute():
            destination = Path(context.current_directory) / destination
        
        if not source.exists():
            return ToolResult(
                success=False,
                error=f"Source not found: {source}",
            )
        
        if destination.exists() and not overwrite:
            return ToolResult(
                success=False,
                error=f"Destination already exists: {destination}. Use overwrite=true to replace.",
            )
        
        try:
            shutil.move(str(source), str(destination))
            return ToolResult(
                success=True,
                message=f"Moved {source.name} to {destination}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error moving: {str(e)}",
            )


@register_tool
class FileInfoTool(BaseTool):
    """Get detailed information about a file or directory."""
    
    @property
    def name(self) -> str:
        return "file_info"
    
    @property
    def description(self) -> str:
        return "Get detailed metadata and information about a file or directory."
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILE_SYSTEM
    
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
                description="Path to the file or directory",
                param_type=ParameterType.PATH,
                required=True,
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["info", "stat", "metadata", "details", "file", "size"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "Get info about config.py",
            "What is the size of the database file?",
            "When was main.py last modified?",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        path = Path(parameters["path"])
        
        if not path.is_absolute():
            path = Path(context.current_directory) / path
        
        if not path.exists():
            return ToolResult(
                success=False,
                error=f"Path not found: {path}",
            )
        
        try:
            stat_info = path.stat()
            
            info = {
                "name": path.name,
                "path": str(path),
                "type": "directory" if path.is_dir() else "file",
                "size_bytes": stat_info.st_size,
                "size_human": self._human_readable_size(stat_info.st_size),
                "created": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat_info.st_atime).isoformat(),
                "readable": os.access(path, os.R_OK),
                "writable": os.access(path, os.W_OK),
                "executable": os.access(path, os.X_OK),
            }
            
            if path.is_file():
                info["extension"] = path.suffix
                # Get line count for text files
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        info["line_count"] = sum(1 for _ in f)
                except:
                    pass
            elif path.is_dir():
                # Count items
                items = list(path.iterdir())
                info["item_count"] = len(items)
                info["subdirectory_count"] = len([i for i in items if i.is_dir()])
                info["file_count"] = len([i for i in items if i.is_file()])
            
            return ToolResult(
                success=True,
                result=info,
                message=f"Info for {path.name}",
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error getting info: {str(e)}",
            )
    
    def _human_readable_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"


@register_tool
class CreateDirectoryTool(BaseTool):
    """Create a new directory (with optional nested creation)."""

    @property
    def name(self) -> str:
        return "create_directory"

    @property
    def description(self) -> str:
        return (
            "Create a new directory at the specified path. Supports creating "
            "nested directory structures when parents=true."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILE_SYSTEM

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
                name="path",
                description="Path to the directory to create",
                param_type=ParameterType.PATH,
                required=True,
            ),
            ToolParameter(
                name="parents",
                description="Create parent directories if they don't exist (default: true)",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
            ),
            ToolParameter(
                name="exist_ok",
                description="Don't raise an error if the directory already exists (default: true)",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
            ),
        ]

    @property
    def tags(self) -> List[str]:
        return ["directory", "create", "mkdir", "folder", "new"]

    @property
    def examples(self) -> List[str]:
        return [
            "Create a new folder called 'output'",
            "Make directory src/utils/helpers",
            "Create the tests directory",
            "mkdir build/release",
        ]

    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        path = Path(parameters["path"])
        parents = parameters.get("parents", True)
        exist_ok = parameters.get("exist_ok", True)

        # Make path absolute if relative
        if not path.is_absolute():
            path = Path(context.current_directory) / path

        if path.exists():
            if path.is_dir():
                if exist_ok:
                    return ToolResult(
                        success=True,
                        message=f"Directory already exists: {path.name}",
                        result={"path": str(path), "action": "already_exists"},
                    )
                else:
                    return ToolResult(
                        success=False,
                        error=f"Directory already exists: {path}",
                    )
            else:
                return ToolResult(
                    success=False,
                    error=f"Path exists but is not a directory: {path}",
                )

        try:
            path.mkdir(parents=parents, exist_ok=exist_ok)
            return ToolResult(
                success=True,
                message=f"Created directory: {path.name}",
                result={"path": str(path), "action": "created"},
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error=(
                    f"Parent directory does not exist: {path.parent}. "
                    "Use parents=true to create parent directories."
                ),
            )
        except PermissionError:
            return ToolResult(
                success=False,
                error=f"Permission denied: cannot create directory at {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error creating directory: {str(e)}",
            )
