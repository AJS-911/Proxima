"""Terminal Tools for Dynamic Tool System.

This module provides terminal/shell execution tools that can be
dynamically discovered and used by the LLM.

Tools included:
- RunCommand: Execute shell commands
- GetWorkingDirectory: Get current directory
- ChangeDirectory: Change working directory
"""

from __future__ import annotations

import os
import subprocess
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import platform

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
from ..execution_context import ExecutionContext, TerminalSession
from ..tool_registry import register_tool


def get_shell_info() -> tuple[str, List[str]]:
    """Get the appropriate shell for the current platform."""
    if platform.system() == "Windows":
        return "powershell", ["powershell", "-NoProfile", "-Command"]
    else:
        # Try to use user's shell, fallback to bash
        shell = os.environ.get("SHELL", "/bin/bash")
        return shell, [shell, "-c"]


@register_tool
class RunCommandTool(BaseTool):
    """Execute a shell command."""
    
    @property
    def name(self) -> str:
        return "run_command"
    
    @property
    def description(self) -> str:
        return (
            "Execute a shell command in the terminal. "
            "Supports both Windows PowerShell and Unix shells. "
            "Returns stdout, stderr, and exit code."
        )
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.TERMINAL
    
    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.EXECUTE
    
    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                description="The command to execute",
                param_type=ParameterType.STRING,
                required=True,
            ),
            ToolParameter(
                name="working_directory",
                description="Directory to execute in (default: current directory)",
                param_type=ParameterType.PATH,
                required=False,
            ),
            ToolParameter(
                name="timeout",
                description="Timeout in seconds (default: 60)",
                param_type=ParameterType.INTEGER,
                required=False,
                default=60,
            ),
            ToolParameter(
                name="capture_output",
                description="Whether to capture and return output",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["terminal", "shell", "command", "execute", "run", "bash", "powershell"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "Run 'pip install requests'",
            "Execute 'ls -la' in the current directory",
            "Run npm install",
            "Execute the test command",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        command = parameters["command"]
        working_dir = parameters.get("working_directory", context.current_directory)
        timeout = parameters.get("timeout", 60)
        capture_output = parameters.get("capture_output", True)
        
        # Resolve working directory
        if working_dir and not Path(working_dir).is_absolute():
            working_dir = str(Path(context.current_directory) / working_dir)
        
        # Validate working directory exists
        if working_dir and not Path(working_dir).is_dir():
            return ToolResult(
                success=False,
                error=f"Working directory not found: {working_dir}",
            )
        
        shell_name, shell_prefix = get_shell_info()
        
        try:
            # Build the command
            if platform.system() == "Windows":
                # PowerShell command
                full_command = shell_prefix + [command]
            else:
                # Unix shell
                full_command = shell_prefix + [command]
            
            # Execute
            result = subprocess.run(
                full_command,
                cwd=working_dir,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                env=dict(os.environ, **context.environment_variables),
            )
            
            output_data = {
                "command": command,
                "exit_code": result.returncode,
                "working_directory": working_dir,
            }
            
            if capture_output:
                output_data["stdout"] = result.stdout
                output_data["stderr"] = result.stderr
            
            success = result.returncode == 0
            
            message = f"Command completed with exit code {result.returncode}"
            if result.stdout:
                message += f"\n{result.stdout[:500]}"
                if len(result.stdout) > 500:
                    message += "\n... (output truncated)"
            
            if result.stderr and not success:
                message += f"\nError: {result.stderr[:200]}"
            
            return ToolResult(
                success=success,
                result=output_data,
                message=message,
                error=result.stderr if not success else None,
            )
        
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout} seconds",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to execute command: {str(e)}",
            )


@register_tool
class GetWorkingDirectoryTool(BaseTool):
    """Get the current working directory."""
    
    @property
    def name(self) -> str:
        return "get_working_directory"
    
    @property
    def description(self) -> str:
        return "Get the current working directory path."
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.TERMINAL
    
    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_ONLY
    
    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.NONE
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return []
    
    @property
    def tags(self) -> List[str]:
        return ["directory", "pwd", "cwd", "path", "location"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "What is the current directory?",
            "Where am I?",
            "Print working directory",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        cwd = context.current_directory
        
        return ToolResult(
            success=True,
            result=cwd,
            message=f"Current directory: {cwd}",
        )


@register_tool
class ChangeDirectoryTool(BaseTool):
    """Change the current working directory."""
    
    @property
    def name(self) -> str:
        return "change_directory"
    
    @property
    def description(self) -> str:
        return "Change the current working directory to a new path."
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.TERMINAL
    
    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_WRITE
    
    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.NONE
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                description="The directory path to change to",
                param_type=ParameterType.PATH,
                required=True,
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["directory", "cd", "navigate", "path", "change"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "Change to the src directory",
            "Go to /home/user/projects",
            "Navigate to the parent directory",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        path = Path(parameters["path"])
        
        # Handle relative paths
        if not path.is_absolute():
            path = Path(context.current_directory) / path
        
        # Resolve .. and .
        path = path.resolve()
        
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
        
        # Update context
        old_dir = context.current_directory
        context.set_current_directory(str(path))
        
        return ToolResult(
            success=True,
            result=str(path),
            message=f"Changed directory: {old_dir} → {path}",
        )


@register_tool
class EnvironmentVariableTool(BaseTool):
    """Get or set environment variables."""
    
    @property
    def name(self) -> str:
        return "environment_variable"
    
    @property
    def description(self) -> str:
        return "Get or set environment variables for the current session."
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.TERMINAL
    
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
                name="action",
                description="Action: get, set, or list",
                param_type=ParameterType.STRING,
                required=True,
            ),
            ToolParameter(
                name="name",
                description="Variable name (for get/set)",
                param_type=ParameterType.STRING,
                required=False,
            ),
            ToolParameter(
                name="value",
                description="Variable value (for set)",
                param_type=ParameterType.STRING,
                required=False,
            ),
        ]
    
    @property
    def tags(self) -> List[str]:
        return ["environment", "variable", "env", "path", "config"]
    
    @property
    def examples(self) -> List[str]:
        return [
            "Get the PATH environment variable",
            "Set MY_VAR to 'hello'",
            "List all environment variables",
        ]
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        action = parameters["action"].lower()
        name = parameters.get("name")
        value = parameters.get("value")
        
        if action == "get":
            if not name:
                return ToolResult(
                    success=False,
                    error="Variable name required for get action",
                )
            
            # Check context first, then os.environ
            env_value = context.environment_variables.get(name, os.environ.get(name))
            
            if env_value is None:
                return ToolResult(
                    success=False,
                    error=f"Environment variable not found: {name}",
                )
            
            return ToolResult(
                success=True,
                result=env_value,
                message=f"{name}={env_value}",
            )
        
        elif action == "set":
            if not name:
                return ToolResult(
                    success=False,
                    error="Variable name required for set action",
                )
            if value is None:
                return ToolResult(
                    success=False,
                    error="Variable value required for set action",
                )
            
            context.environment_variables[name] = value
            os.environ[name] = value
            
            return ToolResult(
                success=True,
                message=f"Set {name}={value}",
            )
        
        elif action == "list":
            # Return a subset of important variables
            important = {
                k: v for k, v in os.environ.items()
                if not k.startswith("_") and len(v) < 200
            }
            
            return ToolResult(
                success=True,
                result=important,
                message=f"Listed {len(important)} environment variables",
            )
        
        return ToolResult(
            success=False,
            error=f"Unknown action: {action}",
        )
