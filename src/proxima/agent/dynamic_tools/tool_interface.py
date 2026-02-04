"""Tool Interface Definitions for Dynamic Tool System.

This module defines the base interfaces and contracts that all tools must implement.
It provides a standardized way to define, validate, and execute tools dynamically
based on LLM reasoning rather than hardcoded patterns.

Key Components:
- ToolInterface: Abstract base class for all tools
- ToolDefinition: Metadata schema for tool discovery
- ToolParameter: Parameter definition with validation
- ToolResult: Standardized execution result
- ToolCategory: Hierarchical tool categorization
- PermissionLevel: Security permission levels
- RiskLevel: Operation risk assessment
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ToolCategory(str, Enum):
    """Hierarchical categories for tool organization.
    
    Categories help the LLM understand which tools are relevant for different
    types of operations without needing hardcoded keyword matching.
    """
    # Primary Categories
    FILE_SYSTEM = "file_system"
    GIT = "git"
    GITHUB = "github"
    TERMINAL = "terminal"
    BACKEND = "backend"
    ANALYSIS = "analysis"
    SYSTEM = "system"
    
    # Sub-categories (can be combined)
    FILE_READ = "file_system.read"
    FILE_WRITE = "file_system.write"
    FILE_SEARCH = "file_system.search"
    DIRECTORY = "file_system.directory"
    
    GIT_BRANCH = "git.branch"
    GIT_COMMIT = "git.commit"
    GIT_REMOTE = "git.remote"
    GIT_HISTORY = "git.history"
    
    GITHUB_AUTH = "github.auth"
    GITHUB_REPO = "github.repo"
    GITHUB_ISSUES = "github.issues"
    GITHUB_PR = "github.pr"
    GITHUB_ACTIONS = "github.actions"
    
    TERMINAL_EXECUTE = "terminal.execute"
    TERMINAL_PROCESS = "terminal.process"
    TERMINAL_STREAM = "terminal.stream"
    
    BACKEND_BUILD = "backend.build"
    BACKEND_RUN = "backend.run"
    BACKEND_CONFIG = "backend.config"
    
    ANALYSIS_RESULT = "analysis.result"
    ANALYSIS_BENCHMARK = "analysis.benchmark"
    ANALYSIS_REPORT = "analysis.report"


class PermissionLevel(str, Enum):
    """Permission levels for tool execution security.
    
    These levels determine what operations a tool can perform and
    whether user consent is required.
    """
    READ_ONLY = "read_only"           # Can only read data
    READ_WRITE = "read_write"         # Can read and write files
    EXECUTE = "execute"               # Can execute commands
    SYSTEM = "system"                 # Can modify system settings
    NETWORK = "network"               # Can access network resources
    ADMIN = "admin"                   # Full administrative access
    FULL_ADMIN = "full_admin"         # Unrestricted access


class RiskLevel(str, Enum):
    """Risk assessment levels for operations.
    
    Determines whether consent is required and what warnings to show.
    """
    NONE = "none"           # No risk (read-only)
    LOW = "low"             # Minimal risk (reversible operations)
    MEDIUM = "medium"       # Moderate risk (may require cleanup)
    HIGH = "high"           # High risk (destructive, irreversible)
    CRITICAL = "critical"   # Critical risk (system-level changes)


class ExecutionStatus(str, Enum):
    """Status of tool execution."""
    PENDING = "pending"
    VALIDATING = "validating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


# =============================================================================
# PARAMETER DEFINITIONS
# =============================================================================

class ParameterType(str, Enum):
    """Supported parameter types with validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    PATH = "path"           # File/directory path
    URL = "url"             # URL with validation
    BRANCH_NAME = "branch"  # Git branch name
    COMMIT_HASH = "commit"  # Git commit hash
    ENUM = "enum"           # Enumerated values


class ToolParameter(BaseModel):
    """Definition of a tool parameter with full validation support.
    
    This replaces hardcoded regex patterns with structured parameter
    definitions that the LLM can understand and validate.
    """
    name: str = Field(..., description="Parameter name")
    param_type: ParameterType = Field(..., description="Parameter type")
    description: str = Field(..., description="Human-readable description for LLM")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if not provided")
    
    # Validation constraints
    enum_values: Optional[List[str]] = Field(default=None, description="Allowed values for enum type")
    min_value: Optional[float] = Field(default=None, description="Minimum value for numbers")
    max_value: Optional[float] = Field(default=None, description="Maximum value for numbers")
    min_length: Optional[int] = Field(default=None, description="Minimum length for strings")
    max_length: Optional[int] = Field(default=None, description="Maximum length for strings")
    pattern: Optional[str] = Field(default=None, description="Regex pattern for validation")
    
    # Inference hints for LLM
    examples: Optional[List[str]] = Field(default=None, description="Example values")
    inference_hint: Optional[str] = Field(default=None, description="Hint for LLM to infer value from context")
    
    class Config:
        use_enum_values = True
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format for LLM function calling."""
        type_map = {
            ParameterType.STRING: "string",
            ParameterType.INTEGER: "integer",
            ParameterType.FLOAT: "number",
            ParameterType.BOOLEAN: "boolean",
            ParameterType.ARRAY: "array",
            ParameterType.OBJECT: "object",
            ParameterType.PATH: "string",
            ParameterType.URL: "string",
            ParameterType.BRANCH_NAME: "string",
            ParameterType.COMMIT_HASH: "string",
            ParameterType.ENUM: "string",
        }
        
        schema: Dict[str, Any] = {
            "type": type_map.get(self.param_type, "string"),
            "description": self.description,
        }
        
        if self.enum_values:
            schema["enum"] = self.enum_values
        if self.default is not None:
            schema["default"] = self.default
        if self.min_value is not None:
            schema["minimum"] = self.min_value
        if self.max_value is not None:
            schema["maximum"] = self.max_value
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.pattern:
            schema["pattern"] = self.pattern
        if self.examples:
            schema["examples"] = self.examples
            
        return schema
    
    def validate_value(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a value against parameter constraints.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            if self.required and self.default is None:
                return False, f"Parameter '{self.name}' is required"
            return True, None
        
        # Type validation
        if self.param_type == ParameterType.STRING:
            if not isinstance(value, str):
                return False, f"Parameter '{self.name}' must be a string"
            if self.min_length and len(value) < self.min_length:
                return False, f"Parameter '{self.name}' must be at least {self.min_length} characters"
            if self.max_length and len(value) > self.max_length:
                return False, f"Parameter '{self.name}' must be at most {self.max_length} characters"
                
        elif self.param_type == ParameterType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"Parameter '{self.name}' must be an integer"
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' must be at least {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' must be at most {self.max_value}"
                
        elif self.param_type == ParameterType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"Parameter '{self.name}' must be a number"
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' must be at least {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' must be at most {self.max_value}"
                
        elif self.param_type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Parameter '{self.name}' must be a boolean"
                
        elif self.param_type == ParameterType.ARRAY:
            if not isinstance(value, list):
                return False, f"Parameter '{self.name}' must be an array"
                
        elif self.param_type == ParameterType.OBJECT:
            if not isinstance(value, dict):
                return False, f"Parameter '{self.name}' must be an object"
                
        elif self.param_type == ParameterType.PATH:
            if not isinstance(value, str):
                return False, f"Parameter '{self.name}' must be a path string"
            # Path normalization will be done by the tool
            
        elif self.param_type == ParameterType.URL:
            if not isinstance(value, str):
                return False, f"Parameter '{self.name}' must be a URL string"
            if not (value.startswith("http://") or value.startswith("https://") or value.startswith("git@")):
                return False, f"Parameter '{self.name}' must be a valid URL"
                
        elif self.param_type == ParameterType.ENUM:
            if self.enum_values and value not in self.enum_values:
                return False, f"Parameter '{self.name}' must be one of: {', '.join(self.enum_values)}"
        
        # Enum validation for any type
        if self.enum_values and value not in self.enum_values:
            return False, f"Parameter '{self.name}' must be one of: {', '.join(map(str, self.enum_values))}"
            
        return True, None


# =============================================================================
# TOOL DEFINITION
# =============================================================================

class ToolDefinition(BaseModel):
    """Complete definition of a tool for LLM discovery and execution.
    
    This schema provides all information needed for an LLM to:
    1. Understand what the tool does
    2. Determine when to use it
    3. Extract required parameters
    4. Validate inputs
    5. Handle results
    """
    # Identity
    name: str = Field(..., description="Unique tool name (snake_case)")
    version: str = Field(default="1.0.0", description="Tool version")
    
    # Description for LLM understanding
    description: str = Field(..., description="Detailed description of what the tool does")
    short_description: str = Field(default="", description="One-line summary")
    
    # Categorization
    category: ToolCategory = Field(..., description="Primary tool category")
    sub_category: Optional[ToolCategory] = Field(default=None, description="Optional sub-category")
    tags: List[str] = Field(default_factory=list, description="Additional tags for discovery")
    
    # Parameters
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    
    # Security
    permission_level: PermissionLevel = Field(default=PermissionLevel.READ_ONLY)
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    requires_consent: bool = Field(default=False, description="Whether user consent is required")
    consent_message: Optional[str] = Field(default=None, description="Message to show for consent")
    
    # Execution hints
    estimated_duration_ms: Optional[int] = Field(default=None, description="Expected execution time")
    supports_streaming: bool = Field(default=False, description="Whether tool supports streaming output")
    supports_cancellation: bool = Field(default=True, description="Whether execution can be cancelled")
    supports_rollback: bool = Field(default=False, description="Whether operation can be undone")
    
    # Dependencies
    dependencies: List[str] = Field(default_factory=list, description="Other tools this depends on")
    conflicts_with: List[str] = Field(default_factory=list, description="Tools that conflict with this")
    
    # Output description
    output_description: str = Field(default="", description="Description of what the tool returns")
    output_type: str = Field(default="object", description="Type of output (string, object, array)")
    
    # Examples for few-shot learning
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Example usages")
    
    class Config:
        use_enum_values = True
    
    def get_required_parameters(self) -> List[ToolParameter]:
        """Get list of required parameters."""
        return [p for p in self.parameters if p.required]
    
    def get_optional_parameters(self) -> List[ToolParameter]:
        """Get list of optional parameters."""
        return [p for p in self.parameters if not p.required]
    
    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }
    
    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic Claude tool format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    
    def to_gemini_function(self) -> Dict[str, Any]:
        """Convert to Google Gemini function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    
    def to_llm_description(self) -> str:
        """Generate a natural language description for LLM understanding.
        
        This creates a detailed description that helps the LLM understand
        when and how to use this tool without keyword matching.
        """
        desc = f"**{self.name}**\n"
        desc += f"{self.description}\n\n"
        
        if self.tags:
            desc += f"Tags: {', '.join(self.tags)}\n"
        
        if self.parameters:
            desc += "\nParameters:\n"
            for param in self.parameters:
                req = "(required)" if param.required else "(optional)"
                desc += f"  - {param.name} [{param.param_type}] {req}: {param.description}"
                if param.default is not None:
                    desc += f" (default: {param.default})"
                if param.examples:
                    desc += f" Examples: {', '.join(param.examples[:3])}"
                desc += "\n"
        
        if self.examples:
            desc += "\nExamples:\n"
            for i, ex in enumerate(self.examples[:3], 1):
                desc += f"  {i}. Input: {ex.get('input', 'N/A')}\n"
                desc += f"     Output: {ex.get('output', 'N/A')}\n"
        
        if self.risk_level != RiskLevel.NONE:
            desc += f"\nRisk Level: {self.risk_level}\n"
        
        return desc


# =============================================================================
# TOOL RESULT
# =============================================================================

@dataclass
class ToolResult:
    """Standardized result of tool execution.
    
    Provides a consistent format for all tool outputs that can be
    processed by the result processor and understood by the LLM.
    
    Fields:
        success: Whether the operation succeeded
        result/data: The result data (both names accepted for convenience)
        message: Optional human-readable message
        error: Error message if failed
        tool_name: Name of the tool (auto-filled by BaseTool)
    """
    success: bool
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    tool_name: str = ""
    message: str = ""  # Human-readable summary
    
    # Execution metadata
    execution_time_ms: float = 0.0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    # Status tracking
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Rollback information
    can_rollback: bool = False
    rollback_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Handle field aliases and defaults."""
        # Set status based on success if not explicitly set
        if not self.success and self.status == ExecutionStatus.COMPLETED:
            self.status = ExecutionStatus.FAILED
    
    @property
    def data(self) -> Any:
        """Alias for result - for convenience."""
        return self.result
    
    @data.setter
    def data(self, value: Any):
        """Set result via data alias."""
        self.result = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "message": self.message,
            "error": self.error,
            "error_type": self.error_type,
            "execution_time_ms": self.execution_time_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "can_rollback": self.can_rollback,
        }
    
    def to_llm_response(self) -> str:
        """Convert to a string suitable for LLM context.
        
        This generates a natural language response that the LLM can use
        to understand what happened and respond to the user.
        """
        if self.success:
            # Start with message if available
            if self.message:
                response = self.message + "\n\n"
            else:
                response = ""
            
            # Add result data
            if self.result is not None:
                if isinstance(self.result, str):
                    response += self.result
                elif isinstance(self.result, dict):
                    response += json.dumps(self.result, indent=2, default=str)
                elif isinstance(self.result, list):
                    response += json.dumps(self.result, indent=2, default=str)
                else:
                    response += str(self.result)
            
            if self.warnings:
                response += f"\n\nWarnings:\n" + "\n".join(f"- {w}" for w in self.warnings)
            
            if self.suggestions:
                response += f"\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in self.suggestions)
                
            return response.strip()
        else:
            error_response = f"Error executing {self.tool_name}: {self.error}"
            if self.error_type:
                error_response = f"[{self.error_type}] {error_response}"
            if self.suggestions:
                error_response += f"\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in self.suggestions)
            return error_response
    
    @classmethod
    def success_result(
        cls,
        tool_name: str,
        result: Any,
        execution_time_ms: float = 0.0,
        **kwargs
    ) -> "ToolResult":
        """Create a successful result."""
        return cls(
            tool_name=tool_name,
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            completed_at=datetime.now().isoformat(),
            status=ExecutionStatus.COMPLETED,
            **kwargs
        )
    
    @classmethod
    def error_result(
        cls,
        tool_name: str,
        error: str,
        error_type: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        **kwargs
    ) -> "ToolResult":
        """Create an error result."""
        return cls(
            tool_name=tool_name,
            success=False,
            error=error,
            error_type=error_type,
            completed_at=datetime.now().isoformat(),
            status=ExecutionStatus.FAILED,
            suggestions=suggestions or [],
            **kwargs
        )


# =============================================================================
# TOOL INTERFACE
# =============================================================================

class ToolInterface(ABC):
    """Abstract base class that all tools must implement.
    
    This interface defines the contract for tool implementation,
    ensuring consistent behavior across all dynamic tools.
    """
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool's definition for discovery and documentation."""
        pass
    
    @abstractmethod
    def execute(
        self,
        parameters: Dict[str, Any],
        context: "ExecutionContext"
    ) -> ToolResult:
        """Execute the tool with given parameters and context.
        
        Args:
            parameters: Validated parameters for the tool
            context: Execution context with system state
            
        Returns:
            ToolResult with execution outcome
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters before execution (synchronous).
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Dict with 'valid' boolean and 'errors' list
        """
        errors = []
        definition = self.definition
        
        # Check required parameters
        for param in definition.parameters:
            if param.required and param.name not in parameters:
                if param.default is None:
                    errors.append(f"Missing required parameter: {param.name}")
            elif param.name in parameters:
                is_valid, error = param.validate_value(parameters[param.name])
                if not is_valid:
                    errors.append(error)
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    async def validate(self, parameters: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate parameters before execution (async wrapper).
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        result = self.validate_parameters(parameters)
        return result["valid"], result["errors"]
    
    async def rollback(
        self,
        result: ToolResult,
        context: "ExecutionContext"
    ) -> ToolResult:
        """Rollback the operation if supported.
        
        Args:
            result: The result of the original execution
            context: Execution context
            
        Returns:
            ToolResult indicating rollback success/failure
        """
        if not self.definition.supports_rollback:
            return ToolResult.error_result(
                tool_name=self.definition.name,
                error="This tool does not support rollback",
                error_type="UnsupportedOperation"
            )
        
        if not result.rollback_data:
            return ToolResult.error_result(
                tool_name=self.definition.name,
                error="No rollback data available",
                error_type="MissingRollbackData"
            )
        
        # Subclasses should override this method
        return ToolResult.error_result(
            tool_name=self.definition.name,
            error="Rollback not implemented",
            error_type="NotImplemented"
        )
    
    async def get_progress(self) -> Optional[float]:
        """Get execution progress (0.0 to 1.0).
        
        Returns:
            Progress percentage or None if not applicable
        """
        return None
    
    async def cancel(self) -> bool:
        """Cancel ongoing execution.
        
        Returns:
            True if cancellation was successful
        """
        return False
    
    def get_consent_message(self, parameters: Dict[str, Any]) -> str:
        """Generate consent message for the operation.
        
        Args:
            parameters: The parameters being used
            
        Returns:
            Human-readable consent message
        """
        if self.definition.consent_message:
            return self.definition.consent_message
        
        return f"This operation ({self.definition.name}) requires your approval."


class BaseTool(ToolInterface):
    """Base implementation of ToolInterface with common functionality.
    
    Provides default implementations and utilities for tool development.
    Subclasses should implement _execute() and define tool metadata via properties.
    
    Required properties to override:
    - name: str - Unique tool name
    - description: str - What the tool does
    - category: ToolCategory - Primary category
    - parameters: List[ToolParameter] - Tool parameters
    
    Optional properties:
    - required_permission: PermissionLevel (default: READ_ONLY)
    - risk_level: RiskLevel (default: LOW)
    - tags: List[str] - Additional tags
    - examples: List[str] - Example usages
    - sub_category: ToolCategory - Sub-category
    """
    
    _is_executing: bool = False
    _is_cancelled: bool = False
    _progress: float = 0.0
    _cached_definition: Optional[ToolDefinition] = None
    
    def __init__(self):
        """Initialize the tool."""
        self._is_executing = False
        self._is_cancelled = False
        self._progress = 0.0
        self._cached_definition = None
    
    # =========================================================================
    # Properties to override in subclasses
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Unique tool name (snake_case)."""
        raise NotImplementedError("Subclasses must define 'name' property")
    
    @property
    def description(self) -> str:
        """Detailed description of what the tool does."""
        raise NotImplementedError("Subclasses must define 'description' property")
    
    @property
    def category(self) -> ToolCategory:
        """Primary tool category."""
        raise NotImplementedError("Subclasses must define 'category' property")
    
    @property
    def parameters(self) -> List[ToolParameter]:
        """Tool parameters."""
        return []
    
    @property
    def required_permission(self) -> PermissionLevel:
        """Required permission level."""
        return PermissionLevel.READ_ONLY
    
    @property
    def risk_level(self) -> RiskLevel:
        """Operation risk level."""
        return RiskLevel.LOW
    
    @property
    def tags(self) -> List[str]:
        """Additional tags for discovery."""
        return []
    
    @property
    def examples(self) -> List[str]:
        """Example usages for the LLM."""
        return []
    
    @property
    def sub_category(self) -> Optional[ToolCategory]:
        """Optional sub-category."""
        return None
    
    # =========================================================================
    # ToolInterface implementation
    # =========================================================================
    
    @property
    def definition(self) -> ToolDefinition:
        """Build and return the tool definition from properties."""
        if self._cached_definition is None:
            self._cached_definition = ToolDefinition(
                name=self.name,
                description=self.description,
                category=self.category,
                parameters=self.parameters,
                permission_level=self.required_permission,
                risk_level=self.risk_level,
                tags=self.tags,
                examples=[{"description": ex} for ex in self.examples] if self.examples else [],
            )
        return self._cached_definition
    
    def get_definition(self) -> ToolDefinition:
        """Get the tool definition (for registry)."""
        return self.definition
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters synchronously.
        
        Returns:
            Dict with 'valid' boolean and 'errors' list
        """
        errors = []
        
        for param in self.parameters:
            if param.required and param.name not in parameters:
                if param.default is None:
                    errors.append(f"Missing required parameter: {param.name}")
            elif param.name in parameters:
                is_valid, error = param.validate_value(parameters[param.name])
                if not is_valid:
                    errors.append(error)
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def execute(
        self,
        parameters: Dict[str, Any],
        context: "ExecutionContext"
    ) -> ToolResult:
        """Execute the tool synchronously.
        
        This is the main entry point for tool execution.
        Subclasses should override _execute() for their implementation.
        """
        self._is_executing = True
        self._is_cancelled = False
        self._progress = 0.0
        
        start_time = time.time()
        
        try:
            # Validate parameters first
            validation = self.validate_parameters(parameters)
            if not validation["valid"]:
                return ToolResult.error_result(
                    tool_name=self.name,
                    error="; ".join(validation["errors"]),
                    error_type="ValidationError"
                )
            
            # Call the implementation
            result = self._execute(parameters, context)
            
            # Ensure tool_name is set
            if not result.tool_name:
                result.tool_name = self.name
            
            execution_time = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time
            result.completed_at = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            return ToolResult.error_result(
                tool_name=self.name,
                error=str(e),
                error_type=type(e).__name__
            )
        finally:
            self._is_executing = False
    
    async def execute_async(
        self,
        parameters: Dict[str, Any],
        context: "ExecutionContext"
    ) -> ToolResult:
        """Execute the tool asynchronously.
        
        Wraps synchronous execute for async contexts.
        """
        return self.execute(parameters, context)
    
    def _execute(
        self,
        parameters: Dict[str, Any],
        context: "ExecutionContext"
    ) -> ToolResult:
        """Implementation method for subclasses.
        
        Subclasses MUST override this method to provide their functionality.
        """
        raise NotImplementedError("Subclasses must implement _execute()")
    
    async def validate(self, parameters: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Async validation wrapper."""
        result = self.validate_parameters(parameters)
        return result["valid"], result["errors"]
    
    async def get_progress(self) -> Optional[float]:
        """Get current progress."""
        return self._progress if self._is_executing else None
    
    async def cancel(self) -> bool:
        """Request cancellation of the operation."""
        if self._is_executing and self.definition.supports_cancellation:
            self._is_cancelled = True
            return True
        return False
    
    def _set_progress(self, progress: float):
        """Update progress (0.0 to 1.0)."""
        self._progress = max(0.0, min(1.0, progress))
    
    def _check_cancelled(self) -> bool:
        """Check if operation was cancelled."""
        return self._is_cancelled
    
    def _normalize_path(self, path: str) -> Path:
        """Normalize a file path."""
        import os
        expanded = os.path.expanduser(os.path.expandvars(path))
        return Path(expanded).resolve()
    
    def _safe_json_serialize(self, obj: Any) -> Any:
        """Safely serialize an object to JSON-compatible format."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._safe_json_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._safe_json_serialize(i) for i in obj]
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return str(obj)


# Type alias for execution context (defined in execution_context.py)
ExecutionContext = Any  # Will be properly typed when imported
