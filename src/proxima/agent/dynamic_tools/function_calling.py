"""Function Calling Integration for LLM Providers.

This module provides function/tool calling support across different LLM providers,
enabling native function calling for supported models and fallback for others.

Phase 2.1.2: Function Calling Integration
=========================================
- Native function calling for OpenAI, Claude, Gemini
- Function definition auto-generation from tool registry
- Function call execution pipeline with parameter mapping
- Parallel function calling support
- Function call chaining for multi-step operations
- Debugging and logging system

Key Features:
------------
- Provider-agnostic function calling abstraction
- Auto-generation of function schemas from tools
- Execution pipeline with validation
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Type, Union

from .tool_interface import ToolDefinition, ToolParameter, ParameterType, ToolResult
from .tool_registry import ToolRegistry, RegisteredTool, get_tool_registry
from .execution_context import ExecutionContext, get_current_context
from .structured_output import StructuredOutputParser, ParseResult, get_structured_output_parser

logger = logging.getLogger(__name__)


class FunctionCallFormat(Enum):
    """Supported function calling formats."""
    OPENAI = "openai"          # OpenAI function/tool calling
    ANTHROPIC = "anthropic"    # Anthropic tool_use
    GOOGLE = "google"          # Google/Gemini function declarations
    OLLAMA = "ollama"          # Ollama (OpenAI-compatible)
    TEXT = "text"              # Text-based fallback


@dataclass
class FunctionDefinition:
    """A function definition for LLM function calling."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format
    required_params: List[str]
    source_tool: Optional[str] = None
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_params,
                },
            },
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params,
            },
        }
    
    def to_google_format(self) -> Dict[str, Any]:
        """Convert to Google/Gemini function declaration."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params,
            },
        }
    
    def to_text_format(self) -> str:
        """Convert to text description for non-native function calling."""
        lines = [
            f"### {self.name}",
            f"Description: {self.description}",
            "Parameters:",
        ]
        
        for param_name, param_schema in self.parameters.items():
            required = "(required)" if param_name in self.required_params else "(optional)"
            param_type = param_schema.get("type", "any")
            param_desc = param_schema.get("description", "")
            lines.append(f"  - {param_name} {required}: {param_type} - {param_desc}")
        
        return "\n".join(lines)


@dataclass
class FunctionCall:
    """A function call from an LLM response."""
    id: str
    name: str
    arguments: Dict[str, Any]
    raw_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class FunctionCallResult:
    """Result of executing a function call."""
    call_id: str
    function_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "function_name": self.function_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }
    
    def to_openai_response(self) -> Dict[str, Any]:
        """Format for OpenAI tool response."""
        content = json.dumps(self.result) if self.success else self.error
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "content": content or "",
        }
    
    def to_anthropic_response(self) -> Dict[str, Any]:
        """Format for Anthropic tool_result."""
        return {
            "type": "tool_result",
            "tool_use_id": self.call_id,
            "content": json.dumps(self.result) if self.success else (self.error or ""),
            "is_error": not self.success,
        }
    
    def to_google_response(self) -> Dict[str, Any]:
        """Format for Google/Gemini function response."""
        return {
            "functionResponse": {
                "name": self.function_name,
                "response": {
                    "result": self.result if self.success else None,
                    "error": self.error if not self.success else None,
                },
            },
        }


class FunctionDefinitionGenerator:
    """Generates function definitions from tool registry."""
    
    def __init__(self, registry: Optional[ToolRegistry] = None):
        """Initialize the generator.
        
        Args:
            registry: Tool registry to generate from
        """
        self._registry = registry or get_tool_registry()
        self._type_mapping = {
            ParameterType.STRING: "string",
            ParameterType.INTEGER: "integer",
            ParameterType.FLOAT: "number",
            ParameterType.BOOLEAN: "boolean",
            ParameterType.PATH: "string",
            ParameterType.URL: "string",
            ParameterType.ARRAY: "array",
            ParameterType.OBJECT: "object",
            ParameterType.BRANCH_NAME: "string",
            ParameterType.COMMIT_HASH: "string",
            ParameterType.ENUM: "string",
        }
    
    def generate_all(self) -> List[FunctionDefinition]:
        """Generate function definitions for all registered tools.
        
        Returns:
            List of function definitions
        """
        definitions = []
        
        for registered in self._registry.get_all_tools():
            definition = self.generate_for_tool(registered)
            if definition:
                definitions.append(definition)
        
        return definitions
    
    def generate_for_tool(
        self,
        registered: RegisteredTool,
    ) -> Optional[FunctionDefinition]:
        """Generate a function definition for a specific tool.
        
        Args:
            registered: The registered tool
            
        Returns:
            Function definition or None if generation fails
        """
        try:
            defn = registered.definition
            
            # Convert parameters to JSON Schema format
            properties: Dict[str, Any] = {}
            required: List[str] = []
            
            for param in defn.parameters:
                param_schema = self._parameter_to_schema(param)
                properties[param.name] = param_schema
                
                if param.required:
                    required.append(param.name)
            
            return FunctionDefinition(
                name=defn.name,
                description=defn.description,
                parameters=properties,
                required_params=required,
                source_tool=defn.name,
            )
            
        except Exception as e:
            logger.error(f"Failed to generate function definition for {registered.definition.name}: {e}")
            return None
    
    def _parameter_to_schema(self, param: ToolParameter) -> Dict[str, Any]:
        """Convert a tool parameter to JSON Schema."""
        schema: Dict[str, Any] = {
            "type": self._type_mapping.get(param.param_type, "string"),
            "description": param.description,
        }
        
        # Add constraints
        if param.default is not None:
            schema["default"] = param.default
        
        if param.enum_values:
            schema["enum"] = param.enum_values
        
        # Add format hints for special types
        if param.param_type == ParameterType.PATH:
            schema["format"] = "path"
        elif param.param_type == ParameterType.URL:
            schema["format"] = "uri"
        
        return schema


class FunctionCallParser:
    """Parses function calls from LLM responses."""
    
    def __init__(self):
        """Initialize the parser."""
        self._output_parser = get_structured_output_parser()
        self._call_counter = 0
    
    def parse(
        self,
        response: Any,
        format: FunctionCallFormat,
    ) -> List[FunctionCall]:
        """Parse function calls from an LLM response.
        
        Args:
            response: The raw LLM response
            format: The expected function calling format
            
        Returns:
            List of parsed function calls
        """
        if format == FunctionCallFormat.OPENAI:
            return self._parse_openai(response)
        elif format == FunctionCallFormat.ANTHROPIC:
            return self._parse_anthropic(response)
        elif format == FunctionCallFormat.GOOGLE:
            return self._parse_google(response)
        elif format == FunctionCallFormat.OLLAMA:
            return self._parse_openai(response)  # Ollama uses OpenAI format
        elif format == FunctionCallFormat.TEXT:
            return self._parse_text(response)
        else:
            logger.warning(f"Unknown format: {format}")
            return []
    
    def _parse_openai(self, response: Any) -> List[FunctionCall]:
        """Parse OpenAI function/tool calls."""
        calls: List[FunctionCall] = []
        
        if isinstance(response, dict):
            # Handle message format
            message = response.get("choices", [{}])[0].get("message", response)
            tool_calls = message.get("tool_calls", [])
            
            for tc in tool_calls:
                if tc.get("type") == "function":
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {"_raw": args_str}
                    
                    calls.append(FunctionCall(
                        id=tc.get("id", self._generate_id()),
                        name=func.get("name", ""),
                        arguments=args,
                        raw_data=tc,
                    ))
            
            # Also check legacy function_call format
            if not calls and "function_call" in message:
                fc = message["function_call"]
                args_str = fc.get("arguments", "{}")
                
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {"_raw": args_str}
                
                calls.append(FunctionCall(
                    id=self._generate_id(),
                    name=fc.get("name", ""),
                    arguments=args,
                    raw_data=fc,
                ))
        
        return calls
    
    def _parse_anthropic(self, response: Any) -> List[FunctionCall]:
        """Parse Anthropic tool_use blocks."""
        calls: List[FunctionCall] = []
        
        if isinstance(response, dict):
            content = response.get("content", [])
            
            for block in content:
                if block.get("type") == "tool_use":
                    calls.append(FunctionCall(
                        id=block.get("id", self._generate_id()),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                        raw_data=block,
                    ))
        
        return calls
    
    def _parse_google(self, response: Any) -> List[FunctionCall]:
        """Parse Google/Gemini function calls."""
        calls: List[FunctionCall] = []
        
        if isinstance(response, dict):
            candidates = response.get("candidates", [])
            
            for candidate in candidates:
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                
                for part in parts:
                    fc = part.get("functionCall")
                    if fc:
                        calls.append(FunctionCall(
                            id=fc.get("id", self._generate_id()),
                            name=fc.get("name", ""),
                            arguments=fc.get("args", {}),
                            raw_data=fc,
                        ))
        
        return calls
    
    def _parse_text(self, response: Any) -> List[FunctionCall]:
        """Parse text-based function calls.
        
        Looks for function calls in format:
        ```tool_call
        {"tool": "tool_name", "arguments": {...}}
        ```
        """
        calls: List[FunctionCall] = []
        text = str(response)
        
        # Find tool_call blocks
        import re
        pattern = r"```tool_call\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match.strip())
                calls.append(FunctionCall(
                    id=self._generate_id(),
                    name=data.get("tool", data.get("function", "")),
                    arguments=data.get("arguments", data.get("args", {})),
                    raw_data=data,
                ))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse text function call: {match[:100]}")
        
        return calls
    
    def _generate_id(self) -> str:
        """Generate a unique call ID."""
        self._call_counter += 1
        return f"call_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._call_counter}"


@dataclass
class FunctionCallingConfig:
    """Configuration for function calling."""
    # Execution settings
    allow_parallel_calls: bool = True
    max_parallel_calls: int = 5
    validate_arguments: bool = True
    
    # Retry settings
    retry_on_error: bool = True
    max_retries: int = 2
    
    # Logging
    log_all_calls: bool = True
    log_arguments: bool = True


class FunctionCallExecutor:
    """Executes function calls against the tool registry.
    
    This executor:
    1. Validates function arguments
    2. Maps to registered tools
    3. Executes with proper context
    4. Formats results for LLM consumption
    """
    
    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        config: Optional[FunctionCallingConfig] = None,
    ):
        """Initialize the executor.
        
        Args:
            registry: Tool registry
            config: Execution configuration
        """
        self._registry = registry or get_tool_registry()
        self._config = config or FunctionCallingConfig()
        self._execution_log: List[Dict[str, Any]] = []
    
    def execute(
        self,
        call: FunctionCall,
        context: Optional[ExecutionContext] = None,
    ) -> FunctionCallResult:
        """Execute a single function call.
        
        Args:
            call: The function call to execute
            context: Execution context
            
        Returns:
            Function call result
        """
        context = context or get_current_context()
        start_time = datetime.now()
        
        # Log the call
        if self._config.log_all_calls:
            self._log_call(call, "started")
        
        # Find the tool
        registered = self._registry.get_tool(call.name)
        if not registered:
            return FunctionCallResult(
                call_id=call.id,
                function_name=call.name,
                success=False,
                result=None,
                error=f"Unknown function: {call.name}",
            )
        
        # Validate arguments
        if self._config.validate_arguments:
            validation_error = self._validate_arguments(
                call.arguments,
                registered.definition.parameters,
            )
            if validation_error:
                return FunctionCallResult(
                    call_id=call.id,
                    function_name=call.name,
                    success=False,
                    result=None,
                    error=validation_error,
                )
        
        # Execute the tool
        try:
            tool_instance = registered.get_instance()
            result = tool_instance.execute(call.arguments, context)
            
            end_time = datetime.now()
            execution_ms = (end_time - start_time).total_seconds() * 1000
            
            # Log success
            if self._config.log_all_calls:
                self._log_call(call, "completed", execution_ms)
            
            return FunctionCallResult(
                call_id=call.id,
                function_name=call.name,
                success=result.success,
                result=result.result if result.success else None,
                error=result.error if not result.success else None,
                execution_time_ms=execution_ms,
            )
            
        except Exception as e:
            end_time = datetime.now()
            execution_ms = (end_time - start_time).total_seconds() * 1000
            
            # Log error
            if self._config.log_all_calls:
                self._log_call(call, "failed", execution_ms, str(e))
            
            return FunctionCallResult(
                call_id=call.id,
                function_name=call.name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=execution_ms,
            )
    
    def execute_batch(
        self,
        calls: List[FunctionCall],
        context: Optional[ExecutionContext] = None,
    ) -> List[FunctionCallResult]:
        """Execute multiple function calls.
        
        If parallel execution is enabled and calls are independent,
        they may be executed concurrently.
        
        Args:
            calls: List of function calls
            context: Execution context
            
        Returns:
            List of results (in same order as calls)
        """
        context = context or get_current_context()
        results: List[FunctionCallResult] = []
        
        if self._config.allow_parallel_calls and len(calls) > 1:
            # For now, execute sequentially but could add async/threading
            # Parallel execution would require careful context handling
            for call in calls:
                result = self.execute(call, context)
                results.append(result)
        else:
            for call in calls:
                result = self.execute(call, context)
                results.append(result)
        
        return results
    
    def _validate_arguments(
        self,
        arguments: Dict[str, Any],
        parameters: List[ToolParameter],
    ) -> Optional[str]:
        """Validate function call arguments.
        
        Returns:
            Error message if validation fails, None otherwise
        """
        param_map = {p.name: p for p in parameters}
        
        # Check required parameters
        for param in parameters:
            if param.required and param.name not in arguments:
                return f"Missing required parameter: {param.name}"
        
        # Validate provided arguments
        for arg_name, arg_value in arguments.items():
            if arg_name not in param_map:
                continue  # Allow extra arguments
            
            param = param_map[arg_name]
            
            # Basic type validation
            if param.param_type == ParameterType.INTEGER:
                if not isinstance(arg_value, int):
                    try:
                        int(arg_value)
                    except (ValueError, TypeError):
                        return f"Parameter {arg_name} must be an integer"
            
            elif param.param_type == ParameterType.BOOLEAN:
                if not isinstance(arg_value, bool):
                    if str(arg_value).lower() not in ("true", "false", "1", "0"):
                        return f"Parameter {arg_name} must be a boolean"
            
            # Enum validation
            if param.enum_values and arg_value not in param.enum_values:
                return f"Parameter {arg_name} must be one of: {param.enum_values}"
        
        return None
    
    def _log_call(
        self,
        call: FunctionCall,
        status: str,
        execution_ms: float = 0,
        error: Optional[str] = None,
    ):
        """Log a function call."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "call_id": call.id,
            "function": call.name,
            "status": status,
            "execution_ms": execution_ms,
        }
        
        if self._config.log_arguments:
            log_entry["arguments"] = call.arguments
        
        if error:
            log_entry["error"] = error
        
        self._execution_log.append(log_entry)
        logger.debug(f"Function call: {call.name} - {status}")
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the execution log."""
        return list(self._execution_log)
    
    def clear_execution_log(self):
        """Clear the execution log."""
        self._execution_log.clear()


class FunctionCallingIntegration:
    """High-level integration for function calling with LLMs.
    
    This class provides a unified interface for:
    1. Generating function definitions from tools
    2. Parsing function calls from responses
    3. Executing function calls
    4. Formatting results for LLMs
    """
    
    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        config: Optional[FunctionCallingConfig] = None,
    ):
        """Initialize the integration.
        
        Args:
            registry: Tool registry
            config: Configuration
        """
        self._registry = registry or get_tool_registry()
        self._config = config or FunctionCallingConfig()
        
        self._generator = FunctionDefinitionGenerator(self._registry)
        self._parser = FunctionCallParser()
        self._executor = FunctionCallExecutor(self._registry, self._config)
        
        # Cached function definitions
        self._definitions: Optional[List[FunctionDefinition]] = None
    
    def get_functions(
        self,
        format: FunctionCallFormat = FunctionCallFormat.OPENAI,
        filter_tools: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get function definitions for LLM.
        
        Args:
            format: Output format
            filter_tools: Optional list of tool names to include
            
        Returns:
            List of function definitions in the specified format
        """
        # Generate definitions if not cached
        if self._definitions is None:
            self._definitions = self._generator.generate_all()
        
        # Filter if specified
        definitions = self._definitions
        if filter_tools:
            definitions = [d for d in definitions if d.source_tool in filter_tools]
        
        # Convert to format
        if format == FunctionCallFormat.OPENAI or format == FunctionCallFormat.OLLAMA:
            return [d.to_openai_format() for d in definitions]
        elif format == FunctionCallFormat.ANTHROPIC:
            return [d.to_anthropic_format() for d in definitions]
        elif format == FunctionCallFormat.GOOGLE:
            return [{"function_declarations": [d.to_google_format() for d in definitions]}]
        elif format == FunctionCallFormat.TEXT:
            # Return as a single text block
            text = "\n\n".join(d.to_text_format() for d in definitions)
            return [{"text": text}]
        
        return []
    
    def process_response(
        self,
        response: Any,
        format: FunctionCallFormat,
        context: Optional[ExecutionContext] = None,
    ) -> tuple[List[FunctionCall], List[FunctionCallResult]]:
        """Process an LLM response containing function calls.
        
        Args:
            response: The LLM response
            format: The response format
            context: Execution context
            
        Returns:
            Tuple of (parsed calls, execution results)
        """
        # Parse function calls
        calls = self._parser.parse(response, format)
        
        if not calls:
            return [], []
        
        # Execute calls
        results = self._executor.execute_batch(calls, context)
        
        return calls, results
    
    def format_results(
        self,
        results: List[FunctionCallResult],
        format: FunctionCallFormat,
    ) -> Any:
        """Format execution results for LLM.
        
        Args:
            results: Execution results
            format: Output format
            
        Returns:
            Formatted results for LLM consumption
        """
        if format == FunctionCallFormat.OPENAI or format == FunctionCallFormat.OLLAMA:
            return [r.to_openai_response() for r in results]
        elif format == FunctionCallFormat.ANTHROPIC:
            return [r.to_anthropic_response() for r in results]
        elif format == FunctionCallFormat.GOOGLE:
            return {"parts": [r.to_google_response() for r in results]}
        else:
            # Text format
            lines = ["# Function Results", ""]
            for r in results:
                status = "✓" if r.success else "✗"
                lines.append(f"## {status} {r.function_name}")
                if r.success:
                    lines.append(f"```json\n{json.dumps(r.result, indent=2)}\n```")
                else:
                    lines.append(f"Error: {r.error}")
                lines.append("")
            return "\n".join(lines)
    
    def refresh_definitions(self):
        """Refresh cached function definitions."""
        self._definitions = None


# Module-level integration instance
_global_function_calling: Optional[FunctionCallingIntegration] = None


def get_function_calling_integration(
    registry: Optional[ToolRegistry] = None,
    config: Optional[FunctionCallingConfig] = None,
) -> FunctionCallingIntegration:
    """Get the global function calling integration.
    
    Args:
        registry: Optional tool registry
        config: Optional configuration
        
    Returns:
        The function calling integration instance
    """
    global _global_function_calling
    if _global_function_calling is None:
        _global_function_calling = FunctionCallingIntegration(registry, config)
    return _global_function_calling


def configure_function_calling(
    config: FunctionCallingConfig,
    registry: Optional[ToolRegistry] = None,
):
    """Configure the global function calling integration.
    
    Args:
        config: Configuration
        registry: Optional tool registry
    """
    global _global_function_calling
    _global_function_calling = FunctionCallingIntegration(registry, config)
