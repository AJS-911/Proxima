"""LLM Integration Layer for Dynamic Tool System.

This module provides the integration layer between LLMs and the tool system.
It enables any compatible LLM (Ollama/Gemini/OpenAI/Anthropic/etc.) to:
- Discover available tools dynamically
- Understand tool capabilities through schemas
- Select appropriate tools based on reasoning
- Execute tools with proper parameter handling

The integration is provider-agnostic and supports:
- OpenAI function calling format
- Anthropic tool use format  
- Google/Gemini function declarations
- Generic prompt-based tool selection

Key Features:
- NO hardcoded keyword matching
- Tools self-describe their capabilities
- LLM uses reasoning to select tools
- Supports multi-turn tool conversations
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .tool_interface import ToolDefinition, ToolResult, RiskLevel
from .tool_registry import ToolRegistry, get_tool_registry
from .execution_context import ExecutionContext, get_current_context
from .tool_orchestrator import ToolOrchestrator, get_tool_orchestrator
from .result_processor import ResultProcessor, ProcessedResult, get_result_processor

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    AZURE = "azure"
    COHERE = "cohere"
    GENERIC = "generic"


@dataclass
class ToolCall:
    """Represents a tool call from an LLM."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str = ""
    raw_response: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }


@dataclass
class ToolCallResult:
    """Result of a tool call to send back to LLM."""
    call_id: str
    tool_name: str
    result: ProcessedResult
    formatted_for_llm: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "result": self.result.to_dict(),
            "formatted_for_llm": self.formatted_for_llm,
        }


@dataclass
class LLMToolConfig:
    """Configuration for LLM tool integration."""
    provider: LLMProvider
    max_tools_per_request: int = 20
    include_examples: bool = True
    max_risk_level: RiskLevel = RiskLevel.HIGH
    require_confirmation_above_risk: RiskLevel = RiskLevel.MEDIUM
    allow_parallel_tool_calls: bool = True
    system_prompt_tools_section: bool = True
    structured_output_format: bool = True


class LLMToolIntegration:
    """Integrates tools with LLM providers.
    
    This class provides the bridge between the tool system and LLMs,
    enabling dynamic tool discovery and execution based on LLM reasoning.
    
    The integration does NOT use keyword matching. Instead:
    1. Tools are presented to the LLM with full descriptions
    2. LLM uses reasoning to determine which tool(s) to use
    3. LLM generates structured tool calls
    4. Integration executes tools and returns results
    5. LLM continues reasoning with results
    """
    
    def __init__(
        self,
        config: Optional[LLMToolConfig] = None,
        registry: Optional[ToolRegistry] = None,
        orchestrator: Optional[ToolOrchestrator] = None,
        processor: Optional[ResultProcessor] = None,
    ):
        """Initialize the LLM tool integration.
        
        Args:
            config: Configuration for the integration
            registry: Tool registry to use
            orchestrator: Tool orchestrator for execution
            processor: Result processor for formatting
        """
        self.config = config or LLMToolConfig(provider=LLMProvider.GENERIC)
        self._registry = registry or get_tool_registry()
        self._orchestrator = orchestrator or get_tool_orchestrator()
        self._processor = processor or get_result_processor()
        
        # Confirmation callback for risky operations
        self._confirmation_callback: Optional[Callable[[ToolDefinition, Dict[str, Any]], bool]] = None
    
    def set_confirmation_callback(
        self, 
        callback: Callable[[ToolDefinition, Dict[str, Any]], bool]
    ):
        """Set callback for confirming risky operations.
        
        Args:
            callback: Function that receives (tool_definition, parameters)
                     and returns True if operation should proceed
        """
        self._confirmation_callback = callback
    
    def get_tools_for_llm(
        self,
        categories: Optional[List[str]] = None,
        specific_tools: Optional[List[str]] = None,
    ) -> Union[List[Dict[str, Any]], str]:
        """Get tools formatted for the configured LLM provider.
        
        This method returns tool definitions in the format expected
        by the configured LLM provider's function/tool calling API.
        
        Args:
            categories: Optional list of categories to include
            specific_tools: Optional list of specific tool names
            
        Returns:
            For structured providers: List of tool definitions
            For generic providers: String description of tools
        """
        tools = []
        
        # Get filtered tools
        for registered in self._registry.get_all_tools():
            defn = registered.definition
            
            # Apply category filter
            if categories:
                if defn.category.value not in categories:
                    continue
            
            # Apply specific tools filter
            if specific_tools:
                if defn.name not in specific_tools:
                    continue
            
            # Apply risk filter
            risk_order = list(RiskLevel)
            if risk_order.index(defn.risk_level) > risk_order.index(self.config.max_risk_level):
                continue
            
            tools.append(registered)
        
        # Limit tools
        tools = tools[:self.config.max_tools_per_request]
        
        # Format based on provider
        if self.config.provider == LLMProvider.OPENAI:
            return self._format_openai_tools(tools)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return self._format_anthropic_tools(tools)
        elif self.config.provider == LLMProvider.GOOGLE:
            return self._format_google_tools(tools)
        elif self.config.provider == LLMProvider.OLLAMA:
            # Ollama can use either OpenAI format or text description
            return self._format_openai_tools(tools)
        else:
            return self._format_text_tools(tools)
    
    def _format_openai_tools(self, tools) -> List[Dict[str, Any]]:
        """Format tools for OpenAI function calling."""
        return [
            {"type": "function", "function": t.definition.to_openai_function()}
            for t in tools
        ]
    
    def _format_anthropic_tools(self, tools) -> List[Dict[str, Any]]:
        """Format tools for Anthropic tool use."""
        return [t.definition.to_anthropic_tool() for t in tools]
    
    def _format_google_tools(self, tools) -> List[Dict[str, Any]]:
        """Format tools for Google/Gemini function declarations."""
        return [{
            "function_declarations": [t.definition.to_gemini_function() for t in tools]
        }]
    
    def _format_text_tools(self, tools) -> str:
        """Format tools as text for generic LLMs.
        
        This format enables LLMs without structured function calling
        to understand and use tools through a text-based protocol.
        """
        lines = [
            "# Available Tools",
            "",
            "You have access to the following tools. To use a tool, respond with:",
            "```tool_call",
            '{"tool": "tool_name", "arguments": {"param1": "value1", ...}}',
            "```",
            "",
            "## Tools:",
            "",
        ]
        
        for registered in tools:
            defn = registered.definition
            lines.append(defn.to_llm_description())
            lines.append("")
        
        return "\n".join(lines)
    
    def get_tools_for_provider(
        self,
        provider: LLMProvider,
        categories: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get tools formatted for a specific LLM provider.
        
        Convenience method that creates a new config for the provider
        and returns formatted tools.
        
        Args:
            provider: The LLM provider to format tools for
            categories: Optional list of categories to include
            
        Returns:
            List of tool definitions in the provider's format
        """
        original_provider = self.config.provider
        self.config.provider = provider
        try:
            result = self.get_tools_for_llm(categories=categories)
            # Ensure we return a list
            if isinstance(result, str):
                return []  # Text format doesn't return a list
            return result
        finally:
            self.config.provider = original_provider
    
    def get_system_prompt_section(self) -> str:
        """Generate the tools section for a system prompt.
        
        Returns a natural language description of available tools
        that helps the LLM understand what capabilities are available.
        
        Returns:
            System prompt section describing tools
        """
        lines = [
            "## Tool Capabilities",
            "",
            "You have access to tools for interacting with the system. "
            "Analyze the user's request and determine which tools, if any, "
            "would help accomplish the task.",
            "",
            "When deciding which tool to use:",
            "1. Consider what the user is trying to accomplish",
            "2. Review the available tools and their capabilities",
            "3. Select the most appropriate tool(s) for the task",
            "4. Provide the required parameters based on context",
            "",
            "Available tool categories:",
        ]
        
        # Summarize by category
        categories: Dict[str, List[str]] = {}
        for registered in self._registry.get_all_tools():
            cat = registered.definition.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(registered.definition.name)
        
        for cat, tool_names in categories.items():
            lines.append(f"- **{cat}**: {', '.join(tool_names[:5])}")
            if len(tool_names) > 5:
                lines.append(f"  (and {len(tool_names) - 5} more)")
        
        lines.append("")
        lines.append("Use tools thoughtfully - not every request requires a tool.")
        
        return "\n".join(lines)
    
    def parse_tool_calls(
        self,
        llm_response: Any,
        provider: Optional[LLMProvider] = None,
    ) -> List[ToolCall]:
        """Parse tool calls from an LLM response.
        
        This method extracts structured tool calls from various
        LLM response formats.
        
        Args:
            llm_response: The raw LLM response
            provider: Provider override (uses config if not specified)
            
        Returns:
            List of parsed tool calls
        """
        provider = provider or self.config.provider
        
        try:
            if provider == LLMProvider.OPENAI:
                return self._parse_openai_response(llm_response)
            elif provider == LLMProvider.ANTHROPIC:
                return self._parse_anthropic_response(llm_response)
            elif provider == LLMProvider.GOOGLE:
                return self._parse_google_response(llm_response)
            elif provider == LLMProvider.OLLAMA:
                # Ollama typically uses OpenAI format
                return self._parse_openai_response(llm_response)
            else:
                return self._parse_text_response(llm_response)
        
        except Exception as e:
            logger.error(f"Failed to parse tool calls: {e}")
            return []
    
    def _parse_openai_response(self, response: Any) -> List[ToolCall]:
        """Parse OpenAI format tool calls."""
        calls = []
        
        # Handle various OpenAI response structures
        if isinstance(response, dict):
            # Direct message with tool_calls
            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                # Check in choices
                choices = response.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    tool_calls = message.get("tool_calls", [])
            
            for tc in tool_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}
                
                calls.append(ToolCall(
                    tool_name=func.get("name", ""),
                    arguments=args,
                    call_id=tc.get("id", ""),
                    raw_response=tc,
                ))
        
        return calls
    
    def _parse_anthropic_response(self, response: Any) -> List[ToolCall]:
        """Parse Anthropic format tool use."""
        calls = []
        
        if isinstance(response, dict):
            content = response.get("content", [])
            for block in content:
                if block.get("type") == "tool_use":
                    calls.append(ToolCall(
                        tool_name=block.get("name", ""),
                        arguments=block.get("input", {}),
                        call_id=block.get("id", ""),
                        raw_response=block,
                    ))
        
        return calls
    
    def _parse_google_response(self, response: Any) -> List[ToolCall]:
        """Parse Google/Gemini format function calls."""
        calls = []
        
        if isinstance(response, dict):
            candidates = response.get("candidates", [])
            for candidate in candidates:
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    fc = part.get("functionCall")
                    if fc:
                        calls.append(ToolCall(
                            tool_name=fc.get("name", ""),
                            arguments=fc.get("args", {}),
                            call_id=fc.get("id", f"call_{len(calls)}"),
                            raw_response=fc,
                        ))
        
        return calls
    
    def _parse_text_response(self, response: Any) -> List[ToolCall]:
        """Parse text-based tool calls.
        
        Looks for tool calls in the format:
        ```tool_call
        {"tool": "tool_name", "arguments": {...}}
        ```
        """
        calls = []
        
        text = str(response)
        
        # Find tool_call blocks
        import re
        pattern = r"```tool_call\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match.strip())
                calls.append(ToolCall(
                    tool_name=data.get("tool", ""),
                    arguments=data.get("arguments", {}),
                    call_id=f"text_call_{len(calls)}",
                    raw_response=data,
                ))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {match}")
        
        return calls
    
    def execute_tool_call(
        self,
        call: ToolCall,
        context: Optional[ExecutionContext] = None,
    ) -> ToolCallResult:
        """Execute a single tool call.
        
        Args:
            call: The tool call to execute
            context: Execution context
            
        Returns:
            Tool call result formatted for LLM
        """
        context = context or get_current_context()
        start_time = datetime.now()
        
        # Get tool definition for confirmation check
        registered = self._registry.get_tool(call.tool_name)
        if not registered:
            error_result = ProcessedResult(
                tool_name=call.tool_name,
                operation_id=call.call_id,
                success=False,
                severity=ResultSeverity.ERROR,
                result_type=ResultType.ERROR,
                content=f"Tool not found: {call.tool_name}",
                summary=f"Unknown tool: {call.tool_name}",
                llm_context=f"Error: Tool '{call.tool_name}' is not available.",
            )
            return ToolCallResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                result=error_result,
                formatted_for_llm=error_result.llm_context,
            )
        
        # Check if confirmation needed
        risk_order = list(RiskLevel)
        defn = registered.definition
        
        if (self._confirmation_callback and 
            risk_order.index(defn.risk_level) > risk_order.index(self.config.require_confirmation_above_risk)):
            if not self._confirmation_callback(defn, call.arguments):
                cancelled_result = ProcessedResult(
                    tool_name=call.tool_name,
                    operation_id=call.call_id,
                    success=False,
                    severity=ResultSeverity.WARNING,
                    result_type=ResultType.TEXT,
                    content="Operation cancelled by user",
                    summary="User cancelled this operation",
                    llm_context="The user cancelled this operation. Please acknowledge and ask if they'd like to do something else.",
                )
                return ToolCallResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    result=cancelled_result,
                    formatted_for_llm=cancelled_result.llm_context,
                )
        
        # Execute the tool
        raw_result = self._orchestrator.execute_single(
            call.tool_name,
            call.arguments,
            context,
        )
        
        end_time = datetime.now()
        execution_ms = (end_time - start_time).total_seconds() * 1000
        
        # Process the result
        processed = self._processor.process(
            raw_result,
            call.tool_name,
            call.call_id,
            execution_ms,
        )
        
        return ToolCallResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            result=processed,
            formatted_for_llm=processed.llm_context,
        )
    
    def execute_tool_calls(
        self,
        calls: List[ToolCall],
        context: Optional[ExecutionContext] = None,
    ) -> List[ToolCallResult]:
        """Execute multiple tool calls.
        
        Args:
            calls: List of tool calls
            context: Execution context
            
        Returns:
            List of tool call results
        """
        results = []
        context = context or get_current_context()
        
        for call in calls:
            result = self.execute_tool_call(call, context)
            results.append(result)
        
        return results
    
    def format_results_for_llm(
        self,
        results: List[ToolCallResult],
        provider: Optional[LLMProvider] = None,
    ) -> Any:
        """Format tool results for sending back to LLM.
        
        Args:
            results: List of tool call results
            provider: Provider override
            
        Returns:
            Formatted results for the LLM
        """
        provider = provider or self.config.provider
        
        if provider == LLMProvider.OPENAI:
            return [
                {
                    "role": "tool",
                    "tool_call_id": r.call_id,
                    "content": r.formatted_for_llm,
                }
                for r in results
            ]
        
        elif provider == LLMProvider.ANTHROPIC:
            return [
                {
                    "type": "tool_result",
                    "tool_use_id": r.call_id,
                    "content": r.formatted_for_llm,
                }
                for r in results
            ]
        
        elif provider == LLMProvider.GOOGLE:
            return {
                "parts": [
                    {
                        "functionResponse": {
                            "name": r.tool_name,
                            "response": {"result": r.formatted_for_llm},
                        }
                    }
                    for r in results
                ]
            }
        
        else:
            # Text format
            lines = ["# Tool Results", ""]
            for r in results:
                lines.append(f"## {r.tool_name}")
                lines.append(r.formatted_for_llm)
                lines.append("")
            return "\n".join(lines)


# Import for convenience
from .result_processor import ResultSeverity, ResultType


# Global integration instance
_global_integration: Optional[LLMToolIntegration] = None


def get_llm_tool_integration(
    config: Optional[LLMToolConfig] = None
) -> LLMToolIntegration:
    """Get the global LLM tool integration instance.
    
    Args:
        config: Optional configuration (only used on first call)
        
    Returns:
        The LLM tool integration instance
    """
    global _global_integration
    if _global_integration is None:
        _global_integration = LLMToolIntegration(config)
    return _global_integration


def configure_llm_integration(config: LLMToolConfig):
    """Configure the global LLM tool integration.
    
    Args:
        config: Configuration to use
    """
    global _global_integration
    _global_integration = LLMToolIntegration(config)
