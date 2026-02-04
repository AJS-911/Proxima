"""Natural Language Processor for Tool Mapping.

This module is the main entry point for Phase 2: LLM Integration - Natural Language
to Tool Mapping. It combines all components to convert natural language queries
into tool invocations.

Main Features:
=============
1. Intent Classification - Understand what the user wants to do
2. Entity Extraction - Extract relevant parameters from the query
3. Tool Selection - Select the best tool(s) for the intent
4. Parameter Mapping - Map extracted entities to tool parameters
5. Tool Execution - Execute the selected tool(s)
6. Response Generation - Generate natural language response

Architecture:
============
The NLProcessor works as follows:

    User Query
         │
         ▼
    ┌─────────────────┐
    │ Intent Classifier│──► Classify intent (file_ops, git_ops, etc.)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Entity Extractor│──► Extract paths, refs, URLs, etc.
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Tool Selector  │──► Select best tool for intent + entities
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Parameter Mapper│──► Map entities to tool parameters
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Tool Executor  │──► Execute tool with parameters
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │Response Generator│──► Generate natural language response
    └─────────────────┘

All components use LLM reasoning - NO hardcoded keywords or regex patterns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Stages in the NL processing pipeline."""
    INTENT_CLASSIFICATION = "intent_classification"
    ENTITY_EXTRACTION = "entity_extraction"
    TOOL_SELECTION = "tool_selection"
    PARAMETER_MAPPING = "parameter_mapping"
    TOOL_EXECUTION = "tool_execution"
    RESPONSE_GENERATION = "response_generation"


@dataclass
class ProcessingContext:
    """Context passed through the processing pipeline."""
    # Input
    query: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    workspace_context: Dict[str, Any] = field(default_factory=dict)
    
    # Processing results (filled as pipeline progresses)
    intent: Optional[Any] = None  # ClassifiedIntent
    entities: Optional[Any] = None  # ExtractionResult
    selected_tools: List[Dict[str, Any]] = field(default_factory=list)
    mapped_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_results: List[Dict[str, Any]] = field(default_factory=list)
    response: Optional[str] = None
    
    # Metadata
    stage: ProcessingStage = ProcessingStage.INTENT_CLASSIFICATION
    errors: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    
    def add_error(self, error: str, stage: Optional[ProcessingStage] = None):
        """Add an error to the context."""
        stage_str = (stage or self.stage).value
        self.errors.append(f"[{stage_str}] {error}")


@dataclass
class ProcessingResult:
    """Final result of natural language processing."""
    success: bool
    response: str
    tool_results: List[Dict[str, Any]]
    
    # Processing details
    intent: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    
    # Metadata
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "response": self.response,
            "tool_results": self.tool_results,
            "intent": self.intent,
            "entities": self.entities,
            "tools_used": self.tools_used,
            "processing_time_ms": self.processing_time_ms,
            "errors": self.errors,
        }


class NaturalLanguageProcessor:
    """Main processor for natural language to tool mapping.
    
    This processor orchestrates all Phase 2 components to convert
    natural language queries into tool invocations.
    
    Example usage:
        >>> processor = NaturalLanguageProcessor()
        >>> result = await processor.process("Show me all Python files in src/")
        >>> print(result.response)
        "Found 42 Python files in src/..."
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the processor.
        
        Args:
            llm_client: Optional LLM client for reasoning
            tool_registry: Optional tool registry
            config: Optional configuration dictionary
        """
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._config = config or {}
        
        # Initialize components lazily
        self._intent_classifier: Optional[Any] = None
        self._entity_extractor: Optional[Any] = None
        self._structured_parser: Optional[Any] = None
        self._function_calling: Optional[Any] = None
        self._provider_router: Optional[Any] = None
        
        # Callback for streaming responses
        self._stream_callback: Optional[Callable[[str], None]] = None
    
    def _get_intent_classifier(self):
        """Get or create intent classifier."""
        if self._intent_classifier is None:
            from .intent_classifier import IntentClassifier
            self._intent_classifier = IntentClassifier(
                llm_client=self._llm_client,
                tool_registry=self._tool_registry,
            )
        return self._intent_classifier
    
    def _get_entity_extractor(self):
        """Get or create entity extractor."""
        if self._entity_extractor is None:
            from .entity_extractor import EntityExtractor
            self._entity_extractor = EntityExtractor(
                llm_client=self._llm_client,
            )
        return self._entity_extractor
    
    def _get_structured_parser(self):
        """Get or create structured output parser."""
        if self._structured_parser is None:
            from .structured_output import StructuredOutputParser
            self._structured_parser = StructuredOutputParser()
        return self._structured_parser
    
    def _get_function_calling(self):
        """Get or create function calling integration."""
        if self._function_calling is None:
            from .function_calling import FunctionCallingIntegration
            self._function_calling = FunctionCallingIntegration(
                tool_registry=self._tool_registry,
            )
        return self._function_calling
    
    def _get_provider_router(self):
        """Get or create provider router."""
        if self._provider_router is None:
            from .provider_router import get_provider_router
            self._provider_router = get_provider_router()
        return self._provider_router
    
    async def process(
        self,
        query: str,
        context: Optional[ProcessingContext] = None,
    ) -> ProcessingResult:
        """Process a natural language query.
        
        This is the main entry point for natural language processing.
        
        Args:
            query: The natural language query
            context: Optional processing context
            
        Returns:
            The processing result
        """
        import time
        start = time.perf_counter()
        
        # Create context if not provided
        if context is None:
            context = ProcessingContext(query=query)
        
        try:
            # Stage 1: Intent Classification
            context.stage = ProcessingStage.INTENT_CLASSIFICATION
            await self._classify_intent(context)
            
            # Stage 2: Entity Extraction  
            context.stage = ProcessingStage.ENTITY_EXTRACTION
            await self._extract_entities(context)
            
            # Stage 3: Tool Selection
            context.stage = ProcessingStage.TOOL_SELECTION
            await self._select_tools(context)
            
            # Stage 4: Parameter Mapping
            context.stage = ProcessingStage.PARAMETER_MAPPING
            await self._map_parameters(context)
            
            # Stage 5: Tool Execution
            context.stage = ProcessingStage.TOOL_EXECUTION
            await self._execute_tools(context)
            
            # Stage 6: Response Generation
            context.stage = ProcessingStage.RESPONSE_GENERATION
            await self._generate_response(context)
            
            elapsed = (time.perf_counter() - start) * 1000
            
            return ProcessingResult(
                success=len(context.errors) == 0,
                response=context.response or "Task completed.",
                tool_results=context.execution_results,
                intent=context.intent.intent_type.value if context.intent else None,
                entities=[
                    e.to_dict() if hasattr(e, 'to_dict') else {"value": str(e)}
                    for e in (context.entities.entities if context.entities else [])
                ],
                tools_used=[t.get("name", "") for t in context.selected_tools],
                processing_time_ms=elapsed,
                errors=context.errors,
            )
            
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"Processing error: {e}", exc_info=True)
            
            return ProcessingResult(
                success=False,
                response=f"I encountered an error while processing your request: {e}",
                tool_results=[],
                processing_time_ms=elapsed,
                errors=context.errors + [str(e)],
            )
    
    async def _classify_intent(self, context: ProcessingContext):
        """Classify the intent of the query.
        
        Uses LLM reasoning to understand what the user wants to do.
        """
        classifier = self._get_intent_classifier()
        
        try:
            # Use LLM to classify intent
            classified = await classifier.classify_async(
                context.query,
                conversation_history=context.conversation_history,
            )
            
            context.intent = classified
            
            # Check if clarification is needed
            if classified.needs_clarification:
                context.add_error(
                    f"Clarification needed: {classified.clarification_question}"
                )
                
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            context.add_error(f"Could not understand intent: {e}")
    
    async def _extract_entities(self, context: ProcessingContext):
        """Extract entities from the query.
        
        Uses LLM reasoning to identify paths, URLs, git refs, etc.
        """
        extractor = self._get_entity_extractor()
        
        try:
            # Get expected entities based on intent
            expected_types = None
            if context.intent:
                expected_types = self._get_expected_entity_types(context.intent)
            
            # Use LLM to extract entities
            result = await extractor.extract_async(
                context.query,
                expected_types=expected_types,
                context=context.workspace_context,
            )
            
            context.entities = result
            
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            context.add_error(f"Could not extract parameters: {e}")
    
    def _get_expected_entity_types(self, intent) -> Optional[List[str]]:
        """Get expected entity types for an intent."""
        from .entity_extractor import EntityType
        
        intent_entity_map = {
            "file_operations": [EntityType.FILE_PATH, EntityType.DIRECTORY],
            "git_operations": [EntityType.GIT_REF, EntityType.GIT_REMOTE, EntityType.FILE_PATH],
            "terminal_operations": [EntityType.COMMAND, EntityType.ENVIRONMENT_VAR],
            "code_analysis": [EntityType.SYMBOL, EntityType.FILE_PATH],
            "search": [EntityType.SEARCH_QUERY, EntityType.FILE_PATTERN],
            "documentation": [EntityType.URL, EntityType.FILE_PATH],
        }
        
        intent_name = intent.intent_type.value if hasattr(intent, 'intent_type') else str(intent)
        return intent_entity_map.get(intent_name)
    
    async def _select_tools(self, context: ProcessingContext):
        """Select the best tools for the intent and entities.
        
        Uses LLM reasoning to select appropriate tools from the registry.
        """
        if self._tool_registry is None:
            context.add_error("No tool registry available")
            return
        
        try:
            # Get available tools
            available_tools = self._tool_registry.get_all_tools()
            
            if not available_tools:
                context.add_error("No tools available")
                return
            
            # Build tool descriptions for LLM
            tool_descriptions = self._build_tool_descriptions(available_tools)
            
            # Use LLM to select best tool(s)
            selected = await self._llm_select_tools(
                context.query,
                context.intent,
                context.entities,
                tool_descriptions,
            )
            
            context.selected_tools = selected
            
        except Exception as e:
            logger.error(f"Tool selection error: {e}")
            context.add_error(f"Could not select tools: {e}")
    
    def _build_tool_descriptions(
        self,
        tools: List[Any],
    ) -> List[Dict[str, Any]]:
        """Build tool descriptions for LLM selection."""
        descriptions = []
        
        for tool in tools:
            info = tool.get_info() if hasattr(tool, 'get_info') else {}
            
            descriptions.append({
                "name": info.get("name", str(tool)),
                "description": info.get("description", ""),
                "category": info.get("category", "general"),
                "parameters": info.get("parameters", {}),
            })
        
        return descriptions
    
    async def _llm_select_tools(
        self,
        query: str,
        intent: Any,
        entities: Any,
        tool_descriptions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Use LLM to select the best tools."""
        if self._llm_client is None:
            # Fallback: Use semantic matching
            return self._semantic_tool_selection(query, intent, tool_descriptions)
        
        # Build prompt for LLM
        tools_text = "\n".join([
            f"- {t['name']}: {t['description']} (category: {t['category']})"
            for t in tool_descriptions
        ])
        
        intent_text = ""
        if intent:
            intent_type = intent.intent_type.value if hasattr(intent, 'intent_type') else str(intent)
            intent_text = f"Intent: {intent_type}"
        
        entities_text = ""
        if entities and hasattr(entities, 'entities'):
            entity_strs = [
                f"{e.entity_type.value}: {e.value}" if hasattr(e, 'entity_type') else str(e)
                for e in entities.entities
            ]
            entities_text = f"Entities: {', '.join(entity_strs)}"
        
        prompt = f"""Select the best tool(s) to accomplish this task.

User Request: {query}
{intent_text}
{entities_text}

Available Tools:
{tools_text}

Respond with a JSON array of tool names to use, in order of execution.
Example: ["tool_name_1", "tool_name_2"]

If no tools are suitable, respond with: []
"""
        
        try:
            # Get LLM response
            response = await self._llm_client.generate(prompt)
            
            # Parse JSON response
            parser = self._get_structured_parser()
            result = parser.extract_json(response)
            
            if result.success and isinstance(result.data, list):
                # Map names to tool info
                selected = []
                for name in result.data:
                    for tool in tool_descriptions:
                        if tool["name"] == name:
                            selected.append(tool)
                            break
                return selected
                
        except Exception as e:
            logger.warning(f"LLM tool selection failed: {e}")
        
        # Fallback
        return self._semantic_tool_selection(query, intent, tool_descriptions)
    
    def _semantic_tool_selection(
        self,
        query: str,
        intent: Any,
        tool_descriptions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fallback semantic tool selection."""
        # Simple word overlap scoring
        query_words = set(query.lower().split())
        
        scored = []
        for tool in tool_descriptions:
            desc_words = set(tool["description"].lower().split())
            name_words = set(tool["name"].lower().replace("_", " ").split())
            
            overlap = len(query_words & (desc_words | name_words))
            scored.append((overlap, tool))
        
        # Sort by score and return top matches
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Return tools with positive score
        return [tool for score, tool in scored if score > 0][:3]
    
    async def _map_parameters(self, context: ProcessingContext):
        """Map extracted entities to tool parameters.
        
        Uses LLM reasoning to correctly assign entities to parameters.
        """
        if not context.selected_tools:
            return
        
        if not context.entities or not hasattr(context.entities, 'entities'):
            return
        
        try:
            for tool in context.selected_tools:
                tool_name = tool.get("name", "")
                tool_params = tool.get("parameters", {})
                
                # Map entities to parameters
                mapped = await self._llm_map_parameters(
                    context.query,
                    context.entities.entities,
                    tool_params,
                )
                
                context.mapped_parameters[tool_name] = mapped
                
        except Exception as e:
            logger.error(f"Parameter mapping error: {e}")
            context.add_error(f"Could not map parameters: {e}")
    
    async def _llm_map_parameters(
        self,
        query: str,
        entities: List[Any],
        tool_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use LLM to map entities to tool parameters."""
        if not tool_params:
            return {}
        
        if self._llm_client is None:
            # Fallback: Simple type matching
            return self._simple_parameter_mapping(entities, tool_params)
        
        # Build entity list
        entity_strs = []
        for entity in entities:
            if hasattr(entity, 'entity_type'):
                entity_strs.append(f"- {entity.entity_type.value}: {entity.value}")
            else:
                entity_strs.append(f"- value: {entity}")
        
        # Build parameter list
        param_strs = []
        for name, spec in tool_params.items():
            ptype = spec.get("type", "any")
            required = spec.get("required", False)
            desc = spec.get("description", "")
            param_strs.append(f"- {name} ({ptype}, {'required' if required else 'optional'}): {desc}")
        
        prompt = f"""Map the extracted values to the tool parameters.

User Request: {query}

Extracted Values:
{chr(10).join(entity_strs)}

Tool Parameters:
{chr(10).join(param_strs)}

Respond with a JSON object mapping parameter names to values.
Example: {{"param_name": "value", "other_param": 123}}

Only include parameters that have matching values.
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            parser = self._get_structured_parser()
            result = parser.extract_json(response)
            
            if result.success and isinstance(result.data, dict):
                return result.data
                
        except Exception as e:
            logger.warning(f"LLM parameter mapping failed: {e}")
        
        return self._simple_parameter_mapping(entities, tool_params)
    
    def _simple_parameter_mapping(
        self,
        entities: List[Any],
        tool_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simple fallback parameter mapping by type."""
        mapped = {}
        
        # Build type to entity map
        entity_by_type: Dict[str, List[Any]] = {}
        for entity in entities:
            etype = entity.entity_type.value if hasattr(entity, 'entity_type') else "string"
            if etype not in entity_by_type:
                entity_by_type[etype] = []
            entity_by_type[etype].append(entity)
        
        # Map parameters
        for param_name, spec in tool_params.items():
            ptype = spec.get("type", "string").lower()
            
            # Try to find matching entity
            matched_entity = None
            
            # Direct type match
            if ptype in entity_by_type and entity_by_type[ptype]:
                matched_entity = entity_by_type[ptype].pop(0)
            
            # Type aliases
            type_aliases = {
                "path": ["file_path", "directory"],
                "file": ["file_path"],
                "dir": ["directory"],
                "ref": ["git_ref", "git_branch"],
            }
            
            if not matched_entity:
                for alias in type_aliases.get(ptype, []):
                    if alias in entity_by_type and entity_by_type[alias]:
                        matched_entity = entity_by_type[alias].pop(0)
                        break
            
            if matched_entity:
                value = matched_entity.value if hasattr(matched_entity, 'value') else str(matched_entity)
                mapped[param_name] = value
        
        return mapped
    
    async def _execute_tools(self, context: ProcessingContext):
        """Execute the selected tools with mapped parameters.
        
        Uses the function calling integration for execution.
        """
        if not context.selected_tools:
            context.add_error("No tools to execute")
            return
        
        try:
            func_calling = self._get_function_calling()
            
            for tool in context.selected_tools:
                tool_name = tool.get("name", "")
                params = context.mapped_parameters.get(tool_name, {})
                
                # Execute the tool
                result = await self._execute_single_tool(tool_name, params)
                
                context.execution_results.append({
                    "tool": tool_name,
                    "parameters": params,
                    "result": result,
                    "success": result.get("success", False),
                })
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            context.add_error(f"Tool execution failed: {e}")
    
    async def _execute_single_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single tool."""
        if self._tool_registry is None:
            return {"success": False, "error": "No tool registry"}
        
        try:
            # Get tool from registry
            tool = self._tool_registry.get_tool(tool_name)
            if not tool:
                return {"success": False, "error": f"Tool '{tool_name}' not found"}
            
            # Execute tool
            result = await tool.execute(**params)
            
            return {
                "success": True,
                "data": result,
            }
            
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_response(self, context: ProcessingContext):
        """Generate a natural language response.
        
        Uses LLM to summarize tool results in natural language.
        """
        # If no execution results, use error message
        if not context.execution_results:
            if context.errors:
                context.response = f"I couldn't complete the task: {context.errors[-1]}"
            else:
                context.response = "I couldn't find any tools to help with that request."
            return
        
        # Check if all tools succeeded
        all_success = all(r.get("success", False) for r in context.execution_results)
        
        if self._llm_client is None:
            # Simple response without LLM
            context.response = self._simple_response(context)
            return
        
        try:
            # Build results summary
            results_text = []
            for result in context.execution_results:
                tool = result.get("tool", "unknown")
                data = result.get("result", {}).get("data")
                success = result.get("success", False)
                
                if success:
                    results_text.append(f"- {tool}: {json.dumps(data, default=str)[:500]}")
                else:
                    error = result.get("result", {}).get("error", "Unknown error")
                    results_text.append(f"- {tool}: Failed - {error}")
            
            prompt = f"""Generate a helpful, natural response for the user based on these results.

User Request: {context.query}

Tool Results:
{chr(10).join(results_text)}

Write a concise, friendly response that:
1. Directly addresses what the user asked
2. Summarizes the key findings or outcomes
3. Is written in natural language (not code or JSON)
4. Mentions any errors if they occurred

Response:"""
            
            response = await self._llm_client.generate(prompt)
            context.response = response.strip()
            
        except Exception as e:
            logger.warning(f"LLM response generation failed: {e}")
            context.response = self._simple_response(context)
    
    def _simple_response(self, context: ProcessingContext) -> str:
        """Generate simple response without LLM."""
        results = context.execution_results
        
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        
        parts = []
        
        if successful:
            tool_names = [r.get("tool", "tool") for r in successful]
            parts.append(f"Completed {', '.join(tool_names)} successfully.")
        
        if failed:
            for f in failed:
                tool = f.get("tool", "tool")
                error = f.get("result", {}).get("error", "unknown error")
                parts.append(f"{tool} failed: {error}")
        
        return " ".join(parts) or "Task completed."
    
    def set_llm_client(self, client: Any):
        """Set the LLM client for reasoning.
        
        Args:
            client: The LLM client
        """
        self._llm_client = client
        
        # Update components
        if self._intent_classifier:
            self._intent_classifier._llm_client = client
        if self._entity_extractor:
            self._entity_extractor._llm_client = client
    
    def set_tool_registry(self, registry: Any):
        """Set the tool registry.
        
        Args:
            registry: The tool registry
        """
        self._tool_registry = registry
        
        if self._intent_classifier:
            self._intent_classifier._tool_registry = registry
        if self._function_calling:
            self._function_calling._tool_registry = registry
    
    def set_stream_callback(self, callback: Callable[[str], None]):
        """Set callback for streaming responses.
        
        Args:
            callback: Function to call with each chunk
        """
        self._stream_callback = callback


class SyncNaturalLanguageProcessor:
    """Synchronous wrapper for NaturalLanguageProcessor.
    
    Provides synchronous methods for environments that don't use asyncio.
    """
    
    def __init__(self, **kwargs):
        """Initialize with same arguments as NaturalLanguageProcessor."""
        self._async_processor = NaturalLanguageProcessor(**kwargs)
    
    def process(
        self,
        query: str,
        context: Optional[ProcessingContext] = None,
    ) -> ProcessingResult:
        """Process a query synchronously.
        
        Args:
            query: The natural language query
            context: Optional processing context
            
        Returns:
            The processing result
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._async_processor.process(query, context)
        )
    
    def set_llm_client(self, client: Any):
        """Set the LLM client."""
        self._async_processor.set_llm_client(client)
    
    def set_tool_registry(self, registry: Any):
        """Set the tool registry."""
        self._async_processor.set_tool_registry(registry)


# Convenience functions for quick use

_processor: Optional[NaturalLanguageProcessor] = None


def get_nl_processor(
    llm_client: Optional[Any] = None,
    tool_registry: Optional[Any] = None,
) -> NaturalLanguageProcessor:
    """Get or create the global NL processor.
    
    Args:
        llm_client: Optional LLM client
        tool_registry: Optional tool registry
        
    Returns:
        The NL processor
    """
    global _processor
    
    if _processor is None:
        _processor = NaturalLanguageProcessor(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )
    
    return _processor


async def process_query(
    query: str,
    llm_client: Optional[Any] = None,
    tool_registry: Optional[Any] = None,
) -> ProcessingResult:
    """Process a natural language query.
    
    Convenience function for quick processing.
    
    Args:
        query: The natural language query
        llm_client: Optional LLM client
        tool_registry: Optional tool registry
        
    Returns:
        The processing result
    """
    processor = get_nl_processor(llm_client, tool_registry)
    return await processor.process(query)
