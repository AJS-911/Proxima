"""Tool Registry for Dynamic Tool System.

This module provides the central registry for all available tools.
The registry supports:
- Dynamic tool registration
- Tool discovery by category, capability, or semantic search
- Tool schema generation for different LLM providers
- Tool validation and security checks

The registry is designed to be the single source of truth for all
tools available to the AI assistant, enabling dynamic tool selection
without hardcoded keyword matching.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

from .tool_interface import (
    ToolInterface,
    ToolDefinition,
    ToolCategory,
    PermissionLevel,
    RiskLevel,
    BaseTool,
)

logger = logging.getLogger(__name__)


@dataclass
class RegisteredTool:
    """A tool registered in the registry."""
    tool_class: Type[ToolInterface]
    definition: ToolDefinition
    instance: Optional[ToolInterface] = None
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    success_count: int = 0
    last_used_at: Optional[str] = None
    
    def get_instance(self) -> ToolInterface:
        """Get or create a tool instance."""
        if self.instance is None:
            self.instance = self.tool_class()
        return self.instance
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate."""
        if self.usage_count == 0:
            return 1.0
        return self.success_count / self.usage_count


@dataclass
class ToolSearchResult:
    """Result from a tool search."""
    tool: RegisteredTool
    relevance_score: float
    match_reason: str


class ToolRegistry:
    """Central registry for all dynamic tools.
    
    The registry provides:
    - Tool registration and management
    - Tool discovery by various criteria
    - Schema generation for LLM function calling
    - Usage tracking and analytics
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, RegisteredTool] = {}
        self._category_index: Dict[ToolCategory, Set[str]] = {}
        self._permission_index: Dict[PermissionLevel, Set[str]] = {}
        self._risk_index: Dict[RiskLevel, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}
        self._capability_keywords: Dict[str, Set[str]] = {}  # keyword -> tool names
        self._initialized = False
    
    def register(
        self,
        tool_class: Type[ToolInterface],
        override: bool = False
    ) -> bool:
        """Register a tool class.
        
        Args:
            tool_class: The tool class to register
            override: Whether to override existing registration
            
        Returns:
            True if registration succeeded
        """
        try:
            # Create temporary instance to get definition
            temp_instance = tool_class()
            definition = temp_instance.get_definition()
            
            if definition.name in self._tools and not override:
                logger.warning(f"Tool '{definition.name}' already registered")
                return False
            
            # Create registered tool
            registered = RegisteredTool(
                tool_class=tool_class,
                definition=definition,
            )
            
            # Store in main registry
            self._tools[definition.name] = registered
            
            # Update indices
            self._update_indices(definition.name, definition)
            
            logger.info(f"Registered tool: {definition.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register tool {tool_class}: {e}")
            return False
    
    def _update_indices(self, tool_name: str, definition: ToolDefinition):
        """Update all indices for a tool."""
        # Category index
        if definition.category not in self._category_index:
            self._category_index[definition.category] = set()
        self._category_index[definition.category].add(tool_name)
        
        # Sub-category index (treat as category)
        if definition.sub_category:
            sub_cat = definition.sub_category
            if sub_cat not in self._category_index:
                self._category_index[sub_cat] = set()
            self._category_index[sub_cat].add(tool_name)
        
        # Permission index
        if definition.permission_level not in self._permission_index:
            self._permission_index[definition.permission_level] = set()
        self._permission_index[definition.permission_level].add(tool_name)
        
        # Risk index
        if definition.risk_level not in self._risk_index:
            self._risk_index[definition.risk_level] = set()
        self._risk_index[definition.risk_level].add(tool_name)
        
        # Tag index
        for tag in definition.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = set()
            self._tag_index[tag_lower].add(tool_name)
        
        # Capability keyword index - extract from description and examples
        keywords = self._extract_keywords(definition)
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self._capability_keywords:
                self._capability_keywords[keyword_lower] = set()
            self._capability_keywords[keyword_lower].add(tool_name)
    
    def _extract_keywords(self, definition: ToolDefinition) -> Set[str]:
        """Extract searchable keywords from a tool definition."""
        keywords = set()
        
        # Add tool name parts
        keywords.update(definition.name.lower().replace("_", " ").split())
        
        # Add tags
        keywords.update(tag.lower() for tag in definition.tags)
        
        # Extract from description
        description_words = definition.description.lower().split()
        # Filter out common words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "this", "that", "these",
            "those", "it", "its", "which", "who", "whom", "whose", "what", "where",
            "when", "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "and", "or", "but", "if",
        }
        keywords.update(w for w in description_words if w not in stop_words and len(w) > 2)
        
        # Add example phrases (if any)
        for example in definition.examples:
            # examples is a list of dicts, extract description
            example_text = example.get("description", "") if isinstance(example, dict) else str(example)
            keywords.update(
                w.lower() for w in example_text.split() 
                if w.lower() not in stop_words and len(w) > 2
            )
        
        return keywords
    
    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if unregistration succeeded
        """
        if tool_name not in self._tools:
            return False
        
        registered = self._tools[tool_name]
        definition = registered.definition
        
        # Remove from all indices
        self._category_index.get(definition.category, set()).discard(tool_name)
        if definition.sub_category:
            self._category_index.get(definition.sub_category, set()).discard(tool_name)
        self._permission_index.get(definition.permission_level, set()).discard(tool_name)
        self._risk_index.get(definition.risk_level, set()).discard(tool_name)
        
        for tag in definition.tags:
            self._tag_index.get(tag.lower(), set()).discard(tool_name)
        
        for keyword_set in self._capability_keywords.values():
            keyword_set.discard(tool_name)
        
        # Remove from main registry
        del self._tools[tool_name]
        
        logger.info(f"Unregistered tool: {tool_name}")
        return True
    
    def get_tool(self, tool_name: str) -> Optional[RegisteredTool]:
        """Get a registered tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            The registered tool or None
        """
        return self._tools.get(tool_name)
    
    def get_tool_instance(self, tool_name: str) -> Optional[ToolInterface]:
        """Get a tool instance by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            The tool instance or None
        """
        registered = self._tools.get(tool_name)
        if registered:
            return registered.get_instance()
        return None
    
    def get_all_tools(self) -> List[RegisteredTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_tools_by_category(self, category: ToolCategory) -> List[RegisteredTool]:
        """Get tools by category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of matching tools
        """
        tool_names = self._category_index.get(category, set())
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_tools_by_permission(
        self, 
        max_permission: PermissionLevel
    ) -> List[RegisteredTool]:
        """Get tools accessible with given permission level.
        
        Args:
            max_permission: Maximum permission level
            
        Returns:
            List of accessible tools
        """
        accessible = []
        permission_order = list(PermissionLevel)
        max_index = permission_order.index(max_permission)
        
        for perm in permission_order[:max_index + 1]:
            for tool_name in self._permission_index.get(perm, set()):
                if tool_name in self._tools:
                    accessible.append(self._tools[tool_name])
        
        return accessible
    
    def get_tools_by_risk(self, max_risk: RiskLevel) -> List[RegisteredTool]:
        """Get tools with risk at or below given level.
        
        Args:
            max_risk: Maximum risk level
            
        Returns:
            List of matching tools
        """
        acceptable = []
        risk_order = list(RiskLevel)
        max_index = risk_order.index(max_risk)
        
        for risk in risk_order[:max_index + 1]:
            for tool_name in self._risk_index.get(risk, set()):
                if tool_name in self._tools:
                    acceptable.append(self._tools[tool_name])
        
        return acceptable
    
    def get_tools_by_tag(self, tag: str) -> List[RegisteredTool]:
        """Get tools with a specific tag.
        
        Args:
            tag: The tag to search for
            
        Returns:
            List of matching tools
        """
        tool_names = self._tag_index.get(tag.lower(), set())
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def search_tools(
        self,
        query: str,
        categories: Optional[List[ToolCategory]] = None,
        max_risk: Optional[RiskLevel] = None,
        max_permission: Optional[PermissionLevel] = None,
        limit: int = 10,
    ) -> List[ToolSearchResult]:
        """Search for tools using semantic matching.
        
        This method provides intelligent tool discovery by analyzing
        the query and matching against tool capabilities, descriptions,
        and examples. This is the primary method for LLM-driven tool selection.
        
        Args:
            query: Natural language query describing desired capability
            categories: Optional category filter
            max_risk: Optional maximum risk level
            max_permission: Optional maximum permission level
            limit: Maximum results to return
            
        Returns:
            List of search results with relevance scores
        """
        results: List[ToolSearchResult] = []
        query_words = set(query.lower().split())
        
        # Get candidate tools
        candidates = list(self._tools.values())
        
        # Apply filters
        if categories:
            candidates = [
                t for t in candidates 
                if t.definition.category in categories
            ]
        
        if max_risk:
            risk_order = list(RiskLevel)
            max_risk_index = risk_order.index(max_risk)
            candidates = [
                t for t in candidates
                if risk_order.index(t.definition.risk_level) <= max_risk_index
            ]
        
        if max_permission:
            perm_order = list(PermissionLevel)
            max_perm_index = perm_order.index(max_permission)
            candidates = [
                t for t in candidates
                if perm_order.index(t.definition.permission_level) <= max_perm_index
            ]
        
        # Score each candidate
        for registered in candidates:
            score, reason = self._calculate_relevance(query_words, registered.definition)
            if score > 0:
                results.append(ToolSearchResult(
                    tool=registered,
                    relevance_score=score,
                    match_reason=reason,
                ))
        
        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        return results[:limit]
    
    def _calculate_relevance(
        self, 
        query_words: Set[str], 
        definition: ToolDefinition
    ) -> tuple[float, str]:
        """Calculate relevance score between query and tool.
        
        Returns:
            Tuple of (score, match_reason)
        """
        score = 0.0
        reasons = []
        
        # Exact name match
        if definition.name.lower() in query_words:
            score += 10.0
            reasons.append("exact name match")
        
        # Name partial match
        name_words = set(definition.name.lower().replace("_", " ").split())
        name_overlap = len(query_words & name_words)
        if name_overlap:
            score += name_overlap * 3.0
            reasons.append(f"name words: {name_overlap}")
        
        # Tag match
        tag_words = {t.lower() for t in definition.tags}
        tag_overlap = len(query_words & tag_words)
        if tag_overlap:
            score += tag_overlap * 2.5
            reasons.append(f"tag match: {tag_overlap}")
        
        # Description match
        desc_words = set(definition.description.lower().split())
        desc_overlap = len(query_words & desc_words)
        if desc_overlap:
            score += desc_overlap * 0.5
            reasons.append(f"description words: {desc_overlap}")
        
        # Example match
        for example in definition.examples:
            # examples is a list of dicts, extract description
            example_text = example.get("description", "") if isinstance(example, dict) else str(example)
            example_words = set(example_text.lower().split())
            example_overlap = len(query_words & example_words)
            if example_overlap:
                score += example_overlap * 1.5
                reasons.append(f"example match")
                break
        
        # Boost based on success rate
        # This will work once tools have usage history
        
        return score, ", ".join(reasons) if reasons else "no match"
    
    def record_usage(
        self, 
        tool_name: str, 
        success: bool
    ):
        """Record tool usage for analytics.
        
        Args:
            tool_name: Name of the tool used
            success: Whether execution was successful
        """
        if tool_name in self._tools:
            registered = self._tools[tool_name]
            registered.usage_count += 1
            if success:
                registered.success_count += 1
            registered.last_used_at = datetime.now().isoformat()
    
    def get_openai_functions(
        self,
        tools: Optional[List[str]] = None,
        categories: Optional[List[ToolCategory]] = None,
        max_permission: Optional[PermissionLevel] = None,
    ) -> List[Dict[str, Any]]:
        """Generate OpenAI function calling schema.
        
        Args:
            tools: Specific tools to include (None for all)
            categories: Categories to filter by
            max_permission: Maximum permission level
            
        Returns:
            List of OpenAI function definitions
        """
        functions = []
        
        for name, registered in self._tools.items():
            # Apply filters
            if tools and name not in tools:
                continue
            if categories and registered.definition.category not in categories:
                continue
            if max_permission:
                perm_order = list(PermissionLevel)
                if perm_order.index(registered.definition.permission_level) > perm_order.index(max_permission):
                    continue
            
            functions.append(registered.definition.to_openai_function())
        
        return functions
    
    def get_anthropic_tools(
        self,
        tools: Optional[List[str]] = None,
        categories: Optional[List[ToolCategory]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate Anthropic tool schema.
        
        Args:
            tools: Specific tools to include
            categories: Categories to filter by
            
        Returns:
            List of Anthropic tool definitions
        """
        tool_defs = []
        
        for name, registered in self._tools.items():
            if tools and name not in tools:
                continue
            if categories and registered.definition.category not in categories:
                continue
            
            tool_defs.append(registered.definition.to_anthropic_tool())
        
        return tool_defs
    
    def get_gemini_functions(
        self,
        tools: Optional[List[str]] = None,
        categories: Optional[List[ToolCategory]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate Gemini function calling schema.
        
        Args:
            tools: Specific tools to include
            categories: Categories to filter by
            
        Returns:
            List of Gemini function declarations
        """
        functions = []
        
        for name, registered in self._tools.items():
            if tools and name not in tools:
                continue
            if categories and registered.definition.category not in categories:
                continue
            
            functions.append(registered.definition.to_gemini_function())
        
        return functions
    
    def get_all_tool_descriptions(self) -> str:
        """Get a human-readable description of all tools.
        
        This generates a comprehensive description suitable for
        inclusion in an LLM system prompt.
        
        Returns:
            Formatted string describing all tools
        """
        descriptions = []
        
        # Group by category
        by_category: Dict[ToolCategory, List[RegisteredTool]] = {}
        for registered in self._tools.values():
            cat = registered.definition.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(registered)
        
        for category in ToolCategory:
            if category not in by_category:
                continue
            
            descriptions.append(f"\n## {category.value} Tools\n")
            
            for registered in by_category[category]:
                defn = registered.definition
                descriptions.append(defn.to_llm_description())
                descriptions.append("")
        
        return "\n".join(descriptions)
    
    def export_registry(self, path: Path):
        """Export registry to JSON file.
        
        Args:
            path: Path to export file
        """
        data = {
            "tools": {
                name: {
                    "definition": {
                        "name": reg.definition.name,
                        "description": reg.definition.description,
                        "category": reg.definition.category.value,
                        "parameters": [
                            {
                                "name": p.name,
                                "description": p.description,
                                "param_type": p.param_type.value,
                                "required": p.required,
                                "default": p.default,
                            }
                            for p in reg.definition.parameters
                        ],
                        "tags": reg.definition.tags,
                        "examples": reg.definition.examples,
                    },
                    "usage_count": reg.usage_count,
                    "success_count": reg.success_count,
                    "registered_at": reg.registered_at,
                    "last_used_at": reg.last_used_at,
                }
                for name, reg in self._tools.items()
            },
            "exported_at": datetime.now().isoformat(),
        }
        
        path.write_text(json.dumps(data, indent=2))
    
    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools
    
    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool_class: Type[ToolInterface]) -> Type[ToolInterface]:
    """Decorator to register a tool class.
    
    Usage:
        @register_tool
        class MyTool(BaseTool):
            ...
    """
    get_tool_registry().register(tool_class)
    return tool_class
