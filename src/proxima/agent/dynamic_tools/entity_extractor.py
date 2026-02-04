"""Entity Extraction System for Dynamic Tool System.

This module extracts entities from natural language without regex patterns.
It uses LLM reasoning and context awareness to identify and extract entities.

Phase 2.2.2: Entity Extraction System
=====================================
- Named Entity Recognition using LLM reasoning
- File path extraction with fuzzy matching
- Git branch name extraction with repository context
- URL extraction with validation
- Parameter value extraction with type inference
- Entity linking to resolve references

Key Design Principle:
--------------------
NO HARDCODED REGEX PATTERNS - All extraction happens through:
1. LLM reasoning about the text
2. Context-aware inference
3. Type-specific validation
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Union
from urllib.parse import urlparse

from .tool_interface import ParameterType, ToolParameter
from .execution_context import ExecutionContext, get_current_context

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities that can be extracted."""
    FILE_PATH = "file_path"
    DIRECTORY_PATH = "directory_path"
    FILE_NAME = "file_name"
    FILE_PATTERN = "file_pattern"
    GIT_BRANCH = "git_branch"
    GIT_COMMIT = "git_commit"
    GIT_REMOTE = "git_remote"
    URL = "url"
    COMMAND = "command"
    VARIABLE_NAME = "variable_name"
    VARIABLE_VALUE = "variable_value"
    NUMBER = "number"
    BOOLEAN = "boolean"
    TEXT_CONTENT = "text_content"
    CODE_SNIPPET = "code_snippet"
    DATE = "date"
    TIME = "time"
    REFERENCE = "reference"  # Reference to something mentioned earlier
    UNKNOWN = "unknown"


@dataclass
class ExtractedEntity:
    """An entity extracted from natural language."""
    entity_type: EntityType
    value: Any
    raw_text: str
    confidence: float
    start_position: int = -1
    end_position: int = -1
    validation_status: str = "valid"  # valid, invalid, uncertain
    normalized_value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type.value,
            "value": self.value,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "validation_status": self.validation_status,
            "normalized_value": self.normalized_value,
            "metadata": self.metadata,
        }
    
    @property
    def is_valid(self) -> bool:
        return self.validation_status == "valid"
    
    def get_usable_value(self) -> Any:
        """Get the value to use - normalized if available, otherwise raw."""
        return self.normalized_value if self.normalized_value is not None else self.value


@dataclass
class ExtractionResult:
    """Result of entity extraction from text."""
    text: str
    entities: List[ExtractedEntity]
    mapped_parameters: Dict[str, Any]
    missing_required: List[str]
    extraction_notes: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "mapped_parameters": self.mapped_parameters,
            "missing_required": self.missing_required,
            "extraction_notes": self.extraction_notes,
            "timestamp": self.timestamp,
        }
    
    def get_entity(self, entity_type: EntityType) -> Optional[ExtractedEntity]:
        """Get first entity of the specified type."""
        for entity in self.entities:
            if entity.entity_type == entity_type:
                return entity
        return None
    
    def get_all_entities(self, entity_type: EntityType) -> List[ExtractedEntity]:
        """Get all entities of the specified type."""
        return [e for e in self.entities if e.entity_type == entity_type]
    
    @property
    def has_all_required(self) -> bool:
        """Check if all required parameters were extracted."""
        return len(self.missing_required) == 0


class LLMBackend(Protocol):
    """Protocol for LLM backends used by the extractor."""
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response from the LLM."""
        ...


@dataclass
class ExtractorConfig:
    """Configuration for entity extraction."""
    # Extraction settings
    enable_llm_extraction: bool = True
    enable_context_inference: bool = True
    enable_fuzzy_matching: bool = True
    
    # Validation settings
    validate_file_paths: bool = True
    validate_urls: bool = True
    validate_git_refs: bool = True
    
    # LLM settings
    extraction_temperature: float = 0.1
    extraction_max_tokens: int = 1024
    
    # Path handling
    expand_home_directory: bool = True
    resolve_relative_paths: bool = True


class EntityValidator(ABC):
    """Base class for entity validators."""
    
    @abstractmethod
    def validate(
        self,
        entity: ExtractedEntity,
        context: ExecutionContext,
    ) -> ExtractedEntity:
        """Validate and potentially normalize an entity."""
        pass


class FilePathValidator(EntityValidator):
    """Validates and normalizes file paths."""
    
    def validate(
        self,
        entity: ExtractedEntity,
        context: ExecutionContext,
    ) -> ExtractedEntity:
        """Validate a file path entity."""
        path_str = str(entity.value)
        
        # Expand home directory
        if path_str.startswith("~"):
            path_str = os.path.expanduser(path_str)
        
        # Resolve relative to working directory
        if not os.path.isabs(path_str):
            base = context.working_directory or os.getcwd()
            path_str = os.path.join(base, path_str)
        
        # Normalize the path
        normalized = os.path.normpath(path_str)
        
        # Check if it exists
        exists = os.path.exists(normalized)
        
        entity.normalized_value = normalized
        entity.validation_status = "valid" if exists else "uncertain"
        entity.metadata["exists"] = exists
        entity.metadata["is_absolute"] = os.path.isabs(normalized)
        
        if exists:
            entity.metadata["is_file"] = os.path.isfile(normalized)
            entity.metadata["is_directory"] = os.path.isdir(normalized)
        
        return entity


class GitRefValidator(EntityValidator):
    """Validates git references (branches, commits, remotes)."""
    
    def validate(
        self,
        entity: ExtractedEntity,
        context: ExecutionContext,
    ) -> ExtractedEntity:
        """Validate a git reference entity."""
        ref = str(entity.value)
        
        # Check against known branches if git state available
        if context.git_state and context.git_state.branches:
            if ref in context.git_state.branches:
                entity.validation_status = "valid"
                entity.metadata["is_known_branch"] = True
            else:
                # Could be a new branch or typo
                entity.validation_status = "uncertain"
                entity.metadata["is_known_branch"] = False
                
                # Find similar branches (basic fuzzy match)
                similar = [
                    b for b in context.git_state.branches
                    if ref.lower() in b.lower() or b.lower() in ref.lower()
                ]
                if similar:
                    entity.metadata["similar_branches"] = similar[:3]
        else:
            entity.validation_status = "uncertain"
        
        entity.normalized_value = ref.strip()
        return entity


class URLValidator(EntityValidator):
    """Validates URLs."""
    
    def validate(
        self,
        entity: ExtractedEntity,
        context: ExecutionContext,
    ) -> ExtractedEntity:
        """Validate a URL entity."""
        url_str = str(entity.value)
        
        try:
            parsed = urlparse(url_str)
            
            # Check for valid scheme
            if parsed.scheme in ("http", "https", "git", "ssh"):
                entity.validation_status = "valid"
                entity.metadata["scheme"] = parsed.scheme
                entity.metadata["host"] = parsed.netloc
                entity.metadata["path"] = parsed.path
            else:
                entity.validation_status = "uncertain"
                entity.metadata["scheme"] = parsed.scheme or "missing"
            
            entity.normalized_value = url_str
            
        except Exception as e:
            entity.validation_status = "invalid"
            entity.metadata["error"] = str(e)
        
        return entity


class EntityExtractor:
    """Extracts entities from natural language using LLM reasoning.
    
    This extractor does NOT use hardcoded regex patterns. Instead, it:
    1. Uses LLM reasoning to identify entities
    2. Leverages context for inference
    3. Validates entities using type-specific validators
    4. Links entities to conversation references
    """
    
    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
        llm_backend: Optional[LLMBackend] = None,
    ):
        """Initialize the entity extractor.
        
        Args:
            config: Extractor configuration
            llm_backend: LLM backend for extraction
        """
        self._config = config or ExtractorConfig()
        self._llm_backend = llm_backend
        
        # Validators for different entity types
        self._validators: Dict[EntityType, EntityValidator] = {
            EntityType.FILE_PATH: FilePathValidator(),
            EntityType.DIRECTORY_PATH: FilePathValidator(),
            EntityType.GIT_BRANCH: GitRefValidator(),
            EntityType.GIT_COMMIT: GitRefValidator(),
            EntityType.URL: URLValidator(),
        }
        
        # Build extraction prompt
        self._extraction_prompt = self._build_extraction_prompt()
    
    def _build_extraction_prompt(self) -> str:
        """Build the system prompt for entity extraction."""
        return """You are an entity extraction system. Extract structured information from natural language.

## Entity Types to Extract:
- FILE_PATH: File or directory paths (absolute or relative)
- DIRECTORY_PATH: Directory/folder paths
- FILE_NAME: Just the filename without path
- FILE_PATTERN: Glob patterns or wildcards (*.py, **/*.txt)
- GIT_BRANCH: Git branch names
- GIT_COMMIT: Git commit hashes or references (HEAD, HEAD~1)
- GIT_REMOTE: Git remote names (origin, upstream)
- URL: URLs (http, https, git, ssh)
- COMMAND: Shell/terminal commands
- VARIABLE_NAME: Environment variable or config names
- VARIABLE_VALUE: Values to assign to variables
- NUMBER: Numeric values
- BOOLEAN: True/false values
- TEXT_CONTENT: Text content to write/create
- CODE_SNIPPET: Code blocks
- REFERENCE: References to earlier context ("that file", "the branch")

## Guidelines:
- Extract ALL relevant entities from the text
- Infer implicit values from context
- "This folder" or "here" typically means current directory (.)
- "That file" refers to recently mentioned files
- Paths can be explicit or implied
- Confidence should reflect certainty (0.0 - 1.0)

## Response Format:
Respond with JSON:
{
    "entities": [
        {
            "type": "ENTITY_TYPE",
            "value": "extracted_value",
            "raw_text": "original text portion",
            "confidence": 0.9,
            "notes": "any clarification"
        }
    ],
    "inferences": [
        "explanation of any inferred values"
    ]
}"""
    
    def extract(
        self,
        text: str,
        expected_parameters: Optional[List[ToolParameter]] = None,
        context: Optional[ExecutionContext] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> ExtractionResult:
        """Extract entities from natural language text.
        
        Args:
            text: The text to extract entities from
            expected_parameters: Tool parameters to map to (optional)
            context: Execution context for validation
            conversation_history: Recent conversation for reference resolution
            
        Returns:
            Extraction result with entities and mapped parameters
        """
        context = context or get_current_context()
        entities: List[ExtractedEntity] = []
        notes: List[str] = []
        
        # Use LLM for extraction if available
        if self._llm_backend and self._config.enable_llm_extraction:
            try:
                llm_entities = self._extract_with_llm(text, context, conversation_history)
                entities.extend(llm_entities)
                notes.append("Extracted using LLM reasoning")
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
                notes.append(f"LLM extraction failed: {e}")
        
        # Fallback/supplement: Context-based inference
        if self._config.enable_context_inference:
            inferred = self._infer_from_context(text, context, conversation_history)
            # Add inferred entities not already found
            existing_values = {e.value for e in entities}
            for entity in inferred:
                if entity.value not in existing_values:
                    entities.append(entity)
                    notes.append(f"Inferred {entity.entity_type.value} from context")
        
        # Validate all entities
        entities = self._validate_entities(entities, context)
        
        # Map entities to expected parameters if provided
        mapped_params: Dict[str, Any] = {}
        missing_required: List[str] = []
        
        if expected_parameters:
            mapped_params, missing_required = self._map_to_parameters(
                entities, expected_parameters, context
            )
        
        return ExtractionResult(
            text=text,
            entities=entities,
            mapped_parameters=mapped_params,
            missing_required=missing_required,
            extraction_notes=notes,
        )
    
    def _extract_with_llm(
        self,
        text: str,
        context: ExecutionContext,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> List[ExtractedEntity]:
        """Extract entities using LLM reasoning."""
        # Build extraction request with context
        parts = [f"Text to analyze: {text}"]
        
        parts.append(f"\nCurrent directory: {context.working_directory}")
        
        if context.git_state:
            if context.git_state.current_branch:
                parts.append(f"Current git branch: {context.git_state.current_branch}")
            if context.git_state.branches:
                parts.append(f"Known branches: {', '.join(context.git_state.branches[:10])}")
        
        # Add recent conversation for reference resolution
        if conversation_history:
            recent = conversation_history[-3:]
            history_text = "\n".join([
                f"{m['role']}: {m['content'][:150]}"
                for m in recent
            ])
            parts.append(f"\nRecent context:\n{history_text}")
        
        parts.append("\nExtract all entities and respond with JSON:")
        
        prompt = "\n".join(parts)
        
        response = self._llm_backend.generate(
            prompt=prompt,
            system_prompt=self._extraction_prompt,
            temperature=self._config.extraction_temperature,
            max_tokens=self._config.extraction_max_tokens,
        )
        
        return self._parse_extraction_response(response, text)
    
    def _parse_extraction_response(
        self,
        response: str,
        original_text: str,
    ) -> List[ExtractedEntity]:
        """Parse the LLM's extraction response."""
        entities: List[ExtractedEntity] = []
        
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
            else:
                return entities
            
            for item in data.get("entities", []):
                entity_type_str = item.get("type", "UNKNOWN").upper()
                try:
                    entity_type = EntityType[entity_type_str]
                except KeyError:
                    entity_type = EntityType.UNKNOWN
                
                entity = ExtractedEntity(
                    entity_type=entity_type,
                    value=item.get("value", ""),
                    raw_text=item.get("raw_text", str(item.get("value", ""))),
                    confidence=item.get("confidence", 0.7),
                    metadata={"notes": item.get("notes", "")},
                )
                entities.append(entity)
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse extraction response: {e}")
        
        return entities
    
    def _infer_from_context(
        self,
        text: str,
        context: ExecutionContext,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> List[ExtractedEntity]:
        """Infer entities from context without LLM.
        
        This uses simple heuristics as a fallback when LLM is not available.
        """
        entities: List[ExtractedEntity] = []
        text_lower = text.lower()
        
        # Infer current directory references
        current_dir_phrases = [
            "this folder", "this directory", "current folder", "current directory",
            "here", "this location", "current location", "this path",
        ]
        for phrase in current_dir_phrases:
            if phrase in text_lower:
                entities.append(ExtractedEntity(
                    entity_type=EntityType.DIRECTORY_PATH,
                    value=".",
                    raw_text=phrase,
                    confidence=0.8,
                    normalized_value=context.working_directory,
                    metadata={"inferred": True, "reason": "current directory reference"},
                ))
                break
        
        # Infer parent directory references
        parent_dir_phrases = ["parent folder", "parent directory", "up a level", "one level up"]
        for phrase in parent_dir_phrases:
            if phrase in text_lower:
                entities.append(ExtractedEntity(
                    entity_type=EntityType.DIRECTORY_PATH,
                    value="..",
                    raw_text=phrase,
                    confidence=0.8,
                    metadata={"inferred": True, "reason": "parent directory reference"},
                ))
                break
        
        # Infer git branch references
        if context.git_state and context.git_state.current_branch:
            branch_phrases = ["this branch", "current branch", "the branch"]
            for phrase in branch_phrases:
                if phrase in text_lower:
                    entities.append(ExtractedEntity(
                        entity_type=EntityType.GIT_BRANCH,
                        value=context.git_state.current_branch,
                        raw_text=phrase,
                        confidence=0.9,
                        metadata={"inferred": True, "reason": "current branch reference"},
                    ))
                    break
        
        # Look for references in conversation history
        if conversation_history:
            reference_phrases = ["that file", "the file", "that folder", "the folder", "that branch"]
            for phrase in reference_phrases:
                if phrase in text_lower:
                    # Search backwards for mentioned entities
                    for msg in reversed(conversation_history[-5:]):
                        content = msg.get("content", "")
                        # This is a simplified lookup - full implementation would parse previous extractions
                        entities.append(ExtractedEntity(
                            entity_type=EntityType.REFERENCE,
                            value=phrase,
                            raw_text=phrase,
                            confidence=0.5,
                            metadata={
                                "inferred": True,
                                "reason": "reference to previous context",
                                "search_context": content[:100],
                            },
                        ))
                        break
        
        return entities
    
    def _validate_entities(
        self,
        entities: List[ExtractedEntity],
        context: ExecutionContext,
    ) -> List[ExtractedEntity]:
        """Validate all extracted entities."""
        validated = []
        
        for entity in entities:
            validator = self._validators.get(entity.entity_type)
            if validator:
                try:
                    entity = validator.validate(entity, context)
                except Exception as e:
                    logger.warning(f"Validation failed for {entity.entity_type}: {e}")
                    entity.validation_status = "uncertain"
            validated.append(entity)
        
        return validated
    
    def _map_to_parameters(
        self,
        entities: List[ExtractedEntity],
        parameters: List[ToolParameter],
        context: ExecutionContext,
    ) -> tuple[Dict[str, Any], List[str]]:
        """Map extracted entities to tool parameters.
        
        Args:
            entities: Extracted entities
            parameters: Expected tool parameters
            context: Execution context
            
        Returns:
            Tuple of (mapped parameters dict, list of missing required params)
        """
        mapped: Dict[str, Any] = {}
        missing: List[str] = []
        
        # Build a mapping from parameter type to entity types
        param_type_mapping = {
            ParameterType.PATH: [EntityType.FILE_PATH, EntityType.DIRECTORY_PATH, EntityType.FILE_NAME],
            ParameterType.STRING: [EntityType.TEXT_CONTENT, EntityType.FILE_NAME, EntityType.VARIABLE_VALUE],
            ParameterType.INTEGER: [EntityType.NUMBER],
            ParameterType.BOOLEAN: [EntityType.BOOLEAN],
            ParameterType.URL: [EntityType.URL],
        }
        
        used_entities: Set[int] = set()
        
        for param in parameters:
            # Find matching entity for this parameter
            matched_entity = None
            
            # First, try to find by parameter name in entity metadata
            for i, entity in enumerate(entities):
                if i in used_entities:
                    continue
                if param.name.lower() in str(entity.metadata.get("notes", "")).lower():
                    matched_entity = entity
                    used_entities.add(i)
                    break
            
            # If no match by name, try by type
            if matched_entity is None:
                expected_entity_types = param_type_mapping.get(param.param_type, [EntityType.UNKNOWN])
                for i, entity in enumerate(entities):
                    if i in used_entities:
                        continue
                    if entity.entity_type in expected_entity_types:
                        matched_entity = entity
                        used_entities.add(i)
                        break
            
            if matched_entity:
                mapped[param.name] = matched_entity.get_usable_value()
            elif param.default is not None:
                mapped[param.name] = param.default
            elif param.required:
                missing.append(param.name)
        
        return mapped, missing
    
    def extract_for_tool(
        self,
        text: str,
        parameters: List[ToolParameter],
        context: Optional[ExecutionContext] = None,
    ) -> ExtractionResult:
        """Convenience method to extract entities for a specific tool.
        
        Args:
            text: User's natural language input
            parameters: Tool's expected parameters
            context: Execution context
            
        Returns:
            Extraction result with mapped parameters
        """
        return self.extract(
            text=text,
            expected_parameters=parameters,
            context=context,
        )
    
    def resolve_reference(
        self,
        reference: ExtractedEntity,
        conversation_history: List[Dict[str, str]],
        context: ExecutionContext,
    ) -> Optional[ExtractedEntity]:
        """Resolve a reference entity to a concrete value.
        
        Uses conversation history to find what the reference points to.
        
        Args:
            reference: A REFERENCE type entity
            conversation_history: Previous conversation
            context: Execution context
            
        Returns:
            Resolved entity or None if cannot resolve
        """
        if reference.entity_type != EntityType.REFERENCE:
            return reference
        
        if not self._llm_backend:
            return None
        
        # Use LLM to resolve the reference
        prompt = f"""Resolve this reference to a specific value:

Reference: "{reference.raw_text}"

Recent conversation:
{chr(10).join(f"{m['role']}: {m['content'][:200]}" for m in conversation_history[-5:])}

What specific file, path, branch, or value does "{reference.raw_text}" refer to?

Respond with JSON:
{{
    "resolved_type": "FILE_PATH|DIRECTORY_PATH|GIT_BRANCH|etc",
    "resolved_value": "the actual value",
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}}"""
        
        try:
            response = self._llm_backend.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=256,
            )
            
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                
                entity_type_str = data.get("resolved_type", "UNKNOWN").upper()
                try:
                    entity_type = EntityType[entity_type_str]
                except KeyError:
                    entity_type = EntityType.UNKNOWN
                
                return ExtractedEntity(
                    entity_type=entity_type,
                    value=data.get("resolved_value", ""),
                    raw_text=reference.raw_text,
                    confidence=data.get("confidence", 0.5),
                    metadata={
                        "resolved_from": "reference",
                        "reasoning": data.get("reasoning", ""),
                    },
                )
        except Exception as e:
            logger.warning(f"Failed to resolve reference: {e}")
        
        return None


# Module-level extractor instance
_global_extractor: Optional[EntityExtractor] = None


def get_entity_extractor(
    config: Optional[ExtractorConfig] = None,
) -> EntityExtractor:
    """Get the global entity extractor instance.
    
    Args:
        config: Optional extractor configuration
        
    Returns:
        The entity extractor instance
    """
    global _global_extractor
    if _global_extractor is None:
        _global_extractor = EntityExtractor(config)
    return _global_extractor


def configure_entity_extractor(
    config: ExtractorConfig,
    llm_backend: Optional[LLMBackend] = None,
):
    """Configure the global entity extractor.
    
    Args:
        config: Extractor configuration
        llm_backend: Optional LLM backend
    """
    global _global_extractor
    _global_extractor = EntityExtractor(
        config=config,
        llm_backend=llm_backend,
    )
