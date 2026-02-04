"""Structured Output Support for LLM Responses.

This module provides structured output parsing from LLM responses,
supporting JSON mode, schema validation, and fallback strategies.

Phase 2.1.1: Structured Output Support
======================================
- JSON mode support for OpenAI, Anthropic, Google Gemini
- Schema validation using Pydantic
- Fallback parsing strategies for models without native JSON support
- Retry logic with schema correction prompts
- Response validation pipeline

Key Features:
------------
- Provider-agnostic structured output parsing
- Pydantic-based schema validation
- Graceful degradation for non-JSON responses
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union

try:
    from pydantic import BaseModel, Field, ValidationError, create_model
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OutputFormat(Enum):
    """Supported output formats."""
    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"


class ParseStatus(Enum):
    """Status of parsing attempt."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    RETRY_NEEDED = "retry_needed"


@dataclass
class ParseResult(Generic[T]):
    """Result of parsing an LLM response."""
    status: ParseStatus
    data: Optional[T] = None
    raw_response: str = ""
    errors: List[str] = field(default_factory=list)
    corrections_applied: List[str] = field(default_factory=list)
    parse_attempts: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "data": self.data if not hasattr(self.data, "model_dump") else self.data.model_dump(),  # type: ignore
            "raw_response": self.raw_response,
            "errors": self.errors,
            "corrections_applied": self.corrections_applied,
            "parse_attempts": self.parse_attempts,
            "timestamp": self.timestamp,
        }
    
    @property
    def is_success(self) -> bool:
        return self.status == ParseStatus.SUCCESS


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output parsing."""
    # Parsing settings
    max_retry_attempts: int = 3
    enable_auto_correction: bool = True
    strict_validation: bool = False
    
    # JSON extraction settings
    extract_json_from_markdown: bool = True
    extract_json_from_text: bool = True
    allow_trailing_commas: bool = True
    allow_single_quotes: bool = True
    
    # Schema settings
    coerce_types: bool = True
    ignore_extra_fields: bool = True
    fill_defaults: bool = True


class JSONExtractor:
    """Extracts JSON from various response formats."""
    
    def __init__(self, config: Optional[StructuredOutputConfig] = None):
        self._config = config or StructuredOutputConfig()
    
    def extract(self, text: str) -> tuple[Optional[Dict[str, Any]], List[str]]:
        """Extract JSON from text.
        
        Tries multiple strategies:
        1. Direct JSON parse
        2. Extract from markdown code blocks
        3. Extract JSON object/array from surrounding text
        4. Apply corrections and retry
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Tuple of (parsed JSON or None, list of errors)
        """
        errors: List[str] = []
        
        # Strategy 1: Direct parse
        try:
            return json.loads(text), []
        except json.JSONDecodeError as e:
            errors.append(f"Direct parse failed: {e}")
        
        # Strategy 2: Extract from markdown code blocks
        if self._config.extract_json_from_markdown:
            json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1).strip()), []
                except json.JSONDecodeError as e:
                    errors.append(f"Markdown extraction failed: {e}")
        
        # Strategy 3: Find JSON object or array boundaries
        if self._config.extract_json_from_text:
            # Find outermost braces/brackets
            json_str = self._find_json_boundaries(text)
            if json_str:
                try:
                    return json.loads(json_str), []
                except json.JSONDecodeError as e:
                    errors.append(f"Boundary extraction failed: {e}")
                    
                    # Strategy 4: Apply corrections
                    if self._config.enable_auto_correction:
                        corrected, corrections = self._apply_corrections(json_str)
                        try:
                            return json.loads(corrected), []
                        except json.JSONDecodeError as e2:
                            errors.append(f"Correction failed: {e2}, applied: {corrections}")
        
        return None, errors
    
    def _find_json_boundaries(self, text: str) -> Optional[str]:
        """Find the JSON object or array in text."""
        # Find start of JSON
        obj_start = text.find("{")
        arr_start = text.find("[")
        
        if obj_start == -1 and arr_start == -1:
            return None
        
        # Determine which comes first and is valid
        if obj_start >= 0 and (arr_start == -1 or obj_start < arr_start):
            # Object
            depth = 0
            for i, char in enumerate(text[obj_start:], obj_start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[obj_start:i + 1]
        
        elif arr_start >= 0:
            # Array
            depth = 0
            for i, char in enumerate(text[arr_start:], arr_start):
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    if depth == 0:
                        return text[arr_start:i + 1]
        
        return None
    
    def _apply_corrections(self, json_str: str) -> tuple[str, List[str]]:
        """Apply common JSON corrections."""
        corrections: List[str] = []
        corrected = json_str
        
        # Remove trailing commas
        if self._config.allow_trailing_commas:
            new_str = re.sub(r",\s*([}\]])", r"\1", corrected)
            if new_str != corrected:
                corrections.append("removed trailing commas")
                corrected = new_str
        
        # Convert single quotes to double quotes
        if self._config.allow_single_quotes:
            # Simple replacement (may not handle all edge cases)
            new_str = corrected.replace("'", '"')
            if new_str != corrected:
                corrections.append("converted single quotes")
                corrected = new_str
        
        # Fix unquoted keys (basic)
        # This is a simple fix - a full solution would need a proper parser
        new_str = re.sub(r"(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', corrected)
        if new_str != corrected:
            corrections.append("quoted unquoted keys")
            corrected = new_str
        
        return corrected, corrections


class SchemaValidator:
    """Validates parsed data against a schema."""
    
    def __init__(self, config: Optional[StructuredOutputConfig] = None):
        self._config = config or StructuredOutputConfig()
    
    def validate(
        self,
        data: Any,
        schema: Optional[Type[BaseModel]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, List[str]]:
        """Validate data against a schema.
        
        Args:
            data: Data to validate
            schema: Pydantic model class
            json_schema: JSON Schema dict (fallback if no Pydantic model)
            
        Returns:
            Tuple of (validated data, list of errors)
        """
        errors: List[str] = []
        
        if schema and PYDANTIC_AVAILABLE:
            try:
                if isinstance(data, dict):
                    validated = schema(**data)
                    return validated, []
                else:
                    errors.append(f"Expected dict for Pydantic validation, got {type(data)}")
            except ValidationError as e:
                errors.append(f"Pydantic validation failed: {e}")
                
                # Try to coerce types if enabled
                if self._config.coerce_types:
                    coerced, coerce_errors = self._coerce_types(data, schema)
                    if not coerce_errors:
                        try:
                            validated = schema(**coerced)
                            return validated, []
                        except ValidationError as e2:
                            errors.append(f"Coercion failed: {e2}")
        
        elif json_schema:
            # Basic JSON schema validation without jsonschema library
            validation_errors = self._validate_json_schema(data, json_schema)
            if validation_errors:
                errors.extend(validation_errors)
            else:
                return data, []
        
        # If strict validation is disabled, return data with errors
        if not self._config.strict_validation:
            return data, errors
        
        return None, errors
    
    def _coerce_types(
        self,
        data: Dict[str, Any],
        schema: Type[BaseModel],
    ) -> tuple[Dict[str, Any], List[str]]:
        """Try to coerce types to match schema."""
        coerced = dict(data)
        errors: List[str] = []
        
        # Get schema field types
        if hasattr(schema, "model_fields"):
            fields = schema.model_fields
        elif hasattr(schema, "__fields__"):
            fields = schema.__fields__
        else:
            return coerced, ["Cannot determine schema fields"]
        
        for field_name, field_info in fields.items():
            if field_name not in coerced:
                continue
            
            value = coerced[field_name]
            
            # Get expected type
            if hasattr(field_info, "annotation"):
                expected_type = field_info.annotation
            elif hasattr(field_info, "outer_type_"):
                expected_type = field_info.outer_type_
            else:
                continue
            
            # Try to coerce
            try:
                if expected_type == int and isinstance(value, str):
                    coerced[field_name] = int(value)
                elif expected_type == float and isinstance(value, str):
                    coerced[field_name] = float(value)
                elif expected_type == bool and isinstance(value, str):
                    coerced[field_name] = value.lower() in ("true", "yes", "1")
                elif expected_type == str and not isinstance(value, str):
                    coerced[field_name] = str(value)
            except (ValueError, TypeError) as e:
                errors.append(f"Cannot coerce {field_name}: {e}")
        
        return coerced, errors
    
    def _validate_json_schema(
        self,
        data: Any,
        schema: Dict[str, Any],
    ) -> List[str]:
        """Basic JSON schema validation without external library."""
        errors: List[str] = []
        
        schema_type = schema.get("type")
        
        # Type validation
        if schema_type:
            type_map = {
                "object": dict,
                "array": list,
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
                "null": type(None),
            }
            expected = type_map.get(schema_type)
            if expected and not isinstance(data, expected):
                errors.append(f"Expected {schema_type}, got {type(data).__name__}")
        
        # Object property validation
        if schema_type == "object" and isinstance(data, dict):
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            # Check required properties
            for prop in required:
                if prop not in data:
                    errors.append(f"Missing required property: {prop}")
            
            # Validate each property
            for prop, prop_schema in properties.items():
                if prop in data:
                    prop_errors = self._validate_json_schema(data[prop], prop_schema)
                    errors.extend([f"{prop}: {e}" for e in prop_errors])
        
        # Array item validation
        if schema_type == "array" and isinstance(data, list):
            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(data):
                    item_errors = self._validate_json_schema(item, items_schema)
                    errors.extend([f"[{i}]: {e}" for e in item_errors])
        
        return errors


class StructuredOutputParser:
    """Parses and validates structured output from LLM responses.
    
    This parser provides:
    - Multi-strategy JSON extraction
    - Schema validation with Pydantic
    - Automatic type coercion
    - Retry logic with correction prompts
    """
    
    def __init__(self, config: Optional[StructuredOutputConfig] = None):
        """Initialize the parser.
        
        Args:
            config: Parser configuration
        """
        self._config = config or StructuredOutputConfig()
        self._extractor = JSONExtractor(config)
        self._validator = SchemaValidator(config)
    
    def parse(
        self,
        response: str,
        schema: Optional[Type[BaseModel]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ParseResult:
        """Parse a structured response.
        
        Args:
            response: LLM response text
            schema: Optional Pydantic model for validation
            json_schema: Optional JSON schema dict
            
        Returns:
            ParseResult with parsed data or errors
        """
        # Extract JSON
        data, extract_errors = self._extractor.extract(response)
        
        if data is None:
            return ParseResult(
                status=ParseStatus.FAILED,
                raw_response=response,
                errors=extract_errors,
            )
        
        # Validate if schema provided
        if schema or json_schema:
            validated, validate_errors = self._validator.validate(
                data, schema, json_schema
            )
            
            if validate_errors:
                if validated is not None and not self._config.strict_validation:
                    # Partial success - data extracted but validation issues
                    return ParseResult(
                        status=ParseStatus.PARTIAL,
                        data=validated,
                        raw_response=response,
                        errors=validate_errors,
                    )
                return ParseResult(
                    status=ParseStatus.FAILED,
                    raw_response=response,
                    errors=extract_errors + validate_errors,
                )
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=validated,
                raw_response=response,
            )
        
        # No validation - just return extracted data
        return ParseResult(
            status=ParseStatus.SUCCESS,
            data=data,
            raw_response=response,
        )
    
    def parse_with_retry(
        self,
        response: str,
        retry_callback: Callable[[str], str],
        schema: Optional[Type[BaseModel]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ParseResult:
        """Parse with automatic retry on failure.
        
        Args:
            response: Initial LLM response
            retry_callback: Function to call for retry (receives correction prompt)
            schema: Optional Pydantic model
            json_schema: Optional JSON schema
            
        Returns:
            ParseResult after all attempts
        """
        attempts = 0
        current_response = response
        all_errors: List[str] = []
        corrections: List[str] = []
        
        while attempts < self._config.max_retry_attempts:
            attempts += 1
            
            result = self.parse(current_response, schema, json_schema)
            
            if result.is_success:
                result.parse_attempts = attempts
                return result
            
            if result.status == ParseStatus.PARTIAL and not self._config.strict_validation:
                result.parse_attempts = attempts
                return result
            
            all_errors.extend(result.errors)
            
            # Generate correction prompt
            correction_prompt = self._generate_correction_prompt(
                current_response, result.errors, schema, json_schema
            )
            corrections.append(f"Attempt {attempts}: {correction_prompt[:100]}...")
            
            # Call retry callback
            try:
                current_response = retry_callback(correction_prompt)
            except Exception as e:
                all_errors.append(f"Retry callback failed: {e}")
                break
        
        return ParseResult(
            status=ParseStatus.FAILED,
            raw_response=current_response,
            errors=all_errors,
            corrections_applied=corrections,
            parse_attempts=attempts,
        )
    
    def _generate_correction_prompt(
        self,
        response: str,
        errors: List[str],
        schema: Optional[Type[BaseModel]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a prompt to correct the response."""
        parts = [
            "Your previous response had formatting issues:",
            "",
        ]
        
        for error in errors[:3]:  # Limit error messages
            parts.append(f"- {error}")
        
        parts.append("")
        parts.append("Please provide a corrected response with valid JSON.")
        
        if schema and PYDANTIC_AVAILABLE:
            # Include schema information
            parts.append("")
            parts.append("Expected schema:")
            if hasattr(schema, "model_json_schema"):
                parts.append(json.dumps(schema.model_json_schema(), indent=2))
            elif hasattr(schema, "schema"):
                parts.append(json.dumps(schema.schema(), indent=2))
        elif json_schema:
            parts.append("")
            parts.append("Expected schema:")
            parts.append(json.dumps(json_schema, indent=2))
        
        return "\n".join(parts)


def create_response_model(
    name: str,
    fields: Dict[str, tuple[type, Any]],
) -> Type[BaseModel]:
    """Create a dynamic Pydantic model for response validation.
    
    Args:
        name: Model name
        fields: Dictionary of field_name -> (type, default or ...)
        
    Returns:
        Dynamic Pydantic model class
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic is required for dynamic model creation")
    
    return create_model(name, **fields)  # type: ignore


# Common response models
if PYDANTIC_AVAILABLE:
    class ToolCallResponse(BaseModel):
        """Standard response for tool call requests."""
        tool: str = Field(description="Name of the tool to call")
        arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
        reasoning: Optional[str] = Field(None, description="Reasoning for the tool selection")
    
    class IntentResponse(BaseModel):
        """Standard response for intent classification."""
        primary_intent: str = Field(description="Primary intent category")
        sub_intent: Optional[str] = Field(None, description="Specific sub-intent")
        confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
        target_tools: List[str] = Field(default_factory=list, description="Relevant tools")
        reasoning: Optional[str] = Field(None, description="Classification reasoning")
    
    class EntityResponse(BaseModel):
        """Standard response for entity extraction."""
        entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
        inferences: List[str] = Field(default_factory=list, description="Inferred information")
else:
    # Fallback when Pydantic not available
    ToolCallResponse = None  # type: ignore
    IntentResponse = None  # type: ignore
    EntityResponse = None  # type: ignore


# Module-level parser instance
_global_parser: Optional[StructuredOutputParser] = None


def get_structured_output_parser(
    config: Optional[StructuredOutputConfig] = None,
) -> StructuredOutputParser:
    """Get the global structured output parser.
    
    Args:
        config: Optional parser configuration
        
    Returns:
        The parser instance
    """
    global _global_parser
    if _global_parser is None:
        _global_parser = StructuredOutputParser(config)
    return _global_parser


def configure_structured_output(config: StructuredOutputConfig):
    """Configure the global structured output parser.
    
    Args:
        config: Parser configuration
    """
    global _global_parser
    _global_parser = StructuredOutputParser(config)
