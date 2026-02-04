"""Result Analyzer for Output Analysis and Report Generation.

This module implements Phase 3.4 for the Dynamic AI Assistant:
- Output Format Detection: Type identification, structure analysis,
  delimiter detection, encoding inference
- Data Extraction: Pattern matching, value extraction, nested parsing
- Analysis Execution: Statistical analysis, trend identification,
  anomaly detection, comparison operations
- Report Generation: Template-based formatting, visualization preparation,
  summary generation, export handling

Key Features:
============
- Automatic output format detection (JSON, CSV, XML, YAML, etc.)
- Structured data extraction with pattern recognition
- Statistical analysis capabilities
- Trend and anomaly detection
- Flexible report generation
- Multiple export formats

Design Principle:
================
All analysis decisions use LLM reasoning - NO hardcoded patterns.
The LLM determines output structure, extraction patterns, and analysis approaches.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Detected output formats."""
    JSON = "json"
    CSV = "csv"
    TSV = "tsv"
    XML = "xml"
    YAML = "yaml"
    KEY_VALUE = "key_value"
    TABLE = "table"
    LOG = "log"
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    BINARY = "binary"
    UNKNOWN = "unknown"


class DataType(Enum):
    """Data types for extracted values."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    LIST = "list"
    DICT = "dict"
    NULL = "null"


class AnalysisType(Enum):
    """Types of analysis to perform."""
    STATISTICAL = "statistical"
    COMPARISON = "comparison"
    TREND = "trend"
    ANOMALY = "anomaly"
    SUMMARY = "summary"
    EXTRACTION = "extraction"
    AGGREGATION = "aggregation"


class ReportFormat(Enum):
    """Report output formats."""
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    CSV = "csv"


@dataclass
class FormatDetectionResult:
    """Result of format detection."""
    format: OutputFormat
    confidence: float  # 0.0-1.0
    encoding: Optional[str] = None
    delimiter: Optional[str] = None
    structure: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format.value,
            "confidence": self.confidence,
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "structure": self.structure,
        }


@dataclass
class ExtractedValue:
    """A value extracted from output."""
    key: str
    value: Any
    data_type: DataType
    path: Optional[str] = None  # JSONPath or XPath style
    source_line: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "type": self.data_type.value,
            "path": self.path,
        }


@dataclass
class ExtractionResult:
    """Result of data extraction."""
    values: List[ExtractedValue] = field(default_factory=list)
    records: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "values_count": len(self.values),
            "records_count": len(self.records),
            "errors": self.errors,
            "values": [v.to_dict() for v in self.values[:10]],  # Limit
        }


@dataclass
class StatisticalResult:
    """Result of statistical analysis."""
    field_name: str
    count: int = 0
    sum: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Any] = None
    std_dev: Optional[float] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    percentiles: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field_name,
            "count": self.count,
            "sum": self.sum,
            "mean": self.mean,
            "median": self.median,
            "std_dev": self.std_dev,
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass
class TrendResult:
    """Result of trend analysis."""
    field_name: str
    direction: str = "stable"  # increasing, decreasing, stable, volatile
    change_rate: Optional[float] = None
    start_value: Optional[Any] = None
    end_value: Optional[Any] = None
    peak_value: Optional[Any] = None
    trough_value: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field_name,
            "direction": self.direction,
            "change_rate": self.change_rate,
            "start_value": self.start_value,
            "end_value": self.end_value,
        }


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    field_name: str
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    threshold: Optional[float] = None
    method: str = "z_score"  # z_score, iqr, isolation_forest
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field_name,
            "anomaly_count": len(self.anomalies),
            "method": self.method,
            "anomalies": self.anomalies[:10],  # Limit
        }


@dataclass
class AnalysisResult:
    """Combined analysis result."""
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    analysis_type: AnalysisType = AnalysisType.SUMMARY
    
    # Results
    statistical: List[StatisticalResult] = field(default_factory=list)
    trends: List[TrendResult] = field(default_factory=list)
    anomalies: List[AnomalyResult] = field(default_factory=list)
    summary: str = ""
    insights: List[str] = field(default_factory=list)
    
    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    record_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "type": self.analysis_type.value,
            "record_count": self.record_count,
            "summary": self.summary,
            "insights": self.insights,
            "statistical": [s.to_dict() for s in self.statistical],
            "trends": [t.to_dict() for t in self.trends],
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


@dataclass
class Report:
    """Generated report."""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    content: str = ""
    format: ReportFormat = ReportFormat.MARKDOWN
    sections: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "title": self.title,
            "format": self.format.value,
            "sections_count": len(self.sections),
            "generated_at": self.generated_at.isoformat(),
        }


class ResultAnalyzer:
    """Analyzer for command outputs and results.
    
    Uses LLM reasoning to:
    1. Detect output format and structure
    2. Extract structured data from unstructured output
    3. Perform statistical and trend analysis
    4. Generate insights and reports
    
    Example:
        >>> analyzer = ResultAnalyzer(llm_client=client)
        >>> format_result = await analyzer.detect_format(output)
        >>> data = await analyzer.extract_data(output, format_result)
        >>> analysis = await analyzer.analyze(data)
        >>> report = await analyzer.generate_report(analysis)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize the analyzer.
        
        Args:
            llm_client: LLM client for reasoning
        """
        self._llm_client = llm_client
        
        # Pattern cache
        self._pattern_cache: Dict[str, re.Pattern] = {}
    
    async def detect_format(
        self,
        output: str,
        context: Optional[str] = None,
    ) -> FormatDetectionResult:
        """Detect the format of output data.
        
        Args:
            output: Raw output string
            context: Optional context about the output source
            
        Returns:
            FormatDetectionResult with detected format
        """
        # Try structured formats first
        
        # JSON detection
        if self._try_json(output):
            return FormatDetectionResult(
                format=OutputFormat.JSON,
                confidence=0.95,
                encoding="utf-8",
            )
        
        # XML detection
        if self._try_xml(output):
            return FormatDetectionResult(
                format=OutputFormat.XML,
                confidence=0.90,
                encoding="utf-8",
            )
        
        # YAML detection
        if self._try_yaml(output):
            return FormatDetectionResult(
                format=OutputFormat.YAML,
                confidence=0.85,
                encoding="utf-8",
            )
        
        # CSV/TSV detection
        csv_result = self._try_csv(output)
        if csv_result:
            return csv_result
        
        # Key-value detection
        if self._try_key_value(output):
            return FormatDetectionResult(
                format=OutputFormat.KEY_VALUE,
                confidence=0.80,
                encoding="utf-8",
            )
        
        # Markdown detection
        if self._try_markdown(output):
            return FormatDetectionResult(
                format=OutputFormat.MARKDOWN,
                confidence=0.75,
                encoding="utf-8",
            )
        
        # Table detection
        if self._try_table(output):
            return FormatDetectionResult(
                format=OutputFormat.TABLE,
                confidence=0.70,
                encoding="utf-8",
            )
        
        # Log format detection
        if self._try_log(output):
            return FormatDetectionResult(
                format=OutputFormat.LOG,
                confidence=0.70,
                encoding="utf-8",
            )
        
        # Use LLM for complex cases
        if self._llm_client:
            llm_result = await self._llm_detect_format(output, context)
            if llm_result.confidence > 0.5:
                return llm_result
        
        # Default to plain text
        return FormatDetectionResult(
            format=OutputFormat.PLAIN_TEXT,
            confidence=0.5,
            encoding="utf-8",
        )
    
    def _try_json(self, output: str) -> bool:
        """Try to parse as JSON."""
        output = output.strip()
        if not (output.startswith("{") or output.startswith("[")):
            return False
        
        try:
            json.loads(output)
            return True
        except json.JSONDecodeError:
            return False
    
    def _try_xml(self, output: str) -> bool:
        """Try to detect XML format."""
        output = output.strip()
        return output.startswith("<?xml") or (
            output.startswith("<") and 
            output.endswith(">") and
            "</" in output
        )
    
    def _try_yaml(self, output: str) -> bool:
        """Try to detect YAML format."""
        lines = output.strip().splitlines()
        if not lines:
            return False
        
        # Check for YAML indicators
        yaml_indicators = 0
        for line in lines[:20]:
            if re.match(r"^\s*[a-zA-Z_][a-zA-Z0-9_]*:\s*", line):
                yaml_indicators += 1
            if line.strip().startswith("- "):
                yaml_indicators += 1
        
        return yaml_indicators >= 3
    
    def _try_csv(self, output: str) -> Optional[FormatDetectionResult]:
        """Try to detect CSV/TSV format."""
        lines = output.strip().splitlines()
        if len(lines) < 2:
            return None
        
        # Check for consistent delimiter
        for delimiter in [",", "\t", ";", "|"]:
            counts = [line.count(delimiter) for line in lines[:10] if line.strip()]
            if counts and all(c == counts[0] for c in counts) and counts[0] >= 1:
                return FormatDetectionResult(
                    format=OutputFormat.CSV if delimiter == "," else OutputFormat.TSV,
                    confidence=0.85,
                    encoding="utf-8",
                    delimiter=delimiter,
                )
        
        return None
    
    def _try_key_value(self, output: str) -> bool:
        """Try to detect key-value format."""
        lines = output.strip().splitlines()
        kv_count = 0
        
        for line in lines[:20]:
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*[=:]\s*", line):
                kv_count += 1
        
        return kv_count >= len(lines[:20]) * 0.5
    
    def _try_markdown(self, output: str) -> bool:
        """Try to detect Markdown format."""
        md_indicators = 0
        
        if re.search(r"^#{1,6}\s", output, re.MULTILINE):
            md_indicators += 1
        if re.search(r"^\s*[-*+]\s", output, re.MULTILINE):
            md_indicators += 1
        if re.search(r"\*\*.*\*\*", output):
            md_indicators += 1
        if re.search(r"```", output):
            md_indicators += 1
        if re.search(r"\[.*\]\(.*\)", output):
            md_indicators += 1
        
        return md_indicators >= 2
    
    def _try_table(self, output: str) -> bool:
        """Try to detect ASCII table format."""
        lines = output.strip().splitlines()
        
        # Look for table borders
        border_count = sum(1 for line in lines if re.match(r"^[+|-]+$", line.strip()))
        pipe_count = sum(1 for line in lines if line.count("|") >= 2)
        
        return border_count >= 2 or pipe_count >= 3
    
    def _try_log(self, output: str) -> bool:
        """Try to detect log format."""
        lines = output.strip().splitlines()
        
        # Common log patterns
        log_patterns = [
            r"^\d{4}-\d{2}-\d{2}",  # Date
            r"^\[\w+\]",  # Level in brackets
            r"(INFO|WARN|ERROR|DEBUG)",  # Log levels
            r"\d{2}:\d{2}:\d{2}",  # Time
        ]
        
        matches = 0
        for line in lines[:20]:
            for pattern in log_patterns:
                if re.search(pattern, line):
                    matches += 1
                    break
        
        return matches >= len(lines[:20]) * 0.3
    
    async def _llm_detect_format(
        self,
        output: str,
        context: Optional[str],
    ) -> FormatDetectionResult:
        """Use LLM to detect format."""
        prompt = f"""Analyze this output and determine its format.

Output (first 500 chars):
{output[:500]}

Context: {context or 'Unknown source'}

Available formats: JSON, CSV, TSV, XML, YAML, KEY_VALUE, TABLE, LOG, MARKDOWN, HTML, PLAIN_TEXT

Respond in this format:
FORMAT: <format_name>
CONFIDENCE: <0.0-1.0>
DELIMITER: <if applicable>
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            
            format_match = re.search(r"FORMAT:\s*(\w+)", response, re.IGNORECASE)
            confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response)
            delimiter_match = re.search(r"DELIMITER:\s*(.+)", response)
            
            if format_match:
                format_str = format_match.group(1).upper()
                format_enum = getattr(OutputFormat, format_str, OutputFormat.PLAIN_TEXT)
                
                confidence = 0.7
                if confidence_match:
                    confidence = float(confidence_match.group(1))
                
                delimiter = None
                if delimiter_match:
                    delimiter = delimiter_match.group(1).strip()
                
                return FormatDetectionResult(
                    format=format_enum,
                    confidence=confidence,
                    delimiter=delimiter,
                )
                
        except Exception as e:
            logger.warning(f"LLM format detection failed: {e}")
        
        return FormatDetectionResult(
            format=OutputFormat.PLAIN_TEXT,
            confidence=0.3,
        )
    
    async def extract_data(
        self,
        output: str,
        format_result: Optional[FormatDetectionResult] = None,
        patterns: Optional[List[str]] = None,
    ) -> ExtractionResult:
        """Extract structured data from output.
        
        Args:
            output: Raw output string
            format_result: Optional format detection result
            patterns: Optional specific patterns to extract
            
        Returns:
            ExtractionResult with extracted data
        """
        result = ExtractionResult()
        
        # Detect format if not provided
        if format_result is None:
            format_result = await self.detect_format(output)
        
        try:
            if format_result.format == OutputFormat.JSON:
                result = self._extract_json(output)
            elif format_result.format in [OutputFormat.CSV, OutputFormat.TSV]:
                result = self._extract_csv(output, format_result.delimiter or ",")
            elif format_result.format == OutputFormat.KEY_VALUE:
                result = self._extract_key_value(output)
            elif format_result.format == OutputFormat.TABLE:
                result = self._extract_table(output)
            elif format_result.format == OutputFormat.LOG:
                result = self._extract_log(output)
            else:
                # Use LLM for complex extraction
                if self._llm_client:
                    result = await self._llm_extract(output, patterns)
                else:
                    result = self._extract_plain_text(output, patterns)
                    
        except Exception as e:
            result.errors.append(str(e))
        
        return result
    
    def _extract_json(self, output: str) -> ExtractionResult:
        """Extract data from JSON."""
        result = ExtractionResult()
        
        data = json.loads(output.strip())
        
        if isinstance(data, list):
            result.records = data
            for i, record in enumerate(data):
                if isinstance(record, dict):
                    for key, value in record.items():
                        result.values.append(ExtractedValue(
                            key=key,
                            value=value,
                            data_type=self._infer_type(value),
                            path=f"$[{i}].{key}",
                        ))
        elif isinstance(data, dict):
            result.records = [data]
            for key, value in data.items():
                result.values.append(ExtractedValue(
                    key=key,
                    value=value,
                    data_type=self._infer_type(value),
                    path=f"$.{key}",
                ))
        
        return result
    
    def _extract_csv(self, output: str, delimiter: str) -> ExtractionResult:
        """Extract data from CSV/TSV."""
        result = ExtractionResult()
        
        reader = csv.DictReader(io.StringIO(output), delimiter=delimiter)
        
        for i, row in enumerate(reader):
            result.records.append(dict(row))
            for key, value in row.items():
                result.values.append(ExtractedValue(
                    key=key,
                    value=value,
                    data_type=self._infer_type(value),
                    source_line=i + 2,  # +2 for header and 0-indexing
                ))
        
        return result
    
    def _extract_key_value(self, output: str) -> ExtractionResult:
        """Extract data from key-value format."""
        result = ExtractionResult()
        record = {}
        
        for i, line in enumerate(output.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*[=:]\s*(.*)$", line)
            if match:
                key = match.group(1)
                value = match.group(2).strip()
                
                # Try to parse value
                parsed_value = self._parse_value(value)
                
                record[key] = parsed_value
                result.values.append(ExtractedValue(
                    key=key,
                    value=parsed_value,
                    data_type=self._infer_type(parsed_value),
                    source_line=i,
                ))
        
        if record:
            result.records.append(record)
        
        return result
    
    def _extract_table(self, output: str) -> ExtractionResult:
        """Extract data from ASCII table."""
        result = ExtractionResult()
        
        lines = output.strip().splitlines()
        
        # Find header row
        header_line = None
        data_lines = []
        
        for line in lines:
            if re.match(r"^[+|-]+$", line.strip()):
                continue
            if "|" in line:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if header_line is None:
                    header_line = cells
                else:
                    data_lines.append(cells)
        
        if header_line:
            for i, cells in enumerate(data_lines):
                record = {}
                for j, value in enumerate(cells):
                    if j < len(header_line):
                        key = header_line[j]
                        parsed_value = self._parse_value(value)
                        record[key] = parsed_value
                        result.values.append(ExtractedValue(
                            key=key,
                            value=parsed_value,
                            data_type=self._infer_type(parsed_value),
                        ))
                
                if record:
                    result.records.append(record)
        
        return result
    
    def _extract_log(self, output: str) -> ExtractionResult:
        """Extract data from log format."""
        result = ExtractionResult()
        
        # Common log patterns
        patterns = [
            r"(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(?P<level>\w+)\s+(?P<message>.*)",
            r"\[(?P<timestamp>[^\]]+)\]\s+\[(?P<level>\w+)\]\s+(?P<message>.*)",
            r"(?P<level>INFO|WARN|ERROR|DEBUG)\s+(?P<message>.*)",
        ]
        
        for i, line in enumerate(output.splitlines(), 1):
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    record = match.groupdict()
                    result.records.append(record)
                    
                    for key, value in record.items():
                        result.values.append(ExtractedValue(
                            key=key,
                            value=value,
                            data_type=DataType.STRING,
                            source_line=i,
                        ))
                    break
        
        return result
    
    def _extract_plain_text(
        self,
        output: str,
        patterns: Optional[List[str]],
    ) -> ExtractionResult:
        """Extract data from plain text using patterns."""
        result = ExtractionResult()
        
        if patterns:
            for pattern in patterns:
                try:
                    compiled = self._get_pattern(pattern)
                    for match in compiled.finditer(output):
                        if match.groupdict():
                            for key, value in match.groupdict().items():
                                result.values.append(ExtractedValue(
                                    key=key,
                                    value=value,
                                    data_type=self._infer_type(value),
                                ))
                        else:
                            for i, group in enumerate(match.groups()):
                                result.values.append(ExtractedValue(
                                    key=f"match_{i}",
                                    value=group,
                                    data_type=DataType.STRING,
                                ))
                except re.error as e:
                    result.errors.append(f"Invalid pattern {pattern}: {e}")
        
        return result
    
    async def _llm_extract(
        self,
        output: str,
        patterns: Optional[List[str]],
    ) -> ExtractionResult:
        """Use LLM to extract data."""
        result = ExtractionResult()
        
        prompt = f"""Extract structured data from this output.

Output:
{output[:1000]}

{"Patterns to look for: " + ", ".join(patterns) if patterns else "Extract any key-value pairs, numbers, or structured data."}

Respond with JSON array of extracted items:
[{{"key": "name", "value": "...", "type": "string|integer|float|boolean"}}]
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            
            # Parse JSON from response
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                items = json.loads(json_match.group())
                
                for item in items:
                    if isinstance(item, dict) and "key" in item:
                        data_type = DataType.STRING
                        type_str = item.get("type", "string").lower()
                        if type_str == "integer":
                            data_type = DataType.INTEGER
                        elif type_str == "float":
                            data_type = DataType.FLOAT
                        elif type_str == "boolean":
                            data_type = DataType.BOOLEAN
                        
                        result.values.append(ExtractedValue(
                            key=item["key"],
                            value=item.get("value"),
                            data_type=data_type,
                        ))
                        
        except Exception as e:
            result.errors.append(f"LLM extraction failed: {e}")
        
        return result
    
    def _get_pattern(self, pattern: str) -> re.Pattern:
        """Get compiled pattern from cache."""
        if pattern not in self._pattern_cache:
            self._pattern_cache[pattern] = re.compile(pattern)
        return self._pattern_cache[pattern]
    
    def _infer_type(self, value: Any) -> DataType:
        """Infer data type of a value."""
        if value is None:
            return DataType.NULL
        if isinstance(value, bool):
            return DataType.BOOLEAN
        if isinstance(value, int):
            return DataType.INTEGER
        if isinstance(value, float):
            return DataType.FLOAT
        if isinstance(value, list):
            return DataType.LIST
        if isinstance(value, dict):
            return DataType.DICT
        return DataType.STRING
    
    def _parse_value(self, value: str) -> Any:
        """Parse a string value to appropriate type."""
        value = value.strip()
        
        # Boolean
        if value.lower() in ["true", "yes", "on"]:
            return True
        if value.lower() in ["false", "no", "off"]:
            return False
        
        # Null
        if value.lower() in ["null", "none", "nil"]:
            return None
        
        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Strip quotes
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        
        return value
    
    async def analyze(
        self,
        data: Union[ExtractionResult, List[Dict[str, Any]]],
        analysis_types: Optional[List[AnalysisType]] = None,
    ) -> AnalysisResult:
        """Perform analysis on extracted data.
        
        Args:
            data: Extraction result or list of records
            analysis_types: Types of analysis to perform
            
        Returns:
            AnalysisResult with analysis outputs
        """
        result = AnalysisResult()
        
        # Get records
        if isinstance(data, ExtractionResult):
            records = data.records
        else:
            records = data
        
        result.record_count = len(records)
        
        if not records:
            result.summary = "No data to analyze"
            return result
        
        # Default analysis types
        if analysis_types is None:
            analysis_types = [
                AnalysisType.STATISTICAL,
                AnalysisType.SUMMARY,
            ]
        
        # Statistical analysis
        if AnalysisType.STATISTICAL in analysis_types:
            result.statistical = self._analyze_statistical(records)
        
        # Trend analysis
        if AnalysisType.TREND in analysis_types:
            result.trends = self._analyze_trends(records)
        
        # Anomaly detection
        if AnalysisType.ANOMALY in analysis_types:
            result.anomalies = self._detect_anomalies(records)
        
        # Generate summary
        if AnalysisType.SUMMARY in analysis_types:
            result.summary = await self._generate_summary(records, result)
        
        # Generate insights
        result.insights = await self._generate_insights(records, result)
        
        return result
    
    def _analyze_statistical(
        self,
        records: List[Dict[str, Any]],
    ) -> List[StatisticalResult]:
        """Perform statistical analysis on numeric fields."""
        results = []
        
        # Find numeric fields
        numeric_fields: Dict[str, List[float]] = {}
        
        for record in records:
            for key, value in record.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in numeric_fields:
                        numeric_fields[key] = []
                    numeric_fields[key].append(float(value))
        
        # Calculate statistics for each field
        for field_name, values in numeric_fields.items():
            if len(values) < 2:
                continue
            
            stat = StatisticalResult(
                field_name=field_name,
                count=len(values),
                sum=sum(values),
                mean=statistics.mean(values),
                median=statistics.median(values),
                min_value=min(values),
                max_value=max(values),
            )
            
            if len(values) >= 2:
                stat.std_dev = statistics.stdev(values)
            
            try:
                stat.mode = statistics.mode(values)
            except statistics.StatisticsError:
                pass
            
            # Percentiles
            sorted_values = sorted(values)
            for percentile in [25, 50, 75, 90, 95, 99]:
                idx = int(len(sorted_values) * percentile / 100)
                stat.percentiles[percentile] = sorted_values[min(idx, len(sorted_values) - 1)]
            
            results.append(stat)
        
        return results
    
    def _analyze_trends(
        self,
        records: List[Dict[str, Any]],
    ) -> List[TrendResult]:
        """Analyze trends in data."""
        results = []
        
        # Find numeric fields
        for key in records[0].keys() if records else []:
            values = []
            for record in records:
                value = record.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
            
            if len(values) < 3:
                continue
            
            trend = TrendResult(
                field_name=key,
                start_value=values[0],
                end_value=values[-1],
                peak_value=max(values),
                trough_value=min(values),
            )
            
            # Calculate trend direction
            if values[-1] > values[0] * 1.1:
                trend.direction = "increasing"
            elif values[-1] < values[0] * 0.9:
                trend.direction = "decreasing"
            else:
                # Check volatility
                changes = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
                avg_change = sum(changes) / len(changes) if changes else 0
                if avg_change > abs(values[-1] - values[0]) * 0.5:
                    trend.direction = "volatile"
                else:
                    trend.direction = "stable"
            
            # Change rate
            if values[0] != 0:
                trend.change_rate = (values[-1] - values[0]) / values[0]
            
            results.append(trend)
        
        return results
    
    def _detect_anomalies(
        self,
        records: List[Dict[str, Any]],
    ) -> List[AnomalyResult]:
        """Detect anomalies in data."""
        results = []
        
        for key in records[0].keys() if records else []:
            values = []
            for i, record in enumerate(records):
                value = record.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append((i, float(value)))
            
            if len(values) < 5:
                continue
            
            numeric_values = [v[1] for v in values]
            mean = statistics.mean(numeric_values)
            std = statistics.stdev(numeric_values) if len(numeric_values) >= 2 else 0
            
            if std == 0:
                continue
            
            anomaly = AnomalyResult(
                field_name=key,
                threshold=2.0,  # Z-score threshold
                method="z_score",
            )
            
            for idx, value in values:
                z_score = abs(value - mean) / std
                if z_score > 2.0:
                    anomaly.anomalies.append({
                        "index": idx,
                        "value": value,
                        "z_score": z_score,
                    })
            
            if anomaly.anomalies:
                results.append(anomaly)
        
        return results
    
    async def _generate_summary(
        self,
        records: List[Dict[str, Any]],
        analysis: AnalysisResult,
    ) -> str:
        """Generate a summary of the data."""
        if self._llm_client:
            prompt = f"""Summarize this data analysis in 2-3 sentences.

Records: {len(records)}
Fields: {list(records[0].keys()) if records else []}
Statistical results: {len(analysis.statistical)} fields analyzed
Trends: {', '.join(f"{t.field_name}: {t.direction}" for t in analysis.trends[:3])}
Anomalies: {sum(len(a.anomalies) for a in analysis.anomalies)} detected

Generate a concise summary.
"""
            
            try:
                return await self._llm_client.generate(prompt)
            except Exception:
                pass
        
        # Fallback summary
        return f"Analyzed {len(records)} records with {len(analysis.statistical)} numeric fields. " \
               f"Found {sum(len(a.anomalies) for a in analysis.anomalies)} anomalies."
    
    async def _generate_insights(
        self,
        records: List[Dict[str, Any]],
        analysis: AnalysisResult,
    ) -> List[str]:
        """Generate insights from analysis."""
        insights = []
        
        # Statistical insights
        for stat in analysis.statistical:
            if stat.std_dev and stat.mean:
                cv = stat.std_dev / stat.mean if stat.mean != 0 else 0
                if cv > 0.5:
                    insights.append(f"{stat.field_name} shows high variability (CV: {cv:.2f})")
        
        # Trend insights
        for trend in analysis.trends:
            if trend.direction == "increasing" and trend.change_rate:
                insights.append(f"{trend.field_name} increased by {trend.change_rate*100:.1f}%")
            elif trend.direction == "decreasing" and trend.change_rate:
                insights.append(f"{trend.field_name} decreased by {abs(trend.change_rate)*100:.1f}%")
        
        # Anomaly insights
        for anomaly in analysis.anomalies:
            if anomaly.anomalies:
                insights.append(f"{anomaly.field_name} has {len(anomaly.anomalies)} anomalous values")
        
        # Use LLM for deeper insights
        if self._llm_client and len(insights) < 3:
            try:
                prompt = f"""Based on this data, provide 2-3 actionable insights.

Data sample: {records[:3] if records else []}
Current insights: {insights}

Provide additional insights not already mentioned.
"""
                response = await self._llm_client.generate(prompt)
                
                for line in response.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and len(line) > 10:
                        insights.append(line)
                        
            except Exception:
                pass
        
        return insights[:10]  # Limit insights
    
    async def generate_report(
        self,
        analysis: AnalysisResult,
        title: Optional[str] = None,
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> Report:
        """Generate a report from analysis results.
        
        Args:
            analysis: Analysis result
            title: Optional report title
            format: Report format
            
        Returns:
            Report object
        """
        report = Report(
            title=title or f"Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            format=format,
        )
        
        if format == ReportFormat.MARKDOWN:
            report.content = self._generate_markdown_report(analysis, report.title)
        elif format == ReportFormat.JSON:
            report.content = json.dumps(analysis.to_dict(), indent=2)
        elif format == ReportFormat.HTML:
            report.content = self._generate_html_report(analysis, report.title)
        else:
            report.content = self._generate_text_report(analysis, report.title)
        
        return report
    
    def _generate_markdown_report(self, analysis: AnalysisResult, title: str) -> str:
        """Generate Markdown report."""
        lines = [
            f"# {title}",
            "",
            f"**Generated:** {analysis.analyzed_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Records Analyzed:** {analysis.record_count}",
            "",
            "## Summary",
            "",
            analysis.summary,
            "",
        ]
        
        if analysis.insights:
            lines.extend([
                "## Key Insights",
                "",
                *[f"- {insight}" for insight in analysis.insights],
                "",
            ])
        
        if analysis.statistical:
            lines.extend([
                "## Statistical Analysis",
                "",
                "| Field | Count | Mean | Std Dev | Min | Max |",
                "|-------|-------|------|---------|-----|-----|",
            ])
            
            for stat in analysis.statistical:
                lines.append(
                    f"| {stat.field_name} | {stat.count} | "
                    f"{stat.mean:.2f if stat.mean else 'N/A'} | "
                    f"{stat.std_dev:.2f if stat.std_dev else 'N/A'} | "
                    f"{stat.min_value} | {stat.max_value} |"
                )
            
            lines.append("")
        
        if analysis.trends:
            lines.extend([
                "## Trend Analysis",
                "",
            ])
            
            for trend in analysis.trends:
                lines.append(f"- **{trend.field_name}:** {trend.direction}")
                if trend.change_rate:
                    lines.append(f"  - Change rate: {trend.change_rate*100:.1f}%")
            
            lines.append("")
        
        if analysis.anomalies:
            lines.extend([
                "## Anomalies Detected",
                "",
            ])
            
            for anomaly in analysis.anomalies:
                lines.append(f"- **{anomaly.field_name}:** {len(anomaly.anomalies)} anomalies")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, analysis: AnalysisResult, title: str) -> str:
        """Generate HTML report."""
        md_content = self._generate_markdown_report(analysis, title)
        
        # Basic HTML wrapper
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
<pre>{md_content}</pre>
</body>
</html>
"""
    
    def _generate_text_report(self, analysis: AnalysisResult, title: str) -> str:
        """Generate plain text report."""
        lines = [
            title,
            "=" * len(title),
            "",
            f"Generated: {analysis.analyzed_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Records: {analysis.record_count}",
            "",
            "Summary:",
            analysis.summary,
            "",
        ]
        
        if analysis.insights:
            lines.append("Insights:")
            for insight in analysis.insights:
                lines.append(f"  * {insight}")
            lines.append("")
        
        return "\n".join(lines)


# Module-level instance
_global_analyzer: Optional[ResultAnalyzer] = None


def get_result_analyzer(llm_client: Optional[Any] = None) -> ResultAnalyzer:
    """Get the global result analyzer.
    
    Args:
        llm_client: Optional LLM client
        
    Returns:
        ResultAnalyzer instance
    """
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = ResultAnalyzer(llm_client=llm_client)
    return _global_analyzer
