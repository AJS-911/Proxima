"""Result Processor for Dynamic Tool System.

This module processes tool execution results to:
- Standardize output formats
- Extract key information for LLM understanding
- Aggregate results from multi-tool executions
- Generate human-readable summaries
- Track result history for context

The processor ensures that tool results are formatted optimally
for both LLM consumption and user presentation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from .tool_interface import ToolResult

logger = logging.getLogger(__name__)


class ResultType(Enum):
    """Types of result content."""
    TEXT = "text"
    CODE = "code"
    JSON = "json"
    TABLE = "table"
    LIST = "list"
    TREE = "tree"
    DIFF = "diff"
    ERROR = "error"
    BINARY = "binary"
    MIXED = "mixed"


class ResultSeverity(Enum):
    """Severity level of result."""
    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ProcessedResult:
    """A processed tool result optimized for consumption."""
    # Source information
    tool_name: str
    operation_id: str
    
    # Result status
    success: bool
    severity: ResultSeverity
    
    # Content
    result_type: ResultType
    content: Any
    summary: str
    
    # Structured data (if applicable)
    structured_data: Optional[Dict[str, Any]] = None
    
    # For LLM
    llm_context: str = ""  # Optimized context for LLM understanding
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_time_ms: Optional[float] = None
    
    # Navigation aids
    related_files: List[str] = field(default_factory=list)
    related_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "operation_id": self.operation_id,
            "success": self.success,
            "severity": self.severity.value,
            "result_type": self.result_type.value,
            "content": self.content,
            "summary": self.summary,
            "structured_data": self.structured_data,
            "llm_context": self.llm_context,
            "timestamp": self.timestamp,
            "execution_time_ms": self.execution_time_ms,
            "related_files": self.related_files,
            "related_actions": self.related_actions,
        }
    
    def to_user_message(self) -> str:
        """Format for display to user."""
        icon = {
            ResultSeverity.SUCCESS: "✅",
            ResultSeverity.INFO: "ℹ️",
            ResultSeverity.WARNING: "⚠️",
            ResultSeverity.ERROR: "❌",
            ResultSeverity.CRITICAL: "🚨",
        }.get(self.severity, "")
        
        lines = [f"{icon} **{self.summary}**"]
        
        if self.result_type == ResultType.CODE:
            lines.append(f"\n```\n{self.content}\n```")
        elif self.result_type == ResultType.JSON:
            if isinstance(self.content, dict):
                lines.append(f"\n```json\n{json.dumps(self.content, indent=2)}\n```")
            else:
                lines.append(f"\n```json\n{self.content}\n```")
        elif self.result_type == ResultType.TABLE:
            lines.append(self._format_table(self.content))
        elif self.result_type == ResultType.LIST:
            for item in self.content[:20]:  # Limit displayed items
                lines.append(f"  • {item}")
            if len(self.content) > 20:
                lines.append(f"  ... and {len(self.content) - 20} more items")
        elif self.result_type == ResultType.DIFF:
            lines.append(f"\n```diff\n{self.content}\n```")
        elif self.result_type == ResultType.ERROR:
            lines.append(f"\n**Error:** {self.content}")
        else:
            if self.content:
                content_str = str(self.content)
                if len(content_str) > 500:
                    lines.append(f"\n{content_str[:500]}...")
                else:
                    lines.append(f"\n{content_str}")
        
        if self.related_files:
            lines.append(f"\n**Related files:** {', '.join(self.related_files[:5])}")
        
        return "\n".join(lines)
    
    def _format_table(self, data: Any) -> str:
        """Format tabular data."""
        if not data:
            return ""
        
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                # List of dicts - format as table
                headers = list(data[0].keys())
                rows = []
                
                # Header row
                rows.append(" | ".join(headers))
                rows.append(" | ".join("-" * len(h) for h in headers))
                
                # Data rows
                for item in data[:20]:
                    row = [str(item.get(h, "")) for h in headers]
                    rows.append(" | ".join(row))
                
                if len(data) > 20:
                    rows.append(f"... and {len(data) - 20} more rows")
                
                return "\n" + "\n".join(rows)
        
        return str(data)


@dataclass
class AggregatedResult:
    """Results aggregated from multiple tool executions."""
    plan_name: str
    total_steps: int
    successful_steps: int
    failed_steps: int
    results: List[ProcessedResult]
    overall_success: bool
    overall_summary: str
    combined_llm_context: str
    execution_time_total_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "results": [r.to_dict() for r in self.results],
            "overall_success": self.overall_success,
            "overall_summary": self.overall_summary,
            "combined_llm_context": self.combined_llm_context,
            "execution_time_total_ms": self.execution_time_total_ms,
            "timestamp": self.timestamp,
        }
    
    def to_user_message(self) -> str:
        """Format for display to user."""
        icon = "✅" if self.overall_success else "❌"
        
        lines = [
            f"{icon} **{self.plan_name}**",
            f"Completed: {self.successful_steps}/{self.total_steps} steps",
            "",
        ]
        
        for result in self.results:
            step_icon = "✅" if result.success else "❌"
            lines.append(f"{step_icon} {result.tool_name}: {result.summary}")
        
        return "\n".join(lines)


class ResultProcessor:
    """Processes tool results for optimal consumption.
    
    The processor provides:
    - Result standardization
    - Content type detection
    - Summary generation
    - LLM context optimization
    - Result aggregation
    """
    
    def __init__(self):
        """Initialize the result processor."""
        self._result_history: List[ProcessedResult] = []
        self._max_history_size = 100
    
    def process(
        self,
        tool_result: ToolResult,
        tool_name: str,
        operation_id: str,
        execution_time_ms: Optional[float] = None,
    ) -> ProcessedResult:
        """Process a tool result.
        
        Args:
            tool_result: The raw tool result
            tool_name: Name of the tool that produced result
            operation_id: Operation identifier
            execution_time_ms: Execution time in milliseconds
            
        Returns:
            Processed result
        """
        # Determine result type
        result_type = self._detect_result_type(tool_result)
        
        # Determine severity
        severity = self._determine_severity(tool_result)
        
        # Extract content
        content = self._extract_content(tool_result)
        
        # Generate summary
        summary = self._generate_summary(tool_result, tool_name)
        
        # Extract structured data
        structured_data = self._extract_structured_data(tool_result)
        
        # Generate LLM context
        llm_context = self._generate_llm_context(tool_result, tool_name, result_type)
        
        # Extract related files
        related_files = self._extract_related_files(tool_result)
        
        # Suggest related actions
        related_actions = self._suggest_related_actions(tool_result, tool_name)
        
        processed = ProcessedResult(
            tool_name=tool_name,
            operation_id=operation_id,
            success=tool_result.success,
            severity=severity,
            result_type=result_type,
            content=content,
            summary=summary,
            structured_data=structured_data,
            llm_context=llm_context,
            execution_time_ms=execution_time_ms,
            related_files=related_files,
            related_actions=related_actions,
        )
        
        # Store in history
        self._result_history.append(processed)
        if len(self._result_history) > self._max_history_size:
            self._result_history = self._result_history[-self._max_history_size:]
        
        return processed
    
    def _detect_result_type(self, result: ToolResult) -> ResultType:
        """Detect the type of result content."""
        if not result.success:
            return ResultType.ERROR
        
        data = result.data
        
        if data is None:
            return ResultType.TEXT
        
        if isinstance(data, dict):
            # Check for specific patterns
            if "diff" in data or "changes" in data:
                return ResultType.DIFF
            if "rows" in data or "table" in data:
                return ResultType.TABLE
            if "tree" in data or "structure" in data:
                return ResultType.TREE
            return ResultType.JSON
        
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                return ResultType.TABLE
            return ResultType.LIST
        
        if isinstance(data, str):
            # Check content patterns
            if data.startswith(("diff ", "---", "+++")):
                return ResultType.DIFF
            if data.startswith(("{", "[")):
                try:
                    json.loads(data)
                    return ResultType.JSON
                except:
                    pass
            if "```" in data or self._looks_like_code(data):
                return ResultType.CODE
        
        return ResultType.TEXT
    
    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like code."""
        code_indicators = [
            "def ", "class ", "import ", "from ",  # Python
            "function ", "const ", "let ", "var ",  # JavaScript
            "public ", "private ", "void ", "int ",  # Java/C++
            "fn ", "let ", "use ", "mod ",  # Rust
        ]
        return any(indicator in text for indicator in code_indicators)
    
    def _determine_severity(self, result: ToolResult) -> ResultSeverity:
        """Determine result severity."""
        if not result.success:
            error = result.error or ""
            if "critical" in error.lower() or "fatal" in error.lower():
                return ResultSeverity.CRITICAL
            return ResultSeverity.ERROR
        
        message = result.message or ""
        if "warning" in message.lower():
            return ResultSeverity.WARNING
        
        return ResultSeverity.SUCCESS
    
    def _extract_content(self, result: ToolResult) -> Any:
        """Extract the main content from result."""
        if not result.success:
            return result.error
        
        if result.data is not None:
            return result.data
        
        return result.message or ""
    
    def _generate_summary(self, result: ToolResult, tool_name: str) -> str:
        """Generate a concise summary of the result."""
        if not result.success:
            return f"{tool_name} failed: {result.error or 'Unknown error'}"[:100]
        
        if result.message:
            return result.message[:100]
        
        # Generate based on data
        data = result.data
        
        if isinstance(data, list):
            return f"{tool_name} returned {len(data)} items"
        
        if isinstance(data, dict):
            keys = list(data.keys())[:3]
            return f"{tool_name} returned data with keys: {', '.join(keys)}"
        
        if isinstance(data, str):
            if len(data) > 100:
                return f"{tool_name} returned {len(data)} characters"
            return data
        
        return f"{tool_name} completed successfully"
    
    def _extract_structured_data(self, result: ToolResult) -> Optional[Dict[str, Any]]:
        """Extract structured data from result."""
        if not result.success or result.data is None:
            return None
        
        if isinstance(result.data, dict):
            return result.data
        
        if isinstance(result.data, list) and len(result.data) > 0:
            if isinstance(result.data[0], dict):
                return {"items": result.data, "count": len(result.data)}
        
        return None
    
    def _generate_llm_context(
        self, 
        result: ToolResult, 
        tool_name: str,
        result_type: ResultType,
    ) -> str:
        """Generate optimized context for LLM understanding."""
        lines = [f"Tool '{tool_name}' execution result:"]
        
        if not result.success:
            lines.append(f"Status: FAILED")
            lines.append(f"Error: {result.error}")
        else:
            lines.append(f"Status: SUCCESS")
            
            if result.message:
                lines.append(f"Message: {result.message}")
            
            if result.data:
                data = result.data
                
                if result_type == ResultType.LIST:
                    lines.append(f"Returned {len(data)} items")
                    if len(data) <= 10:
                        for item in data:
                            lines.append(f"  - {item}")
                    else:
                        for item in data[:5]:
                            lines.append(f"  - {item}")
                        lines.append(f"  ... and {len(data) - 5} more")
                
                elif result_type == ResultType.TABLE:
                    if isinstance(data, list) and len(data) > 0:
                        lines.append(f"Returned table with {len(data)} rows")
                        if isinstance(data[0], dict):
                            lines.append(f"Columns: {', '.join(data[0].keys())}")
                
                elif result_type == ResultType.JSON:
                    # Truncate large JSON
                    json_str = json.dumps(data, indent=2, default=str)
                    if len(json_str) > 500:
                        lines.append(f"Data (truncated):\n{json_str[:500]}...")
                    else:
                        lines.append(f"Data:\n{json_str}")
                
                elif result_type == ResultType.TEXT:
                    text = str(data)
                    if len(text) > 300:
                        lines.append(f"Output (truncated):\n{text[:300]}...")
                    else:
                        lines.append(f"Output:\n{text}")
        
        return "\n".join(lines)
    
    def _extract_related_files(self, result: ToolResult) -> List[str]:
        """Extract file paths mentioned in result."""
        files = []
        
        if not result.success or not result.data:
            return files
        
        def find_paths(obj: Any, depth: int = 0):
            if depth > 5:
                return
            
            if isinstance(obj, str):
                # Look for path patterns
                if "/" in obj or "\\" in obj:
                    if obj.endswith((".py", ".js", ".ts", ".json", ".yaml", ".yml", 
                                    ".md", ".txt", ".cfg", ".ini", ".sh", ".ps1")):
                        files.append(obj)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    if key.lower() in ("path", "file", "filename", "filepath"):
                        if isinstance(value, str):
                            files.append(value)
                    find_paths(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj[:20]:  # Limit search
                    find_paths(item, depth + 1)
        
        find_paths(result.data)
        
        return list(set(files))[:10]  # Unique, limited
    
    def _suggest_related_actions(
        self, 
        result: ToolResult, 
        tool_name: str
    ) -> List[str]:
        """Suggest related actions based on result."""
        actions = []
        
        if not result.success:
            actions.append("Retry operation")
            actions.append("Check error details")
            return actions
        
        # Suggest based on tool type
        tool_lower = tool_name.lower()
        
        if "read" in tool_lower or "list" in tool_lower:
            actions.append("Edit file")
            actions.append("Search in files")
        
        if "write" in tool_lower or "create" in tool_lower:
            actions.append("Verify changes")
            actions.append("Commit changes")
        
        if "git" in tool_lower:
            actions.append("View git status")
            actions.append("Push changes")
        
        if "terminal" in tool_lower or "execute" in tool_lower:
            actions.append("Run another command")
            actions.append("Check output")
        
        return actions[:3]
    
    def aggregate_results(
        self,
        results: List[ProcessedResult],
        plan_name: str,
    ) -> AggregatedResult:
        """Aggregate multiple results.
        
        Args:
            results: List of processed results
            plan_name: Name of the execution plan
            
        Returns:
            Aggregated result
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_time = sum(
            r.execution_time_ms or 0 
            for r in results
        )
        
        # Generate combined LLM context
        context_parts = [f"Execution plan '{plan_name}' results:"]
        context_parts.append(f"Total steps: {len(results)}")
        context_parts.append(f"Successful: {len(successful)}")
        context_parts.append(f"Failed: {len(failed)}")
        context_parts.append("")
        
        for r in results:
            status = "✓" if r.success else "✗"
            context_parts.append(f"{status} {r.tool_name}: {r.summary}")
        
        # Generate overall summary
        if failed:
            overall_summary = f"Plan completed with {len(failed)} failure(s)"
        else:
            overall_summary = f"Plan completed successfully ({len(successful)} steps)"
        
        return AggregatedResult(
            plan_name=plan_name,
            total_steps=len(results),
            successful_steps=len(successful),
            failed_steps=len(failed),
            results=results,
            overall_success=len(failed) == 0,
            overall_summary=overall_summary,
            combined_llm_context="\n".join(context_parts),
            execution_time_total_ms=total_time,
        )
    
    def get_history(
        self, 
        tool_name: Optional[str] = None,
        limit: int = 20,
    ) -> List[ProcessedResult]:
        """Get result history.
        
        Args:
            tool_name: Filter by tool name
            limit: Maximum results to return
            
        Returns:
            List of processed results
        """
        results = self._result_history
        
        if tool_name:
            results = [r for r in results if r.tool_name == tool_name]
        
        return results[-limit:]
    
    def clear_history(self):
        """Clear result history."""
        self._result_history = []


# Global processor instance
_global_processor: Optional[ResultProcessor] = None


def get_result_processor() -> ResultProcessor:
    """Get the global result processor instance."""
    global _global_processor
    if _global_processor is None:
        _global_processor = ResultProcessor()
    return _global_processor
