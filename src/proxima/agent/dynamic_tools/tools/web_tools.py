"""Web Tools — Phase 16, Step 16.2: Web Fetch and Agentic Fetch.

Provides ``WebFetchTool`` for fetching content from a URL, and
``AgenticFetchTool`` which spawns a read-only sub-agent to analyse
fetched web content.

Architecture Note
-----------------
These tools use only ``urllib.request`` (stdlib) — no external HTTP
dependency is required.  The assistant architecture remains stable;
any integrated model operates dynamically for content analysis.
"""

from __future__ import annotations

import hashlib
import html.parser
import logging
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..tool_interface import (
    BaseTool,
    ToolParameter,
    ToolResult,
    ToolCategory,
    PermissionLevel,
    RiskLevel,
    ParameterType,
)
from ..execution_context import ExecutionContext
from ..tool_registry import register_tool

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

#: Maximum response body size (1 MB).
_MAX_RESPONSE_BYTES = 1_048_576

#: Content exceeding this size is saved to a temp file.
LARGE_CONTENT_THRESHOLD = 50_000  # characters

#: Truncated preview length for large content.
_PREVIEW_LENGTH = 5_000

#: Cache directory under the Proxima data folder.
_FETCH_CACHE_DIR = ".proxima/fetch_cache"

#: User-Agent header.
_USER_AGENT = "Proxima/1.0"


# ═══════════════════════════════════════════════════════════════════════════
#  HTML text extraction (stdlib-only)
# ═══════════════════════════════════════════════════════════════════════════

class _HTMLTextExtractor(html.parser.HTMLParser):
    """Minimal HTML → plain-text extractor using the stdlib parser."""

    _SKIP_TAGS = frozenset({"script", "style", "nav", "footer", "header", "noscript", "svg"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = " ".join(self._parts)
        # Collapse whitespace
        return re.sub(r"\s+", " ", raw).strip()


def _extract_text_from_html(html_content: str) -> str:
    """Strip HTML tags and return plain text."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html_content)
        return extractor.get_text()
    except Exception:
        # Fallback: brute-force regex strip
        text = re.sub(r"<[^>]+>", " ", html_content)
        return re.sub(r"\s+", " ", text).strip()


def _ensure_cache_dir() -> Path:
    """Create and return the fetch-cache directory."""
    cache = Path.home() / _FETCH_CACHE_DIR.replace("/", os.sep)
    cache.mkdir(parents=True, exist_ok=True)
    return cache


# ═══════════════════════════════════════════════════════════════════════════
#  WebFetchTool
# ═══════════════════════════════════════════════════════════════════════════

@register_tool
class WebFetchTool(BaseTool):
    """Fetch content from a URL and return the extracted text.

    Uses ``urllib.request`` (stdlib) — no external dependency needed.
    HTML responses are automatically stripped of tags.
    """

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch content from a URL and return the text"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SYSTEM

    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.NETWORK

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                param_type=ParameterType.URL,
                description="The URL to fetch",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                param_type=ParameterType.INTEGER,
                description="Timeout in seconds (max 60)",
                required=False,
                default=30,
            ),
        ]

    def _execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
    ) -> ToolResult:
        """Fetch the URL and return extracted text."""
        url: str = parameters.get("url", "")
        timeout: int = min(int(parameters.get("timeout", 30)), 60)

        # Validate URL scheme
        if not url.startswith(("http://", "https://")):
            return ToolResult.error_result(
                tool_name=self.name,
                error="Invalid URL: must start with http:// or https://",
            )

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": _USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw_bytes = resp.read(_MAX_RESPONSE_BYTES)
                # Best-effort decode
                charset = "utf-8"
                if "charset=" in content_type:
                    charset = content_type.split("charset=")[-1].split(";")[0].strip()
                body = raw_bytes.decode(charset, errors="replace")

        except urllib.error.HTTPError as exc:
            return ToolResult.error_result(
                tool_name=self.name,
                error=f"HTTP {exc.code}: {exc.reason}",
            )
        except urllib.error.URLError as exc:
            return ToolResult.error_result(
                tool_name=self.name,
                error=f"URL error: {exc.reason}",
            )
        except Exception as exc:
            return ToolResult.error_result(
                tool_name=self.name,
                error=f"Fetch failed: {exc}",
            )

        # Extract text if HTML
        if "html" in content_type.lower():
            text = _extract_text_from_html(body)
        else:
            text = body

        # Truncate to 50K chars
        text = text[:50_000]

        # If content is large, save full version to cache
        saved_path: Optional[str] = None
        if len(text) > LARGE_CONTENT_THRESHOLD:
            try:
                cache_dir = _ensure_cache_dir()
                url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                cache_file = cache_dir / f"fetch_{url_hash}_{int(time.time())}.txt"
                cache_file.write_text(text, encoding="utf-8")
                saved_path = str(cache_file)
            except Exception:
                pass  # Non-critical — still return the content

            preview = text[:_PREVIEW_LENGTH]
            note = f"\n\n[Full content ({len(text)} chars) saved to: {saved_path}]" if saved_path else ""
            return ToolResult.success_result(
                tool_name=self.name,
                result=preview + note,
                message=f"Fetched {url} — {len(text)} chars (preview returned)",
                metadata={"url": url, "chars": len(text), "cached_path": saved_path},
            )

        return ToolResult.success_result(
            tool_name=self.name,
            result=text,
            message=f"Fetched {url} — {len(text)} chars",
            metadata={"url": url, "chars": len(text)},
        )


# ═══════════════════════════════════════════════════════════════════════════
#  AgenticFetchTool
# ═══════════════════════════════════════════════════════════════════════════

@register_tool
class AgenticFetchTool(BaseTool):
    """Fetch a URL and analyse the content with an AI sub-agent.

    Combines ``WebFetchTool`` with a read-only sub-agent that can reason
    about the fetched content and extract the information the user needs.
    """

    @property
    def name(self) -> str:
        return "agentic_fetch"

    @property
    def description(self) -> str:
        return (
            "Search the web and/or fetch a URL, then analyze the content "
            "with an AI sub-agent"
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SYSTEM

    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.NETWORK

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                param_type=ParameterType.URL,
                description="URL to fetch (optional — if omitted, the prompt is used as-is)",
                required=False,
            ),
            ToolParameter(
                name="prompt",
                param_type=ParameterType.STRING,
                description="What information to find or extract",
                required=True,
            ),
        ]

    def _execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
    ) -> ToolResult:
        """Fetch content and run a sub-agent analysis."""
        url: str = parameters.get("url", "")
        prompt: str = parameters.get("prompt", "")

        if not prompt:
            return ToolResult.error_result(
                tool_name=self.name,
                error="A 'prompt' parameter is required.",
            )

        # Step 1: Fetch URL content (if provided)
        fetched_content = ""
        if url:
            fetch_tool = WebFetchTool()
            fetch_result = fetch_tool._execute({"url": url, "timeout": 30}, context)
            if fetch_result.success:
                fetched_content = str(fetch_result.result or "")
            else:
                return ToolResult.error_result(
                    tool_name=self.name,
                    error=f"Failed to fetch {url}: {fetch_result.error}",
                )

        # Step 2: Build an analysis prompt for the sub-agent
        if fetched_content:
            analysis_prompt = (
                f"The user wants to know: {prompt}\n\n"
                f"Here is the web content from {url}:\n---\n"
                f"{fetched_content[:20_000]}\n---\n\n"
                f"Analyze this content and provide a clear, concise answer to "
                f"the user's question. If the content doesn't contain the "
                f"answer, say so clearly."
            )
        else:
            analysis_prompt = (
                f"The user wants to know: {prompt}\n\n"
                f"No URL was provided. Answer based on your knowledge, or "
                f"explain what URL the user should fetch to find this information."
            )

        # Step 3: Run a sub-agent for analysis (if available)
        try:
            from proxima.agent.sub_agent import SubAgent, SubAgentConfig

            config = SubAgentConfig(
                name="Fetch Analysis",
                allowed_tools=["web_fetch", "read_file", "search_files"],
                model_preference="small",
                max_iterations=5,
                timeout_seconds=60,
            )

            # Try to get the LLM router from context or globals
            llm_router = None
            tool_registry = None
            if context is not None:
                llm_router = getattr(context, "llm_router", None)
                tool_registry = getattr(context, "tool_registry", None)

            if llm_router is not None:
                agent = SubAgent(
                    config=config,
                    llm_router=llm_router,
                    tool_registry=tool_registry,
                )
                result_text = agent.run(analysis_prompt)
                return ToolResult.success_result(
                    tool_name=self.name,
                    result=result_text,
                    message=f"Agentic fetch completed for {url or 'no URL'}",
                    metadata={"url": url, "prompt": prompt},
                )
        except ImportError:
            logger.debug("SubAgent not available for agentic fetch")
        except Exception as exc:
            logger.debug("Sub-agent analysis failed: %s", exc)

        # Fallback: return raw content with the prompt
        if fetched_content:
            return ToolResult.success_result(
                tool_name=self.name,
                result=(
                    f"Fetched content from {url}:\n\n"
                    f"{fetched_content[:10_000]}\n\n"
                    f"(Sub-agent unavailable — returning raw content. "
                    f"Original question: {prompt})"
                ),
                message="Agentic fetch completed (fallback — no sub-agent)",
                metadata={"url": url, "prompt": prompt, "fallback": True},
            )

        return ToolResult.error_result(
            tool_name=self.name,
            error="No URL provided and sub-agent unavailable.",
        )
