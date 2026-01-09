"""
AI/ML components module.

Phase 3: Intelligence & Decision Systems
- LLM routing (local/remote with consent)
- Backend auto-selection with explanation
- Insight engine for result interpretation
"""

from proxima.intelligence.llm_router import (
    LLMRouter,
    LLMRequest,
    LLMResponse,
    LLMProvider,
    ProviderRegistry,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    LMStudioProvider,
    LocalLLMDetector,
    APIKeyManager,
    ConsentGate,
    build_router,
    ProviderName,
)
from proxima.intelligence.selector import (
    BackendSelector,
    SelectionResult,
    SelectionInput,
    BackendScore,
)
from proxima.intelligence.insights import (
    InsightEngine,
    InsightReport,
    StatisticalMetrics,
)

__all__ = [
    # LLM Router
    "LLMRouter",
    "LLMRequest",
    "LLMResponse",
    "LLMProvider",
    "ProviderRegistry",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "LMStudioProvider",
    "LocalLLMDetector",
    "APIKeyManager",
    "ConsentGate",
    "build_router",
    "ProviderName",
    # Backend Selector
    "BackendSelector",
    "SelectionResult",
    "SelectionInput",
    "BackendScore",
    # Insight Engine
    "InsightEngine",
    "InsightReport",
    "StatisticalMetrics",
]
