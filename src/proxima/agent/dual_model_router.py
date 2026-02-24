"""Dual-Model Router — Phase 16, Step 16.5.

Wraps ``LLMRouter`` to support configuring separate models for
different purposes (large = main reasoning; small = summaries,
sub-agents, titles).

Architecture Note
-----------------
This is a new file in ``agent/`` — NOT in ``llm_integration.py`` —
because ``DualModelRouter`` wraps ``LLMRouter`` from
``src/proxima/intelligence/llm_router.py``, a different package.
Placing it in ``llm_integration.py`` would create a cross-package
dependency.
"""

from __future__ import annotations

import copy
import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from proxima.config.settings import Settings
    from proxima.intelligence.llm_router import LLMRouter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Model Role
# ═══════════════════════════════════════════════════════════════════════════

class ModelRole(Enum):
    """Identifies the purpose that a model is used for."""

    LARGE = "large"   # Complex reasoning, code generation, main agent
    SMALL = "small"   # Title generation, summarization, sub-agents


# ═══════════════════════════════════════════════════════════════════════════
#  DualModelRouter
# ═══════════════════════════════════════════════════════════════════════════

class DualModelRouter:
    """Provides two ``LLMRouter`` instances — one for *large* (main agent)
    tasks and one for *small* (summarization / sub-agent) tasks.

    The large router uses whatever is already configured in ``Settings.llm``.
    The small router reads ``agent.models.small`` from ``configs/default.yaml``
    (accessed via the raw config dict since the ``Settings`` Pydantic model does
    not have an ``agent`` field).

    Parameters
    ----------
    settings : Settings
        The application settings object.
    consent_prompt : callable or None
        Consent prompt function passed through to each ``LLMRouter``.
    agent_config : dict or None
        The ``agent`` subsection from the raw YAML config.  If *None*,
        falls back to a same-model setup (large == small).
    """

    def __init__(
        self,
        settings: "Settings",
        consent_prompt: Optional[Callable[[str], bool]] = None,
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Lazy import to avoid circular deps
        try:
            from proxima.intelligence.llm_router import LLMRouter
        except ImportError:
            LLMRouter = None  # type: ignore[misc]

        self._consent_prompt = consent_prompt
        self._LLMRouter = LLMRouter

        # ── Large router (uses base settings as-is) ───────────────────
        if LLMRouter is not None:
            self._large: Optional["LLMRouter"] = LLMRouter(
                settings=settings,
                consent_prompt=consent_prompt,
            )
        else:
            self._large = None

        # ── Small router ──────────────────────────────────────────────
        agent_cfg = agent_config or {}
        models_cfg = agent_cfg.get("models", {}) if isinstance(agent_cfg, dict) else {}
        small_cfg = models_cfg.get("small", None)

        if small_cfg and isinstance(small_cfg, dict) and LLMRouter is not None:
            try:
                small_settings = settings.model_copy(deep=True)
                small_provider = small_cfg.get("provider")
                small_model = small_cfg.get("model")
                if small_provider:
                    small_settings.llm.provider = small_provider
                if small_model:
                    small_settings.llm.model = small_model
                self._small: Optional["LLMRouter"] = LLMRouter(
                    settings=small_settings,
                    consent_prompt=consent_prompt,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to create small-model router: %s. "
                    "Falling back to large model.",
                    exc,
                )
                self._small = self._large
        else:
            # No small config — same model for both roles
            self._small = self._large

    # ── Public API ────────────────────────────────────────────────────

    def get_router(self, role: ModelRole = ModelRole.LARGE) -> Optional["LLMRouter"]:
        """Return the ``LLMRouter`` for the requested role.

        Parameters
        ----------
        role : ModelRole
            ``LARGE`` for the main agent, ``SMALL`` for summaries,
            sub-agents, and titles.

        Returns
        -------
        LLMRouter or None
            The router instance, or *None* if LLM is unavailable.
        """
        if role == ModelRole.SMALL:
            return self._small
        return self._large

    def set_models(
        self,
        settings: "Settings",
        small_settings: Optional["Settings"] = None,
    ) -> None:
        """Replace both routers at runtime.

        Parameters
        ----------
        settings : Settings
            Settings for the large model.
        small_settings : Settings or None
            Settings for the small model. If *None*, the small router
            mirrors the large one.
        """
        if self._LLMRouter is None:
            logger.warning("LLMRouter unavailable — cannot set models.")
            return

        try:
            self._large = self._LLMRouter(
                settings=settings,
                consent_prompt=self._consent_prompt,
            )
            if small_settings is not None:
                self._small = self._LLMRouter(
                    settings=small_settings,
                    consent_prompt=self._consent_prompt,
                )
            else:
                self._small = self._large
        except Exception as exc:
            logger.warning("Failed to set new models: %s", exc)

    @property
    def large(self) -> Optional["LLMRouter"]:
        """Direct access to the large-model router."""
        return self._large

    @property
    def small(self) -> Optional["LLMRouter"]:
        """Direct access to the small-model router."""
        return self._small
