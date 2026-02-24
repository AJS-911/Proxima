"""Agent inter-screen messages sub-package.

Re-exports all message classes from ``agent_messages`` so that
consumers can import directly from ``proxima.tui.messages``:

    from proxima.tui.messages import AgentTerminalStarted

or from the canonical module:

    from proxima.tui.messages.agent_messages import AgentTerminalStarted
"""

from .agent_messages import (  # noqa: F401
    AgentPlanStarted,
    AgentPlanStepCompleted,
    AgentResultReady,
    AgentTerminalCompleted,
    AgentTerminalOutput,
    AgentTerminalStarted,
)

__all__ = [
    "AgentTerminalStarted",
    "AgentTerminalOutput",
    "AgentTerminalCompleted",
    "AgentResultReady",
    "AgentPlanStarted",
    "AgentPlanStepCompleted",
]
