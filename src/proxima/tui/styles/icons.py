"""Unicode icons and symbols for Proxima TUI.

A comprehensive collection of icons for status, progress, and quantum concepts.
"""

# Status Icons
ICON_CHECK = "✓"
ICON_CROSS = "✗"
ICON_RUNNING = "●"
ICON_PAUSED = "⏸"
ICON_IDLE = "○"
ICON_WARNING = "⚠"
ICON_ERROR = "✗"
ICON_INFO = "ℹ"
ICON_SUCCESS = "✓"

# Backend Icons
ICON_BACKEND = "⬡"
ICON_HEALTHY = "●"
ICON_DEGRADED = "◐"
ICON_UNHEALTHY = "○"
ICON_UNAVAILABLE = "○"

# Progress Icons
ICON_STAGE_DONE = "✓"
ICON_STAGE_CURRENT = "●"
ICON_STAGE_PENDING = "○"
ICON_STAGE_ERROR = "✗"

# Quantum Icons
ICON_QUBIT = "⟩"
ICON_ENTANGLE = "⊗"
ICON_SUPERPOS = "∿"
ICON_MEASURE = "⟨⟩"
ICON_GATE = "▢"

# Model/LLM Icons
ICON_MODEL = "◈"
ICON_THINKING = "◆"
ICON_CONNECTED = "●"
ICON_DISCONNECTED = "○"

# Memory Icons
ICON_MEMORY = "▓"
ICON_MEMORY_EMPTY = "░"
ICON_MEMORY_WARNING = "[!]"
ICON_MEMORY_CRITICAL = "[!!]"
ICON_MEMORY_ABORT = "[!!!]"

# Master dictionary for easy access
ICONS = {
    "file": "📄",
    "folder": "📁",
    "edit": "✏️",
    "terminal": "💻",
    "cloud": "☁️",
    "check": ICON_CHECK,
    "cross": ICON_CROSS,
    "running": ICON_RUNNING,
    "paused": ICON_PAUSED,
    "idle": ICON_IDLE,
    "warning": ICON_WARNING,
    "error": ICON_ERROR,
    "info": ICON_INFO,
    "success": ICON_SUCCESS,
    "backend": ICON_BACKEND,
    "healthy": ICON_HEALTHY,
    "degraded": ICON_DEGRADED,
    "unhealthy": ICON_UNHEALTHY,
    "unavailable": ICON_UNAVAILABLE,
    "qubit": ICON_QUBIT,
    "entangle": ICON_ENTANGLE,
    "superpos": ICON_SUPERPOS,
    "measure": ICON_MEASURE,
    "gate": ICON_GATE,
    "model": ICON_MODEL,
    "thinking": ICON_THINKING,
    "connected": ICON_CONNECTED,
    "disconnected": ICON_DISCONNECTED,
}

# Radio Buttons
RADIO_ON = "●"
RADIO_OFF = "○"

# Checkboxes
CHECKBOX_ON = "☑"
CHECKBOX_OFF = "☐"

# Navigation
ICON_ARROW_UP = "↑"
ICON_ARROW_DOWN = "↓"
ICON_ARROW_LEFT = "←"
ICON_ARROW_RIGHT = "→"
ICON_ENTER = "↵"
ICON_TAB = "⇥"

# File/Session Icons
ICON_FILE = "📄"
ICON_FOLDER = "📁"
ICON_SESSION = "◉"
ICON_CHECKPOINT = "⚑"
ICON_NEW = "+new"
ICON_MODIFIED = "*"

# Action Icons
ICON_PLAY = "▶"
ICON_PAUSE = "⏸"
ICON_STOP = "⏹"
ICON_RELOAD = "⟳"
ICON_ROLLBACK = "↶"
ICON_EXPORT = "⤓"
ICON_IMPORT = "⤒"

# Diff Icons
ICON_DIFF_ADD = "+"
ICON_DIFF_REMOVE = "-"
ICON_DIFF_CHANGE = "~"

# Box Drawing Characters
BOX_TOP_LEFT = "┌"
BOX_TOP_RIGHT = "┐"
BOX_BOTTOM_LEFT = "└"
BOX_BOTTOM_RIGHT = "┘"
BOX_HORIZONTAL = "─"
BOX_VERTICAL = "│"
BOX_T_DOWN = "┬"
BOX_T_UP = "┴"
BOX_T_RIGHT = "├"
BOX_T_LEFT = "┤"
BOX_CROSS = "┼"

# Progress Bar Characters
PROGRESS_FILLED = "▓"
PROGRESS_EMPTY = "░"
PROGRESS_HALF = "▒"

# Spinners
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_DOTS = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
SPINNER_LINE = ["—", "\\", "|", "/"]

# Probability Bar Characters
PROB_BAR_FULL = "▓"
PROB_BAR_EMPTY = "░"

# Separators
SEP_THIN = "─"
SEP_THICK = "━"
SEP_DOUBLE = "═"
SEP_DOTTED = "┄"

# Ellipsis
ELLIPSIS = "…"
MORE_ITEMS = "… and {n} more"


def get_spinner_frame(index: int, style: str = "dots") -> str:
    """Get a spinner frame for animation.
    
    Args:
        index: Current frame index
        style: Spinner style ('dots', 'line', 'braille')
    
    Returns:
        Spinner character
    """
    spinners = {
        "dots": SPINNER_DOTS,
        "line": SPINNER_LINE,
        "braille": SPINNER_FRAMES,
    }
    frames = spinners.get(style, SPINNER_DOTS)
    return frames[index % len(frames)]


def get_progress_bar(percent: float, width: int = 10) -> str:
    """Generate a progress bar string.
    
    Args:
        percent: Progress percentage (0-100)
        width: Width of the bar in characters
    
    Returns:
        Progress bar string
    """
    filled = int(width * percent / 100)
    empty = width - filled
    return PROGRESS_FILLED * filled + PROGRESS_EMPTY * empty


def get_memory_indicator(level: str) -> str:
    """Get memory level indicator.
    
    Args:
        level: Memory level ('OK', 'INFO', 'WARNING', 'CRITICAL', 'ABORT')
    
    Returns:
        Memory indicator string
    """
    indicators = {
        "OK": "",
        "INFO": "",
        "WARNING": ICON_MEMORY_WARNING,
        "CRITICAL": ICON_MEMORY_CRITICAL,
        "ABORT": ICON_MEMORY_ABORT,
    }
    return indicators.get(level.upper(), "")


def get_health_icon(status: str) -> str:
    """Get backend health icon.
    
    Args:
        status: Health status ('HEALTHY', 'DEGRADED', 'UNHEALTHY', 'UNKNOWN')
    
    Returns:
        Health icon
    """
    icons = {
        "HEALTHY": ICON_HEALTHY,
        "DEGRADED": ICON_DEGRADED,
        "UNHEALTHY": ICON_UNHEALTHY,
        "UNKNOWN": ICON_UNAVAILABLE,
    }
    return icons.get(status.upper(), ICON_UNAVAILABLE)


def get_stage_icon(status: str) -> str:
    """Get stage status icon.
    
    Args:
        status: Stage status ('done', 'current', 'pending', 'error')
    
    Returns:
        Stage icon
    """
    icons = {
        "done": ICON_STAGE_DONE,
        "current": ICON_STAGE_CURRENT,
        "pending": ICON_STAGE_PENDING,
        "error": ICON_STAGE_ERROR,
    }
    return icons.get(status.lower(), ICON_STAGE_PENDING)
