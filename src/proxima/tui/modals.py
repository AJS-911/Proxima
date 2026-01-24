"""Backward compatibility shim for proxima.tui.modals.

This module provides backward compatibility with the old TUI architecture.
New code should use proxima.tui.dialogs instead.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from textual.screen import ModalScreen
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Button, Input


class DialogResult(Enum):
    """Result of a dialog interaction."""
    OK = "ok"
    CANCEL = "cancel"
    YES = "yes"
    NO = "no"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    DISMISSED = "dismissed"
    DECLINED = "declined"


@dataclass
class ModalResponse:
    """Response from a modal dialog."""
    result: DialogResult
    data: Optional[Any] = None
    
    @property
    def confirmed(self) -> bool:
        """Check if dialog was confirmed."""
        return self.result in (DialogResult.OK, DialogResult.YES, DialogResult.CONFIRMED)
    
    @property
    def cancelled(self) -> bool:
        """Check if dialog was cancelled."""
        return self.result in (DialogResult.CANCEL, DialogResult.NO, DialogResult.CANCELLED, DialogResult.DECLINED)


class ConfirmModal(ModalScreen):
    """Confirmation modal dialog."""
    
    def __init__(
        self,
        title: str = "Confirm",
        message: str = "Are you sure?",
        confirm_text: str = "OK",
        cancel_text: str = "Cancel",
        **kwargs,
    ):
        super().__init__()
        self._title = title
        self._message = message
        self._confirm_text = confirm_text
        self._cancel_text = cancel_text


class InputModal(ModalScreen):
    """Input modal dialog."""
    
    def __init__(
        self,
        title: str = "Input",
        prompt: str = "",
        label: str = "Enter value:",
        default: str = "",
        default_value: str = "",
        placeholder: str = "",
        **kwargs,
    ):
        super().__init__()
        self._title = title
        self._prompt = prompt or label
        self._label = label
        self._default = default or default_value
        self._default_value = default_value or default
        self._placeholder = placeholder


class ChoiceModal(ModalScreen):
    """Choice modal dialog."""
    
    def __init__(
        self,
        title: str = "Choose",
        message: str = "Select an option:",
        choices: Optional[List] = None,
        allow_multiple: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._title = title
        self._message = message
        self._choices = choices or []
        self._allow_multiple = allow_multiple


class ProgressModal(ModalScreen):
    """Progress modal dialog."""
    
    def __init__(
        self,
        title: str = "Progress",
        message: str = "Processing...",
        cancelable: bool = True,
        **kwargs,
    ):
        super().__init__()
        self._title = title
        self._message = message
        self._cancelable = cancelable


class ErrorModal(ModalScreen):
    """Error modal dialog."""
    
    def __init__(
        self,
        title: str = "Error",
        message: str = "An error occurred",
        details: str = "",
        error_details: str = "",
        **kwargs,
    ):
        super().__init__()
        self._title = title
        self._message = message
        self._details = details or error_details
        self._error_details = error_details or details


class ConsentModal(ModalScreen):
    """Consent modal dialog."""
    
    def __init__(
        self,
        title: str = "Consent Required",
        operation: str = "",
        implications: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        self._title = title
        self._operation = operation
        self._implications = implications or []
        self._details = details or {}


@dataclass
class FormField:
    """Form field definition."""
    key: str = ""
    name: str = ""
    label: str = ""
    field_type: str = "text"
    default: str = ""
    default_value: str = ""
    required: bool = False
    placeholder: str = ""
    choices: List[Tuple[str, str]] = field(default_factory=list)
    
    def __post_init__(self):
        # Support both 'key' and 'name' for backward compat
        if not self.name and self.key:
            self.name = self.key
        if not self.key and self.name:
            self.key = self.name
        if not self.default and self.default_value:
            self.default = self.default_value
        if not self.default_value and self.default:
            self.default_value = self.default


class FormModal(ModalScreen):
    """Form modal dialog."""
    
    def __init__(
        self,
        title: str = "Form",
        fields: Optional[List[FormField]] = None,
        **kwargs,
    ):
        super().__init__()
        self._title = title
        self._fields = fields or []


__all__ = [
    "DialogResult",
    "ModalResponse",
    "ConfirmModal",
    "InputModal",
    "ChoiceModal",
    "ProgressModal",
    "ErrorModal",
    "ConsentModal",
    "FormField",
    "FormModal",
]
