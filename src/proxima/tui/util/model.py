"""Base model interface for Proxima TUI.

Provides base class for data models used in the TUI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime


@dataclass
class BaseModel(ABC):
    """Base class for TUI data models.
    
    Provides common functionality for serialization,
    validation, and change tracking.
    """
    
    _created_at: datetime = field(default_factory=datetime.now, repr=False)
    _modified_at: Optional[datetime] = field(default=None, repr=False)
    _dirty: bool = field(default=False, repr=False)
    
    def mark_modified(self) -> None:
        """Mark the model as modified."""
        self._modified_at = datetime.now()
        self._dirty = True
    
    def mark_clean(self) -> None:
        """Mark the model as clean (saved)."""
        self._dirty = False
    
    @property
    def is_dirty(self) -> bool:
        """Check if model has unsaved changes."""
        return self._dirty
    
    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        return self._created_at
    
    @property
    def modified_at(self) -> Optional[datetime]:
        """Get last modification timestamp."""
        return self._modified_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary.
        
        Returns:
            Dictionary representation
        """
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, BaseModel):
                    result[key] = value.to_dict()
                elif isinstance(value, datetime):
                    result[key] = value.isoformat()
                elif isinstance(value, list):
                    result[key] = [
                        v.to_dict() if isinstance(v, BaseModel) else v
                        for v in value
                    ]
                else:
                    result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModel":
        """Create model from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            Model instance
        """
        # Filter to only known fields
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls) if not f.name.startswith('_')}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)
    
    def copy(self) -> "BaseModel":
        """Create a copy of the model.
        
        Returns:
            Copy of the model
        """
        import copy
        return copy.deepcopy(self)
    
    def update(self, **kwargs) -> None:
        """Update model fields.
        
        Args:
            **kwargs: Fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key) and not key.startswith('_'):
                setattr(self, key, value)
                self.mark_modified()
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the model.
        
        Returns:
            True if valid, False otherwise
        """
        pass


@dataclass
class TimestampedModel(BaseModel):
    """Model with explicit timestamp tracking."""
    
    id: str = ""
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize timestamps."""
        if self.created is None:
            self.created = datetime.now()
        if self.updated is None:
            self.updated = self.created
    
    def mark_modified(self) -> None:
        """Mark as modified and update timestamp."""
        super().mark_modified()
        self.updated = datetime.now()
    
    def validate(self) -> bool:
        """Validate the model."""
        return bool(self.id)


@dataclass
class NamedModel(TimestampedModel):
    """Model with name and description."""
    
    name: str = ""
    description: str = ""
    
    def validate(self) -> bool:
        """Validate the model."""
        return super().validate() and bool(self.name)
