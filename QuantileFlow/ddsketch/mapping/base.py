"""Base class for DDSketch mapping schemes."""

from abc import ABC, abstractmethod


class MappingScheme(ABC):
    """Abstract base class for different mapping schemes."""
    
    @abstractmethod
    def compute_bucket_index(self, value: float) -> int:
        """Compute the bucket index for a given value."""
        pass

    @abstractmethod
    def compute_value_from_index(self, index: int) -> float:
        """Compute the representative value for a given bucket index."""
        pass