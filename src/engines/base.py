from abc import ABC, abstractmethod
import numpy as np


class MachiningEngine(ABC):
    @abstractmethod
    def initialize(self, workpiece_polygon, tool_radius: float) -> None:
        """Set up workpiece geometry and tool."""
        ...

    @abstractmethod
    def apply_move(self, x: float, y: float) -> dict:
        """
        Move tool to (x, y) in world coordinates.
        Returns dict with keys:
          - removed_area: float  (how much material was cut)
          - collision:    bool   (tool hit a fixture or out of bounds)
          - done:         bool   (all material removed)
        """
        ...

    @abstractmethod
    def get_material_grid(self, grid_size: int) -> np.ndarray:
        """Rasterize remaining material to a (grid_size x grid_size) float32 array."""
        ...

    @abstractmethod
    def get_remaining_area(self) -> float:
        """Return remaining material area as fraction of original [0, 1]."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Restore workpiece to initial state."""
        ...
