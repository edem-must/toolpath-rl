import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.affinity import scale
from engines.base import MachiningEngine
from shapely.ops import unary_union


class Engine2D(MachiningEngine):
    """
    2D milling simulation using Shapely polygon Boolean operations.
    The workpiece is a Shapely Polygon. Each tool move subtracts a
    circular disk (tool cross-section) from the remaining material.
    """

    def __init__(self) -> None:
        self._initial_workpiece: BaseGeometry = Polygon()
        self._remaining: BaseGeometry = Polygon()
        self._tool_radius: float = 1.0
        self._initial_area: float = 1.0
        self._bounds: tuple[float, float, float, float] = (-10, -10, 10, 10)

    def initialize(self, workpiece_polygon, tool_radius: float) -> None:
        self._initial_workpiece = workpiece_polygon
        self._remaining = workpiece_polygon
        self._tool_radius = tool_radius
        self._initial_area = workpiece_polygon.area
        self._bounds = workpiece_polygon.bounds

    def apply_move(self, x: float, y: float) -> dict:
        tool_circle = Point(x, y).buffer(self._tool_radius)

        minx, miny, maxx, maxy = self._bounds
        out_of_bounds = not (minx <= x <= maxx and miny <= y <= maxy)

        area_before = self._remaining.area
        self._remaining = self._remaining.difference(tool_circle)
        if self._remaining.geom_type == "MultiPolygon":
            self._remaining = unary_union(self._remaining)  # merge back
        area_after = self._remaining.area

        removed_area = (area_before - area_after) / max(self._initial_area, 1e-9)
        done = self._remaining.area / max(self._initial_area, 1e-9) < 0.01

        return {
            "removed_area": float(removed_area),
            "collision": bool(out_of_bounds),
            "done": bool(done),
        }

    def get_material_grid(self, grid_size: int) -> np.ndarray:
        minx, miny, maxx, maxy = self._bounds

        if self._remaining.is_empty:
            return np.zeros((grid_size, grid_size), dtype=np.float32)

        xs = np.linspace(minx, maxx, grid_size)
        ys = np.linspace(miny, maxy, grid_size)
        xx, yy = np.meshgrid(xs, ys)
        coords = np.column_stack([xx.ravel(), yy.ravel()])

        from shapely import contains_xy

        mask = contains_xy(self._remaining, coords[:, 0], coords[:, 1])

        return mask.reshape(grid_size, grid_size).astype(np.float32)

    def get_remaining_area(self) -> float:
        if self._remaining is None or self._initial_area == 0:
            return 0.0
        return float(self._remaining.area / self._initial_area)

    def reset(self) -> None:
        self._remaining = self._initial_workpiece
