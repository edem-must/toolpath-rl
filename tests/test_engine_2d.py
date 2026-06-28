import math

import pytest
from shapely.geometry import Polygon

from engines.engine_2d import Engine2D


def make_engine() -> Engine2D:
    engine = Engine2D()
    engine.initialize(
        Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)]), tool_radius=0.5
    )
    return engine


def test_first_move_cuts_disk():
    engine = make_engine()
    result = engine.apply_move(0.0, 0.0)
    disk_fraction = math.pi * 0.5**2 / 100.0
    assert result["removed_area"] == pytest.approx(disk_fraction, rel=0.01)
    assert result["collision"] is False
    assert result["done"] is False


def test_capsule_cut_removes_material_along_path():
    engine = make_engine()
    engine.set_tool_position(-4.0, 0.0)
    result = engine.apply_move(4.0, 0.0)
    # Swept capsule: 8x1 rectangle + two half-disks ~ 8.78 of 100
    capsule_fraction = (8.0 * 1.0 + math.pi * 0.5**2) / 100.0
    assert result["removed_area"] == pytest.approx(capsule_fraction, rel=0.01)
    # Mid-path material is gone even though the tool never stopped there
    grid = engine.get_material_grid(64)
    assert grid[32, 32] == 0.0


def test_out_of_bounds_collision_flag():
    engine = make_engine()
    result = engine.apply_move(20.0, 0.0)
    assert result["collision"] is True


def test_reset_restores_material():
    engine = make_engine()
    engine.apply_move(0.0, 0.0)
    assert engine.get_remaining_area() < 1.0
    engine.reset()
    assert engine.get_remaining_area() == pytest.approx(1.0)
