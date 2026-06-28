import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon
from engines.engine_2d import Engine2D


class ToolpathEnv2D(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    HALF_EXTENT = 5.0  # workpiece spans [-5, 5] world units in x and y

    def __init__(self, config: dict, render_mode=None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        self.grid_size = config.get("grid_size", 64)
        self.max_steps = config.get("max_steps", 500)
        self.step_size = config.get("step_size", 0.4)  # world units per step
        self.tool_radius = config.get("tool_radius", 0.5)  # world units

        # Tool center limit (normalized) so the full tool disk stays inside
        self._pos_limit = 1.0 - self.tool_radius / self.HALF_EXTENT

        # Reward weights
        self.w_material = config.get("material_weight", 50.0)
        self.w_coverage = config.get("coverage_weight", 5.0)
        self.w_progress = config.get("progress_weight", 4.0)
        self.w_step = config.get("step_penalty", 0.03)
        self.w_collision = config.get("collision_penalty", 0.2)
        self.w_revisit = config.get("revisit_penalty", 0.02)
        self.w_idle = config.get("idle_penalty", 0.3)
        self.milestone_bonus = config.get("milestone_bonus", 5.0)
        self.shaping_gamma = config.get("gamma", 0.99)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "material_map": spaces.Box(
                    0, 1, shape=(self.grid_size, self.grid_size), dtype=np.float32
                ),
                "visited_map": spaces.Box(
                    0, 1, shape=(self.grid_size, self.grid_size), dtype=np.float32
                ),
                "collision_map": spaces.Box(
                    0, 1, shape=(self.grid_size, self.grid_size), dtype=np.float32
                ),
                "tool_pos": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "step_progress": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "material_remaining": spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        # Build default rectangular workpiece (10x10 centered at origin)
        workpiece = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        self.engine = Engine2D()
        self.engine.initialize(workpiece, tool_radius=self.tool_radius)

        self.tool_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self.current_step: int = 0
        self.collision_count: int = 0
        self.visited_map: np.ndarray = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float32
        )
        self.collision_map: np.ndarray = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float32
        )
        self._pending_milestones: list[float] = [0.25, 0.50, 0.75]

    def reset(  # type: ignore[override]
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self.engine.reset()
        self.current_step = 0
        self.collision_count = 0
        self.visited_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.collision_map = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float32
        )

        # Start each episode at a random valid position so the policy must
        # learn position-conditional coverage instead of one fixed trajectory.
        limit = self._pos_limit
        self.tool_pos = self.np_random.uniform(-limit, limit, size=2).astype(np.float32)
        wx = float(self.tool_pos[0] * self.HALF_EXTENT)
        wy = float(self.tool_pos[1] * self.HALF_EXTENT)
        self.engine.set_tool_position(wx, wy)
        # Mark the start cell visited so coverage reward counts only new ground
        self._grid_block(self.tool_pos)  # populates self._last_block bounds
        self.visited_map[self._by0 : self._by1, self._bx0 : self._bx1] = 1.0

        # Full workpiece -> tool sits on material -> distance 0 -> potential 0
        self._prev_potential = 0.0
        # Removal fractions whose one-time milestone bonus is still unclaimed
        self._pending_milestones = [0.25, 0.50, 0.75]

        return self._get_obs(), {}

    def _grid_block(self, tool_pos: np.ndarray, r: int = 2) -> None:
        """Compute the grid-cell block covered by the tool at tool_pos."""
        gx = int((tool_pos[0] + 1) / 2 * (self.grid_size - 1))
        gy = int((tool_pos[1] + 1) / 2 * (self.grid_size - 1))
        self._bgx, self._bgy = gx, gy
        self._bx0, self._bx1 = max(0, gx - r), min(self.grid_size, gx + r + 1)
        self._by0, self._by1 = max(0, gy - r), min(self.grid_size, gy + r + 1)

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        # Action is a normalized direction; step_size is in world units
        delta = action[:2] * (self.step_size / self.HALF_EXTENT)
        intended = self.tool_pos + delta

        # Collision: the move would push the tool disk past the workpiece
        # boundary. Clamp the center so the disk stays fully inside the box.
        collision = bool(np.any(np.abs(intended) > self._pos_limit + 1e-9))
        self.tool_pos = np.clip(intended, -self._pos_limit, self._pos_limit).astype(
            np.float32
        )
        world_x = float(self.tool_pos[0] * self.HALF_EXTENT)
        world_y = float(self.tool_pos[1] * self.HALF_EXTENT)

        result = self.engine.apply_move(world_x, world_y)
        result["collision"] = bool(collision or result["collision"])
        self.current_step += 1

        # Mark current tool position as visited, measuring newly-covered ground
        self._grid_block(self.tool_pos)
        x0, x1, y0, y1 = self._bx0, self._bx1, self._by0, self._by1
        already_visited = bool(self.visited_map[self._bgy, self._bgx] > 0.5)
        block = self.visited_map[y0:y1, x0:x1]
        new_cells = int(block.size - block.sum())  # cells flipping 0 -> 1
        coverage_gain = new_cells / (self.grid_size**2)
        self.visited_map[y0:y1, x0:x1] = 1.0

        # Remember where collisions happened (episode memory for the policy)
        if result["collision"]:
            self.collision_map[y0:y1, x0:x1] = 1.0
            self.collision_count += 1

        revisit = already_visited and result["removed_area"] == 0.0
        # Idle: neither cutting material nor covering new ground -> the move
        # made no progress of any kind. Penalized hard so parking in a cleared
        # region is strictly worse than traversing it toward fresh material.
        idle = result["removed_area"] == 0.0 and new_cells == 0
        terminated = bool(result["done"])
        truncated = bool(self.current_step >= self.max_steps)

        # Rasterize remaining material once; reused for shaping and observation
        material_map = self.engine.get_material_grid(self.grid_size)

        # Potential-based shaping toward the nearest remaining material.
        # Φ = 0 at terminal states preserves policy-invariance.
        phi_new = 0.0 if (terminated or truncated) else self._potential(material_map)
        shaping = self.w_progress * (
            self.shaping_gamma * phi_new - self._prev_potential
        )
        self._prev_potential = phi_new

        reward = self._compute_reward(result, coverage_gain, revisit, idle) + shaping

        # One-time milestone bonuses give reachable intermediate goals on the
        # way to the (initially never-observed) terminal completion bonus.
        removed_fraction = 1.0 - self.engine.get_remaining_area()
        while self._pending_milestones and removed_fraction >= self._pending_milestones[0]:
            self._pending_milestones.pop(0)
            reward += self.milestone_bonus

        if terminated:
            reward += 50.0

        if truncated and not terminated:
            reward -= 10.0

        info = {
            "collisions": self.collision_count,
            "progress_potential": phi_new,
        }
        return self._get_obs(material_map), reward, terminated, truncated, info

    def _potential(self, material_map: np.ndarray) -> float:
        """Negative normalized distance from the tool *rim* to the nearest material.

        Measured beyond the tool's reach: while material is within tool_radius
        the potential is 0, so a tool standing in its own freshly cut hole is
        not treated as "far from material" mid-cut.
        """
        mat = material_map > 0.5
        if not mat.any():
            return 0.0  # no material left
        dist = float(np.asarray(distance_transform_edt(~mat))[self._bgy, self._bgx])
        reach = self.tool_radius / (2 * self.HALF_EXTENT) * self.grid_size
        return -max(0.0, dist - reach) / self.grid_size

    def _get_obs(self, material_map: np.ndarray | None = None) -> dict:
        if material_map is None:
            material_map = self.engine.get_material_grid(self.grid_size)
        return {
            "material_map": material_map,
            "visited_map": self.visited_map.copy(),
            "collision_map": self.collision_map.copy(),
            "tool_pos": self.tool_pos.copy(),
            "step_progress": np.array(
                [self.current_step / self.max_steps], dtype=np.float32
            ),
            "material_remaining": np.array(
                [self.engine.get_remaining_area()], dtype=np.float32
            ),
        }

    def _compute_reward(
        self, result: dict, coverage_gain: float, revisit: bool, idle: bool
    ) -> float:
        r_material = self.w_material * result["removed_area"]
        r_coverage = self.w_coverage * coverage_gain
        r_step = -self.w_step
        r_collision = -self.w_collision * float(result["collision"])
        # Idle (no removal, no new coverage) is penalized hard; a mere revisit
        # while still covering new ground stays cheap so traversal is viable.
        r_progressless = -self.w_idle if idle else -self.w_revisit * float(revisit)
        return r_material + r_coverage + r_step + r_collision + r_progressless

    def render(self) -> None:
        if self.render_mode not in ("human", "rgb_array"):
            return

        # Imported lazily so training subprocesses never load matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        grid = self.engine.get_material_grid(self.grid_size)

        if not hasattr(self, "_fig"):
            plt.ion()
            self._fig, self._ax = plt.subplots(1, 1, figsize=(6, 6))

        self._ax.clear()

        # Draw material grid (white = material, black = removed)
        self._ax.imshow(
            grid, origin="lower", cmap="gray", vmin=0, vmax=1, extent=(-5, 5, -5, 5)
        )

        # Draw tool position
        world_x = float(self.tool_pos[0] * self.HALF_EXTENT)
        world_y = float(self.tool_pos[1] * self.HALF_EXTENT)
        tool_circle = patches.Circle(
            (world_x, world_y),
            radius=self.tool_radius,
            color="red",
            alpha=0.8,
            zorder=5,
        )
        self._ax.add_patch(tool_circle)

        # Labels
        self._ax.set_title(
            f"Step: {self.current_step} | "
            f"Material: {self.engine.get_remaining_area()*100:.1f}% | "
            f"Collisions: {self.collision_count}"
        )
        self._ax.set_xlim(-5, 5)
        self._ax.set_ylim(-5, 5)
        self._ax.set_xlabel("X (world)")
        self._ax.set_ylabel("Y (world)")

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.01)

    def close(self) -> None:
        import matplotlib.pyplot as plt

        if hasattr(self, "_fig"):
            plt.close(self._fig)
            del self._fig
            del self._ax
