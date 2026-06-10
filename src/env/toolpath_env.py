import gymnasium as gym
import numpy as np
from gymnasium import spaces
from shapely.geometry import Polygon
from engines.engine_2d import Engine2D
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ToolpathEnv2D(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: dict, render_mode=None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        self.grid_size = config.get("grid_size", 64)
        self.max_steps = config.get("max_steps", 500)

        # Reward weights
        self.w_material = config.get("material_weight", 10.0)
        self.w_step = config.get("step_penalty", 0.01)
        self.w_collision = config.get("collision_penalty", 5.0)
        self.w_wasted = config.get("wasted_move_penalty", 0.5)

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
        self.engine.initialize(workpiece, tool_radius=0.5)

        self.tool_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self.current_step: int = 0
        self.visited_map: np.ndarray = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float32
        )

    def reset(  # type: ignore[override]
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self.engine.reset()
        self.tool_pos = np.zeros(2, dtype=np.float32)
        self.current_step = 0
        self.visited_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        # Map normalized [-1,1] action to world coordinates [-5,5]
        delta = action[:2] * 0.4
        self.tool_pos = np.clip(self.tool_pos + delta, -1.0, 1.0)
        world_x = float(self.tool_pos[0] * 5)
        world_y = float(self.tool_pos[1] * 5)

        result = self.engine.apply_move(world_x, world_y)
        self.current_step += 1

        # Mark current tool position as visited
        gx = int((self.tool_pos[0] + 1) / 2 * (self.grid_size - 1))
        gy = int((self.tool_pos[1] + 1) / 2 * (self.grid_size - 1))
        r = 2
        x0, x1 = max(0, gx - r), min(self.grid_size, gx + r + 1)
        y0, y1 = max(0, gy - r), min(self.grid_size, gy + r + 1)
        self.visited_map[y0:y1, x0:x1] = 1.0

        reward = self._compute_reward(result)
        terminated = bool(result["done"])
        truncated = bool(self.current_step >= self.max_steps)

        if terminated:
            reward += 50.0

        if truncated and not terminated:
            reward -= 10.0

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> dict:
        return {
            "material_map": self.engine.get_material_grid(self.grid_size),
            "visited_map": self.visited_map.copy(),
            "tool_pos": self.tool_pos.copy(),
            "step_progress": np.array(
                [self.current_step / self.max_steps], dtype=np.float32
            ),
            "material_remaining": np.array(
                [self.engine.get_remaining_area()], dtype=np.float32
            ),
        }

    def _compute_reward(self, result: dict) -> float:
        r_material = self.w_material * result["removed_area"]
        r_step = -self.w_step
        r_collision = -self.w_collision * float(result["collision"])
        r_wasted = -self.w_wasted * float(result["removed_area"] == 0.0)
        return r_material + r_step + r_collision + r_wasted

    def render(self) -> None:
        if self.render_mode not in ("human", "rgb_array"):
            return

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
        world_x = float(self.tool_pos[0] * 5)
        world_y = float(self.tool_pos[1] * 5)
        tool_circle = patches.Circle(
            (world_x, world_y),
            radius=0.5,  # must match engine tool_radius
            color="red",
            alpha=0.8,
            zorder=5,
        )
        self._ax.add_patch(tool_circle)

        # Labels
        self._ax.set_title(
            f"Step: {self.current_step} | "
            f"Material: {self.engine.get_remaining_area()*100:.1f}%"
        )
        self._ax.set_xlim(-5, 5)
        self._ax.set_ylim(-5, 5)
        self._ax.set_xlabel("X (world)")
        self._ax.set_ylabel("Y (world)")

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.01)

    def close(self) -> None:
        if hasattr(self, "_fig"):
            plt.close(self._fig)
            del self._fig
            del self._ax
