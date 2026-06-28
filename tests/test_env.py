import numpy as np
from stable_baselines3.common.env_checker import check_env

from env.toolpath_env import ToolpathEnv2D

CONFIG = {"grid_size": 64, "max_steps": 500}


def test_check_env():
    env = ToolpathEnv2D(config=CONFIG)
    check_env(env, warn=True)
    env.close()


def test_random_rollout():
    env = ToolpathEnv2D(config=CONFIG)
    obs, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert "collision_map" in obs
        if terminated or truncated:
            break
    env.close()


def test_wall_collision_detected_and_remembered():
    env = ToolpathEnv2D(config=CONFIG)
    env.reset(seed=0)
    obs, info = {}, {}
    # Push into the bottom-left corner long enough to hit the wall
    for _ in range(40):
        obs, reward, terminated, truncated, info = env.step(
            np.array([-1.0, -1.0, 0.0], dtype=np.float32)
        )
    # Tool disk must stay fully inside the workpiece
    limit = 1.0 - env.tool_radius / env.HALF_EXTENT
    assert np.all(np.abs(obs["tool_pos"]) <= limit + 1e-6)
    # Collisions were detected and remembered in the observation
    assert info["collisions"] > 0
    assert obs["collision_map"].sum() > 0
    env.close()


def test_random_start_varies():
    env = ToolpathEnv2D(config=CONFIG)
    obs_a, _ = env.reset(seed=1)
    obs_b, _ = env.reset(seed=2)
    assert not np.allclose(obs_a["tool_pos"], obs_b["tool_pos"])
    env.close()


def test_fresh_move_beats_revisit():
    env = ToolpathEnv2D(config=CONFIG)
    env.reset(seed=0)
    # Move in +x into fresh material
    fwd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    _, fresh_reward, _, _, _ = env.step(fwd)
    # Step back over the just-cut / just-visited strip
    back = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    _, revisit_reward, _, _, _ = env.step(back)
    assert fresh_reward > 0.0
    assert fresh_reward > revisit_reward
    env.close()


def test_idle_penalized_harder_than_traversal():
    env = ToolpathEnv2D(config=CONFIG)
    env.reset(seed=0)
    stay = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    env.step(stay)  # first stationary step still cuts the disk under the tool
    _, idle_reward, _, _, _ = env.step(stay)  # no removal, no new coverage
    # Parking must cost clearly more than a cheap revisit/traversal step
    assert idle_reward < -(env.w_step + env.w_revisit)

    env.reset(seed=0)
    fwd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    _, fresh_reward, _, _, _ = env.step(fwd)
    # Cutting fresh material is rewarded; idling is strictly worse
    assert fresh_reward > 0.0 > idle_reward


def test_milestone_bonus_once_per_threshold():
    env = ToolpathEnv2D(config=CONFIG)
    env.reset(seed=0)
    assert env._pending_milestones == [0.25, 0.50, 0.75]
    # Force-clear 30% of the workpiece, then take a no-op cut step
    from shapely.geometry import Polygon

    env.engine._remaining = Polygon([(-5, -2), (5, -2), (5, 5), (-5, 5)])  # 70% left
    _, reward, _, _, _ = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert env._pending_milestones == [0.50, 0.75]  # 25% bonus claimed exactly once
    assert reward > env.milestone_bonus / 2  # bonus dominates the step reward
    _, reward2, _, _, _ = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert reward2 < 0  # no second payout for the same milestone
    env.close()


def test_potential_increases_toward_material():
    env = ToolpathEnv2D(config=CONFIG)
    env.reset(seed=0)
    gs = env.grid_size
    # Material only in the right half of the workpiece
    mat = np.zeros((gs, gs), dtype=np.float32)
    mat[:, gs // 2 :] = 1.0

    env._grid_block(np.array([0.4, 0.0], dtype=np.float32))  # near the material
    phi_near = env._potential(mat)
    env._grid_block(np.array([-0.9, 0.0], dtype=np.float32))  # far from material
    phi_far = env._potential(mat)

    # Potential is higher (less negative) when the tool is closer to material
    assert phi_near > phi_far
    env.close()


def test_progress_shaping_rewards_approach():
    env = ToolpathEnv2D(config=CONFIG)
    env.reset(seed=0)
    gs = env.grid_size
    mat = np.zeros((gs, gs), dtype=np.float32)
    mat[:, gs // 2 :] = 1.0  # material on the right

    env._grid_block(np.array([-0.5, 0.0], dtype=np.float32))
    phi0 = env._potential(mat)
    env._grid_block(np.array([-0.4, 0.0], dtype=np.float32))  # moved toward material
    phi_closer = env._potential(mat)
    env._grid_block(np.array([-0.6, 0.0], dtype=np.float32))  # moved away
    phi_farther = env._potential(mat)

    shaping_toward = env.w_progress * (env.shaping_gamma * phi_closer - phi0)
    shaping_away = env.w_progress * (env.shaping_gamma * phi_farther - phi0)

    # Approaching remaining material is rewarded; retreating into cleared space hurts
    assert shaping_toward > 0.0 > shaping_away
    env.close()


if __name__ == "__main__":
    test_check_env()
    print("Environment check passed!")
    test_random_rollout()
    test_wall_collision_detected_and_remembered()
    test_random_start_varies()
    test_fresh_move_beats_revisit()
    test_idle_penalized_harder_than_traversal()
    test_milestone_bonus_once_per_threshold()
    test_potential_increases_toward_material()
    test_progress_shaping_rewards_approach()
    print("All env tests passed!")
