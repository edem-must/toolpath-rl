from stable_baselines3.common.env_checker import check_env
from env.toolpath_env import ToolpathEnv2D

config = {"grid_size": 64, "max_steps": 500}
env = ToolpathEnv2D(config=config)

check_env(env, warn=True)
print("Environment check passed!")

obs, info = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"reward: {reward:.4f}, material left: {obs['material_map'].sum():.1f}")
    if terminated or truncated:
        break

env.close()
