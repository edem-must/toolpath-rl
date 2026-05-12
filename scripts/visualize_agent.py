import time
import yaml
from stable_baselines3 import PPO
from env.toolpath_env import ToolpathEnv2D


def main():
    with open("configs/env_2d.yaml", "r") as f:
        config = yaml.safe_load(f)

    env = ToolpathEnv2D(config=config["env"], render_mode="human")

    # Load the latest saved model
    model = PPO.load("outputs/models/ppo_toolpath2d_final", env=env)

    obs, _ = env.reset()
    env.render()

    total_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.05)  # slow down so you can actually see it

        if terminated or truncated:
            print(f"Episode done. Total reward: {total_reward:.2f}")
            print(f"Material remaining: {env.engine.get_remaining_area()*100:.1f}%")
            break

    env.close()


if __name__ == "__main__":
    main()
