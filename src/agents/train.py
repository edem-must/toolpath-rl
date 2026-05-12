import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from env.toolpath_env import ToolpathEnv2D


def make_env(config: dict):
    def _init():
        return ToolpathEnv2D(config=config["env"])

    return _init


def main():
    with open("configs/env_2d.yaml", "r") as f:
        config = yaml.safe_load(f)

    env = DummyVecEnv([make_env(config)])

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000, save_path="outputs/models/", name_prefix="ppo_toolpath2d"
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log="outputs/logs",
        device="auto",
    )

    model.learn(
        total_timesteps=100_000,
        callback=checkpoint_cb,
    )

    model.save("outputs/models/ppo_toolpath2d_final")
    print("Training complete")


if __name__ == "__main__":
    main()
