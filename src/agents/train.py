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

    with open("configs/train_ppo.yaml", "r") as f:
        train_config = yaml.safe_load(f)

    env = DummyVecEnv([make_env(config)])

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000, save_path="outputs/models/", name_prefix="ppo_toolpath2d"
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=train_config["ppo"]["verbose"],
        learning_rate=train_config["ppo"]["learning_rate"],
        n_steps=train_config["ppo"]["n_steps"],
        batch_size=train_config["ppo"]["batch_size"],
        n_epochs=train_config["ppo"]["n_epochs"],
        gamma=train_config["ppo"]["gamma"],
        tensorboard_log="outputs/logs/",
        device="auto",
    )

    model.learn(
        total_timesteps=train_config["train"]["total_timesteps"],
        callback=checkpoint_cb,
    )

    model.save("outputs/models/ppo_toolpath2d_final")
    print("Training complete")


if __name__ == "__main__":
    main()
