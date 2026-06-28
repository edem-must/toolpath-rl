import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from agents.feature_extractor import ToolpathCombinedExtractor
from env.toolpath_env import ToolpathEnv2D


def make_env(config: dict):
    env_config = {**config["env"], **config["reward"]}

    def _init():
        return ToolpathEnv2D(config=env_config)

    return _init


def linear_schedule(initial_value: float, floor_fraction: float = 0.2):
    """Linearly decay from initial_value, never below floor_fraction of it.

    Decaying all the way to 0 froze earlier runs in a local optimum (approx_kl
    and clip_fraction hit exactly 0 at the end of training).
    """

    def func(progress_remaining: float) -> float:
        return max(progress_remaining, floor_fraction) * initial_value

    return func


def main():
    # The policy net is tiny; single-thread torch avoids contention with the
    # env subprocesses (standard SB3 guidance for SubprocVecEnv on CPU).
    torch.set_num_threads(1)

    with open("configs/env_2d.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("configs/train_ppo.yaml", "r") as f:
        train_config = yaml.safe_load(f)

    n_envs = train_config["train"]["n_envs"]
    # VecMonitor records episode reward/length so rollout/ep_rew_mean is logged.
    env = VecMonitor(SubprocVecEnv([make_env(config) for _ in range(n_envs)]))

    # CheckpointCallback counts per-env steps; divide so we save every ~N total
    save_freq = max(train_config["train"]["checkpoint_freq"] // n_envs, 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq, save_path="outputs/models/", name_prefix="ppo_toolpath2d"
    )

    # CNN over the spatial maps; no LSTM — the visited/collision maps already
    # give the policy episode memory, and plain PPO trains ~8x faster.
    policy_kwargs = dict(
        features_extractor_class=ToolpathCombinedExtractor,
        features_extractor_kwargs=dict(cnn_features=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        # std ~0.37: with the default std=1 on a [-1,1] action space, most
        # samples are clipped by the env while PPO credits the unclipped
        # values, so the policy mean never learns (it collapsed to wall-
        # pressing in runs 14-16 while exploration noise did the cutting).
        log_std_init=-1.0,
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=train_config["ppo"]["verbose"],
        learning_rate=linear_schedule(train_config["ppo"]["learning_rate"]),
        n_steps=train_config["ppo"]["n_steps"],
        batch_size=train_config["ppo"]["batch_size"],
        n_epochs=train_config["ppo"]["n_epochs"],
        gamma=train_config["ppo"]["gamma"],
        ent_coef=train_config["ppo"]["ent_coef"],
        tensorboard_log="outputs/logs/",
        device="cpu",
    )

    model.learn(
        total_timesteps=train_config["train"]["total_timesteps"],
        callback=checkpoint_cb,
    )

    model.save("outputs/models/ppo_toolpath2d_final")
    print("Training complete")


if __name__ == "__main__":
    main()
