"""
Generate an animated visualization showing agent learning progress across training stages.

This script:
1. Loads agent checkpoints from different training steps (10k, 50k, ..., 1M)
2. Runs each agent for one episode
3. Records frames showing tool position and material removal
4. Saves as MP4 video + sequence of PNG images
5. Creates a summary figure showing learning metrics

Usage:
    python scripts/animate_learning.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import yaml
from stable_baselines3 import PPO

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from env.toolpath_env import ToolpathEnv2D
from agents.feature_extractor import ToolpathCombinedExtractor  # noqa: F401


def extract_step_count(filename: str) -> int:
    """Extract step count from model filename like 'ppo_toolpath2d_500000_steps.zip'."""
    try:
        # Format: ppo_toolpath2d_500000_steps.zip
        # After split: ["ppo", "toolpath2d", "500000", "steps.zip"]
        return int(filename.split("_")[2])
    except (IndexError, ValueError):
        return 0


def get_checkpoint_steps() -> list[int]:
    """Get sorted list of available checkpoint steps."""
    # Resolve path relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / "outputs" / "models"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    checkpoints = []
    for f in model_dir.glob("ppo_toolpath2d_*_steps.zip"):
        step = extract_step_count(f.name)
        if step > 0:
            checkpoints.append(step)

    return sorted(checkpoints)


def run_episode_with_frames(agent, env, render_mode="rgb_array", max_frames=None):
    """
    Run an episode and collect frames.

    Returns:
        frames: list of (material_map, visited_map, tool_pos, reward_sum, material_pct_removed)
        episode_reward: total episode reward
    """
    obs, _ = env.reset()
    frames = []
    episode_reward = 0.0

    step = 0
    while True:
        # Get action from agent
        action, _ = agent.predict(obs, deterministic=True)

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        # Collect frame data
        material_map = obs["material_map"].copy()
        visited_map = obs["visited_map"].copy()
        tool_pos = obs["tool_pos"].copy()

        # Compute material removed percentage (1 - remaining)
        initial_area = 100.0
        remaining_area = env.engine.get_remaining_area()
        material_pct_removed = (initial_area - remaining_area) / initial_area * 100.0

        frames.append({
            "material_map": material_map,
            "visited_map": visited_map,
            "tool_pos": tool_pos,
            "step": step,
            "material_pct_removed": material_pct_removed,
        })

        step += 1
        if max_frames and len(frames) >= max_frames:
            break
        if terminated or truncated:
            break

    return frames, episode_reward, material_pct_removed


def create_frame_image(frames_data: dict, tool_radius: float = 0.5) -> np.ndarray:
    """
    Create a matplotlib figure for a single frame.

    Returns RGB array suitable for video.
    """
    from PIL import Image
    import io

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

    material_map = frames_data["material_map"]
    visited_map = frames_data["visited_map"]
    tool_pos = frames_data["tool_pos"]
    material_pct = frames_data["material_pct_removed"]
    step = frames_data["step"]

    # Left: material + tool position
    ax1.imshow(material_map, cmap="gray", extent=[-1, 1, -1, 1], origin="lower")

    # Draw tool position as red circle
    circle = Circle((tool_pos[0], tool_pos[1]), tool_radius,
                   color="red", fill=False, linewidth=2, label="Tool")
    ax1.add_patch(circle)

    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X (normalized)")
    ax1.set_ylabel("Y (normalized)")
    ax1.set_title(f"Material & Tool Position (Step {step})")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Right: coverage progress
    ax2.barh(["Material\nRemoved"], [material_pct], color="green", alpha=0.7, height=0.5)
    ax2.barh(["Material\nRemoved"], [100 - material_pct], left=[material_pct],
            color="white", alpha=0.7, height=0.5, edgecolor="black")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Percentage (%)")
    ax2.set_title("Workpiece Coverage")
    ax2.text(50, 0, f"{material_pct:.1f}%", ha="center", va="center",
            fontweight="bold", fontsize=12)

    fig.tight_layout()

    # Convert to RGB array via PIL
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    image = np.array(img)
    plt.close(fig)
    buf.close()

    return image


def create_learning_progression_video(checkpoints: list[int], output_path: str = "outputs/learning_animation.mp4"):
    """
    Create a video showing agent progression across training stages.

    For each checkpoint, runs an episode and creates a frame.
    """
    import cv2

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    with open(project_root / "configs" / "env_2d.yaml", "r") as f:
        config = yaml.safe_load(f)

    env_config = {**config["env"], **config["reward"]}
    # Load first model to determine expected grid size, then create env with that size
    first_model_path = str(project_root / "outputs" / "models" / f"ppo_toolpath2d_{checkpoints[0]}_steps")
    first_model = PPO.load(first_model_path)
    grid_size = first_model.observation_space.spaces["material_map"].shape[0]
    env_config["grid_size"] = grid_size
    print(f"  Using grid size {grid_size}x{grid_size} (from trained model)")

    env = ToolpathEnv2D(config=env_config, render_mode="human")

    frames_list = []
    metrics = {"steps": [], "rewards": [], "coverage": []}

    print("Running learning progression animation...")
    print(f"Processing {len(checkpoints)} checkpoints...")

    for i, step_count in enumerate(checkpoints):
        model_path = project_root / "outputs" / "models" / f"ppo_toolpath2d_{step_count}_steps"

        if not Path(f"{model_path}.zip").exists():
            print(f"  Checkpoint {step_count} not found, skipping...")
            continue

        print(f"  [{i+1}/{len(checkpoints)}] Loading agent at {step_count} steps...", end="")
        # Load without env param to avoid space mismatch—model carries its own spaces
        agent = PPO.load(str(model_path))

        # Get agent's expected grid size; adjust env if needed
        agent_grid_size = agent.observation_space.spaces["material_map"].shape[0]
        current_grid_size = env.grid_size
        if agent_grid_size != current_grid_size:
            print(f"\n    Adjusting grid size {current_grid_size} -> {agent_grid_size}...", end="")
            env_config["grid_size"] = agent_grid_size
            env = ToolpathEnv2D(config=env_config, render_mode="human")

        print(" Running episode...", end="")
        frames, episode_reward, final_coverage = run_episode_with_frames(agent, env)

        print(f" Coverage: {final_coverage:.1f}%")

        # Take frames at regular intervals (every 5 steps) to avoid too many frames
        sample_every = max(1, len(frames) // 20)  # Target ~20 frames per agent
        sampled_frames = frames[::sample_every]

        # Create images for sampled frames
        for frame_data in sampled_frames:
            img = create_frame_image(frame_data, tool_radius=env.tool_radius)
            frames_list.append(img)

        # Record metrics
        metrics["steps"].append(step_count)
        metrics["rewards"].append(episode_reward)
        metrics["coverage"].append(final_coverage)

    # Create video (only if we have frames and frames are properly sized)
    print(f"\nCreating video with {len(frames_list)} frames...")

    if frames_list and len(frames_list) > 1:
        try:
            # Ensure output directory exists
            output_file = project_root / output_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            h, w = frames_list[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_file), fourcc, 4.0, (w, h))

            for frame in frames_list:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"[OK] Video saved: {output_file}")
        except Exception as e:
            print(f"  Warning: Could not create video: {e}")
            print(f"  (Continuing with other outputs...)")
    else:
        print(f"  Skipping video: need frames to generate")

    env.close()

    return metrics


def create_metrics_summary(metrics: dict, output_path: str = "outputs/learning_metrics.png"):
    """
    Create a summary figure showing learning metrics.
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_file = project_root / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = np.array(metrics["steps"]) / 1e6  # Convert to millions
    rewards = metrics["rewards"]
    coverage = metrics["coverage"]

    # Plot 1: Episode Reward
    ax1.plot(steps, rewards, "o-", color="blue", linewidth=2, markersize=8)
    ax1.set_xlabel("Training Steps (Millions)", fontsize=12)
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.set_title("Agent Performance: Episode Reward Over Training", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(steps, rewards, alpha=0.2, color="blue")

    # Annotate max
    max_idx = np.argmax(rewards)
    ax1.annotate(f"Max: {rewards[max_idx]:.1f}",
                xy=(steps[max_idx], rewards[max_idx]),
                xytext=(steps[max_idx], rewards[max_idx] + 10),
                ha="center", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="blue"))

    # Plot 2: Coverage Percentage
    ax2.plot(steps, coverage, "o-", color="green", linewidth=2, markersize=8)
    ax2.set_xlabel("Training Steps (Millions)", fontsize=12)
    ax2.set_ylabel("Material Removed (%)", fontsize=12)
    ax2.set_title("Agent Performance: Workpiece Coverage Over Training", fontsize=13, fontweight="bold")
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(steps, coverage, alpha=0.2, color="green")

    # Annotate max
    max_idx = np.argmax(coverage)
    ax2.annotate(f"Max: {coverage[max_idx]:.1f}%",
                xy=(steps[max_idx], coverage[max_idx]),
                xytext=(steps[max_idx], coverage[max_idx] - 10),
                ha="center", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="green"))

    fig.suptitle("Toolpath-RL: Learning Progression (0 → 1M steps)",
                fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"[OK] Metrics summary saved: {output_file}")
    plt.close()


def create_stage_comparison(checkpoints: list[int], output_dir: str = "outputs/stage_comparison"):
    """
    Create side-by-side comparison of agent behavior at different training stages.
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_path = project_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    with open(project_root / "configs" / "env_2d.yaml", "r") as f:
        config = yaml.safe_load(f)

    env_config = {**config["env"], **config["reward"]}
    # Load first model to determine expected grid size
    first_model_path = str(project_root / "outputs" / "models" / f"ppo_toolpath2d_{checkpoints[0]}_steps")
    first_model = PPO.load(first_model_path)
    grid_size = first_model.observation_space.spaces["material_map"].shape[0]
    env_config["grid_size"] = grid_size

    env = ToolpathEnv2D(config=env_config, render_mode="human")

    # Select stages: early, mid, late
    stage_indices = [0, len(checkpoints) // 2, len(checkpoints) - 1]
    stages = ["Early (10-50k steps)", "Mid (400-500k steps)", "Late (900k+ steps)"]

    print("\nCreating stage comparison images...")

    for stage_idx, (checkpoint_idx, stage_name) in enumerate(zip(stage_indices, stages)):
        step_count = checkpoints[checkpoint_idx]
        model_path = str(project_root / "outputs" / "models" / f"ppo_toolpath2d_{step_count}_steps")

        print(f"  Stage {stage_idx + 1}: {step_count} steps...", end="")
        agent = PPO.load(model_path)

        # Adjust env grid size if needed
        agent_grid_size = agent.observation_space.spaces["material_map"].shape[0]
        if agent_grid_size != env.grid_size:
            print(f"\n    Adjusting grid {env.grid_size} -> {agent_grid_size}...", end="")
            env_config["grid_size"] = agent_grid_size
            env = ToolpathEnv2D(config=env_config, render_mode="human")

        frames, episode_reward, final_coverage = run_episode_with_frames(agent, env, max_frames=10)

        # Create figure with 3 snapshots: start, middle, end
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        snapshot_indices = [0, len(frames) // 2, -1]
        for ax_idx, frame_idx in enumerate(snapshot_indices):
            frame_data = frames[frame_idx]
            material_map = frame_data["material_map"]
            tool_pos = frame_data["tool_pos"]

            ax = axes[ax_idx]
            ax.imshow(material_map, cmap="gray", extent=[-1, 1, -1, 1], origin="lower")

            # Draw tool
            circle = Circle((tool_pos[0], tool_pos[1]), 0.5,
                          color="red", fill=False, linewidth=2)
            ax.add_patch(circle)

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal")
            ax.set_title(f"Step {frame_data['step']}", fontsize=10)
            ax.axis("off")

        fig.suptitle(f"{stage_name}\nReward: {episode_reward:.1f} | Coverage: {final_coverage:.1f}%",
                    fontsize=12, fontweight="bold")
        fig.tight_layout()

        output_file = output_path / f"stage_{stage_idx+1:02d}_{step_count}steps.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f" → {output_file.name}")
        plt.close()

    env.close()


def main():
    """Main entry point."""
    # Create output directories
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/stage_comparison").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Toolpath-RL: Learning Progression Animation")
    print("=" * 70)

    # Get available checkpoints
    checkpoints = get_checkpoint_steps()
    print(f"\nFound {len(checkpoints)} checkpoints")
    print(f"  Steps: {checkpoints[0]} -> {checkpoints[-1]}")

    if not checkpoints:
        print("ERROR: No checkpoints found in outputs/models/")
        return

    # Create learning progression video
    print("\n" + "=" * 70)
    print("Phase 1: Creating Learning Progression Video")
    print("=" * 70)
    metrics = create_learning_progression_video(checkpoints)

    # Create metrics summary
    print("\n" + "=" * 70)
    print("Phase 2: Creating Metrics Summary")
    print("=" * 70)
    create_metrics_summary(metrics)

    # Create stage comparison
    print("\n" + "=" * 70)
    print("Phase 3: Creating Stage Comparison Images")
    print("=" * 70)
    create_stage_comparison(checkpoints)

    print("\n" + "=" * 70)
    print("Animation Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • outputs/learning_animation.mp4 — Full learning progression video")
    print("  • outputs/learning_metrics.png — Reward & coverage metrics")
    print("  • outputs/stage_comparison/*.png — Early/mid/late stage comparisons")
    print("\nRecommended presentation flow:")
    print("  1. Show learning_metrics.png for overview of learning curve")
    print("  2. Play learning_animation.mp4 to demonstrate agent behavior evolution")
    print("  3. Show stage_comparison images for detailed behavioral analysis")
    print("=" * 70)


if __name__ == "__main__":
    main()
