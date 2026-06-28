# Reinforcement Learning for CNC Toolpath Planning

A deep reinforcement learning system that trains a PPO agent to plan optimal CNC machining toolpaths in 2D. The agent learns to move a circular milling tool across a square workpiece, removing ≥99% of material while avoiding boundary collisions and minimizing wasted motion.

## Overview

The project solves a practical robotic machining problem: given a 10×10 world-unit workpiece and a circular tool with radius 0.5, learn a navigation policy that clears the workpiece efficiently within an 800-step episode budget. The tool sweeps a capsule-shaped contact region along each movement path, and remaining material is tracked using exact 2D polygon Boolean operations (Shapely).

**Key metrics:**
- Optimal lawnmower sweep: ~250 steps
- Episode budget: 800 steps
- Target clearance: ≥99% of workpiece material
- Physics: Exact 2D geometry (no voxels)

## Key Design Decisions

1. **Spatial memory maps instead of LSTM**  
   The observation includes persistent `visited_map` and `collision_map` grids, making the MDP Markovian without recurrence. This achieves ~8× faster training compared to LSTM-based PPO while keeping the feature dimensionality manageable.

2. **Log-std initialization (`log_std_init=-1.0`)**  
   Critical to prevent the Gaussian policy from saturating at action clipping boundaries. Earlier runs (PPO 14–16) with default log_std=0 saw the policy mean collapse to wall-pressing. Setting log_std=-1 (std ≈ 0.37) keeps the policy exploration balanced.

3. **Potential-based reward shaping with matching γ**  
   Reward includes an EDT (Euclidean Distance Transform) shaping term that guides the tool toward uncut material. The shaping discount `γ=0.99` must match the PPO discount to preserve policy-invariance. Shaping is zeroed at terminal states.

## Project Structure

| Path | Purpose |
|------|---------|
| `src/agents/train.py` | Main PPO training loop |
| `src/env/toolpath_env.py` | Gymnasium environment wrapper |
| `src/engines/engine_2d.py` | Shapely-based 2D machining physics |
| `src/engines/base.py` | Abstract engine interface |
| `src/agents/feature_extractor.py` | Custom CNN+scalar feature extractor |
| `configs/env_2d.yaml` | Environment and reward configuration |
| `configs/train_ppo.yaml` | PPO hyperparameters |
| `scripts/visualize_agent.py` | Interactive single-episode playback |
| `scripts/animate_learning.py` | Checkpoint-to-MP4 learning animation |
| `tests/` | Unit and integration tests |
| `outputs/models/` | Trained model checkpoints (created at runtime) |
| `outputs/logs/` | TensorBoard event files (created at runtime) |

## Installation

**Requirements:** Python ≥ 3.11

```bash
# Clone the repository and navigate to it
cd toolpath-rl

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate

# Install package with dev dependencies
pip install -e ".[dev]"
```

Optional: install 3D visualization support
```bash
pip install -e ".[3d]"
```

## Usage

### Train the Agent

```bash
# Windows (PowerShell)
$env:PYTHONPATH="src"; python src/agents/train.py

# Linux/macOS
PYTHONPATH=src python src/agents/train.py
```

Configuration is loaded from `configs/train_ppo.yaml` and `configs/env_2d.yaml`. Checkpoints are saved every 100k steps to `outputs/models/` and TensorBoard logs to `outputs/logs/`.

### Visualize a Trained Agent

```bash
PYTHONPATH=src python scripts/visualize_agent.py
```

Loads the trained model from `outputs/models/ppo_toolpath2d_final.zip` (if available) and renders one episode with matplotlib. Requires a display.

### Generate Learning Animation

```bash
PYTHONPATH=src python scripts/animate_learning.py
```

Produces:
- `outputs/learning_animation.mp4` — learning curve over all available checkpoints
- `outputs/learning_metrics.png` — aggregate metrics
- `outputs/stage_comparison/*.png` — per-stage performance snapshots

### Run Tests

```bash
pytest
```

Tests include environment compliance checks, engine physics validation, and policy integration tests.

## Environment Details

### Observation Space

| Key | Shape | Description |
|---|---|---|
| `material_map` | (64, 64) | Rasterized remaining material (1 = uncut, 0 = removed) |
| `visited_map` | (64, 64) | Cells the tool has traversed (episode memory) |
| `collision_map` | (64, 64) | Cells where collisions occurred (episode memory) |
| `tool_pos` | (2,) | Normalized tool center in [-1, 1] |
| `step_progress` | (1,) | `current_step / max_steps` in [0, 1] |
| `material_remaining` | (1,) | Remaining material area fraction in [0, 1] |

### Action Space

Continuous `Box([-1, -1, 0], [1, 1, 1])`:
- First two components: normalized movement direction `[dx, dy]`
- Third component: reserved for future engage signal (not currently used)
- Actual world displacement: `direction * step_size / HALF_EXTENT`

### Reward Signal

| Component | Default Weight | Trigger |
|---|---|---|
| Material removal | 50.0 | Fraction of total area removed this step |
| Coverage gain | 5.0 | New grid cells entered |
| Step penalty | -0.03 | Every step |
| Collision penalty | -0.2 | Tool center would leave valid range |
| Revisit penalty | -0.02 | Moving over visited ground with no material removal |
| Idle penalty | -0.3 | No material removed AND no new coverage |
| Progress shaping | 4.0 (γ=0.99) | Potential-based reward toward uncut material (EDT) |
| Milestone bonus | +5.0 | One-time at 25%, 50%, 75% material cleared |
| Completion bonus | +50.0 | Terminal: material < 1% remaining |
| Truncation penalty | -10.0 | Hit 800 steps without finishing |

All weights are configurable in `configs/env_2d.yaml`.

## Training Configuration

### PPO Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `n_envs` | 8 | SubprocVecEnv parallel processes |
| `total_timesteps` | 100,000 | Configurable; ~30 min on CPU |
| `learning_rate` | 0.0003 | Linearly decayed with 20% floor |
| `n_steps` | 256 | Per-env rollout; 2048 total buffer |
| `batch_size` | 256 | |
| `n_epochs` | 10 | Gradient updates per batch |
| `gamma` | 0.99 | Discount factor (matches reward shaping) |
| `ent_coef` | 0.0 | Entropy regularization disabled |

### Architecture

**Feature Extractor** (`ToolpathCombinedExtractor`):
- **CNN branch:** Stacks 3-channel map input (material, visited, collision)
  - Conv2d(3→16, 3×3, stride=2) + ReLU
  - Conv2d(16→32, 3×3, stride=2) + ReLU
  - Conv2d(32→32, 3×3, stride=2) + ReLU
  - Flatten → Linear(flatten_dim, 128) + ReLU
- **Vector branch:** Concatenates tool_pos (2), step_progress (1), material_remaining (1) = 4 scalars
- **Combined:** 132-dim feature vector
- **Policy/value heads:** MLP [128, 128] for both

### Output Locations

- **Model checkpoints:** `outputs/models/ppo_toolpath2d_<step>_steps.zip`
- **TensorBoard logs:** `outputs/logs/` (view with `tensorboard --logdir outputs/logs/`)
- **Training throughput:** ~1,100 fps on CPU

## Requirements

- **Python:** ≥ 3.11
- **Core dependencies:**
  - `stable-baselines3>=2.0` — PPO implementation
  - `gymnasium>=0.28` — RL environment API
  - `shapely>=2.0` — 2D geometry and Boolean operations
  - `torch>=2.0` — neural networks (CPU mode default)
  - `scipy>=1.10` — distance transform for reward shaping
  - `matplotlib>=3.7` — visualization
  - `tensorboard>=2.13` — training monitoring
  - `pyyaml` — config file parsing
- **Optional (`dev`):** pytest, ruff, black, ipykernel
- **Optional (`3d`):** open3d ≥ 0.18 (for planned 3D support)

See `pyproject.toml` for the complete dependency specification.

## Notes

- **Single-threaded PyTorch:** Set via `torch.set_num_threads(1)` to avoid contention with 8-process VecEnv.
- **CPU-only training:** The project trains on CPU by default. GPU support is available via Stable-Baselines3 by setting `device="cuda"` in the training config.
- **VSCode integration:** Launch configs for "Train Agent (PPO 2D)" and "Test Environment" are defined in `.vscode/launch.json`.

## License

See LICENSE file (if present) for terms.
