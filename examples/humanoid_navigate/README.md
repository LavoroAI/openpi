# Humanoid Navigate — π₀.₅ fine-tune (action_horizon=1)

Fine-tune of **π₀.₅ base** for Humanoid Navigate navigation tasks: **6-dim action vector** (`[x, z, u, w, vel, curvature]`), **1 base camera**, and **`action_horizon=1`**.

## Files

- `src/openpi/policies/humanoid_navigate_policy.py` — `HumanoidNavigateInputs` / `HumanoidNavigateOutputs`.
- `src/openpi/training/config.py` — `HumanoidNavigate` data config + `humanoid_navigate` train config.

## 1. Point openpi at your local dataset

LeRobot resolves a dataset by `repo_id` under `HF_LEROBOT_HOME` (default `~/.cache/huggingface/lerobot`).
For a local dataset, place or symlink the folder layout to match `repo_id`:

```bash
mkdir -p ~/.cache/huggingface/lerobot/simnav
ln -s /path/to/simnav/output_lerobot ~/.cache/huggingface/lerobot/simnav/vla_dataset
```

Or set the environment variable:

```bash
export HF_LEROBOT_HOME=/path/to/lerobot_datasets
# dataset lives at: /path/to/lerobot_datasets/simnav/vla_dataset
```

`repo_id="simnav/vla_dataset"` in the `humanoid_navigate` config (`src/openpi/training/config.py`).

## 2. Compute normalization stats

Before training, compute normalization stats for `actions`:

```bash
uv run scripts/compute_norm_stats.py --config-name humanoid_navigate
```

Stats will be written to `assets/humanoid_navigate/simnav/vla_dataset/norm_stats.json`.

## 3. Train

```bash
# Blackwell (RTX PRO 6000) NCCL workaround. Without these, multi-GPU DDP aborts
# with SIGABRT / illegal memory access. IB_DISABLE + ALGO=Ring only matter at 3+
# GPUs. See https://github.com/isaac-sim/IsaacLab/issues/4011
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py humanoid_navigate \
    --exp-name=humanoid_nav_experiment --overwrite
```

Smoke test (10 steps, no wandb):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py humanoid_navigate \
    --exp-name=smoke_test --overwrite --num-train-steps=10 --wandb-enabled=False
```


## Key Configuration Notes

- **Action Horizon = 1**: Set `action_horizon=1` in `Pi0Config` as trajectory bundles consist of single datapoints.
- **Single Base Camera**: Wrist camera inputs are zero-padded and masked out (`False`).
- **Action Dimensions**: `action_dim=32` is maintained to match the pretrained `pi05_base` checkpoint, and the 6-dim predictions are sliced out in `HumanoidNavigateOutputs`.
