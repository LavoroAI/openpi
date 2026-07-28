# Lite6 Station — π₀.₅ full fine-tune

Full fine-tune of **π₀.₅ base** for the Lite6 station: **6-dim joint-velocity** actions and **2 cameras**
(one base/exterior view + one wrist view), trained from a **local** LeRobot dataset.

## Files

- `src/openpi/policies/lite6_policy.py` — `Lite6Inputs` / `Lite6Outputs` (camera + action mapping).
- `src/openpi/training/config.py` — `Lite6Station` data config + the `lite6_station` train config.

## 1. Point openpi at your local dataset

LeRobot resolves a dataset by `repo_id` under `HF_LEROBOT_HOME` (default `~/.cache/huggingface/lerobot`).
For a locally-hosted dataset, put it (or symlink it) so the folder layout matches the `repo_id`:

```
$HF_LEROBOT_HOME/<namespace>/<dataset_name>/
```

Set the env var to wherever your datasets live, e.g.:

```bash
export HF_LEROBOT_HOME=/path/to/lerobot_datasets
# dataset then lives at: /path/to/lerobot_datasets/lite6/station
```

Then set `repo_id="lite6/station"` in the `lite6_station` config (`src/openpi/training/config.py`).
No Hugging Face Hub access is needed — the local folder is used directly.

## 2. Match your dataset columns

In the `Lite6Station` data config (`src/openpi/training/config.py`), update the `RepackTransform`
right-hand side to your dataset's actual column names:

```python
"observation/image":       "observation.images.base",   # your base/exterior cam column
"observation/wrist_image": "observation.images.wrist",   # your wrist cam column
"observation/state":       "observation.state",          # your 6-dim state column
"actions":                 "action",                     # your 6-dim joint-velocity column
```

Tip: inspect the columns with `python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata as M; print(M('lite6/station').features)"`.

## 3. Compute normalization stats

Required before training (fresh stats — this custom robot doesn't match any provided asset):

```bash
uv run scripts/compute_norm_stats.py --config-name lite6_station
```

Writes to `assets/lite6_station/lite6/station/norm_stats.json`. Quick sanity pass on a large dataset:
add `--max-frames 100000`.

## 4. Train

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py lite6_station \
    --exp-name=my_experiment --overwrite
```

Smoke test first (10 steps, no wandb):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py lite6_station \
    --exp-name=smoke_test --overwrite --num-train-steps=10 --wandb-enabled=False
```

Checkpoints land in `checkpoints/lite6_station/my_experiment/<step>/`.

### Notes
- Keep `action_dim=32` (matches `pi05_base`); your 6 dims are zero-padded automatically and sliced
  back in `Lite6Outputs`.
- `action_horizon=16` is the predicted action-chunk length — tune to your control frequency.
- Out of GPU memory? Lower `--batch-size`, add `--fsdp-devices <num_gpus>`, or disable EMA.
- If your dataset has no per-episode task string, remove `prompt_from_task=True` from the config and
  set a `default_prompt` instead (π₀.₅ always expects a prompt).
