import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# TODO: Edit the source and target path 
EPS = 1e-3
DATA_FOLDER = Path("").expanduser()
OUTPUT_PATH = Path("").expanduser()
TARGET_STATE_DIM = 32
ACTION_HORIZON = 50
# Joint mask for YAM: [left_6_joints, left_gripper, right_6_joints, right_gripper]
# Joints are delta (True), grippers are absolute (False).
JOINT_DELTA_MASK = np.array([True]*6 + [False] + [True]*6 + [False])  # 14D
ROBOT_ACTION_DIM = len(JOINT_DELTA_MASK)  # 14

DATA_DIR = DATA_FOLDER / "data"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_padded_array(arr: np.ndarray, target_dim: int) -> np.ndarray:
    if arr.shape[-1] >= target_dim:
        return arr[..., :target_dim]
    pad_width = target_dim - arr.shape[-1]
    return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")


all_states = []
all_delta_actions = []
chunk_dirs = sorted(DATA_DIR.glob("chunk-*"))
n_chunks = len(chunk_dirs)

for i, chunk_dir in enumerate(chunk_dirs, start=1):
    print(f"Processing chunk {i} of {n_chunks}: {chunk_dir.name}")
    parquet_files = sorted(chunk_dir.glob("*.parquet"))
    for parquet_file in tqdm(parquet_files, desc=f"Processing {chunk_dir.name}"):
        df = pd.read_parquet(parquet_file)

        states = np.stack(df["state"].to_numpy())    # (T, state_dim)
        actions = np.stack(df["actions"].to_numpy()) # (T, action_dim)
        T = len(states)

        all_states.append(get_padded_array(states, TARGET_STATE_DIM))

        # Compute delta actions relative to the chunk-start state, matching the DeltaActions transform.
        # For each timestep t, all actions in the chunk [t, t+H) are expressed relative to state[t].
        for t in range(T):
            chunk_end = min(t + ACTION_HORIZON, T)
            chunk_actions = actions[t:chunk_end].copy()  # (H, action_dim)
            chunk_start_state = states[t, :ROBOT_ACTION_DIM]  # (ROBOT_ACTION_DIM,)
            chunk_actions[:, :ROBOT_ACTION_DIM] -= np.where(JOINT_DELTA_MASK, chunk_start_state, 0)
            all_delta_actions.append(get_padded_array(chunk_actions, TARGET_STATE_DIM))

all_states = np.concatenate(all_states, axis=0)        # (N, TARGET_STATE_DIM)
all_delta_actions = np.concatenate(all_delta_actions, axis=0)  # (N, TARGET_STATE_DIM)


def compute_stats(data: np.ndarray, eps: float = EPS):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std = np.maximum(std, eps)
    q01 = np.quantile(data, 0.01, axis=0)
    q99 = np.quantile(data, 0.99, axis=0)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
    }


stats = {
    "norm_stats": {
        "state": compute_stats(all_states),
        "actions": compute_stats(all_delta_actions),
    }
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Saved delta-action norm stats to: {OUTPUT_PATH}")
