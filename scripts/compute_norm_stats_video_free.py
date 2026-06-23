import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# TODO: Edit the source and target path 
DATA_FOLDER = Path("").expanduser()
OUTPUT_PATH = Path("").expanduser()
TARGET_STATE_DIM = 32
DATA_DIR = DATA_FOLDER / f"data"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_padded_array(arr: np.ndarray, target_dim: int) -> np.ndarray:
    if arr.shape[-1] >= target_dim:
        return arr[..., :target_dim]
    pad_width = target_dim - arr.shape[-1]
    return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")

all_states = []
all_actions = []
chunk_dirs = sorted(DATA_DIR.glob("chunk-*"))
n_chunks = len(chunk_dirs)

for i, chunk_dir in enumerate(chunk_dirs, start=1):
    print(f"Processing chunk {i} of {n_chunks}: {chunk_dir.name}")    
    parquet_files = sorted(chunk_dir.glob("*.parquet"))
    for parquet_file in tqdm(parquet_files, desc=f"Processing {chunk_dir.name}"):
        df = pd.read_parquet(parquet_file)

        states = np.stack(df["state"].to_numpy())  # (T, dim)
        actions = np.stack(df["actions"].to_numpy())

        states = get_padded_array(states, TARGET_STATE_DIM)
        actions = get_padded_array(actions, TARGET_STATE_DIM)

        all_states.append(states)
        all_actions.append(actions)

all_states = np.concatenate(all_states, axis=0)
all_actions = np.concatenate(all_actions, axis=0)

def compute_stats(data: np.ndarray):
    return {
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist()
    }

stats = {
    "norm_stats": {
        "state": compute_stats(all_states),
        "actions": compute_stats(all_actions)
    }
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Saved norm stats to: {OUTPUT_PATH}")
