"""
RL data loader for offline residual RL training.

Wraps an existing LeRobot dataset to produce RL transitions:
(observation, actions, next_observation, rewards, dones, mc_returns)

All RL signals (reward, done, mc_return) are inferred from episode boundaries
— no new fields need to be stored on disk.
"""

import logging
import os
from typing import SupportsIndex

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

log = logging.getLogger(__name__)


class RLLeRobotDataset:
    """Wraps a LeRobot dataset to produce RL transitions for offline training.

    For each frame at index t:
    - observation: standard observation at t
    - actions: action chunk at t (action_horizon steps)
    - next_observation: observation at t + action_horizon (clamped to episode end)
    - reward: discounted sum of dense rewards over the action chunk
        r_t + gamma*r_{t+1} + ... + gamma^(H-1)*r_{t+H-1}
    - done: 1.0 if episode ends within this action chunk, 0.0 otherwise
    - mc_return: gamma^(steps_remaining) for sparse end-of-episode reward

    The "next" observation corresponds to the state after the full action chunk,
    matching the critic's chunk-level granularity.
    """

    def __init__(
        self,
        dataset,
        episode_data_index: dict,
        dense_rewards: np.ndarray,
        action_horizon: int = 50,
        discount: float = 0.99,
        episode_progress: np.ndarray | None = None,
    ):
        """
        Args:
            episode_progress: optional per-frame progress column. If provided, the
                MC return for frames in episode `ep` is scaled by the progress value
                at the last frame of that episode, i.e.
                    mc_return[t] = progress[end-1] * discount ** steps_remaining
                If None, all terminal rewards are assumed to be 1.0 (legacy behavior).
        """
        self._dataset = dataset
        self._discount = discount
        self._action_horizon = action_horizon

        # Build episode boundary lookup
        self._episode_data_index = episode_data_index
        ep_from = self._episode_data_index["from"]  # tensor of start indices
        ep_to = self._episode_data_index["to"]  # tensor of end indices (exclusive)

        num_episodes = len(ep_from)
        total_frames = len(dataset)
        log.info(f"RLLeRobotDataset: {total_frames} frames, {num_episodes} episodes, action_horizon={action_horizon}")

        # Precompute discount weights for chunk reward: [1, gamma, gamma^2, ..., gamma^(H-1)]
        discount_weights = discount ** np.arange(action_horizon, dtype=np.float64)

        # Build per-frame metadata
        self._is_last = np.zeros(total_frames, dtype=bool)
        self._mc_returns = np.zeros(total_frames, dtype=np.float32)
        self._chunk_rewards = np.zeros(total_frames, dtype=np.float32)
        self._chunk_done = np.zeros(total_frames, dtype=np.float32)
        self._next_idx = np.zeros(total_frames, dtype=np.int64)

        for ep_idx in range(num_episodes):
            start = ep_from[ep_idx].item()
            end = ep_to[ep_idx].item()  # exclusive
            ep_len = end - start

            # Mark last frame
            self._is_last[end - 1] = True

            # Terminal reward: either the dataset-provided last-frame progress, or 1.0.
            if episode_progress is not None:
                p_ep = float(episode_progress[end - 1])
            else:
                p_ep = 1.0

            # MC returns (vectorized): p_ep * gamma^(steps_remaining) for sparse reward
            steps_remaining = np.arange(ep_len - 1, -1, -1)
            self._mc_returns[start:end] = p_ep * (discount ** steps_remaining)

            # Per-frame chunk metadata
            ep_rewards = dense_rewards[start:end]
            for t_local in range(ep_len):
                t = start + t_local
                chunk_len = min(action_horizon, ep_len - t_local)

                # Next obs index
                next_t = t + action_horizon
                if next_t >= end:
                    self._next_idx[t] = end - 1  # clamp to last valid frame
                    self._chunk_done[t] = 1.0
                else:
                    self._next_idx[t] = next_t
                    self._chunk_done[t] = 0.0

                # Chunk reward: dot product of discount weights with rewards slice
                self._chunk_rewards[t] = np.dot(
                    discount_weights[:chunk_len], ep_rewards[t_local:t_local + chunk_len]
                )

        # Valid indices: all frames
        self._valid_indices = np.arange(total_frames)
        log.info(f"RLLeRobotDataset: {len(self._valid_indices)} valid transitions")

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = int(self._valid_indices[int(index)])

        # Current observation
        item = self._dataset[idx]

        # Next observation: state after the full action chunk
        next_idx = int(self._next_idx[idx])
        next_item = self._dataset[next_idx]

        # Build output dict with current obs fields
        out = {}

        # Copy all current observation fields
        for key, value in item.items():
            if key in ("episode_index", "frame_index", "timestamp", "task_index", "index"):
                continue  # skip metadata
            out[key] = value

        # Add next observation fields with "next_" prefix
        for key, value in next_item.items():
            if key in ("episode_index", "frame_index", "timestamp", "task_index", "index"):
                continue
            out[f"next_{key}"] = value

        # RL signals
        out["reward"] = np.float32(self._chunk_rewards[idx])  # dense reward aggregated over chunk
        out["done"] = np.float32(self._chunk_done[idx])  # done if episode ends within chunk
        out["mc_return"] = np.float32(self._mc_returns[idx])  # mc_return based on sparse reward

        return out


class RLTransformedDataset:
    """Applies transforms to both current and next observations in RL transitions."""

    def __init__(
        self,
        dataset: RLLeRobotDataset,
        transforms: list[_transforms.DataTransformFn],
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: SupportsIndex) -> dict:
        raw = self._dataset[index]

        # Separate current and next fields
        current_fields = {}
        next_fields = {}
        rl_fields = {}

        for key, value in raw.items():
            if key in ("reward", "done", "mc_return"):
                rl_fields[key] = value
            elif key.startswith("next_"):
                next_fields[key[5:]] = value  # strip "next_" prefix
            else:
                current_fields[key] = value

        # Apply transforms to current observation
        current_transformed = self._transform(current_fields)

        # Apply same transforms to next observation
        next_transformed = self._transform(next_fields)

        # Combine: current obs fields + next obs fields (re-prefixed) + RL signals
        out = {}
        for key, value in current_transformed.items():
            out[key] = value

        for key, value in next_transformed.items():
            out[f"next_{key}"] = value

        out.update(rl_fields)
        return out


def create_rl_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    discount: float = 0.99,
    use_progress_mc_return: bool = False,
) -> RLLeRobotDataset:
    """Create an RL-wrapped LeRobot dataset.

    If use_progress_mc_return=True, reads a per-frame `progress` column from
    the HuggingFace dataset and uses the last frame's progress of each episode
    as the terminal reward for the MC return. Otherwise, assumes terminal=1.
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set.")

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    # Extract episode boundaries before any wrapping.
    episode_data_index = dataset.episode_data_index

    # Read all dense rewards in bulk from the underlying HuggingFace dataset
    raw_rewards = dataset.hf_dataset["reward"]
    dense_rewards = np.array([float(r.item() if hasattr(r, "item") else r) for r in raw_rewards], dtype=np.float32)
    log.info(f"Loaded {len(dense_rewards)} dense rewards (min={dense_rewards.min():.4f}, max={dense_rewards.max():.4f})")

    episode_progress = None
    if use_progress_mc_return:
        raw_progress = dataset.hf_dataset["progress"]
        episode_progress = np.array(
            [float(p.item() if hasattr(p, "item") else p) for p in raw_progress], dtype=np.float32
        )
        log.info(
            f"Loaded {len(episode_progress)} progress values (min={episode_progress.min():.4f}, "
            f"max={episode_progress.max():.4f}); MC return uses last-frame progress per episode."
        )

    # Apply PromptFromLeRobotTask if configured — this adds a "prompt" key
    # from the task_index, which downstream transforms (RepackTransform) expect.
    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    return RLLeRobotDataset(
        dataset,
        episode_data_index,
        dense_rewards,
        action_horizon=action_horizon,
        discount=discount,
        episode_progress=episode_progress,
    )


def create_rl_data_loader(
    config: _config.TrainConfig,
    *,
    discount: float = 0.99,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = True,
    num_batches: int | None = None,
    num_workers: int = 0,
    use_progress_mc_return: bool = False,
) -> "RLDataLoaderImpl":
    """Create a data loader for offline RL training.

    Returns batches of (observation, actions, next_observation, rewards, dones, mc_returns).
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    # Create RL dataset
    rl_dataset = create_rl_dataset(
        data_config,
        config.model.action_horizon,
        discount=discount,
        use_progress_mc_return=use_progress_mc_return,
    )

    # Build transform pipeline (same as standard training)
    norm_stats = data_config.norm_stats
    if norm_stats is None:
        raise ValueError(
            f"norm_stats not found for asset_id='{data_config.asset_id}' in assets dir. "
            f"Run `python scripts/compute_norm_stats.py --config-name={config.name}` "
            f"or symlink from the original pi0 config's assets directory."
        )

    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]

    # Wrap with transforms (handles both current and next obs)
    transformed_dataset = RLTransformedDataset(rl_dataset, transforms)

    # Use existing TorchDataLoader infrastructure
    torch_loader = _data_loader.TorchDataLoader(
        transformed_dataset,
        config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    return RLDataLoaderImpl(data_config, torch_loader)


class RLDataLoaderImpl:
    """Data loader that yields RL transitions."""

    def __init__(self, data_config: _config.DataConfig, data_loader: _data_loader.TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            # Extract RL signals
            rewards = batch.pop("reward")
            dones = batch.pop("done")
            mc_returns = batch.pop("mc_return")

            # Separate current and next observation fields
            next_fields = {}
            current_fields = {}
            for key, value in batch.items():
                if key.startswith("next_"):
                    next_fields[key[5:]] = value  # strip prefix
                else:
                    current_fields[key] = value

            # Build Observation objects
            obs = _model.Observation.from_dict(current_fields)
            actions = current_fields["actions"]

            next_obs = _model.Observation.from_dict(next_fields)

            yield obs, actions, next_obs, rewards, dones, mc_returns
