#!/usr/bin/env python

"""
Evaluate a trained policy on validation data.
"""

import dataclasses
import logging
import platform

import etils.epath as epath
import jax
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import tqdm

import openpi.models.model as _model
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def load_trained_policy(
    config: _config.TrainConfig, checkpoint_dir: str, checkpoint_step: int | None = None
) -> _policy.Policy:
    """Load a trained policy using the inference-friendly policy loader."""

    # If checkpoint_step is specified, append it to the directory
    if checkpoint_step is not None:
        checkpoint_path = f"{checkpoint_dir}/{checkpoint_step}"
    else:
        checkpoint_path = checkpoint_dir

    logging.info(f"Loading from checkpoint path: {checkpoint_path}")

    # Use the policy config loader which handles device/sharding differences better
    policy = _policy_config.create_trained_policy(config, checkpoint_path, default_prompt=None)

    return policy


def evaluate_policy(
    policy: _policy.Policy,
    observation: _model.Observation,
    actions_gt: _model.Actions,
    rng: at.KeyArrayLike,
) -> tuple[_model.Actions, dict[str, float]]:
    """Evaluate policy on a single batch and compute metrics."""

    # Convert observation to the format expected by policy
    # The observation is already structured properly from the data loader

    # IMPORTANT: The data loader has already applied the YAM input transform,
    # so we need to call the model directly rather than going through policy.infer()
    # which would try to apply the input transform again

    # Apply the model's sample_actions method directly on the full batch
    actions_pred = policy._sample_actions(rng, observation, **policy._sample_kwargs)

    # Compute metrics
    actions_gt_np = np.array(actions_gt)
    actions_pred_np = np.array(actions_pred)

    # For YAM, focus on the first 14 dimensions (actual DOF), ignore padding
    if actions_gt_np.shape[-1] == 32 and actions_pred_np.shape[-1] == 32:
        # This is likely YAM with 32-dim padding - use only first 14 dims for metrics
        logging.info("Using first 14 dimensions for metrics computation (YAM DOF)")
        actions_gt_np = actions_gt_np[:, :, :14]
        actions_pred_np = actions_pred_np[:, :, :14]

    # Handle shape mismatch if needed
    if actions_gt_np.shape != actions_pred_np.shape:
        logging.warning(f"Shape mismatch detected. GT: {actions_gt_np.shape}, Pred: {actions_pred_np.shape}")
        # If predicted actions have fewer dimensions, we might need to handle this
        min_samples = min(actions_gt_np.size, actions_pred_np.size)
        gt_flat = actions_gt_np.flatten()[:min_samples]
        pred_flat = actions_pred_np.flatten()[:min_samples]
    else:
        # Flatten for metric computation
        gt_flat = actions_gt_np.reshape(-1)
        pred_flat = actions_pred_np.reshape(-1)

    metrics = {
        "mse": float(mean_squared_error(gt_flat, pred_flat)),
        "mae": float(mean_absolute_error(gt_flat, pred_flat)),
        "rmse": float(np.sqrt(mean_squared_error(gt_flat, pred_flat))),
        "r2": float(1 - np.sum((gt_flat - pred_flat) ** 2) / np.sum((gt_flat - np.mean(gt_flat)) ** 2)),
    }

    return actions_pred, metrics


def plot_action_comparison(
    actions_gt: np.ndarray,
    actions_pred: np.ndarray,
    save_path: str,
    action_names: list[str] | None = None,
    sample_indices: list[int] | None = None,
    action_stats: dict | None = None,
):
    """Create comprehensive plots comparing ground truth vs predicted actions."""

    batch_size, action_horizon, gt_action_dim = actions_gt.shape
    pred_action_dim = actions_pred.shape[2]

    # Determine plot dimensions, excluding padding for YAM data
    if action_names and len(action_names) >= 14 and any("Pad_" in name for name in action_names[14:]):
        # This is YAM with padding - only use first 14 dimensions
        plot_action_dim = 14
        logging.info(
            f"YAM detected: Plotting only first {plot_action_dim} dimensions for action comparison (excluding padding)"
        )
    else:
        # Use the smaller action dimension for plotting
        plot_action_dim = min(gt_action_dim, pred_action_dim)

    if action_names is None:
        action_names = [f"Action_{i}" for i in range(plot_action_dim)]
    else:
        action_names = action_names[:plot_action_dim]  # Truncate to available dimensions

    if sample_indices is None:
        # Select a few random samples to plot
        sample_indices = np.random.choice(batch_size, min(4, batch_size), replace=False)

    # Create figure with subplots
    fig, axes = plt.subplots(
        len(sample_indices), plot_action_dim, figsize=(4 * plot_action_dim, 3 * len(sample_indices))
    )
    if len(sample_indices) == 1:
        axes = axes.reshape(1, -1)
    if plot_action_dim == 1:
        axes = axes.reshape(-1, 1)

    for i, sample_idx in enumerate(sample_indices):
        for j in range(plot_action_dim):
            ax = axes[i, j]

            # Plot ground truth and predicted actions
            time_steps = np.arange(action_horizon)
            ax.plot(time_steps, actions_gt[sample_idx, :, j], "b-", label="Ground Truth", linewidth=2)
            ax.plot(time_steps, actions_pred[sample_idx, :, j], "r--", label="Predicted", linewidth=2)

            # Set y-axis limits based on action statistics if available
            if action_stats is not None and "actions" in action_stats:
                if j < len(action_stats["actions"]["min"]) and j < len(action_stats["actions"]["max"]):
                    y_min = float(action_stats["actions"]["min"][j])
                    y_max = float(action_stats["actions"]["max"][j])
                    # Add small margin (5% of range)
                    y_range = y_max - y_min
                    margin = y_range * 0.05
                    ax.set_ylim(y_min - margin, y_max + margin)

                    # Debug logging for first few plots
                    if i == 0 and j < 3:
                        logging.info(
                            f"Setting y-axis for {action_names[j]} (dim {j}): [{y_min:.3f}, {y_max:.3f}] + margin"
                        )
                elif i == 0 and j < 3:
                    logging.warning(f"Action stats missing for dimension {j} (action: {action_names[j]})")
            elif i == 0 and j < 3:
                if action_stats is None:
                    logging.warning(f"No action_stats provided for {action_names[j]}")
                elif "actions" not in action_stats:
                    logging.warning(f"No 'actions' key in action_stats. Available keys: {list(action_stats.keys())}")
                    # Try alternative key names
                    for alt_key in ["action", "observation.actions", "policy_actions"]:
                        if alt_key in action_stats:
                            logging.info(f"Found alternative actions key: {alt_key}")
                            break

            ax.set_xlabel("Time Step")
            ax.set_ylabel("Action Value")
            ax.set_title(f"Sample {sample_idx} - {action_names[j]}")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_heatmap(
    actions_gt: np.ndarray,
    actions_pred: np.ndarray,
    save_path: str,
    action_names: list[str] | None = None,
):
    """Create heatmap of prediction errors across time and action dimensions."""

    batch_size, action_horizon, gt_action_dim = actions_gt.shape
    pred_action_dim = actions_pred.shape[2]

    # Determine plot dimensions, excluding padding for YAM data
    if action_names and len(action_names) >= 14 and any("Pad_" in name for name in action_names[14:]):
        # This is YAM with padding - only use first 14 dimensions
        plot_action_dim = 14
        logging.info(f"YAM detected: Error heatmap using only first {plot_action_dim} dimensions (excluding padding)")
    else:
        # Use the smaller action dimension for error computation
        plot_action_dim = min(gt_action_dim, pred_action_dim)

    if action_names is None:
        action_names = [f"Action_{i}" for i in range(plot_action_dim)]
    else:
        action_names = action_names[:plot_action_dim]

    # Truncate both arrays to the same dimensions for error computation
    actions_gt_truncated = actions_gt[:, :, :plot_action_dim]
    actions_pred_truncated = actions_pred[:, :, :plot_action_dim]

    # Compute absolute errors
    errors = np.abs(actions_gt_truncated - actions_pred_truncated)

    # Average errors across batch
    avg_errors = np.mean(errors, axis=0)  # Shape: (action_horizon, plot_action_dim)

    # Create heatmap
    plt.figure(figsize=(max(16, plot_action_dim * 2.0), max(6, action_horizon // 3)))
    sns.heatmap(
        avg_errors.T,  # Transpose so actions are on y-axis
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=[f"T{i}" for i in range(action_horizon)],
        yticklabels=action_names,
        cbar_kws={"label": "Mean Absolute Error"},
    )
    plt.title("Prediction Error Heatmap\n(Averaged across batch)")
    plt.xlabel("Time Step")
    plt.ylabel("Action Dimension")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_distribution(
    actions_gt: np.ndarray,
    actions_pred: np.ndarray,
    save_path: str,
    action_names: list[str] | None = None,
):
    """Plot distribution of prediction errors."""

    # For YAM, only use the first 14 dimensions (actual DOF), ignore padding
    if action_names and len(action_names) >= 14 and any("Pad_" in name for name in action_names[14:]):
        # This is YAM with padding - only use first 14 dimensions
        min_action_dim = 14
        logging.info(
            f"YAM detected: Using only first {min_action_dim} dimensions for error analysis (excluding padding)"
        )
    else:
        # Handle dimension mismatch - use only the overlapping dimensions
        min_action_dim = min(actions_gt.shape[-1], actions_pred.shape[-1])

    actions_gt_trimmed = actions_gt[:, :, :min_action_dim]
    actions_pred_trimmed = actions_pred[:, :, :min_action_dim]

    # Compute errors
    errors = (actions_pred_trimmed - actions_gt_trimmed).flatten()

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of errors
    ax1.hist(errors, bins=50, alpha=0.7, color="blue", edgecolor="black")
    ax1.axvline(0, color="red", linestyle="--", linewidth=2, label="Perfect Prediction")
    ax1.set_xlabel("Prediction Error")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Distribution of Prediction Errors\n(Using {min_action_dim}/{actions_gt.shape[-1]} action dims)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot of absolute errors by action dimension
    abs_errors_by_dim = []
    action_labels = []
    for i in range(min_action_dim):
        abs_errors_by_dim.append(np.abs(actions_pred_trimmed[:, :, i] - actions_gt_trimmed[:, :, i]).flatten())
        if action_names and i < len(action_names):
            action_labels.append(action_names[i])
        else:
            action_labels.append(f"A{i}")

    ax2.boxplot(abs_errors_by_dim, positions=range(min_action_dim))
    ax2.set_xlabel("Action Dimension")
    ax2.set_ylabel("Absolute Error")
    ax2.set_title(f"Error Distribution by Action Dimension\n({min_action_dim} meaningful dims)")
    ax2.set_xticks(range(min_action_dim))
    ax2.set_xticklabels(action_labels, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_summary_metrics_plot(
    all_metrics: list[dict[str, float]],
    save_path: str,
):
    """Create a summary plot of all evaluation metrics."""

    # Aggregate metrics
    metric_names = list(all_metrics[0].keys())
    aggregated = {}

    for metric in metric_names:
        values = [batch_metrics[metric] for batch_metrics in all_metrics]
        aggregated[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(metric_names))
    means = [aggregated[metric]["mean"] for metric in metric_names]
    stds = [aggregated[metric]["std"] for metric in metric_names]

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, color="skyblue", edgecolor="black")

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std,
            f"{mean:.4f}±{std:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Value")
    ax.set_title("Model Evaluation Metrics Summary")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return aggregated


# Apply video loading patch immediately
def patch_video_loading():
    """Patch LeRobot to use pyav instead of torchcodec for video loading."""
    try:
        from lerobot.common.datasets import video_utils

        # Patch the default codec function to return pyav instead of torchcodec
        def patched_get_safe_default_codec():
            """Always return pyav as the default codec."""
            return "pyav"

        # Replace the function in the module
        video_utils.get_safe_default_codec = patched_get_safe_default_codec

        # Also patch the decode_video_frames function directly
        original_decode = video_utils.decode_video_frames

        def patched_decode_video_frames(video_path, timestamps, tolerance_s, backend=None):
            """Force pyav backend."""
            return video_utils.decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend="pyav")

        video_utils.decode_video_frames = patched_decode_video_frames

        logging.info("Successfully patched video loading to use pyav backend")
        return True
    except Exception as e:
        logging.warning(f"Failed to patch video loading: {e}")
        return False


# Apply patch immediately on import
patch_video_loading()


def load_dataset_stats(
    config: _config.TrainConfig, checkpoint_path: str = None, checkpoint_step: int = None
) -> dict | None:
    """Load dataset statistics from the config system."""
    try:
        import json
        from pathlib import Path

        # First, try to get the norm_stats from the data config (same way the config system loads it)
        try:
            data_config = config.data.create(config.assets_dirs, config.model)
            if hasattr(data_config, "norm_stats") and data_config.norm_stats is not None:
                norm_stats_path = Path(data_config.norm_stats)
                logging.info(f"Trying to load norm stats from config path: {norm_stats_path}")

                if norm_stats_path.exists():
                    with open(norm_stats_path) as f:
                        norm_data = json.load(f)

                    if "norm_stats" in norm_data and "actions" in norm_data["norm_stats"]:
                        actions_norm = norm_data["norm_stats"]["actions"]
                        logging.info(f"Successfully loaded norm stats from {norm_stats_path}")
                        logging.info(f"Actions norm stats keys: {list(actions_norm.keys())}")

                        # Convert norm stats to min/max format for plotting
                        if "q01" in actions_norm and "q99" in actions_norm:
                            converted_stats = {
                                "actions": {
                                    "min": actions_norm["q01"],
                                    "max": actions_norm["q99"],
                                    "mean": actions_norm.get("mean", []),
                                    "std": actions_norm.get("std", []),
                                }
                            }
                            logging.info("Converted quantiles (q01/q99) to min/max format")
                            logging.info(
                                f"Actions min (q01): {converted_stats['actions']['min'][:5] if len(converted_stats['actions']['min']) > 5 else converted_stats['actions']['min']}"
                            )
                            logging.info(
                                f"Actions max (q99): {converted_stats['actions']['max'][:5] if len(converted_stats['actions']['max']) > 5 else converted_stats['actions']['max']}"
                            )
                            return converted_stats
                        if "min" in actions_norm and "max" in actions_norm:
                            # Already in min/max format
                            converted_stats = {"actions": actions_norm}
                            logging.info("Found actions stats in min/max format")
                            logging.info(
                                f"Actions min: {converted_stats['actions']['min'][:5] if len(converted_stats['actions']['min']) > 5 else converted_stats['actions']['min']}"
                            )
                            logging.info(
                                f"Actions max: {converted_stats['actions']['max'][:5] if len(converted_stats['actions']['max']) > 5 else converted_stats['actions']['max']}"
                            )
                            return converted_stats
                        logging.warning(
                            f"Norm stats found but missing required keys (q01/q99 or min/max): {list(actions_norm.keys())}"
                        )
                    else:
                        logging.warning("Norm stats file exists but missing 'norm_stats.actions' structure")
                else:
                    logging.warning(f"Norm stats path from config does not exist: {norm_stats_path}")
        except Exception as e:
            logging.debug(f"Could not load norm stats from config path: {e}")

        # Fallback: Try multiple possible locations for norm_stats.json
        logging.info("Trying fallback locations for norm_stats.json...")
        possible_paths = []

        # Check in common asset locations
        if hasattr(config, "assets_dirs"):
            # Handle both single path and list of paths
            assets_dirs = config.assets_dirs
            if not isinstance(assets_dirs, (list, tuple)):
                assets_dirs = [assets_dirs]

            for assets_dir in assets_dirs:
                norm_path = Path(assets_dir) / config.data.repo_id / "norm_stats.json"
                if norm_path.exists():
                    possible_paths.append(norm_path)
                    logging.info(f"Found potential norm_stats at: {norm_path}")

        # Check in checkpoint directory assets (common for trained models)
        if checkpoint_path is not None:
            # Construct the same path as load_trained_policy does
            if checkpoint_step is not None:
                actual_checkpoint_path = f"{checkpoint_path}/{checkpoint_step}"
            else:
                actual_checkpoint_path = checkpoint_path

            checkpoint_assets_path = Path(actual_checkpoint_path) / "assets" / config.data.repo_id / "norm_stats.json"
            if checkpoint_assets_path.exists():
                possible_paths.append(checkpoint_assets_path)
                logging.info(f"Found potential norm_stats at checkpoint assets: {checkpoint_assets_path}")

        # Try to create a minimal dataset instance to access its root
        try:
            dataset = LeRobotDataset(config.data.repo_id)
            if hasattr(dataset, "root") and dataset.root is not None:
                norm_path = Path(dataset.root) / "norm_stats.json"
                if norm_path.exists():
                    possible_paths.append(norm_path)
                    logging.info(f"Found potential norm_stats at dataset root: {norm_path}")
        except Exception as e:
            logging.debug(f"Could not access dataset root: {e}")

        for norm_stats_path in possible_paths:
            try:
                logging.info(f"Trying to load norm stats from: {norm_stats_path}")
                with open(norm_stats_path) as f:
                    norm_data = json.load(f)

                if "norm_stats" in norm_data and "actions" in norm_data["norm_stats"]:
                    actions_norm = norm_data["norm_stats"]["actions"]
                    logging.info(f"Successfully loaded norm stats from {norm_stats_path}")
                    logging.info(f"Actions norm stats keys: {list(actions_norm.keys())}")

                    # Convert norm stats to min/max format for plotting
                    if "q01" in actions_norm and "q99" in actions_norm:
                        converted_stats = {
                            "actions": {
                                "min": actions_norm["q01"],
                                "max": actions_norm["q99"],
                                "mean": actions_norm.get("mean", []),
                                "std": actions_norm.get("std", []),
                            }
                        }
                        logging.info("Converted quantiles (q01/q99) to min/max format")
                        logging.info(
                            f"Actions min (q01): {converted_stats['actions']['min'][:5] if len(converted_stats['actions']['min']) > 5 else converted_stats['actions']['min']}"
                        )
                        logging.info(
                            f"Actions max (q99): {converted_stats['actions']['max'][:5] if len(converted_stats['actions']['max']) > 5 else converted_stats['actions']['max']}"
                        )
                        return converted_stats
                    if "min" in actions_norm and "max" in actions_norm:
                        # Already in min/max format
                        converted_stats = {"actions": actions_norm}
                        logging.info("Found actions stats in min/max format")
                        return converted_stats
                    logging.warning(
                        f"Norm stats found but missing required keys (q01/q99 or min/max): {list(actions_norm.keys())}"
                    )
                else:
                    logging.warning(
                        f"Norm stats file exists but missing 'norm_stats.actions' structure in {norm_stats_path}"
                    )

            except Exception as e:
                logging.debug(f"Could not load norm stats from {norm_stats_path}: {e}")
                continue

        logging.warning("Could not find dataset statistics in any expected location")
        return None

    except Exception as e:
        logging.warning(f"Failed to load dataset statistics: {e}")
        return None


def main(
    config: _config.TrainConfig,
    num_eval_batches: int = 10,
    checkpoint_step: int | None = None,
    checkpoint_dir: str | None = None,
):
    """Main evaluation function."""

    logging.info(f"Running evaluation on: {platform.node()}")

    # Determine checkpoint directory
    checkpoint_path = checkpoint_dir if checkpoint_dir is not None else str(config.checkpoint_dir)

    # Create output directory
    eval_output_dir = epath.Path(checkpoint_path) / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Load trained policy
    logging.info("Loading trained policy...")
    policy = load_trained_policy(config, checkpoint_path, checkpoint_step)
    logging.info("Loaded policy successfully")

    # Create data loader with real data
    logging.info("Creating data loader...")
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    # Apply video backend patch
    logging.info("Applying video backend patches...")
    patch_video_loading()

    # Force video backend to pyav in config
    if hasattr(config.data, "video_backend"):
        config.data.video_backend = "pyav"

    # We'll skip norm stats loading in the data loader and handle statistics loading separately for plots

    # Debug: Print dataset configuration
    logging.info(f"Dataset repo_id: {config.data.repo_id}")
    logging.info(f"Dataset config: {config.data}")

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
        num_batches=num_eval_batches,
        skip_norm_stats=True,  # Skip norm stats loading for evaluation
    )

    # Setup evaluation
    rng = jax.random.key(config.seed)
    all_metrics = []
    all_actions_gt = []
    all_actions_pred = []

    # Run evaluation
    logging.info(f"Running evaluation on {num_eval_batches} batches...")
    data_iter = iter(data_loader)

    for batch_idx in tqdm.tqdm(range(num_eval_batches), desc="Evaluating"):
        # Get batch
        observation, actions_gt = next(data_iter)

        # Debug: Print observation info for first batch
        if batch_idx == 0:
            logging.info("=== OBSERVATION INFO (First Batch) ===")

            if hasattr(observation, "images"):
                logging.info(f"Image keys: {list(observation.images.keys())}")

                # Create directory for saving sample images
                sample_images_dir = eval_output_dir / "sample_images"
                sample_images_dir.mkdir(parents=True, exist_ok=True)

                for key, img in observation.images.items():
                    img_array = np.array(img)
                    logging.info(
                        f"Image '{key}': shape={img_array.shape}, range=[{img_array.min():.2f}, {img_array.max():.2f}]"
                    )

                    # Check for duplicate images in the batch
                    if img_array.shape[0] >= 3:
                        img0 = img_array[0]
                        img1 = img_array[1]
                        img2 = img_array[2]

                        # Compare images using MSE
                        mse_01 = np.mean((img0 - img1) ** 2)
                        mse_02 = np.mean((img0 - img2) ** 2)
                        mse_12 = np.mean((img1 - img2) ** 2)

                        logging.info(f"Image '{key}' similarity check:")
                        logging.info(f"  MSE between sample 0 and 1: {mse_01:.6f}")
                        logging.info(f"  MSE between sample 0 and 2: {mse_02:.6f}")
                        logging.info(f"  MSE between sample 1 and 2: {mse_12:.6f}")

                        if mse_01 < 1e-10 and mse_02 < 1e-10:
                            logging.warning("  ⚠️  Images appear to be IDENTICAL (likely duplicate data)")
                        elif mse_01 < 1e-6 and mse_02 < 1e-6:
                            logging.warning("  ⚠️  Images appear to be nearly identical (very similar)")
                        else:
                            logging.info("  ✅ Images appear to be different (normal)")

                        # Save difference images to visualize the subtle differences
                        if key == "base_0_rgb":  # Only for base camera to avoid clutter
                            diff_01 = np.abs(img0 - img1)
                            diff_02 = np.abs(img0 - img2)

                            # Amplify differences for visualization (scale by 10)
                            diff_01_amplified = np.clip(diff_01 * 10, 0, 1)
                            diff_02_amplified = np.clip(diff_02 * 10, 0, 1)

                            # Convert to [0, 255] for saving
                            diff_01_uint8 = (diff_01_amplified * 255).astype(np.uint8)
                            diff_02_uint8 = (diff_02_amplified * 255).astype(np.uint8)

                            # Save difference images
                            diff_01_img = Image.fromarray(diff_01_uint8)
                            diff_02_img = Image.fromarray(diff_02_uint8)

                            diff_01_path = sample_images_dir / f"{key}_diff_0_vs_1_amplified.png"
                            diff_02_path = sample_images_dir / f"{key}_diff_0_vs_2_amplified.png"

                            diff_01_img.save(diff_01_path)
                            diff_02_img.save(diff_02_path)

                            logging.info("  💡 Saved amplified difference images:")
                            logging.info(f"    - {diff_01_path}")
                            logging.info(f"    - {diff_02_path}")

                    # Save a few sample images from the batch for visual verification
                    for sample_idx in range(min(3, img_array.shape[0])):  # Save first 3 samples
                        sample_img = img_array[sample_idx]  # Shape: (224, 224, 3)

                        # Convert from [-1, 1] back to [0, 255] for saving
                        sample_img_uint8 = ((sample_img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

                        # Save as PNG
                        pil_img = Image.fromarray(sample_img_uint8)
                        image_path = sample_images_dir / f"{key}_sample_{sample_idx}.png"
                        pil_img.save(image_path)

                logging.info(f"Sample images saved to: {sample_images_dir}")

            if hasattr(observation, "state"):
                state_array = np.array(observation.state)
                logging.info(f"State: shape={state_array.shape}")

                # Check if states are also duplicated
                if state_array.shape[0] >= 3:
                    state0 = state_array[0]
                    state1 = state_array[1]
                    state2 = state_array[2]

                    state_mse_01 = np.mean((state0 - state1) ** 2)
                    state_mse_02 = np.mean((state0 - state2) ** 2)

                    # logging.info(f"State similarity check:")
                    # logging.info(f"  MSE between sample 0 and 1: {state_mse_01:.6f}")
                    # logging.info(f"  MSE between sample 0 and 2: {state_mse_02:.6f}")

                    if state_mse_01 < 1e-10 and state_mse_02 < 1e-10:
                        logging.warning("  ⚠️  States appear to be IDENTICAL")
            #         else:
            #             logging.info(f"  ✅ States appear to be different")

            # logging.info(f"Actions GT: shape={np.array(actions_gt).shape}")

            # Check actions for duplication too
            actions_array = np.array(actions_gt)
            if actions_array.shape[0] >= 3:
                action0 = actions_array[0]
                action1 = actions_array[1]
                action2 = actions_array[2]

                action_mse_01 = np.mean((action0 - action1) ** 2)
                action_mse_02 = np.mean((action0 - action2) ** 2)

                # logging.info(f"Action similarity check:")
                # logging.info(f"  MSE between sample 0 and 1: {action_mse_01:.6f}")
                # logging.info(f"  MSE between sample 0 and 2: {action_mse_02:.6f}")

                if action_mse_01 < 1e-10 and action_mse_02 < 1e-10:
                    logging.warning("  ⚠️  Actions appear to be IDENTICAL")
                # else:
                #     logging.info(f"  ✅ Actions appear to be different")

            logging.info("==========================================")

        # Run inference
        eval_rng = jax.random.fold_in(rng, batch_idx)
        actions_pred, metrics = evaluate_policy(policy, observation, actions_gt, eval_rng)

        # Store results
        all_metrics.append(metrics)
        all_actions_gt.append(np.array(actions_gt))
        all_actions_pred.append(np.array(actions_pred))

        # Log progress
        if batch_idx % 5 == 0:
            logging.info(f"Batch {batch_idx}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

    # Concatenate all results
    all_actions_gt = np.concatenate(all_actions_gt, axis=0)
    all_actions_pred = np.concatenate(all_actions_pred, axis=0)

    logging.info(f"Evaluation complete. Total samples: {all_actions_gt.shape[0]}")
    logging.info(f"Action shape: {all_actions_gt.shape}")

    # Load dataset statistics for proper y-axis scaling
    logging.info("Loading dataset statistics for plot scaling...")
    dataset_stats = load_dataset_stats(config, checkpoint_path, checkpoint_step)
    if dataset_stats is not None:
        logging.info("Successfully loaded dataset statistics")
    else:
        logging.warning("Could not load dataset statistics - plots will use auto-scaling")

    # Create visualizations
    logging.info("Creating visualizations...")

    # Determine action names based on the dataset
    action_names = None
    data_config = config.data.create(config.assets_dirs, config.model)
    if hasattr(config.data, "repo_id") and config.data.repo_id:
        if "yam" in config.data.repo_id.lower():
            # YAM has 14 actual DOF but model uses 32-dim actions (padded)
            action_names = [
                "L_Joint_0",
                "L_Joint_1",
                "L_Joint_2",
                "L_Joint_3",
                "L_Joint_4",
                "L_Joint_5",
                "L_Gripper",
                "R_Joint_0",
                "R_Joint_1",
                "R_Joint_2",
                "R_Joint_3",
                "R_Joint_4",
                "R_Joint_5",
                "R_Gripper",
            ]
            # Extend to 32 dimensions with padding names
            action_names.extend([f"Pad_{i}" for i in range(14, 32)])
        elif "xmi" in config.data.repo_id.lower():
            action_names = [
                "L_Rot_0",
                "L_Rot_1",
                "L_Rot_2",
                "L_Rot_3",
                "L_Rot_4",
                "L_Rot_5",
                "L_Pos_X",
                "L_Pos_Y",
                "L_Pos_Z",
                "L_Gripper",
                "R_Rot_0",
                "R_Rot_1",
                "R_Rot_2",
                "R_Rot_3",
                "R_Rot_4",
                "R_Rot_5",
                "R_Pos_X",
                "R_Pos_Y",
                "R_Pos_Z",
                "R_Gripper",
            ]
        elif "aloha" in config.data.repo_id.lower():
            action_names = [
                "L_Joint_0",
                "L_Joint_1",
                "L_Joint_2",
                "L_Joint_3",
                "L_Joint_4",
                "L_Joint_5",
                "L_Gripper",
                "R_Joint_0",
                "R_Joint_1",
                "R_Joint_2",
                "R_Joint_3",
                "R_Joint_4",
                "R_Joint_5",
                "R_Gripper",
            ]

    # Action comparison plots
    logging.info("Creating action comparison plot...")
    logging.info(f"GT actions shape: {all_actions_gt.shape}")
    logging.info(f"Pred actions shape: {all_actions_pred.shape}")

    plot_action_comparison(
        all_actions_gt,
        all_actions_pred,
        str(eval_output_dir / "action_comparison.png"),
        action_names=action_names,
        action_stats=dataset_stats,
    )

    # Error heatmap
    plot_error_heatmap(
        all_actions_gt,
        all_actions_pred,
        str(eval_output_dir / "error_heatmap.png"),
        action_names=action_names,
    )

    # Error distribution
    plot_error_distribution(
        all_actions_gt,
        all_actions_pred,
        str(eval_output_dir / "error_distribution.png"),
        action_names=action_names,
    )

    # Summary metrics
    summary_metrics = create_summary_metrics_plot(
        all_metrics,
        str(eval_output_dir / "summary_metrics.png"),
    )

    # Save detailed results (excluding config which may not be JSON serializable)
    results = {
        "summary_metrics": summary_metrics,
        "all_metrics": all_metrics,
        "evaluation_info": {
            "num_batches": num_eval_batches,
            "total_samples": int(all_actions_gt.shape[0]),
            "action_shape": [int(x) for x in all_actions_gt.shape],
            "checkpoint_step": checkpoint_step if checkpoint_step is not None else "latest",
            "config_name": config.name if hasattr(config, "name") else "unknown",
        },
    }

    import json

    with open(eval_output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logging.info("=" * 60)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 60)
    for metric_name, metric_data in summary_metrics.items():
        logging.info(f"{metric_name.upper()}: {metric_data['mean']:.6f} ± {metric_data['std']:.6f}")
    logging.info("=" * 60)
    logging.info(f"Results saved to: {eval_output_dir}")
    logging.info("Generated plots:")
    logging.info(f"  - Action comparison: {eval_output_dir / 'action_comparison.png'}")
    logging.info(f"  - Error heatmap: {eval_output_dir / 'error_heatmap.png'}")
    logging.info(f"  - Error distribution: {eval_output_dir / 'error_distribution.png'}")
    logging.info(f"  - Summary metrics: {eval_output_dir / 'summary_metrics.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained OpenPI model")
    parser.add_argument("--config-name", type=str, required=True, help="Name of the training config to use")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name (required if not set in config)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory (default: use config.checkpoint_dir)",
    )
    parser.add_argument("--num-eval-batches", type=int, default=10, help="Number of batches to evaluate on")
    parser.add_argument(
        "--checkpoint-step", type=int, default=None, help="Specific checkpoint step to load (default: latest)"
    )

    args = parser.parse_args()

    # Initialize logging early for error messages
    init_logging()

    # Load config
    config = _config.get_config(args.config_name)

    # Set experiment name if provided
    if args.exp_name is not None:
        config = dataclasses.replace(config, exp_name=args.exp_name)

    # Validate that exp_name is set
    try:
        # This will trigger the error if exp_name is missing
        _ = config.checkpoint_dir
    except (TypeError, AttributeError) as e:
        if "PropagatingMissingType" in str(e) or "unsupported operand type" in str(e):
            logging.error("Experiment name (exp_name) is not set in the config and not provided via --exp-name")
            logging.error("Please provide --exp-name argument or ensure the config has exp_name set")
            exit(1)
        else:
            raise

    # Run evaluation
    main(config, args.num_eval_batches, args.checkpoint_step, args.checkpoint_dir)
