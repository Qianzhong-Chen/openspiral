"""
Dataset utilities for YAMS data conversion.

This module contains functions for creating and managing LeRobot datasets with hardware encoding.
"""

from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

try:
    from .data_utils import CAMERA_KEY_MAPPING
    from .data_utils import CAMERA_KEYS
    from .video_utils import encode_video_hardware
except ImportError:
    from data_utils import CAMERA_KEY_MAPPING
    from data_utils import CAMERA_KEYS
    from video_utils import encode_video_hardware


def save_episode_with_hardware_encoding(
    episode_data: dict, episode_idx: int, dataset_path: Path, encoder: str = None
) -> dict:
    """Save episode data with hardware-accelerated video encoding."""

    frames_data = episode_data["frames_data"]
    language_instruction = episode_data["language_instruction"]
    seq_length = episode_data["seq_length"]

    if not frames_data:
        return None

    # Create episode directory
    episode_dir = dataset_path / "videos" / f"episode_{episode_idx:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Group frames by camera for video creation
    camera_frames = {}
    joint_data = []

    for step, frame_data in enumerate(frames_data):
        # Collect joint data
        joint_data.append(
            {
                "joint_positions": frame_data["joint_positions"],
                "actions": frame_data["actions"],
            }
        )

        # Collect camera frames
        for camera_key in CAMERA_KEYS:
            if camera_key in CAMERA_KEY_MAPPING:
                lerobot_key = CAMERA_KEY_MAPPING[camera_key]
                if lerobot_key in frame_data:
                    if lerobot_key not in camera_frames:
                        camera_frames[lerobot_key] = []
                    camera_frames[lerobot_key].append(frame_data[lerobot_key])

    # Save videos using hardware encoding
    video_paths = {}
    for camera_name, frames in camera_frames.items():
        if frames:
            video_path = episode_dir / f"{camera_name}.mp4"
            encode_video_hardware(frames, video_path, fps=30, encoder=encoder)
            video_paths[camera_name] = str(video_path)

    return {
        "episode_idx": episode_idx,
        "joint_data": joint_data,
        "video_paths": video_paths,
        "language_instruction": language_instruction,
        "seq_length": seq_length,
    }


def create_lerobot_dataset_from_encoded_videos(
    dataset_path: Path, episode_results: list[dict], repo_name: str
) -> LeRobotDataset:
    """Create LeRobot dataset from pre-encoded videos."""

    # Create dataset features (no video features since we handle videos separately)
    dataset_features = {
        "joint_positions": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["actions"],
        },
    }

    # Create dataset without video features initially
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="yam",
        fps=30,
        features=dataset_features,
    )

    # Add episodes
    for episode_result in episode_results:
        if episode_result is None:
            continue

        joint_data = episode_result["joint_data"]
        language_instruction = episode_result["language_instruction"]

        # Add frames to dataset (without videos)
        for frame_data in joint_data:
            dataset.add_frame(frame_data, task=language_instruction)

        # Save episode
        dataset.save_episode()

    return dataset


def create_standard_lerobot_dataset(
    repo_name: str, resize_size: int, skip_videos: bool, max_workers: int
) -> LeRobotDataset:
    """Create standard LeRobot dataset with video features."""
    # Create LeRobot dataset for joint space data
    dataset_features = {
        "joint_positions": {
            "dtype": "float32",
            "shape": (14,),  # 6 joints + 1 gripper per arm * 2 arms
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (14,),  # Same as state - joint space actions
            "names": ["actions"],
        },
    }

    # Add camera features dynamically based on available cameras (unless skipping videos)
    if not skip_videos:
        for camera_key in CAMERA_KEYS:
            if camera_key in CAMERA_KEY_MAPPING:
                dataset_features[CAMERA_KEY_MAPPING[camera_key]] = {
                    "dtype": "video",
                    "shape": (resize_size, resize_size, 3),
                    "names": ["height", "width", "channel"],
                }

    print(f"Dataset features: {list(dataset_features.keys())}")

    # Override LeRobot's video encoding settings for hardware acceleration
    video_backend = "pyav" if not skip_videos else "imageio"

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="yam",
        fps=30,
        features=dataset_features,
        image_writer_threads=max_workers if not skip_videos else 1,
        image_writer_processes=max_workers // 2 if not skip_videos else 1,
        video_backend=video_backend,
    )

    return dataset


def cleanup_and_prepare_output_dir(output_path: Path) -> None:
    """Clean up existing dataset directory and prepare for new data."""
    if output_path.exists():
        shutil.rmtree(output_path)
    print(f"Dataset will be saved to: {output_path}")


def finalize_dataset(dataset_path: Path, temp_repo_name: str, final_repo_name: str, hf_lerobot_home: Path) -> None:
    """Move dataset from temporary location to final location."""
    temp_path = hf_lerobot_home / temp_repo_name
    final_path = hf_lerobot_home / final_repo_name

    if temp_path.exists() and temp_path != final_path:
        if final_path.exists():
            shutil.rmtree(final_path)
        shutil.move(str(temp_path), str(final_path))
