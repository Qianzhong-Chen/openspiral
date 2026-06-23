import json
from pathlib import Path

import cv2
import numpy as np


def extract_video_frames(video_path):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return np.array(frames)


def load_episode_data(episode_path: Path):
    """Load data for a specific episode."""
    print(f"Loading episode: {episode_path.name}")

    try:
        # Load action data (4x4 transformation matrices)
        action_data = {}
        action_files = [
            "action-left-hand_in_quest_world_frame.npy",
            "action-left-head.npy",
            "action-left-pos.npy",
            "action-left-quest_world_frame.npy",
            "action-right-hand_in_quest_world_frame.npy",
            "action-right-head.npy",
            "action-right-pos.npy",
            "action-right-quest_world_frame.npy",
        ]

        for action_file in action_files:
            file_path = episode_path / action_file
            if file_path.exists():
                action_data[action_file.replace(".npy", "")] = np.load(file_path)

        # Load joint data
        joint_data = {}
        joint_files = [
            "left-gripper_pos.npy",
            "left-joint_eff.npy",
            "left-joint_pos.npy",
            "left-joint_vel.npy",
            "right-gripper_pos.npy",
            "right-joint_eff.npy",
            "right-joint_pos.npy",
            "right-joint_vel.npy",
        ]

        for joint_file in joint_files:
            file_path = episode_path / joint_file
            if file_path.exists():
                joint_data[joint_file.replace(".npy", "")] = np.load(file_path)

        # Load timestamp data
        timestamp_data = {}
        timestamp_files = [
            "timestamp.npy",
            "timestamp_end.npy",
            "left_camera-timestamp.npy",
            "right_camera-timestamp.npy",
            "top_camera-timestamp.npy",
        ]

        for timestamp_file in timestamp_files:
            file_path = episode_path / timestamp_file
            if file_path.exists():
                timestamp_data[timestamp_file.replace(".npy", "")] = np.load(file_path)

        # Load annotation data if available
        annotation_file = episode_path / "top_camera-images-rgb_annotation.json"
        annotations = None
        if annotation_file.exists():
            with open(annotation_file) as f:
                annotations = json.load(f)

        # Find and extract video files
        video_files = {}
        images = {}
        camera_names = []

        for video_file in episode_path.glob("*.mp4"):
            camera_name = video_file.stem  # e.g., "left_camera-images-rgb"
            video_files[camera_name] = video_file

            # Extract frames from video
            frames = extract_video_frames(video_file)
            if len(frames) > 0:
                images[camera_name] = frames
                camera_names.append(camera_name)

        # Synchronize frame counts across all cameras
        if images:
            min_frames = min(len(frames) for frames in images.values())
            for camera_name in images:
                images[camera_name] = images[camera_name][:min_frames]
            print(f"Loaded {len(images)} camera feeds with {min_frames} frames each")

            # Print resolution info for each camera
            for camera_name in camera_names:
                shape = images[camera_name].shape
                print(f"  {camera_name}: {shape[1]}x{shape[2]} resolution")

        return {
            "action_data": action_data,
            "joint_data": joint_data,
            "timestamp_data": timestamp_data,
            "images": images,
            "camera_names": camera_names,
            "annotations": annotations,
            "video_files": video_files,
        }

    except Exception as e:
        print(f"Error loading episode data: {e}")
        raise


def _load_episode_data(episode_path: Path):
    """Load data for a specific episode (legacy function for backward compatibility)."""
    # This is the original function - keeping for any existing usage
    # But redirecting to the main function
    return load_episode_data(episode_path)
