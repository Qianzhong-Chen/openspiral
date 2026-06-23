"""
Data processing utilities for YAMS data conversion.

This module contains functions for loading, processing, and validating YAMS episode data.
"""

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .video_utils import extract_video_frames_fast
    from .video_utils import resize_frames_vectorized
except ImportError:
    from video_utils import extract_video_frames_fast
    from video_utils import resize_frames_vectorized

# YAMS data configuration
CAMERA_KEYS = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]

CAMERA_KEY_MAPPING = {
    "left_camera-images-rgb": "exterior_image_1_left",
    "right_camera-images-rgb": "exterior_image_2_right",
    "top_camera-images-rgb": "exterior_image_3_top",
}


def find_episode_directories(parent_dir: Path | list[Path]) -> list[Path]:
    """Find all YAMS episode directories."""

    # Handle both single path and list of paths
    if isinstance(parent_dir, list):
        parent_dirs = [Path(dir) for dir in parent_dir]
    else:
        parent_dirs = [Path(parent_dir)]

    episode_dirs = []
    for parent in parent_dirs:
        for item in parent.iterdir():
            # if item.is_dir() and item.name.startswith("episode_") and item.name.endswith(".npy.mp4"):
            if item.is_dir() and item.name.startswith("episode_"):
                episode_dirs.append(item)
    return sorted(episode_dirs)


def load_episode_annotations(episode_path: Path) -> dict:
    """Load annotation data for a YAMS episode to check quality labels."""
    annotations = {}

    # Look for annotation files for each camera
    camera_prefixes = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]

    for camera_prefix in camera_prefixes:
        annotation_file = episode_path / f"{camera_prefix}_annotation.json"
        if annotation_file.exists():
            try:
                with open(annotation_file) as f:
                    annotation_data = json.load(f)
                    annotations[camera_prefix] = annotation_data
            except Exception as e:
                print(f"Warning: Could not load annotation file {annotation_file}: {e}")

    return annotations

# def split_stage_from_annotation(episode_path) -> dict:
#     annotations = load_episode_annotations(episode_path)
#     """
#     Extract stage labels and corresponding frame ranges from annotation data.

#     Args:
#         annotations (dict): Dictionary mapping camera names to their annotation content.

#     Returns:
#         dict: A dictionary mapping each label to a [from_frame, to_frame] list.
#               If multiple cameras exist, it uses the first found one.
#     """
#     try: 
#         for camera_prefix, data in annotations.items():
#             if "annotations" in data:
#                 return {
#                     ann["label"]: [ann["from_frame"], ann["to_frame"]]
#                     for ann in data["annotations"]
#                 }
#     except Exception as e:
#         print(f"Error processing dense annotations for {episode_path}")
#         return None  # Return None if no valid annotation found

def split_stage_from_annotation(episode_path) -> dict:
    annotations = load_episode_annotations(episode_path)
    """
    Extract stage labels and corresponding frame ranges from annotation data.

    Args:
        annotations (dict): Dictionary mapping camera names to their annotation content.

    Returns:
        dict: A dictionary mapping each label to a [from_frame, to_frame] list.
              If multiple cameras exist, it uses the first found one.
    """
    try: 
        for camera_prefix, data in annotations.items():
            if "annotations" in data:
                result = {}
                for ann in data["annotations"]:
                    if ann["label"] not in result:
                        result[ann["label"]] = [[ann["from_frame"], ann["to_frame"]]]
                    else:
                        result[ann["label"]].append([ann["from_frame"], ann["to_frame"]])
                return result
    except Exception as e:
        print(f"Error processing dense annotations for {episode_path}")
        return None  # Return None if no valid annotation found

def split_and_combine_hybird_annotations(episode_path) -> dict | None:
    """
    select sparse or dense annotation processing based on the episode path.
    """
    data = load_episode_annotations(episode_path)
    try:
        data = data["top_camera-images-rgb"]
        anns = data["annotations"]
    except KeyError as e:
        print(f"[ERROR] {episode_path}: Missing expected key in annotations: {e}")
        return None
    if not isinstance(anns, list):
        print(f"[ERROR] {episode_path}: 'annotations' is not a list.")
        return None
    
    if len(anns) == 5:
        return split_and_combine_sparse_annotations(episode_path)
    elif len(anns) == 14:
        return split_and_combine_dense_annotations(episode_path)
    else:
        print(f"[ERROR] {episode_path}: Unexpected number of annotations: {len(anns)}")
        return None

def split_and_combine_sparse_annotations(episode_path) -> dict | None:
    """
    Validate a sparse 5-step annotation sequence, merge the first two steps
    into one, and return:
        { label: [[from_frame, to_frame]], ... }
    Returns None if the raw labels are not exactly the expected 5 in order.
    """
    data = load_episode_annotations(episode_path)  # replace with your loader
    try:
        data = data["top_camera-images-rgb"]
        anns = data["annotations"]
    except KeyError as e:
        print(f"[ERROR] {episode_path}: Missing expected key in annotations: {e}")
        return None

    if not isinstance(anns, list):
        print(f"[ERROR] {episode_path}: 'annotations' is not a list.")
        return None

    # Expected 5-step order
    expected_seq = [
        "Grab the tshirt from the pile",
        "Move the tshirt to the center of the board",
        "Flatten the tshirt out",
        "Fold the tshirt",
        "Neatly place the folded tshirt to the corner",
    ]

    labels_in = [a.get("label") for a in anns]
    if labels_in != expected_seq:
        print(f"[ERROR] {episode_path}: Label sequence mismatch.\nExpected: {expected_seq}\nGot:      {labels_in}")
        return None

    # Combine first two steps
    combined: dict[str, list[list[int]]] = {}

    # Step 1+2 merged
    start = int(anns[0]["from_frame"])
    end = int(anns[1]["to_frame"])
    combined["Grab the tshirt from the pile and move the tshirt to the center of the board"] = [[start, end]]

    # Remaining steps unchanged
    for a in anns[2:]:
        combined[a["label"]] = [[int(a["from_frame"]), int(a["to_frame"])]]

    return combined

def split_and_combine_sparse_annotations_multi(episode_path) -> dict | None:
    """
    Handle multiple sequences where "Move the tshirt to the center of the board"
    might be missing. Merge the first step (or first two if present) into
    "Grab the tshirt from the pile and move the tshirt to the center of the board".
    Returns None if any unexpected label appears.
    """
    data = load_episode_annotations(episode_path)
    try:
        data = data["top_camera-images-rgb"]
        anns = data["annotations"]
    except KeyError as e:
        print(f"[ERROR] {episode_path}: Missing expected key: {e}")
        return None

    if not isinstance(anns, list):
        print(f"[ERROR] {episode_path}: 'annotations' is not a list.")
        return None

    # The remaining steps after the first two
    remaining_steps = [
        "Flatten the tshirt out",
        "Fold the tshirt",
        "Neatly place the folded tshirt to the corner",
    ]

    combined: dict[str, list[list[int]]] = {}
    i = 0

    while i < len(anns):
        # Check first step
        if anns[i]["label"] != "Grab the tshirt from the pile":
            print(f"[ERROR] {episode_path}: Expected 'Grab the tshirt from the pile' at index {i}, got {anns[i]['label']}")
            return None

        start = int(anns[i]["from_frame"])
        end = int(anns[i]["to_frame"])
        i += 1

        # Check if next step is "Move the tshirt to the center of the board"
        if i < len(anns) and anns[i]["label"] == "Move the tshirt to the center of the board":
            end = int(anns[i]["to_frame"])
            i += 1  # consume this step

        # Add merged label
        merged_label = "Grab the tshirt from the pile and move the tshirt to the center of the board"
        combined.setdefault(merged_label, []).append([start, end])

        # Add remaining steps
        for step in remaining_steps:
            if i >= len(anns) or anns[i]["label"] != step:
                print(f"[ERROR] {episode_path}: Expected '{step}' at index {i}, got {anns[i]['label'] if i < len(anns) else 'EOF'}")
                return None
            combined.setdefault(step, []).append([int(anns[i]["from_frame"]), int(anns[i]["to_frame"])])
            i += 1

    return combined



def split_and_combine_dense_annotations(episode_path) -> dict | None:
    """
    Validate a dense 14-step annotation sequence, merge 'reset both arms' and
    'smooth out wrinkles on the folded tshirt' into the previous step, and return
    combined task spans as:
        { combined_label: [[from_frame, to_frame]], ... }
    Returns None if the raw labels are not exactly the expected 14 in order (after merging).
    """
    data = load_episode_annotations(episode_path)
    try:
        data = data["top_camera-images-rgb"]
        anns = data["annotations"]
    except KeyError as e:
        print(f"[ERROR] {episode_path}: Missing expected key in annotations: {e}")
        return None

    if not isinstance(anns, list):
        print(f"[ERROR] {episode_path}: 'annotations' is not a list.")
        return None

    # Raw labels
    L = {
        "grab_bin": "grab crumpled tshirt from bin with the right arm",
        "move_center": "move the tshirt to the center of the board",
        "flatten": "flatten out the tshirt",
        "grab_close": "grab the side of the flattened tshirt closest away from the robot with both arms",
        "fold_1of3_flattened": "fold the flattened tshirt in 1/3 of the way",
        "grab_far": "grab the side of the flattened tshirt furthest away from the robot with both arms",
        "fold_other_side": "fold the other side of the tshirt into a rectangle",
        "rotate_90": "rotate the tshirt 90 degrees",
        "grab_bottom": "grab the bottom of the folded rectangular tshirt with both arms",
        "fold_1of3_folded": "fold the folded rectangular tshirt in 1/3 of the way",
        "grab_2of3": "grab one side of the 2/3 rectangular tshirt with both arms",
        "fold_neat_square": "fold the long partially folded tshirt into a neat square",
        "put_corner": "put the neatly folded tshirt into the corner of the board with the left arm",
    }
    RESET_LABEL = "reset both arms"
    SMOOTH_LABEL = "smooth out wrinkles on the folded tshirt"

    expected_seq = [
        L["grab_bin"],
        L["move_center"],
        L["flatten"],
        L["grab_close"],
        L["fold_1of3_flattened"],
        L["grab_far"],
        L["fold_other_side"],
        L["rotate_90"],
        L["grab_bottom"],
        L["fold_1of3_folded"],
        L["grab_2of3"],
        L["fold_neat_square"],
        L["put_corner"],
        RESET_LABEL,
    ]

    # --- Step 1: merge special labels into the previous step ---
    merged_anns = []
    skip_next = False
    for idx, ann in enumerate(anns):
        if skip_next:
            skip_next = False
            continue
        label = ann["label"]
        if label in (RESET_LABEL, SMOOTH_LABEL):
            if merged_anns:
                # extend previous to_frame
                merged_anns[-1]["to_frame"] = max(merged_anns[-1]["to_frame"], ann["to_frame"])
            # skip adding this step entirely
            continue
        merged_anns.append(ann)

    anns = merged_anns

    # --- Step 2: validate sequence length and order ---
    if len(anns) != len(expected_seq) - 1:  # minus 1 because RESET is merged
        print(f"[ERROR] {episode_path}: Expected {len(expected_seq)-1} annotations after merging, got {len(anns)}.")
        return None

    labels_in = [a.get("label") for a in anns]
    expected_after_merge = expected_seq[:-1]  # without RESET
    if labels_in != expected_after_merge:
        print(f"[ERROR] {episode_path}: Label sequence mismatch after merging.\nExpected: {expected_after_merge}\nGot:      {labels_in}")
        return None

    # --- Step 3: build map: raw label -> (from_frame, to_frame) ---
    try:
        label_to_span = {
            a["label"]: (int(a["from_frame"]), int(a["to_frame"]))
            for a in anns
        }
    except Exception as e:
        print(f"[ERROR] {episode_path}: Failed parsing frame numbers — {e}")
        return None

    # --- Step 4: define combined tasks ---
    COMBINED_TASKS = [
        ("grab crumpled tshirt from bin with the right arm and move the tshirt to the center of the board",
         [L["grab_bin"], L["move_center"]]),
        ("flatten out the tshirt",
         [L["flatten"]]),
        ("grab the side of the flattened tshirt closest away from the robot with both arms and fold the flattened tshirt in 1/3 of the way",
         [L["grab_close"], L["fold_1of3_flattened"]]),
        ("grab the side of the flattened tshirt furthest away from the robot with both arms and fold the other side of the tshirt into a rectangle",
         [L["grab_far"], L["fold_other_side"]]),
        ("rotate the tshirt 90 degrees",
         [L["rotate_90"]]),
        ("grab the bottom of the folded rectangular tshirt with both arms and fold the folded rectangular tshirt in 1/3 of the way",
         [L["grab_bottom"], L["fold_1of3_folded"]]),
        ("grab one side of the 2/3 rectangular tshirt with both arms and fold the long partially folded tshirt into a neat square",
         [L["grab_2of3"], L["fold_neat_square"]]),
        ("put the neatly folded tshirt into the corner of the board with the left arm",
         [L["put_corner"]]),
    ]

    # --- Step 5: combine spans into [[start, end]] per combined task ---
    combined: dict[str, list[list[int]]] = {}
    for out_label, parts in COMBINED_TASKS:
        first_label = parts[0]
        last_label = parts[-1]
        if first_label not in label_to_span or last_label not in label_to_span:
            print(f"[ERROR] {episode_path}: Missing label(s) in combined task '{out_label}' — parts={parts}")
            return None
        start = label_to_span[first_label][0]
        end = label_to_span[last_label][1]
        if start > end:
            print(f"[ERROR] {episode_path}: Invalid frame order in '{out_label}' — start={start}, end={end}")
            return None
        combined[out_label] = [[start, end]]

    return combined


def is_episode_good_quality(episode_path: Path) -> bool:
    """Check if an episode is labeled as 'good' quality based on annotations."""
    annotations = load_episode_annotations(episode_path)

    if not annotations:
        return False

    # Check if any camera has "good" quality label
    for camera_prefix, annotation_data in annotations.items():
        video_labels = annotation_data.get("video_labels", [])
        for label in video_labels:
            if label.get("class_description") == "overall_quality" and label.get("label") == "good":
                return True

    return False


def load_yams_episode_data_fast(episode_path: Path) -> dict | None:
    """Load data for a specific YAMS episode with maximum optimizations."""
    episode_data = {"metadata": {}, "joint_data": {}, "images": {}}

    try:
        # Load metadata first (small file)
        metadata_file = episode_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                episode_data["metadata"] = json.load(f)

        # Load joint data using memory mapping for large files
        essential_joint_files = [
            "left-joint_pos.npy",
            "right-joint_pos.npy",
            "left-gripper_pos.npy",
            "right-gripper_pos.npy",
            "action-left-pos.npy",
            "action-right-pos.npy",
        ]

        # Load all joint files at once
        joint_data = {}
        for joint_file in essential_joint_files:
            file_path = episode_path / joint_file
            if file_path.exists():
                # Use memory mapping for faster loading of large arrays
                joint_data[joint_file.replace(".npy", "")] = np.load(file_path, mmap_mode="r")

        episode_data["joint_data"] = joint_data

        # Load videos with priority and early termination
        video_priorities = ["_crf18.mp4", ".mp4", "_low_res.mp4"]
        camera_prefixes = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]

        # Process videos in parallel within the episode
        for camera_prefix in camera_prefixes:
            for priority_suffix in video_priorities:
                video_file = episode_path / f"{camera_prefix}{priority_suffix}"
                if video_file.exists():
                    frames = extract_video_frames_fast(video_file)
                    if len(frames) > 0:
                        episode_data["images"][camera_prefix] = frames
                    break  # Found video for this camera, move to next

        # Force garbage collection after loading large video data
        gc.collect()

        return episode_data

    except Exception as e:
        print(f"Error loading YAMS episode data from {episode_path}: {e}")
        return None

def load_yams_episode_data_with_reward_fast(episode_path: Path) -> dict | None:
    """Load data for a specific YAMS episode with maximum optimizations."""
    episode_data = {"metadata": {}, "joint_data": {}, "images": {}}

    try:
        # Load metadata first (small file)
        metadata_file = episode_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                episode_data["metadata"] = json.load(f)

        # Load joint data using memory mapping for large files
        essential_joint_files = [
            "left-joint_pos.npy",
            "right-joint_pos.npy",
            "left-gripper_pos.npy",
            "right-gripper_pos.npy",
            "action-left-pos.npy",
            "action-right-pos.npy",
            "reward.npy",
        ]

        # Load all joint files at once
        joint_data = {}
        for joint_file in essential_joint_files:
            file_path = episode_path / joint_file
            if file_path.exists():
                # Use memory mapping for faster loading of large arrays
                joint_data[joint_file.replace(".npy", "")] = np.load(file_path, mmap_mode="r")

        episode_data["joint_data"] = joint_data
        episode_data["reward"] = np.load(episode_path / "reward.npy", mmap_mode="r")

        # Load videos with priority and early termination
        video_priorities = ["_crf18.mp4", ".mp4", "_low_res.mp4"]
        camera_prefixes = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]

        # Process videos in parallel within the episode
        for camera_prefix in camera_prefixes:
            for priority_suffix in video_priorities:
                video_file = episode_path / f"{camera_prefix}{priority_suffix}"
                if video_file.exists():
                    frames = extract_video_frames_fast(video_file)
                    if len(frames) > 0:
                        episode_data["images"][camera_prefix] = frames
                    break  # Found video for this camera, move to next

        # Force garbage collection after loading large video data
        gc.collect()

        return episode_data

    except Exception as e:
        print(f"Error loading YAMS episode data from {episode_path}: {e}")
        return None
    
def load_yams_episode_data_with_reward_progress_fast(episode_path: Path) -> dict | None:
    """Load data for a specific YAMS episode with maximum optimizations."""
    episode_data = {"metadata": {}, "joint_data": {}, "images": {}}

    try:
        # Load metadata first (small file)
        metadata_file = episode_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                episode_data["metadata"] = json.load(f)

        # Load joint data using memory mapping for large files
        essential_joint_files = [
            "left-joint_pos.npy",
            "right-joint_pos.npy",
            "left-gripper_pos.npy",
            "right-gripper_pos.npy",
            "action-left-pos.npy",
            "action-right-pos.npy",
            "reward.npy",
            "progress.npy"
        ]

        # Load all joint files at once
        joint_data = {}
        for joint_file in essential_joint_files:
            file_path = episode_path / joint_file
            if file_path.exists():
                # Use memory mapping for faster loading of large arrays
                joint_data[joint_file.replace(".npy", "")] = np.load(file_path, mmap_mode="r")

        episode_data["joint_data"] = joint_data
        episode_data["reward"] = np.load(episode_path / "reward.npy", mmap_mode="r")
        # Stored progress is in [-1, 0]; shift to [0, 1] for use as a terminal-reward scale.
        episode_data["progress"] = np.load(episode_path / "progress.npy", mmap_mode="r") + 1.0

        # Load videos with priority and early termination
        video_priorities = ["_crf18.mp4", ".mp4", "_low_res.mp4"]
        camera_prefixes = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]

        # Process videos in parallel within the episode
        for camera_prefix in camera_prefixes:
            for priority_suffix in video_priorities:
                video_file = episode_path / f"{camera_prefix}{priority_suffix}"
                if video_file.exists():
                    frames = extract_video_frames_fast(video_file)
                    if len(frames) > 0:
                        episode_data["images"][camera_prefix] = frames
                    break  # Found video for this camera, move to next

        # Force garbage collection after loading large video data
        gc.collect()

        return episode_data

    except Exception as e:
        print(f"Error loading YAMS episode data from {episode_path}: {e}")
        return None


def process_joint_data(joint_data: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """Process and combine joint data from YAMS episode."""
    # Check if we have the required joint data
    required_keys = ["left-joint_pos", "right-joint_pos", "left-gripper_pos", "right-gripper_pos"]
    if not all(key in joint_data for key in required_keys):
        return "error", None
        # raise ValueError("Missing required joint data keys") # cannot know which episode cause the issue

    # Get joint positions and gripper positions (use views, not copies)
    left_joint_pos = joint_data["left-joint_pos"]  # Shape: (N, 6)
    right_joint_pos = joint_data["right-joint_pos"]  # Shape: (N, 6)
    left_gripper_pos = joint_data["left-gripper_pos"]  # Shape: (N, 1)
    right_gripper_pos = joint_data["right-gripper_pos"]  # Shape: (N, 1)

    # Apply joint position flipping (create copies only when needed)
    # left_joint_pos = np.flip(left_joint_pos, axis=1).copy()  # Copy needed for flip # BAD BAD BAD
    # right_joint_pos = np.flip(right_joint_pos, axis=1).copy()  # Copy needed for flip # BAD BAD BAD

    # Pre-allocate full state array to avoid multiple concatenations
    seq_length = len(left_joint_pos)
    full_joint_state = np.empty((seq_length, 14), dtype=np.float32)

    full_joint_state[:, :6] = left_joint_pos
    full_joint_state[:, 6:7] = left_gripper_pos[:, 0:1]
    full_joint_state[:, 7:13] = right_joint_pos
    full_joint_state[:, 13:14] = right_gripper_pos[:, 0:1]

    if (
        "action-left-pos" in joint_data and "action-right-pos" in joint_data
    ):  # Some datasets have action data, some don't
        full_joint_action = np.empty((seq_length, 14), dtype=np.float32)
        left_action_pos = joint_data["action-left-pos"]  # Shape: (N, 7) (includes gripper)
        right_action_pos = joint_data["action-right-pos"]  # Shape: (N, 7) (includes gripper)
        full_joint_action[:, :7] = left_action_pos
        full_joint_action[:, 7:14] = right_action_pos
        return full_joint_state, full_joint_action

    return full_joint_state, None


def process_images(images: dict, seq_length: int, resize_size: int, skip_videos: bool = False) -> dict:
    """Process and resize images from YAMS episode."""
    processed_images = {}

    if not skip_videos:
        for camera_key in CAMERA_KEYS:
            if camera_key in images:
                camera_frames = images[camera_key][:seq_length]

                # Batch resize all frames at once using fastest method
                resized_frames = resize_frames_vectorized(camera_frames, resize_size)
                processed_images[camera_key] = resized_frames

    return processed_images


def calculate_actions(full_joint_state: np.ndarray, full_joint_action: np.ndarray | None, seq_length: int):
    joint_states = full_joint_state[:seq_length]
    if full_joint_action is None:
        joint_actions = joint_states.copy()
    else:  # If no action data, use joint states as actions
        joint_actions = full_joint_action[:seq_length]

    return joint_states, joint_actions


def calculate_actions_cartesian(
    full_joint_state: np.ndarray, full_joint_action: np.ndarray | None, seq_length: int, robot: Any
):
    from openpi.utils.matrix_utils import quat_to_rot_6d

    joint_states = full_joint_state[:seq_length]

    if full_joint_action is None:
        joint_actions = joint_states
    else:  # If no action data, use joint states as actions
        joint_actions = full_joint_action[:seq_length]

    # joint states ordering: left_joint_pos, left_gripper_pos, right_joint_pos, right_gripper_pos
    # T_left_ee, T_right_ee = robot.solve_fk_base(joint_states[:, :6], joint_states[:, 7:13]) # BUGGY, INTRODUCES A POSE DELAY
    cartesian_states_list = []
    cartesian_actions_list = []

    for i in range(
        seq_length
    ):  # For some reason running batched FK is buggy -- camera lags behind pose so delay is introduced somewhere somehow
        T_left_ee_state, T_right_ee_state = robot.solve_fk_base(joint_states[i, :6], joint_states[i, 7:13])
        T_left_ee_6d_state = quat_to_rot_6d(T_left_ee_state.rotation().wxyz[None, :], scalar_first=True)
        T_right_ee_6d_state = quat_to_rot_6d(T_right_ee_state.rotation().wxyz[None, :], scalar_first=True)
        T_left_ee_pos_state = T_left_ee_state.wxyz_xyz[-3:].reshape(1, -1)
        T_right_ee_pos_state = T_right_ee_state.wxyz_xyz[-3:].reshape(1, -1)
        left_gripper_pos_state = joint_states[i, 6].reshape(-1, 1)
        right_gripper_pos_state = joint_states[i, 13].reshape(-1, 1)
        cartesian_states = np.concatenate(
            [
                T_left_ee_6d_state,
                T_left_ee_pos_state,
                left_gripper_pos_state,
                T_right_ee_6d_state,
                T_right_ee_pos_state,
                right_gripper_pos_state,
            ],
            axis=1,
        )[0]
        cartesian_states_list.append(cartesian_states)

        if full_joint_action is not None:
            T_left_ee_action, T_right_ee_action = robot.solve_fk_base(joint_actions[i, :6], joint_actions[i, 7:13])
            T_left_ee_6d_action = quat_to_rot_6d(T_left_ee_action.rotation().wxyz[None, :], scalar_first=True)
            T_right_ee_6d_action = quat_to_rot_6d(T_right_ee_action.rotation().wxyz[None, :], scalar_first=True)
            T_left_ee_pos_action = T_left_ee_action.wxyz_xyz[-3:].reshape(1, -1)
            T_right_ee_pos_action = T_right_ee_action.wxyz_xyz[-3:].reshape(1, -1)
            left_gripper_pos_action = joint_actions[i, 6].reshape(-1, 1)
            right_gripper_pos_action = joint_actions[i, 13].reshape(-1, 1)
            cartesian_actions = np.concatenate(
                [
                    T_left_ee_6d_action,
                    T_left_ee_pos_action,
                    left_gripper_pos_action,
                    T_right_ee_6d_action,
                    T_right_ee_pos_action,
                    right_gripper_pos_action,
                ],
                axis=1,
            )[0]
            cartesian_actions_list.append(cartesian_actions)
        else:
            cartesian_actions_list.append(cartesian_states)

    cartesian_states = np.stack(cartesian_states_list, axis=0)
    cartesian_actions = np.stack(cartesian_actions_list, axis=0)

    return cartesian_states, cartesian_actions


def calculate_actions_delta_cartesian(full_joint_state: np.ndarray, seq_length: int, robot: Any):
    from openpi.utils.matrix_utils import quat_to_rot_6d

    joint_states = full_joint_state[:seq_length]

    # joint states ordering: left_joint_pos, left_gripper_pos, right_joint_pos, right_gripper_pos
    # T_left_ee, T_right_ee = robot.solve_fk_base(joint_states[:, :6], joint_states[:, 7:13]) # BUGGY, INTRODUCES A POSE DELAY
    cartesian_states_list = []
    cartesian_actions_list = []

    for i in range(
        seq_length
    ):  # For some reason running batched FK is buggy -- camera lags behind pose so delay is introduced somewhere somehow
        T_left_ee, T_right_ee = robot.solve_fk_base(joint_states[i, :6], joint_states[i, 7:13])
        T_left_ee_6d = quat_to_rot_6d(T_left_ee.rotation().wxyz[None, :], scalar_first=True)
        T_right_ee_6d = quat_to_rot_6d(T_right_ee.rotation().wxyz[None, :], scalar_first=True)
        T_left_ee_pos = T_left_ee.wxyz_xyz[-3:].reshape(1, -1)
        T_right_ee_pos = T_right_ee.wxyz_xyz[-3:].reshape(1, -1)
        left_gripper_pos = joint_states[i, 6].reshape(-1, 1)
        right_gripper_pos = joint_states[i, 13].reshape(-1, 1)
        cartesian_states = np.concatenate(
            [T_left_ee_6d, T_left_ee_pos, left_gripper_pos, T_right_ee_6d, T_right_ee_pos, right_gripper_pos], axis=1
        )[0]
        cartesian_states_list.append(cartesian_states)
    cartesian_abs_states = np.stack(cartesian_states_list, axis=0)

    for i in range(seq_length):
        cartesian_actions_t = cartesian_abs_states[i + 1] - cartesian_abs_states[i]
        # keep grippers absolute
        cartesian_actions_t[6] = cartesian_abs_states[i + 1, 6]
        cartesian_actions_t[13] = cartesian_abs_states[i + 1, 13]

        assert len(cartesian_actions_t.shape) == 1
        cartesian_actions_list.append(cartesian_actions_t)

    cartesian_delta_actions = np.stack(cartesian_actions_list, axis=0)
    return cartesian_abs_states, cartesian_delta_actions


def create_frame_data(
    joint_states: np.ndarray,
    joint_actions: np.ndarray,
    processed_images: dict,
    seq_length: int,
    skip_videos: bool = False,
) -> list[dict]:
    """Create frame data for LeRobot dataset."""
    frames_data = []

    for step in range(seq_length):
        # Use pre-calculated values (no computation in loop)
        frame_data = {
            "joint_positions": joint_states[step],
            "actions": joint_actions[step],
        }

        # Add camera images (direct indexing, no copies) - only if not skipping videos
        if not skip_videos:
            for camera_key in CAMERA_KEYS:
                if camera_key in processed_images and camera_key in CAMERA_KEY_MAPPING:
                    lerobot_key = CAMERA_KEY_MAPPING[camera_key]
                    frame_data[lerobot_key] = processed_images[camera_key][step]

        frames_data.append(frame_data)

    return frames_data
