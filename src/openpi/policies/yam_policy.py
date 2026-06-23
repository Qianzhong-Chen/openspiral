import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_yam_example() -> dict:
    """Creates a random input example for the YAM policy."""
    return {
        "state": np.ones((14,)),
        "left_camera-images-rgb": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "right_camera-images-rgb": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "top_camera-images-rgb": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Inputs for the YAM policy.

    Expected inputs:
    - left_camera-images-rgb: [channel, height, width] - Left camera
    - right_camera-images-rgb: [channel, height, width] - Right camera
    - top_camera-images-rgb: [channel, height, width] - Top camera
    - state:
    [14] - Joint positions for both arms (6 joints + 1 gripper per arm)
    or
    [20] - Cartesian positions for both arms (6d rot + 3d pos + 1 gripper pos per arm)
    - actions:
    [action_horizon, 14] - Joint actions for both arms
    or
    [action_horizon, 20] - Cartesian actions for both arms
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # The expected camera names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "left_camera-images-rgb",
        "right_camera-images-rgb",
        "top_camera-images-rgb",
    )

    def __call__(self, data: dict) -> dict:
        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        # print(data.keys())
        base_image = _parse_image(data["top_camera-images-rgb"])
        left_wrist_image = _parse_image(data["left_camera-images-rgb"])
        right_wrist_image = _parse_image(data["right_camera-images-rgb"])

        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_wrist_image,
            "right_wrist_0_rgb": right_wrist_image,
        }

        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # YAM uses absolute joint actions - no conversion needed
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Outputs for the YAM policy."""

    robot_action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims (YAM joint space).
        actions = np.asarray(data["actions"][:, : self.robot_action_dim])  # TEMPORARY HALF THE ACTION HORIZON
        # YAM uses absolute joint actions - no conversion needed
        return {"actions": actions}
