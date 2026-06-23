"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import os

# os.environ["LEROBOT_HOME"] = "/mnt/disks/ssd1/lerobot"
os.environ["LEROBOT_HOME"] = "/shared/projects/icrl/data/dpgs/lerobot"
import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client.image_tools import resize_with_pad
from PIL import Image
from tqdm import tqdm
import zarr

RAW_DATASET_FOLDERS = [
    # "/mnt/disks/ssd7/dpgs_dataset/yumi_coffee_maker/successes_041325"
    # "/mnt/disks/ssd7/dpgs_dataset/yumi_faucet/successes_041425",
    # "/mnt/disks/ssd7/dpgs_dataset/yumi_led_light/successes_041425_2334"
    # "/shared/projects/dpgs_dataset/yumi_bin_pickup/successes_041625_2054",
    # "/shared/projects/dpgs_dataset/yumi_drawer_open/successes_041725_2136/successes",
    # "/shared/projects/dpgs_dataset/yumi_faucet/successes_041425",
    # "/shared/projects/dpgs_dataset/yumi_pick_tiger_r2r2r/successes_042225_2121",
    # "/shared/projects/dpgs_dataset/yumi_pick_tiger_r2r2r/successes_042325_1714",
    # "/shared/projects/dpgs_dataset/yumi_pick_tiger_r2r2r/successes_042525_1750",
    "/shared/projects/dpgs_dataset/yumi_cardboard_lift/successes_042525_2016",
    # "/shared/projects/dpgs_dataset/yumi_pick_tiger_r2r2r/successes_041725_2203",
    # "/shared/projects/dpgs_dataset/yumi_led_light/successes_041825_1856",
    # "/shared/projects/dpgs_dataset/yumi_cardboard_lift/successes_041825_2245",
    # "/shared/projects/dpgs_dataset/yumi_faucet/successes_041425" # bajcsy
    # "/shared/projects/dpgs_dataset/yumi_drawer_open/successes_041525_2044" # bajcsy
]
LANGUAGE_INSTRUCTIONS = [
    # # "put the white cup on the coffee machine"
    # "open the drawer"
    # "pick up the tiger"
    # "turn off the faucet"
    # "turn the LED light"
    "pick up the cardboard box"
    # "pick up the bin"
]

# # REPO_NAME = "mlfu7/dpgs_sim_faucet_maker_5k_updated"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_faucet_5k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_drawer_open_1k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_led_5k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_bin_pickup_1k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_drawer_open_v3_1k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_tiger_1k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_tiger_1k_v4"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_tiger_1k_v5"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_tiger_2k_v6"  # use absolute gripper positions that is 0 (open), 1 (close)
# REPO_NAME = "mlfu7/dpgs_sim_tiger_1k_v7"  # use absolute gripper positions that is 0 (open), 1 (close)
# REPO_NAME = "mlfu7/dpgs_sim_led_v2_1k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_bimanual_lift_v2_1k"  # Name of the output dataset, also used for the Hugging Face Hub
REPO_NAME = "mlfu7/dpgs_bimanual_lift_v4_1k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_faucet_v1_1k"  # Name of the output dataset, also used for the Hugging Face Hub

CAMERA_KEYS = ["camera_0/rgb", "camera_1/rgb"]  # folder of rgb images
CAMERA_KEY_MAPPING = {
    "camera_0/rgb": "exterior_image_1_left",
    "camera_1/rgb": "exterior_image_2_left",
}
STATE_KEY = "robot_data/robot_data_joint.zarr"
# GRIPPER_KEY = "robot_data/robot_data_gripper_cmd.zarr"
RESIZE_SIZE = 224
UPTO_N_TRAJ = None
# UPTO_N_TRAJ = 1100


def main():
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    print("Dataset saved to ", output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=15,
        features={
            "exterior_image_1_left": {
                "dtype": "video",
                "shape": (RESIZE_SIZE, RESIZE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "video",
                "shape": (RESIZE_SIZE, RESIZE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["joint_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["actions"],
            },
        },
        image_writer_threads=20,
        image_writer_processes=10,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name, language_instruction in zip(RAW_DATASET_FOLDERS, LANGUAGE_INSTRUCTIONS):
        # get all the tasks that are collected that day
        data_day_dir = raw_dataset_name
        print("Processing folder: ", data_day_dir)
        trajs = os.listdir(data_day_dir)
        if UPTO_N_TRAJ is not None:
            trajs = trajs[:UPTO_N_TRAJ]
        for idx, task in tqdm(enumerate(trajs), desc="Processing trajectories"):
            print(f"Trajectory {idx}/{len(trajs)}: {task} is being processed")
            task_folder = f"{data_day_dir}/{task}"
            proprio_data = zarr.load(f"{task_folder}/{STATE_KEY}")
            # gripper_data = zarr.load(f"{task_folder}/{GRIPPER_KEY}")
            seq_length = (
                proprio_data.shape[0] - 1
            )  # remove the last proprio state since we need to calculate the action
            images = {
                key: [os.path.join(task_folder, key, i) for i in sorted(os.listdir(os.path.join(task_folder, key)))]
                for key in CAMERA_KEYS
            }
            images_per_step = [{key: images[key][i] for key in CAMERA_KEYS} for i in range(seq_length)]

            for step in range(seq_length):
                # load proprio data
                proprio_t = proprio_data[step]

                # # replace proprio_t last two dimensions
                # proprio_t[-2:] = gripper_data[step]

                # create delta action
                action_t = proprio_data[step + 1] - proprio_t

                # change the gripper to absolute position from t+1
                # action_t[-2:] = gripper_data[step + 1]
                action_t[-2:] = proprio_data[step + 1][-2:]

                # get the images for this step
                images_t = {
                    CAMERA_KEY_MAPPING[key]: resize_with_pad(
                        np.array(Image.open(images_per_step[step][key])), RESIZE_SIZE, RESIZE_SIZE
                    )
                    for key in CAMERA_KEYS
                }
                dataset.add_frame({"joint_position": proprio_t, "actions": action_t, **images_t})
            dataset.save_episode(task=language_instruction)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    print("Dataset saved to ", output_path)


if __name__ == "__main__":
    main()
