import torch
from typing import Callable
from pathlib import Path
from .lerobot_dataset import LeRobotDataset
import time
from typing import Tuple
from faker import Faker



class FrameGapLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        episodes: list[int] | None = None,
        n_obs_steps: int = 1,
        frame_gap: int = 1,
        horizon: int = 1,
        max_rewind_steps: int = 0,
        root: str | Path | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        image_names: list[str] = ["top_camera-images-rgb"],
        dense_annotation: bool = False,
        video_eval: bool = False,
        annotation_list: list[str] | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )

        self.n_obs_steps = n_obs_steps
        self.frame_gap = frame_gap
        self.horizon = horizon
        self.max_rewind_steps = max_rewind_steps
        self.timestamp_tensor = torch.tensor(self.hf_dataset["timestamp"]).flatten()
        assert all(img_name in self.meta.video_keys for img_name in image_names), f"Image names {image_names} not found in metadata video keys."
        self.wrapped_video_keys = image_names  # Use only the specified camera for videos
        self.verbs = ['move', 'grasp', 'rotate', 'push', 'pull', 'slide', 'lift', 'place']
        self.fake = Faker()
        self.dense_annotation = dense_annotation
        self.video_eval = video_eval
        self.annotation_list = annotation_list
        self.required_history = self.n_obs_steps * self.frame_gap

    # # add fixed ep start to sequence
    # def __getitem__(self, idx: int) -> dict:
    #     item = self.hf_dataset[idx]
    #     ep_idx = item["episode_index"].item()
    #     if self.episodes is not None:
    #         assert ep_idx in self.episodes, f"Episode {ep_idx} not found in the selected episodes."
    #         global_idx = self.episodes.index(ep_idx)
    #     else:
    #         global_idx = ep_idx

    #     ep_start = self.episode_data_index["from"][global_idx].item()
    #     ep_end = self.episode_data_index["to"][global_idx].item()

    #     # Adjust idx if there's not enough history
    #     required_history = self.n_obs_steps * self.frame_gap
    #     if idx - required_history < ep_start:
    #         idx = ep_start + required_history

    #     # Compute frame indices for observation
    #     obs_indices = [ep_start] + [idx - i * self.frame_gap for i in reversed(range(self.n_obs_steps))]
        
    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        

        # Map to global episode index (unchanged logic)
        if self.episodes is not None:
            assert ep_idx in self.episodes, f"Episode {ep_idx} not found in the selected episodes."
            global_idx = self.episodes.index(ep_idx)
        else:
            global_idx = ep_idx

        ep_start = self.episode_data_index["from"][global_idx].item()
        ep_end = self.episode_data_index["to"][global_idx].item()

        # ---- Key change: keep idx as the last frame, and back-fill with ep-aware padding ----
        # Build [n_obs_steps + 1] indices with last == idx, stepping backwards by frame_gap,
        # but never going earlier than ep_start.
        obs_indices = [ep_start] * (self.n_obs_steps + 1)
        obs_indices[-1] = idx
        for i in range(self.n_obs_steps - 1, -1, -1):
            # the frame BEFORE obs_indices[i+1]
            prev_candidate = obs_indices[i + 1] - self.frame_gap
            obs_indices[i] = max(ep_start, prev_candidate)

        # print(f"start: {ep_start}, idx: {idx}, end: {ep_end}")
        # print(obs_indices)
        sequence = self.hf_dataset.select(obs_indices)

        progress_list = torch.zeros(len(obs_indices), dtype=torch.float32)
        # Extract sequence data
        seq_item = {}
        for key in sequence.features:
            value = sequence[key]
            if key == "actions":
                seq_item[key] = torch.stack(value)
            elif key == "state":
                seq_item[key] = torch.stack(value)
            elif key == "reward":
                progress_list = torch.stack(value).squeeze(-1)
            elif key == "index":
                seq_item["local_index"] = value[-1].item()
            else:
                seq_item[key] = value[0]
            del value
        seq_item["index"] = obs_indices[-1]
        del sequence

        # Query video frames
        obs_ts_range = self.timestamp_tensor[obs_indices].tolist()
        query_ts_dict = {key: obs_ts_range for key in self.wrapped_video_keys}
        video_frames = self._query_videos(query_ts_dict, ep_idx)
        
        if not self.video_eval:
            rewind_flag = torch.rand(1).item() < 0.8 and idx > ep_start + self.required_history
        else:
            rewind_flag = False
        rewind_step = None
        for key in self.wrapped_video_keys:
            frames = video_frames[key]
            if frames.shape[0] < self.n_obs_steps:
                pad_count = self.n_obs_steps - frames.shape[0]
                pad_frame = frames[-1:].repeat(pad_count, 1, 1, 1)
                frames = torch.cat([frames, pad_frame], dim=0)

            if rewind_flag:
                rewind_step, rewind_frames = self._get_rewind(
                    idx, key, ep_idx, rewind_step=rewind_step
                )
                frames = torch.cat([frames, rewind_frames], dim=0)
            else:
                rewind_step = 0
                padding_frames = torch.zeros((self.max_rewind_steps, *frames.shape[1:]), dtype=frames.dtype)
                frames = torch.cat([frames, padding_frames], dim=0)

            seq_item[key] = frames

        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                if cam in seq_item:
                    seq_item[cam] = self.image_transforms(seq_item[cam])

        # Task string
        pertube_task_flag = torch.rand(1).item() < 0.2
        if self.video_eval:
            pertube_task_flag = False
        if pertube_task_flag:
            num_words = torch.randint(1, 6, (1,)).item()
            verb = self.verbs[torch.randint(0, len(self.verbs), (1,)).item()]
            phrase = [verb] + self.fake.words(nb=num_words)
            seq_item["task"] = " ".join(phrase)
        else:
            seq_item["task"] = "fold the tshirt"

        # Progress targets
        seq_item["targets"] = torch.zeros(1 + self.n_obs_steps + self.max_rewind_steps, dtype=torch.float32)
        state_with_rewind = torch.zeros([1 + self.n_obs_steps + self.max_rewind_steps, seq_item["state"].shape[-1]], dtype=torch.float32)
        state_with_rewind[:self.n_obs_steps + 1, :] = seq_item["state"]
        frame_relative_indices = torch.zeros(1 + self.n_obs_steps + self.max_rewind_steps, dtype=torch.float32)

        if not pertube_task_flag:
            seq_item["targets"][:self.n_obs_steps + 1] = progress_list
            for i in range(rewind_step):
                seq_item["targets"][1 + self.n_obs_steps + i] = torch.flip(progress_list, dims=[0])[i + 1]
        
        for i, idx in enumerate(obs_indices):
            frame_relative_indices[i] = (idx - ep_start) / (ep_end - ep_start) if ep_end > ep_start else 0.0
        
        for i in range(rewind_step):
            frame_relative_indices[1 + self.n_obs_steps + i] = torch.flip(frame_relative_indices[:self.n_obs_steps + 1], dims=[0])[i + 1]
            state_with_rewind[1 + self.n_obs_steps + i, :] = torch.flip(seq_item["state"], dims=[0])[i + 1]
        
        seq_item["state"] = state_with_rewind
        seq_item["lengths"] = torch.tensor(1 + self.n_obs_steps + rewind_step, dtype=torch.int32)
        seq_item["frame_relative_indices"] = frame_relative_indices

        if self.dense_annotation:
            if pertube_task_flag:
                seq_item["task"] = [seq_item["task"]] * (1 + self.n_obs_steps + self.max_rewind_steps)
            else:
                seq_item["task"] = [''] * (1 + self.n_obs_steps + self.max_rewind_steps)
                # Two-staged task annotation
                for i in range(0, 1 + self.n_obs_steps + self.max_rewind_steps):
                    stage_idx =  int(torch.floor(seq_item["targets"][i]).item())
                    stage_idx = min(stage_idx, len(self.annotation_list) - 1)
                    seq_item["task"][i] = self.annotation_list[stage_idx]

        del item, video_frames, query_ts_dict, obs_ts_range, progress_list, state_with_rewind, frame_relative_indices

        return seq_item


    def _get_rewind(self, idx: int, key: str, ep_idx: int, rewind_step=None) -> Tuple[int, torch.Tensor]:
        assert self.max_rewind_steps < self.n_obs_steps, "Max rewind steps must be less than n_obs_steps."

        max_valid_step = (idx - self.frame_gap) // self.frame_gap
        max_rewind = min(self.max_rewind_steps, max_valid_step)

        if rewind_step is None:
            rewind_step = torch.randint(1, max_rewind + 1, (1,)).item()

        rewind_indices = list(range(idx - rewind_step * self.frame_gap, idx, self.frame_gap))
        if len(rewind_indices) < rewind_step:
            pad_count = rewind_step - len(rewind_indices)
            rewind_indices += [rewind_indices[-1]] * pad_count

        rewind_ts_range = self.timestamp_tensor[rewind_indices].tolist()
        query_ts_dict = {key: rewind_ts_range}
        rewind_frames = self._query_videos(query_ts_dict, ep_idx)[key]

        if rewind_frames.ndim == 3:
            rewind_frames = rewind_frames.unsqueeze(0)

        rewind_frames = torch.flip(rewind_frames, dims=[0])
        padding_needed = self.max_rewind_steps - rewind_step
        if padding_needed > 0:
            pad = torch.zeros((padding_needed, *rewind_frames.shape[1:]), dtype=rewind_frames.dtype)
            rewind_frames = torch.cat([rewind_frames, pad], dim=0)

        return rewind_step, rewind_frames
