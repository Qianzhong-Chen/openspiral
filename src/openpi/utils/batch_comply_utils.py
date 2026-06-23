import numpy as np
import jax.numpy as jnp
from typing import Any, Dict, List, Mapping

def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()  # torch -> numpy
    return np.asarray(x)

def _ensure_btchw(x: np.ndarray) -> np.ndarray:
    # Accept [B,T,H,W,C] or [B,H,W,C]; return [B,T,C,H,W], float32 in [0,1]
    # if x.ndim == 5:      # [B,T,H,W,C] -> [B,T,C,H,W]
    #     x = np.transpose(x, (0, 1, 4, 2, 3))
    # elif x.ndim == 4:    # [B,H,W,C] -> [B,1,C,H,W]
    #     x = np.transpose(x, (0, 3, 1, 2))
    #     x = x[:, None, ...]
    # else:
    #     raise ValueError(f"Unexpected image shape {x.shape}")
    x = x.astype(np.float32)
    if x.max() > 1.0 or x.dtype == np.uint8:
        # in case it’s uint8 [0,255] or >1.0, map to [0,1]
        x = x / 255.0 if x.dtype == np.uint8 else np.clip(x, 0.0, 1.0)
    return x

def _ensure_bta(x: np.ndarray) -> np.ndarray:  # actions -> [B,T,A]
    if x.ndim == 2: x = x[:, None, :]
    if x.ndim != 3: raise ValueError(f"actions need [B,T,A] or [B,A], got {x.shape}")
    return x.astype(np.float32)

def _ensure_btd(x: np.ndarray) -> np.ndarray:  # state -> [B,T,D]
    if x.ndim == 2: x = x[:, None, :]
    if x.ndim != 3: raise ValueError(f"state need [B,T,D] or [B,D], got {x.shape}")
    return x.astype(np.float32)

def _ensure_b_mask(x: np.ndarray, T: int) -> np.ndarray:  # masks -> [B]
    if x.ndim == 1:
        m = x
    elif x.ndim == 2:
        m = x[:, -1] if x.shape[1] > 1 else x[:, 0]
    elif x.ndim == 3:
        if x.shape[2] != 1: raise ValueError(f"Unexpected mask shape {x.shape}")
        m = x[:, -1, 0]
    else:
        raise ValueError(f"Unexpected mask shape {x.shape}")
    return m.astype(np.float32)

def comply_lerobot_batch_jax(
    batch: Mapping[str, Any],
    camera_names: List[str] = ("left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"),
    *,
    fallback_state_dim: int = 20,
) -> Dict[str, Any]:
    # actions
    actions = _ensure_bta(_to_numpy(batch["actions"]))

    # state (optional)
    if "state" in batch:
        state = _ensure_btd(_to_numpy(batch["state"]))
    else:
        B, T, _ = actions.shape
        state = np.zeros((B, T, fallback_state_dim), dtype=np.float32)

    # masks (optional)
    masks_np = _to_numpy(batch["mask"]) if "mask" in batch else np.ones((actions.shape[0],), dtype=np.float32)
    masks = _ensure_b_mask(masks_np, T=actions.shape[1])

    # images per camera (put directly under obs; DO NOT include nested dict in jnp.asarray)
    obs = {"state": state}
    for cam in camera_names:
        if cam not in batch:
            raise KeyError(f"Camera '{cam}' missing in batch keys: {list(batch.keys())}")
        obs[cam] = _ensure_btchw(_to_numpy(batch[cam]))

    # Cast ONLY array leaves to jax arrays (exclude any nested dicts)
    obs_jax = {k: jnp.asarray(v) for k, v in obs.items()}  # no 'image_frames' here
    return {
        "obs": obs_jax,
        "action": jnp.asarray(actions),
        "masks": jnp.asarray(masks),
    }
