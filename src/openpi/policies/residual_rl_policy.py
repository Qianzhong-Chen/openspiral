"""
Policy wrapper for residual RL inference.

Loads a frozen pi0 + trained residual actor. At inference:
  action = pi0(obs, z) + actor(features, z)
Single sample, no critic or best-of-N needed.
"""

import logging
import pathlib
import time
from collections.abc import Sequence
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

import openpi.models.model as _model
import openpi.models.residual_rl_model as rl_model
from openpi_client import base_policy as _base_policy
import openpi.shared.array_typing as at
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.transforms as _transforms


class _FusedInferenceModule(nnx.Module):
    """Thin wrapper holding pi0 + actor so we can module_jit a single fused method."""

    def __init__(self, pi0_model: _model.BaseModel, actor: rl_model.ResidualActor, rl_config: rl_model.ResidualRLConfig):
        self.pi0_model = pi0_model
        self.actor = actor
        self.rl_config = rl_config

    def infer_actions(self, rng, observation):
        noise_rng, action_rng = jax.random.split(rng)
        B = observation.state.shape[0]
        config = self.rl_config

        noise = jax.random.normal(noise_rng, (B, config.action_horizon, config.action_dim))
        base_actions = self.pi0_model.sample_actions(action_rng, observation, noise=noise)
        features = rl_model.extract_features(self.pi0_model, observation)
        noise_flat = noise.reshape(B, config.flat_action_dim)
        residual = self.actor(features, noise_flat)
        return base_actions, residual


class ResidualRLPolicy(_base_policy.BasePolicy):
    """Policy: action = pi0(obs, z) + actor(features, z).

    Single noise sample per step. No critic needed at inference.
    """

    def __init__(
        self,
        pi0_model: _model.BaseModel,
        actor: rl_model.ResidualActor,
        rl_config: rl_model.ResidualRLConfig,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        metadata: dict[str, Any] | None = None,
        norm_stats: dict | None = None,
        dry_run: bool = False,
    ):
        self._pi0_model = pi0_model
        self._actor = actor
        self._rl_config = rl_config
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._metadata = metadata or {}
        self._dry_run = dry_run

        # Extract action std for dry-run residual scaling
        if dry_run and norm_stats is not None and "actions" in norm_stats:
            self._action_std = np.array(norm_stats["actions"].std) + 1e-6
        else:
            self._action_std = np.ones(1)

        if dry_run:
            logging.info("DRY-RUN MODE: outputting pi0 base actions only, printing unnormalized residuals")

        # Fuse pi0 + actor into a single JIT-compiled function using module_jit.
        # module_jit freezes NNX state via nnx.split, avoiding the massive memory
        # overhead that plain jax.jit causes on NNX module methods.
        fused = _FusedInferenceModule(pi0_model, actor, rl_config)
        self._infer_fn = nnx_utils.module_jit(fused.infer_actions)

        # In dry-run mode, also compile pi0 standalone (same path as regular Policy)
        # for A/B comparison.
        if dry_run:
            self._pi0_sample_actions = nnx_utils.module_jit(pi0_model.sample_actions)

        # Warm up JIT (takes a few minutes on first run, then cached)
        logging.info("Warming up JIT compilation (this is a one-time cost)...")
        dummy_obs = _make_dummy_obs(pi0_model, rl_config)
        result = self._infer_fn(jax.random.key(42), dummy_obs)
        jax.block_until_ready(result)
        if dry_run:
            result2 = self._pi0_sample_actions(jax.random.key(42), dummy_obs)
            jax.block_until_ready(result2)
        logging.info("JIT warmup complete")

    def infer(self, obs: dict) -> dict:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        observation = _model.Observation.from_dict(inputs)
        if self._dry_run:
            # Use standalone pi0 path (identical to regular Policy) for clean baseline
            actions = self._pi0_sample_actions(sample_rng, observation)
            # Also run fused to get residual for logging
            _, residual = self._infer_fn(sample_rng, observation)
        else:
            base_actions, residual = self._infer_fn(sample_rng, observation)
            actions = base_actions + residual

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)

        if self._dry_run:
            # Scale residual by std only (no mean offset) to see actual correction magnitude
            r = np.asarray(residual[0, ...])  # (ah, ad), normalized space
            r_scaled = r * self._action_std
            logging.info(
                f"[DRY-RUN] residual (scaled by std) mean={r_scaled.mean():.4f} std={r_scaled.std():.4f} "
                f"min={r_scaled.min():.4f} max={r_scaled.max():.4f} shape={r_scaled.shape}"
            )

        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def _make_dummy_obs(pi0_model, rl_config):
    """Create a dummy observation for JIT warmup."""
    from openpi.models.model import IMAGE_RESOLUTION, Observation
    dummy_img = jnp.zeros((1, *IMAGE_RESOLUTION, 3), dtype=jnp.float32)
    dummy_mask = jnp.ones((1,), dtype=jnp.bool_)
    return Observation(
        images={"base_0_rgb": dummy_img, "left_wrist_0_rgb": dummy_img, "right_wrist_0_rgb": dummy_img},
        image_masks={"base_0_rgb": dummy_mask, "left_wrist_0_rgb": dummy_mask, "right_wrist_0_rgb": dummy_mask},
        state=jnp.zeros((1, rl_config.state_dim), dtype=jnp.float32),
        tokenized_prompt=jnp.zeros((1, pi0_model.max_token_len), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.zeros((1, pi0_model.max_token_len), dtype=jnp.bool_),
    )


def load_residual_rl_policy(
    pi0_config_name: str,
    rl_checkpoint_dir: str,
    pi0_checkpoint: str | None = None,
    rl_config: rl_model.ResidualRLConfig | None = None,
    *,
    default_prompt: str | None = None,
    dry_run: bool = False,
) -> ResidualRLPolicy:
    """Load a trained residual RL policy for inference.

    Args:
        pi0_config_name: Name of the pi0 TrainConfig (e.g., "residual_rl_pi05_yam").
        rl_checkpoint_dir: Path to RL checkpoint dir (contains rl_state/).
        pi0_checkpoint: Path to frozen pi0 checkpoint. If None, reads from the
            weight_loader in the pi0 TrainConfig.
        rl_config: ResidualRLConfig. If None, uses defaults.
        default_prompt: Default language prompt for the task.
    """
    import json

    rl_checkpoint_dir = pathlib.Path(rl_checkpoint_dir)

    # Load RL config from checkpoint (saved during training), fall back to provided or default
    rl_config_path = rl_checkpoint_dir / "rl_config.json"
    if rl_config is None and rl_config_path.exists():
        with open(rl_config_path) as f:
            config_dict = json.load(f)
        # Convert lists back to tuples for tuple fields
        for key in ("actor_hidden_dims", "critic_hidden_dims"):
            if key in config_dict:
                config_dict[key] = tuple(config_dict[key])
        rl_config = rl_model.ResidualRLConfig(**config_dict)
        logging.info(f"Loaded RL config from checkpoint: {rl_config_path}")
    else:
        rl_config = rl_config or rl_model.ResidualRLConfig()
        logging.info("Using default RL config (no rl_config.json found in checkpoint)")

    # Load frozen pi0
    train_config = _config.get_config(pi0_config_name)
    model_config = train_config.model

    # Resolve pi0 checkpoint: use explicit path, or fall back to config's weight_loader
    if pi0_checkpoint is None:
        if hasattr(train_config.weight_loader, "params_path"):
            pi0_checkpoint = train_config.weight_loader.params_path
            logging.info(f"Using pi0 checkpoint from config: {pi0_checkpoint}")
        else:
            raise ValueError(
                "No pi0_checkpoint provided and config has no CheckpointWeightLoader. "
                "Pass --policy.pi0-dir explicitly."
            )

    pi0_path = _download.maybe_download(pi0_checkpoint)
    logging.info(f"Loading pi0 from: {pi0_path}")
    pi0_params = _model.restore_params(pi0_path, dtype=jnp.bfloat16)
    pi0_model = model_config.load(pi0_params)
    pi0_model.eval()

    # Initialize actor with correct architecture (will be overwritten by checkpoint)
    actor = rl_model.ResidualActor(rl_config, rngs=nnx.Rngs(jax.random.key(0)))

    # Restore trained actor params
    rl_checkpoint_dir = pathlib.Path(rl_checkpoint_dir)
    logging.info(f"Loading RL actor from: {rl_checkpoint_dir}")
    with ocp.PyTreeCheckpointer() as ckptr:
        restored = ckptr.restore(rl_checkpoint_dir / "rl_state")

    actor_graphdef, actor_state = nnx.split(actor)
    actor_state.replace_by_pure_dict(restored["actor_params"])
    actor = nnx.merge(actor_graphdef, actor_state)
    logging.info("RL actor loaded successfully")

    # Build transform pipeline
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    # Load norm_stats from pi0 checkpoint (same as create_trained_policy does),
    # falling back to config assets. Without correct norm_stats, inference produces garbage.
    norm_stats = None
    pi0_ckpt_path = pathlib.Path(pi0_path)
    # pi0_path points to .../params, go up one level to checkpoint root
    pi0_ckpt_root = pi0_ckpt_path.parent if pi0_ckpt_path.name == "params" else pi0_ckpt_path
    if data_config.asset_id and (pi0_ckpt_root / "assets").exists():
        try:
            norm_stats = _checkpoints.load_norm_stats(pi0_ckpt_root / "assets", data_config.asset_id)
            logging.info(f"Loaded norm_stats from pi0 checkpoint: {pi0_ckpt_root / 'assets'}")
        except FileNotFoundError:
            logging.info(f"Norm stats not found in pi0 checkpoint assets, will use config assets")
    if norm_stats is None:
        norm_stats = data_config.norm_stats
    if norm_stats is None:
        raise ValueError(
            "No norm_stats found! Check that either the pi0 checkpoint contains assets/ "
            "or the config's assets directory has norm_stats. "
            "Without norm_stats, pi0 receives unnormalized data and produces garbage actions."
        )

    # Match create_trained_policy: do NOT include repack_transforms (those map
    # dataset keys like "actions" which don't exist at inference time).
    return ResidualRLPolicy(
        pi0_model=pi0_model,
        actor=actor,
        rl_config=rl_config,
        transforms=[
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        norm_stats=norm_stats,
        dry_run=dry_run,
    )
