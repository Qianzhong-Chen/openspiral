"""
Offline residual RL training script for DICE-RL style finetuning of pi0 policies.

Usage:
    python scripts/train_residual_rl.py \
        --pi0-config-name pi05_base \
        --pi0-checkpoint /path/to/pi0/checkpoint/params \
        --batch-size 32 \
        --num-train-steps 100000

The script:
1. Loads a frozen pi0 model from checkpoint
2. Initializes a lightweight residual actor + critic ensemble
3. Trains via offline actor-critic with BC regularization on a LeRobot dataset
"""

import dataclasses
import functools
import logging
import pathlib
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax import struct
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.models.residual_rl_model as rl_model
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.training.rl_data_loader as rl_data_loader


# Config class is defined in openpi.training.config (ResidualRLTrainConfig).
# Use named presets via rl_cli() or get_rl_config(), with CLI overrides.
ResidualRLTrainConfig = _config._RL_CONFIG_CLASS


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------

@at.typecheck
@struct.dataclass
class RLTrainState:
    """Train state for residual RL (actor + critic)."""
    step: at.Int[at.ArrayLike, ""]
    # Actor
    actor_params: nnx.State
    actor_def: nnx.GraphDef[rl_model.ResidualActor]
    actor_opt_state: optax.OptState
    # Critic
    critic_params: nnx.State
    critic_def: nnx.GraphDef[rl_model.CriticEnsemble]
    critic_opt_state: optax.OptState
    # Target critic (Polyak averaged)
    target_critic_params: nnx.State


# ---------------------------------------------------------------------------
# Pi0 precomputation (runs in its own JIT, separate from RL update)
# ---------------------------------------------------------------------------

def precompute_pi0_outputs(
    extract_features_fn,
    sample_actions_fn,
    rng: at.KeyArrayLike,
    obs: _model.Observation,
    next_obs: _model.Observation,
    config_rl: rl_model.ResidualRLConfig,
):
    """Pure-Python orchestrator over pre-JITted pi0 primitives.

    IMPORTANT: This function is NOT wrapped in jax.jit. A single outer jit that
    inlined all four pi0 calls produced an ~8.4 GB compiled artifact and a
    >120 GB host-RAM compile peak. Two additional mitigations are applied here:

    1. extract_features and sample_actions are compiled as *independent* jits.
    2. sample_actions is called *only at batch B* (never at B*K) — the K actor
       rollouts are produced as K sequential micro-batches of size B. This
       means there is only ONE sample_actions compile instead of two, and the
       compiled artifact is ~K× smaller than compiling at B*K directly.

    Args:
        extract_features_fn: jitted (obs) -> features.
        sample_actions_fn:   jitted (rng, obs, noise) -> actions. Only ever
            invoked at batch B so only one shape specialization is compiled.
    """
    K = config_rl.num_noise_samples
    critic_rng, actor_rng = jax.random.split(rng)

    # Feature extraction
    features = extract_features_fn(obs)
    next_features = extract_features_fn(next_obs)
    B = features.shape[0]

    # --- For critic: base actions at next state (batch B) ---
    noise_rng, action_rng = jax.random.split(critic_rng)
    next_noise = jax.random.normal(noise_rng, (B, config_rl.action_horizon, config_rl.action_dim))
    next_base_actions = sample_actions_fn(action_rng, next_obs, next_noise)

    # --- For actor: K base actions at current state ---
    # Micro-batch: run sample_actions K times at batch B, then stack. This keeps
    # the sample_actions compile at a single batch size (B, not B*K).
    noise_rng_k, actor_rng = jax.random.split(actor_rng)
    noise_k = jax.random.normal(noise_rng_k, (K, B, config_rl.action_horizon, config_rl.action_dim))
    actor_rngs = jax.random.split(actor_rng, K)

    base_actions_micro = []
    for k in range(K):
        base_actions_micro.append(sample_actions_fn(actor_rngs[k], obs, noise_k[k]))
    # Stack -> (K, B, ah, ad) then reshape to (B*K, ah, ad) in a way that
    # matches the prior layout: element at index (b*K + k) corresponds to
    # batch b, noise sample k. noise_k was (K, B, ...) so transpose first.
    base_actions_stacked = jnp.stack(base_actions_micro, axis=0)           # (K, B, ah, ad)
    base_actions_k = jnp.transpose(base_actions_stacked, (1, 0, 2, 3))     # (B, K, ah, ad)
    base_actions_k = base_actions_k.reshape(B * K, config_rl.action_horizon, config_rl.action_dim)

    # Noise layout must match actions: (B, K, ...) then flatten to (B*K, ...).
    noise_k_bk = jnp.transpose(noise_k, (1, 0, 2, 3))                      # (B, K, ah, ad)
    noise_k_flat = noise_k_bk.reshape(B * K, config_rl.flat_action_dim)

    features_expanded = jnp.repeat(features[:, None, :], K, axis=1).reshape(B * K, -1)

    return (
        features, next_features,
        next_base_actions, next_noise,
        base_actions_k, noise_k_flat, features_expanded,
    )


# ---------------------------------------------------------------------------
# RL update step (lightweight — only actor/critic MLPs, no pi0)
# ---------------------------------------------------------------------------

def rl_update_step(
    config: ResidualRLTrainConfig,
    actor_tx: optax.GradientTransformation,
    critic_tx: optax.GradientTransformation,
    state: RLTrainState,
    # Batch data
    actions: at.Float[at.Array, "b ah ad"],
    rewards: at.Float[at.Array, " b"],
    dones: at.Float[at.Array, " b"],
    mc_returns: at.Float[at.Array, " b"],
    # Precomputed pi0 outputs
    features: at.Float[at.Array, "b f"],
    next_features: at.Float[at.Array, "b f"],
    next_base_actions: at.Float[at.Array, "b ah ad"],
    next_noise: at.Float[at.Array, "b ah ad"],
    base_actions_k: at.Float[at.Array, "bk ah ad"],
    noise_k_flat: at.Float[at.Array, "bk n"],
    features_expanded: at.Float[at.Array, "bk f"],
) -> tuple[RLTrainState, dict[str, at.Array]]:
    """Single RL update step — only lightweight actor/critic MLPs, no pi0."""
    # Reconstruct modules from state
    actor = nnx.merge(state.actor_def, state.actor_params)
    critic = nnx.merge(state.critic_def, state.critic_params)
    target_critic = nnx.merge(state.critic_def, state.target_critic_params)

    # --- Critic update ---
    def critic_loss_fn(critic_module):
        loss, info = rl_model.critic_loss(
            critic=critic_module,
            target_critic=target_critic,
            actor=actor,
            features=features,
            actions=actions,
            rewards=rewards,
            next_features=next_features,
            next_base_actions=next_base_actions,
            next_noise=next_noise,
            dones=dones,
            discount=config.rl.discount,
            action_horizon=config.rl.action_horizon,
            mc_returns=mc_returns,
            critic_target=config.rl.critic_target,
            mc_weight=config.rl.mc_weight,
        )
        return loss, info

    critic_loss_val, critic_grads, critic_info = _value_grad_and_info(critic_loss_fn, critic)
    critic_params = nnx.state(critic)
    critic_updates, new_critic_opt_state = critic_tx.update(critic_grads, state.critic_opt_state, critic_params)
    new_critic_params_raw = optax.apply_updates(critic_params, critic_updates)
    nnx.update(critic, new_critic_params_raw)
    new_critic_params = nnx.state(critic)

    # --- Actor update ---
    def actor_loss_fn(actor_module):
        loss, info = rl_model.actor_loss(
            actor=actor_module,
            critic=critic,
            features=features,
            base_actions_k=base_actions_k,
            noise_k_flat=noise_k_flat,
            features_expanded=features_expanded,
            config=config.rl,
            training_step=state.step,
        )
        return loss, info

    actor_loss_val, actor_grads, actor_info = _value_grad_and_info(actor_loss_fn, actor)
    actor_params = nnx.state(actor)
    actor_updates, new_actor_opt_state = actor_tx.update(actor_grads, state.actor_opt_state, actor_params)
    new_actor_params_raw = optax.apply_updates(actor_params, actor_updates)
    nnx.update(actor, new_actor_params_raw)
    new_actor_params = nnx.state(actor)

    # --- Target critic Polyak update ---
    new_target_critic_params = rl_model.polyak_update(
        state.target_critic_params, new_critic_params, config.rl.target_tau
    )

    new_state = RLTrainState(
        step=state.step + 1,
        actor_params=new_actor_params,
        actor_def=state.actor_def,
        actor_opt_state=new_actor_opt_state,
        critic_params=new_critic_params,
        critic_def=state.critic_def,
        critic_opt_state=new_critic_opt_state,
        target_critic_params=new_target_critic_params,
    )

    info = {
        **{f"critic/{k}": v for k, v in critic_info.items()},
        **{f"actor/{k}": v for k, v in actor_info.items()},
        "critic_grad_norm": optax.global_norm(critic_grads),
        "actor_grad_norm": optax.global_norm(actor_grads),
    }
    return new_state, info


def _value_grad_and_info(loss_fn, module):
    """Compute loss, gradients, and info dict for an NNX module."""
    (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(module)
    return loss, grads, info


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def load_frozen_pi0(config: ResidualRLTrainConfig) -> _model.BaseModel:
    """Load and freeze the pi0 model from checkpoint."""
    import openpi.shared.download as _download

    # Get the train config to access model config
    train_config = _config.get_config(config.pi0_config_name)
    model_config = train_config.model

    # Load parameters — maybe_download handles GCS paths (gs://...) transparently
    params_path = _download.maybe_download(config.pi0_checkpoint)
    logging.info(f"Loading pi0 params from: {params_path}")
    params = _model.restore_params(params_path, dtype=jnp.bfloat16)

    # Create model with loaded params
    pi0_model = model_config.load(params)
    pi0_model.eval()

    logging.info("Pi0 model loaded and frozen successfully")
    return pi0_model


def init_rl_state(
    config: ResidualRLTrainConfig,
    rng: at.KeyArrayLike,
    actor_tx: optax.GradientTransformation,
    critic_tx: optax.GradientTransformation,
) -> RLTrainState:
    """Initialize the RL train state (actor, critic, target critic)."""
    actor_rng, critic_rng = jax.random.split(rng)

    # Create actor
    actor = rl_model.ResidualActor(config.rl, rngs=nnx.Rngs(actor_rng))
    actor_params = nnx.state(actor)
    actor_def = nnx.graphdef(actor)

    # Create critic
    critic = rl_model.CriticEnsemble(config.rl, rngs=nnx.Rngs(critic_rng))
    critic_params = nnx.state(critic)
    critic_def = nnx.graphdef(critic)

    # Target critic: copy of critic
    target_critic_params = jax.tree.map(lambda x: x.copy(), critic_params)

    return RLTrainState(
        step=0,
        actor_params=actor_params,
        actor_def=actor_def,
        actor_opt_state=actor_tx.init(actor_params),
        critic_params=critic_params,
        critic_def=critic_def,
        critic_opt_state=critic_tx.init(critic_params),
        target_critic_params=target_critic_params,
    )


def save_rl_checkpoint(state: RLTrainState, checkpoint_dir: pathlib.Path, step: int, config: ResidualRLTrainConfig):
    """Save RL train state checkpoint + config."""
    import json
    import orbax.checkpoint as ocp

    ckpt_dir = checkpoint_dir / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(
            ckpt_dir / "rl_state",
            {
                "actor_params": state.actor_params.to_pure_dict(),
                "critic_params": state.critic_params.to_pure_dict(),
                "target_critic_params": state.target_critic_params.to_pure_dict(),
                "step": state.step,
            },
        )

    # Save RL config so inference knows the network architecture
    config_dict = dataclasses.asdict(config.rl)
    # Convert tuples to lists for JSON serialization
    config_dict = {k: list(v) if isinstance(v, tuple) else v for k, v in config_dict.items()}
    with open(ckpt_dir / "rl_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logging.info(f"Saved RL checkpoint at step {step} to {ckpt_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: ResidualRLTrainConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"Config: {config}")

    if not config.pi0_checkpoint:
        raise ValueError("pi0_checkpoint must be specified")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # Initialize wandb
    if config.wandb_enabled:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
    else:
        wandb.init(mode="disabled")

    # Load frozen pi0
    pi0_model = load_frozen_pi0(config)

    # Create data loader
    train_config = _config.get_config(config.pi0_config_name)
    # Override batch size
    train_config = dataclasses.replace(train_config, batch_size=config.batch_size)

    data_loader = rl_data_loader.create_rl_data_loader(
        train_config,
        discount=config.rl.discount,
        shuffle=True,
        use_progress_mc_return=config.use_progress_mc_return,
    )
    data_iter = iter(data_loader)
    logging.info("Data loader initialized")

    # Create optimizers
    actor_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.actor_lr, weight_decay=config.weight_decay),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.critic_lr, weight_decay=config.weight_decay),
    )

    # Initialize RL state
    rl_state = init_rl_state(config, init_rng, actor_tx, critic_tx)
    logging.info("RL state initialized")
    logging.info(f"  Actor params: {jax.tree.map(lambda x: x.shape, rl_state.actor_params)}")
    logging.info(f"  Critic params: {jax.tree.map(lambda x: x.shape, rl_state.critic_params)}")

    # Split pi0 operations into TWO independent JITs so each compile is small.
  
    extract_features_jit = jax.jit(
        functools.partial(rl_model.extract_features, pi0_model),
    )
    sample_actions_jit = jax.jit(
        lambda rng, obs, noise: pi0_model.sample_actions(rng, obs, noise=noise),
    )
    # precompute_pi0_outputs is a pure-Python orchestrator (NOT jit-wrapped).
    pi0_forward = functools.partial(
        precompute_pi0_outputs,
        extract_features_jit,
        sample_actions_jit,
        config_rl=config.rl,
    )

    # JIT the RL update (small — only actor/critic MLPs)
    prl_update = jax.jit(
        functools.partial(rl_update_step, config, actor_tx, critic_tx),
    )

    # Checkpoint directory: checkpoint_dir / config_name / exp_name
    checkpoint_dir = pathlib.Path(config.checkpoint_dir) / config.name / config.exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    pbar = tqdm.tqdm(range(config.num_train_steps), dynamic_ncols=True)
    infos = []

    for step in pbar:
        batch = next(data_iter)
        obs, actions, next_obs, rewards, dones, mc_returns = batch

        # Step 1: precompute pi0 outputs (separate JIT, no grads through pi0)
        step_rng = jax.random.fold_in(train_rng, step)
        pi0_outputs = pi0_forward(step_rng, obs, next_obs)
        (features, next_features,
         next_base_actions, next_noise,
         base_actions_k, noise_k_flat, features_expanded) = pi0_outputs

        # Step 2: RL update (lightweight JIT — only actor/critic)
        rl_state, info = prl_update(
            rl_state,
            actions, rewards, dones, mc_returns,
            features, next_features,
            next_base_actions, next_noise,
            base_actions_k, noise_k_flat, features_expanded,
        )
        infos.append(info)

        if step % config.log_interval == 0 and step > 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if (step % config.save_interval == 0 and step > 0) or step == config.num_train_steps - 1:
            save_rl_checkpoint(rl_state, checkpoint_dir, step, config)

    logging.info("Training complete!")


if __name__ == "__main__":
    config = _config.rl_cli()
    main(config)
