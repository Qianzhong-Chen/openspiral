"""
Residual RL model for DICE-RL style offline finetuning of pi0 flow-matching policies.

Components:
- extract_features: Extract frozen visual features from pi0's SigLIP encoder + state
- ResidualActor: Lightweight MLP that predicts action corrections on top of frozen base policy
- CriticMLP: Single Q-network MLP
- CriticEnsemble: Ensemble of Q-networks for stable value estimation
"""

import dataclasses
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.shared.array_typing as at


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ResidualRLConfig:
    """Configuration for the residual RL model."""

    # Dimensions (must match the frozen pi0 model)
    action_dim: int = 32
    action_horizon: int = 50

    # Feature extraction
    # SigLIP So400m/14 projects to paligemma width=2048, with 3 cameras + state
    siglip_feature_dim: int = 2048
    num_cameras: int = 3
    state_dim: int = 32  # same as action_dim in pi0

    # Actor
    actor_hidden_dims: Sequence[int] = (1024, 1024, 1024)
    actor_output_scale: float = 0.1  # initial residual magnitude

    # Critic
    num_critics: int = 5
    critic_hidden_dims: Sequence[int] = (1024, 1024, 1024)

    # RL hyperparameters
    discount: float = 0.99
    beta_bc: float = 50.0
    num_noise_samples: int = 4  # K latents per state for multi-sample training
    target_tau: float = 0.005
    n_step: int = 3
    q_filter_warmup: int = 999_999_999,  # filter effectively off
    critic_target: str = "mc"  # "td", "mc", or "blend"
    mc_weight: float = 1.0  # blending weight when critic_target="blend"

    @property
    def feature_dim(self) -> int:
        """Total feature dimension after pooling SigLIP + state."""
        return self.siglip_feature_dim * self.num_cameras + self.state_dim

    @property
    def flat_action_dim(self) -> int:
        """Flattened action chunk size."""
        return self.action_horizon * self.action_dim


# ---------------------------------------------------------------------------
# Feature Extraction (frozen, from pi0's SigLIP encoder)
# ---------------------------------------------------------------------------

def extract_features(
    pi0_model: _model.BaseModel,
    observation: _model.Observation,
) -> at.Float[at.Array, "b f"]:
    """Extract compact feature vector from pi0's frozen SigLIP vision encoder.

    For each camera image, runs SigLIP to get patch tokens (B, 196, 2048),
    then global-average-pools to (B, 2048). Concatenates all cameras + state.

    Args:
        pi0_model: Frozen pi0 model (only uses PaliGemma.img).
        observation: Preprocessed observation with images and state.

    Returns:
        features: (B, feature_dim) where feature_dim = num_cameras * 2048 + state_dim.
    """
    observation = _model.preprocess_observation(None, observation, train=False)

    pooled_features = []
    for name in observation.images:
        # SigLIP forward: (B, 196, 2048) patch tokens
        image_tokens, _ = pi0_model.PaliGemma.img(observation.images[name], train=False)
        # Mask invalid images by zeroing their tokens
        mask = observation.image_masks[name]  # (B,)
        image_tokens = image_tokens * mask[:, None, None]
        # Global average pool: (B, 2048)
        pooled = jnp.mean(image_tokens, axis=1)
        pooled_features.append(pooled)

    # Concatenate all camera features + state
    features = jnp.concatenate([*pooled_features, observation.state], axis=-1)
    return jax.lax.stop_gradient(features)


def extract_features_from_img(
    siglip_img,
    observation: _model.Observation,
) -> at.Float[at.Array, "b f"]:
    """Same as extract_features but takes the SigLIP module directly.

    Useful at inference time to avoid passing the full pi0 model,
    allowing separate JIT compilation.
    """
    observation = _model.preprocess_observation(None, observation, train=False)

    pooled_features = []
    for name in observation.images:
        image_tokens, _ = siglip_img(observation.images[name], train=False)
        mask = observation.image_masks[name]
        image_tokens = image_tokens * mask[:, None, None]
        pooled = jnp.mean(image_tokens, axis=1)
        pooled_features.append(pooled)

    features = jnp.concatenate([*pooled_features, observation.state], axis=-1)
    return jax.lax.stop_gradient(features)


# ---------------------------------------------------------------------------
# MLP building block
# ---------------------------------------------------------------------------

class MLPBlock(nnx.Module):
    """MLP block: [Linear -> LayerNorm -> SiLU] x N -> Linear."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        dims = [in_dim, *hidden_dims]
        self.linears = [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]
        self.norms = [nnx.LayerNorm(dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]
        self.out_linear = nnx.Linear(dims[-1], out_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        for linear, norm in zip(self.linears, self.norms):
            x = nnx.silu(norm(linear(x)))
        return self.out_linear(x)


# ---------------------------------------------------------------------------
# Residual Actor
# ---------------------------------------------------------------------------

class ResidualActor(nnx.Module):
    """Lightweight residual actor: predicts action corrections given features and noise.

    total_action = pi0_base_action(obs, z) + output_scale * actor(features, z)

    The noise z is included as input so the residual is aware of which base
    proposal it is correcting.
    """

    def __init__(self, config: ResidualRLConfig, *, rngs: nnx.Rngs):
        self.config = config
        in_dim = config.feature_dim + config.flat_action_dim  # features + flattened noise
        self.mlp = MLPBlock(
            in_dim=in_dim,
            hidden_dims=config.actor_hidden_dims,
            out_dim=config.flat_action_dim,
            rngs=rngs,
        )
        self.output_scale = config.actor_output_scale

    def __call__(
        self,
        features: at.Float[at.Array, "b f"],
        noise_flat: at.Float[at.Array, "b n"],
    ) -> at.Float[at.Array, "b ah ad"]:
        """
        Args:
            features: (B, feature_dim) from extract_features.
            noise_flat: (B, action_horizon * action_dim) flattened noise.

        Returns:
            residual_actions: (B, action_horizon, action_dim) scaled residual.
        """
        x = jnp.concatenate([features, noise_flat], axis=-1)
        out = self.mlp(x)  # (B, flat_action_dim)
        out = out * self.output_scale
        return out.reshape(-1, self.config.action_horizon, self.config.action_dim)


# ---------------------------------------------------------------------------
# Critic Ensemble
# ---------------------------------------------------------------------------

class CriticMLP(nnx.Module):
    """Single Q-network: maps (features, noise, action) -> scalar Q-value."""

    def __init__(self, config: ResidualRLConfig, *, rngs: nnx.Rngs):
        # Input: features + flattened noise + flattened action
        in_dim = config.feature_dim + config.flat_action_dim + config.flat_action_dim
        self.mlp = MLPBlock(
            in_dim=in_dim,
            hidden_dims=config.critic_hidden_dims,
            out_dim=1,
            rngs=rngs,
        )

    def __call__(
        self,
        features: at.Float[at.Array, "b f"],
        noise_flat: at.Float[at.Array, "b n"],
        action_flat: at.Float[at.Array, "b a"],
    ) -> at.Float[at.Array, "b 1"]:
        x = jnp.concatenate([features, noise_flat, action_flat], axis=-1)
        return self.mlp(x)


class CriticEnsemble(nnx.Module):
    """Ensemble of Q-networks for stable value estimation.

    Uses independent MLPs. For target computation, takes min across ensemble.
    """

    def __init__(self, config: ResidualRLConfig, *, rngs: nnx.Rngs):
        self.config = config
        # Create N_Q independent critics with different random seeds
        self.critics = [
            CriticMLP(config, rngs=nnx.Rngs(jax.random.fold_in(rngs.params(), i)))
            for i in range(config.num_critics)
        ]

    def __call__(
        self,
        features: at.Float[at.Array, "b f"],
        noise_flat: at.Float[at.Array, "b n"],
        action_flat: at.Float[at.Array, "b a"],
    ) -> at.Float[at.Array, "nq b 1"]:
        """Forward pass through all critics.

        Returns:
            q_values: (num_critics, B, 1) Q-values from each critic.
        """
        q_values = jnp.stack(
            [critic(features, noise_flat, action_flat) for critic in self.critics],
            axis=0,
        )
        return q_values

    def min_q(
        self,
        features: at.Float[at.Array, "b f"],
        noise_flat: at.Float[at.Array, "b n"],
        action_flat: at.Float[at.Array, "b a"],
    ) -> at.Float[at.Array, "b 1"]:
        """Minimum Q-value across ensemble (conservative estimate)."""
        q_all = self(features, noise_flat, action_flat)  # (N_Q, B, 1)
        return jnp.min(q_all, axis=0)  # (B, 1)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def critic_loss(
    critic: CriticEnsemble,
    target_critic: CriticEnsemble,
    actor: ResidualActor,
    # Precomputed pi0 outputs (computed outside JIT)
    features: at.Float[at.Array, "b f"],
    actions: at.Float[at.Array, "b ah ad"],
    rewards: at.Float[at.Array, " b"],
    next_features: at.Float[at.Array, "b f"],
    next_base_actions: at.Float[at.Array, "b ah ad"],  # pi0(next_obs, z')
    next_noise: at.Float[at.Array, "b ah ad"],          # z' used for next_base_actions
    dones: at.Float[at.Array, " b"],
    # Config
    discount: float = 0.99,
    action_horizon: int = 50,
    mc_returns: at.Float[at.Array, " b"] | None = None,
    critic_target: str = "mc",
    mc_weight: float = 1.0,
) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
    """Compute critic loss.

    critic_target controls the target computation:
      - "td": chunk-level TD bootstrapping (R_chunk + gamma^H * Q_target)
        where R_chunk is the discounted sum of dense rewards over the action chunk
        and H is the action_horizon
      - "mc": pure Monte-Carlo returns from the dataset
      - "blend": mc_weight * mc_return + (1 - mc_weight) * td_target

    All pi0 forward passes are precomputed — this function only uses
    the lightweight actor and critic MLPs.
    """
    B = features.shape[0]
    flat_action_dim = actions.shape[1] * actions.shape[2]
    action_flat = actions.reshape(B, flat_action_dim)

    # Noise = zeros for dataset actions (we don't know the original latent)
    zero_noise = jnp.zeros((B, flat_action_dim))

    # Q predictions for dataset actions from all critics
    q_pred_all = critic(features, zero_noise, action_flat)  # (N_Q, B, 1)

    # --- Compute target Q ---
    next_noise_flat = next_noise.reshape(B, flat_action_dim)

    # Residual actions for next state (stop gradient - don't train actor through critic loss)
    residual_next = jax.lax.stop_gradient(actor(next_features, next_noise_flat))
    total_actions_next = next_base_actions + residual_next
    total_actions_next_flat = total_actions_next.reshape(B, flat_action_dim)

    # Target Q from target critic ensemble (min over ensemble)
    target_q = target_critic.min_q(next_features, next_noise_flat, total_actions_next_flat)  # (B, 1)
    target_q = jax.lax.stop_gradient(target_q)

    # TD target: R_chunk + gamma^H * (1 - done) * Q_target
    # rewards is already the discounted chunk reward from the data loader
    chunk_discount = discount ** action_horizon
    td_target = rewards[:, None] + chunk_discount * (1.0 - dones[:, None]) * target_q  # (B, 1)

    # Select target based on critic_target mode
    if critic_target == "mc":
        target = mc_returns[:, None]  # (B, 1)
    elif critic_target == "blend":
        mc_target = mc_returns[:, None]  # (B, 1)
        target = mc_weight * mc_target + (1.0 - mc_weight) * td_target
    else:  # "td"
        target = td_target

    # MSE loss averaged over ensemble
    td_errors = q_pred_all - target[None, :, :]  # (N_Q, B, 1)
    critic_loss_val = jnp.mean(td_errors ** 2)

    info = {
        "critic_loss": critic_loss_val,
        "q_mean": jnp.mean(q_pred_all),
        "q_min": jnp.min(q_pred_all),
        "q_max": jnp.max(q_pred_all),
        "td_target_mean": jnp.mean(td_target),
        "mc_target_mean": jnp.mean(mc_returns[:, None]) if mc_returns is not None else jnp.float32(0.0),
        "blended_target_mean": jnp.mean(target),
    }
    return critic_loss_val, info


def actor_loss(
    actor: ResidualActor,
    critic: CriticEnsemble,
    # Precomputed pi0 outputs (computed outside JIT)
    features: at.Float[at.Array, "b f"],
    base_actions_k: at.Float[at.Array, "bk ah ad"],  # pi0(obs, z_k) for K samples, shape (B*K, ah, ad)
    noise_k_flat: at.Float[at.Array, "bk n"],         # flattened z_k, shape (B*K, flat_action_dim)
    features_expanded: at.Float[at.Array, "bk f"],     # features repeated K times, shape (B*K, feat_dim)
    # Config
    config: ResidualRLConfig,
    training_step: int = 0,
) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
    """Compute actor loss: L_RL + beta * L_BC with optional BC-loss filter.

    All pi0 forward passes are precomputed — this function only uses
    the lightweight actor and critic MLPs.
    """
    BK = base_actions_k.shape[0]
    K = config.num_noise_samples
    B = BK // K
    flat_action_dim = config.flat_action_dim

    # Residual actions: (B*K, ah, ad)
    residual_actions = actor(features_expanded, noise_k_flat)
    total_actions = base_actions_k + residual_actions
    total_actions_flat = total_actions.reshape(BK, flat_action_dim)

    # Q-values for total actions (min over ensemble): (B*K, 1)
    q_values = critic.min_q(features_expanded, noise_k_flat, total_actions_flat)
    q_values = q_values.reshape(B, K)  # (B, K)

    # RL loss: maximize Q-values (average over K samples)
    rl_loss = -jnp.mean(q_values)

    # BC loss: penalize residual magnitude
    residual_sq = jnp.mean(residual_actions ** 2, axis=(-1, -2))  # (B*K,)
    residual_sq = residual_sq.reshape(B, K)  # (B, K)

    # BC-loss filter: after warmup, disable BC penalty when residual improves Q.
    # Always compute both paths for JIT compatibility, select via jnp.where.
    bc_loss_unfiltered = jnp.mean(residual_sq)

    # Filtered BC loss: disable BC when residual-edited action has higher Q than base
    base_actions_flat = base_actions_k.reshape(BK, flat_action_dim)
    q_base = critic.min_q(features_expanded, noise_k_flat, base_actions_flat)
    q_base = jax.lax.stop_gradient(q_base.reshape(B, K))
    improvement = q_values - q_base  # (B, K)
    bc_mask = jnp.where(improvement > 0, 0.0, 1.0)
    bc_loss_filtered = jnp.mean(bc_mask * residual_sq)

    use_filter = training_step > config.q_filter_warmup
    bc_loss = jnp.where(use_filter, bc_loss_filtered, bc_loss_unfiltered)

    total_loss = rl_loss + config.beta_bc * bc_loss

    # Base pi0 action norm (per-sample mean over horizon and action dims)
    base_sq = jnp.mean(base_actions_k ** 2, axis=(-1, -2))  # (B*K,)
    base_norm = jnp.mean(jnp.sqrt(base_sq + 1e-8))

    info = {
        "actor_total_loss": total_loss,
        "actor_rl_loss": rl_loss,
        "actor_bc_loss": bc_loss,
        "residual_norm": jnp.mean(jnp.sqrt(residual_sq + 1e-8)),
        "base_actor_norm": base_norm,
        "q_actor_mean": jnp.mean(q_values),
    }
    return total_loss, info


# ---------------------------------------------------------------------------
# Target network update (Polyak averaging)
# ---------------------------------------------------------------------------

def polyak_update(
    target_params: nnx.State,
    online_params: nnx.State,
    tau: float,
) -> nnx.State:
    """Polyak averaging: target = tau * online + (1 - tau) * target."""
    return jax.tree.map(lambda t, o: tau * o + (1.0 - tau) * t, target_params, online_params)


# ---------------------------------------------------------------------------
# Best-of-N action selection (for inference)
# ---------------------------------------------------------------------------

def best_of_n_actions(
    pi0_model: _model.BaseModel,
    actor: ResidualActor,
    critic: CriticEnsemble,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    features: at.Float[at.Array, "b f"],
    config: ResidualRLConfig,
    n_samples: int = 10,
) -> at.Float[at.Array, "b ah ad"]:
    """Sample N action candidates and select the one with highest Q-value.

    Args:
        pi0_model: Frozen base policy.
        actor: Residual actor.
        critic: Critic ensemble.
        rng: Random key.
        observation: Current observation.
        features: Precomputed features from extract_features.
        config: RL config.
        n_samples: Number of candidates to evaluate.

    Returns:
        best_actions: (B, action_horizon, action_dim) highest-Q actions.
    """
    B = features.shape[0]
    K = n_samples
    flat_action_dim = config.flat_action_dim

    noise_rng, action_rng = jax.random.split(rng)
    noise_k = jax.random.normal(noise_rng, (B, K, config.action_horizon, config.action_dim))

    # Expand for B*K batch
    features_expanded = jnp.repeat(features[:, None, :], K, axis=1).reshape(B * K, -1)
    noise_flat = noise_k.reshape(B * K, flat_action_dim)

    def repeat_leaf(x):
        if x is None:
            return None
        expanded = jnp.repeat(x[:, None], K, axis=1)
        return expanded.reshape(-1, *x.shape[1:])

    obs_expanded = jax.tree.map(repeat_leaf, observation)

    # Get total actions: base + residual
    base_actions = pi0_model.sample_actions(
        action_rng, obs_expanded,
        noise=noise_k.reshape(B * K, config.action_horizon, config.action_dim),
    )
    base_actions = jax.lax.stop_gradient(base_actions)
    residual_actions = actor(features_expanded, noise_flat)
    total_actions = base_actions + residual_actions  # (B*K, ah, ad)
    total_actions_flat = total_actions.reshape(B * K, flat_action_dim)

    # Q-values: min over ensemble
    q_values = critic.min_q(features_expanded, noise_flat, total_actions_flat)  # (B*K, 1)
    q_values = q_values.reshape(B, K)  # (B, K)

    # Select best per batch element
    best_idx = jnp.argmax(q_values, axis=1)  # (B,)
    total_actions_bk = total_actions.reshape(B, K, config.action_horizon, config.action_dim)
    best_actions = total_actions_bk[jnp.arange(B), best_idx]  # (B, ah, ad)

    return best_actions
