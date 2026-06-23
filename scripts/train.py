import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def save_debug_images(
    observation: _model.Observation, step: int, checkpoint_dir: epath.Path, data_config: _config.DataConfig = None
):
    """Save sample images from the observation for debugging purposes."""
    debug_dir = checkpoint_dir / "debug_images"
    debug_dir.mkdir(exist_ok=True)

    # Convert JAX arrays to numpy for easier handling
    images = jax.device_get(observation.images)
    image_masks = jax.device_get(observation.image_masks)

    # Save first sample from the batch
    sample_idx = 0
    step_dir = debug_dir / f"step_{step:06d}"
    step_dir.mkdir(exist_ok=True)

    # Store images for visualization
    camera_images = {}

    for camera_name, img_batch in images.items():
        if sample_idx < img_batch.shape[0]:  # Check if sample exists in batch
            img = img_batch[sample_idx]  # Shape: (H, W, C)
            mask = image_masks[camera_name][sample_idx] if camera_name in image_masks else True

            # Check if images are actually normalized (they're not in YAM)
            image_is_normalized = (
                data_config is not None
                and data_config.norm_stats is not None
                and f"image/{camera_name}" in data_config.norm_stats
            )

            # Convert from float32 [-1,1] to uint8 [0,255] if needed
            if img.dtype == np.float32:
                img = (np.clip(img, -1, 1) * 127.5 + 127.5).astype(np.uint8)

            # Store for visualization
            camera_images[camera_name] = img

            # Save as .npy file with stats
            img_path = step_dir / f"{camera_name}_sample_{sample_idx}.npy"
            np.save(img_path, img)

            # Log image stats
            stats_path = step_dir / f"{camera_name}_sample_{sample_idx}_stats.txt"
            with open(stats_path, "w") as f:
                f.write(f"Camera: {camera_name}\n")
                f.write(f"Step: {step}\n")
                f.write(f"Sample index: {sample_idx}\n")
                f.write(f"Image mask: {mask}\n")
                f.write(f"Image normalization applied: {image_is_normalized}\n")
                f.write("\n--- TRAINING IMAGE (after pipeline) ---\n")
                f.write(f"Image shape: {img.shape}\n")
                f.write(f"Image dtype: {img.dtype}\n")
                f.write(f"Image min: {img.min()}\n")
                f.write(f"Image max: {img.max()}\n")
                f.write(f"Image mean: {img.mean():.4f}\n")
                f.write(f"Image std: {img.std():.4f}\n")
                f.write(
                    f"Non-zero pixels: {np.count_nonzero(img)}/{img.size} ({100 * np.count_nonzero(img) / img.size:.2f}%)\n"
                )

                if not image_is_normalized:
                    f.write("\nNOTE: Images are NOT normalized in this config.\n")
                    f.write("The training pipeline preserves original image intensity values.\n")
                    f.write("Raw LeRobot data: float32 [0,1] -> Training data: uint8 [0,255]\n")

    # Create visualization if matplotlib is available
    if MATPLOTLIB_AVAILABLE and camera_images:
        try:
            camera_names = list(camera_images.keys())
            num_cameras = len(camera_names)

            fig, axes = plt.subplots(1, num_cameras, figsize=(5 * num_cameras, 5))
            if num_cameras == 1:
                axes = [axes]  # Make it a list for consistency

            for i, camera_name in enumerate(camera_names):
                img = camera_images[camera_name]
                axes[i].imshow(img)
                axes[i].set_title(f"{camera_name}\nStep {step}\nMean: {img.mean():.1f}, Std: {img.std():.1f}")
                axes[i].axis("off")

            plt.tight_layout()

            # Save visualization
            viz_path = step_dir / f"step_{step:06d}_visualization.png"
            plt.savefig(viz_path, dpi=150, bbox_inches="tight")
            plt.close()  # Close to free memory

            logging.info(f"Saved debug images and visualization for step {step} to {step_dir}")
        except Exception as e:
            logging.warning(f"Failed to create visualization for step {step}: {e}")
            logging.info(f"Saved debug images for step {step} to {step_dir}")
    elif not MATPLOTLIB_AVAILABLE:
        logging.info(f"Saved debug images for step {step} to {step_dir} (matplotlib not available for visualization)")
    else:
        logging.info(f"Saved debug images for step {step} to {step_dir}")


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    # import pdb; pdb.set_trace() # observation.images['base_0_rgb'] is float32

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_config = data_loader.data_config()  # Get the data config for debug image unnormalization
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        # Save debug images periodically (every 100 steps for the first 1000 steps, then every 1000 steps)
        if (step < 1000 and step % 100 == 0) or (step >= 1000 and step % 1000 == 0) or step == 0:
            observation, _ = batch
            save_debug_images(observation, step, config.checkpoint_dir, data_config)

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
