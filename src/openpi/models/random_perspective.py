# random_perspective.py
import jax
import jax.numpy as jnp
import augmax


def _compute_homography(src_xy, dst_xy):
    """Compute 3x3 homography H that maps src_xy -> dst_xy (both 4x2, normalized)."""
    # Build linear system A h = b (8x8) for 4 corner correspondences
    x, y = src_xy[:, 0], src_xy[:, 1]
    X, Y = dst_xy[:, 0], dst_xy[:, 1]

    zeros = jnp.zeros_like(x)
    ones = jnp.ones_like(x)

    A_top = jnp.stack([x, y, ones, zeros, zeros, zeros, -x * X, -y * X], axis=1)
    A_bot = jnp.stack([zeros, zeros, zeros, x, y, ones, -x * Y, -y * Y], axis=1)
    A = jnp.vstack([A_top, A_bot])
    b = jnp.concatenate([X, Y], axis=0)

    # Solve for h (8,) then append 1
    h = jnp.linalg.lstsq(A, b, rcond=None)[0]
    H = jnp.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0],
    ])
    return H


def _bilinear_sample(image, xs, ys):
    """Bilinear sampler. image: (H,W,C); xs, ys: normalized in [0,1]. Returns (H,W,C)."""
    H, W, C = image.shape
    # Convert to pixel coords
    xs = xs * (W - 1)
    ys = ys * (H - 1)

    x0 = jnp.floor(xs).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, W - 1)
    y0 = jnp.floor(ys).astype(jnp.int32)
    y1 = jnp.clip(y0 + 1, 0, H - 1)

    x0c = jnp.clip(x0, 0, W - 1)
    y0c = jnp.clip(y0, 0, H - 1)

    Ia = image[y0c, x0c]
    Ib = image[y0c, x1]
    Ic = image[y1, x0c]
    Id = image[y1, x1]

    wa = (x1 - xs) * (y1 - ys)
    wb = (xs - x0) * (y1 - ys)
    wc = (x1 - xs) * (ys - y0)
    wd = (xs - x0) * (ys - y0)

    wa = wa[..., None]
    wb = wb[..., None]
    wc = wc[..., None]
    wd = wd[..., None]

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out


def _warp_perspective(image, H):
    """Apply homography H to image. Coordinates are normalized [0,1]."""
    Hinv = jnp.linalg.inv(H)
    H_, W_, _ = image.shape

    # Build target grid in normalized coords
    ys = jnp.linspace(0.0, 1.0, H_)
    xs = jnp.linspace(0.0, 1.0, W_)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")

    ones = jnp.ones_like(grid_x)
    tgt = jnp.stack([grid_x, grid_y, ones], axis=-1)  # (H,W,3)

    # Map target -> source using H^{-1}
    src = tgt @ Hinv.T  # (H,W,3)
    src_x = src[..., 0] / (src[..., 2] + 1e-8)
    src_y = src[..., 1] / (src[..., 2] + 1e-8)

    # Clamp to [0,1] to avoid NaNs
    src_x = jnp.clip(src_x, 0.0, 1.0)
    src_y = jnp.clip(src_y, 0.0, 1.0)

    return _bilinear_sample(image, src_x, src_y)


class RandomPerspective:
    """
    Random perspective warp compatible with augmax.Chain.
    scale: max normalized corner shift (0..1). (0.2–0.35 is typical)
    p: probability of applying the transform.
    """

    def __init__(self, scale=0.30, p=0.5):
        self.scale = float(scale)
        self.p = float(p)

    def apply(self, rng, inputs, input_types, invert=False):
        # We assume `inputs` is either an image (H,W,C) or a PyTree of images.
        rng, p_key, off_key = jax.random.split(rng, 3)

        def _apply_to_image(img):
            # Corners in normalized coords
            src = jnp.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ])
            # Random offsets per corner
            offsets = jax.random.uniform(
                off_key, (4, 2), minval=-self.scale, maxval=self.scale
            )
            dst = jnp.clip(src + offsets, 0.0, 1.0)

            H = _compute_homography(src, dst)
            return _warp_perspective(img, H)

        def _do(_):
            # If inputs is a PyTree (e.g., dict of images), warp only arrays with shape (...,3)
            def _maybe(img):
                # Basic heuristic: only warp HWC images
                return jax.lax.cond(
                    (img.ndim == 3) & (img.shape[-1] in (1, 3, 4)),
                    lambda x: _apply_to_image(x),
                    lambda x: x,
                    img,
                )

            return jax.tree_util.tree_map(_maybe, inputs)

        def _skip(_):
            return inputs

        return jax.lax.cond(jax.random.bernoulli(p_key, self.p), _do, _skip, operand=None)


