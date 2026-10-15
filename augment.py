"""Token-excision and SCARF augmenters.

Three modes:

  cell    - whole continuous group plus elementwise one-hot masking; embeddings untouched.

  feature - whole-feature token excision: continuous group zeroed, one-hot block
            zeroed, embedding index -> 0 (<unknown>).

  scarf   - SCARF-style corruption: replace each cell value with a value sampled
            from a random batch row (possibly the same row) in the batch (marginal shuffling). This is
            the strongest comparison for the token excision claim: same architecture,
            different augmentation signal.

`make_augmenter(mode, mask_prob, one_hot_groups)` returns a jit-friendly
`augment(batch, key) -> {"view1":..., "view2":...}`.
"""
from __future__ import annotations

from typing import Callable, Dict, List
import jax
import jax.numpy as jnp


def make_augmenter(mode: str, mask_prob: float, one_hot_groups: List[int]) -> Callable:
    if mode not in ("cell", "feature", "scarf"):
        raise ValueError(f"excision mode must be 'cell', 'feature', or 'scarf', got {mode!r}")
    keep_p = 1.0 - mask_prob
    groups = tuple(int(g) for g in one_hot_groups)

    # ── cell mode helpers ───────────────────────────────────────────
    def _cell_cont(c, k):
        if c.shape[1] == 0:
            return c
        return c * jax.random.bernoulli(k, p=keep_p, shape=(c.shape[0], 1))

    def _cell_oh(oh, k):
        if oh.shape[1] == 0:
            return oh
        return oh * jax.random.bernoulli(k, p=keep_p, shape=oh.shape)

    # ── feature mode helpers ────────────────────────────────────────
    def _feat_cont(c, k):
        if c.shape[1] == 0:
            return c
        # zero entire continuous group (one token) as a unit
        keep = jax.random.bernoulli(k, p=keep_p, shape=(c.shape[0], 1))
        return c * keep

    def _feat_oh(oh, k):
        if oh.shape[1] == 0:
            return oh
        B = oh.shape[0]
        g_keep = jax.random.bernoulli(k, p=keep_p, shape=(B, len(groups)))
        col_mask = jnp.concatenate(
            [jnp.broadcast_to(g_keep[:, i:i + 1], (B, sz)) for i, sz in enumerate(groups)],
            axis=1)
        return oh * col_mask

    def _feat_embed(e, k):
        if e.shape[1] == 0:
            return e
        keep = jax.random.bernoulli(k, p=keep_p, shape=e.shape)
        return jnp.where(keep, e, 0)

    # ── SCARF helper ────────────────────────────────────────────────
    def _scarf_float(arr, k):
        """Replace each cell with a value from a random batch row, including itself (marginal shuffle)."""
        if arr.shape[1] == 0:
            return arr
        B, D = arr.shape
        corrupt_mask = jax.random.bernoulli(k, p=mask_prob, shape=(B, D))
        # random row indices for replacement
        row_idx = jax.random.randint(k, shape=(B, D), minval=0, maxval=B)
        col_idx = jnp.broadcast_to(jnp.arange(D), (B, D))
        shuffled = arr[row_idx, col_idx]
        return jnp.where(corrupt_mask, shuffled, arr)

    # ── dispatch ───────────────────────────────────────────────────
    def _view(batch, key):
        kc, ko, ke = jax.random.split(key, 3)
        if mode == "cell":
            return {
                "continuous": _cell_cont(batch["continuous"], kc),
                "one_hot": _cell_oh(batch["one_hot"], ko),
                "embedding": batch["embedding"],  # never masked in cell mode
            }
        elif mode == "feature":
            return {
                "continuous": _feat_cont(batch["continuous"], kc),
                "one_hot": _feat_oh(batch["one_hot"], ko),
                "embedding": _feat_embed(batch["embedding"], ke),
            }
        else:  # scarf
            return {
                "continuous": _scarf_float(batch["continuous"], kc),
                "one_hot": _scarf_float(batch["one_hot"], ko),
                "embedding": batch["embedding"],  # SCARF doesn't corrupt categoricals
            }

    def augment(batch: Dict[str, jnp.ndarray], key) -> Dict[str, Dict]:
        k1, k2 = jax.random.split(key)
        return {"view1": _view(batch, k1), "view2": _view(batch, k2)}

    return augment
