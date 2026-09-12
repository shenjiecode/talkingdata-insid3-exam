"""Position-debiased cross-image matching of already extracted features.

Method: INSID3 section 3.1, equations (2)--(4). Normalization and probe
centering follow the author-code locations pinned in README.md. No feature
encoder, clustering, mask refinement, or image resizing is performed here.
"""

from numbers import Integral

import numpy as np


_EPS = 1e-12  # torch.nn.functional.normalize's default denominator floor


def _feature_grid(value: np.ndarray, name: str) -> np.ndarray:
    """Validate one channels-last grid and return a float64 working copy."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.ndim != 3 or any(size == 0 for size in value.shape):
        raise ValueError(f"{name} must have nonempty shape (H, W, C)")
    if value.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric features")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    result = value.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be representable as finite float64")
    return result


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize the last axis with an epsilon floor and no mutation.

    Scaling before squaring avoids overflow/underflow for large/small finite
    inputs. The final factor preserves x / max(norm(x), EPS), including zero.
    """
    scale = np.max(np.abs(vectors), axis=-1, keepdims=True)
    scaled = np.divide(
        vectors, scale, out=np.zeros_like(vectors), where=scale != 0
    )
    length = np.linalg.norm(scaled, axis=-1, keepdims=True)
    direction = np.divide(
        scaled, length, out=np.zeros_like(scaled), where=length != 0
    )
    # If scale >= EPS, the true norm is >= EPS and this factor is one.
    factor = np.minimum((np.minimum(scale, _EPS) / _EPS) * length, 1.0)
    return direction * factor


def compute_debiased_similarity(
    probe_features: np.ndarray,
    reference_features: np.ndarray,
    reference_mask: np.ndarray,
    target_features: np.ndarray,
    rank: int,
) -> dict[str, np.ndarray]:
    """Return a positional basis, masked prototype, and target similarity map.

    Args:
        probe_features: Finite real features of shape (Hp, Wp, C).
        reference_features: Finite real features of shape (Hr, Wr, C).
        reference_mask: Binary/bool array of shape (Hr, Wr), with foreground.
        target_features: Finite real features of shape (Ht, Wt, C).
        rank: Nonnegative integer; capped at min(Hp * Wp, C).

    Returns:
        Float64 arrays: basis (C, r), reference_prototype (C,), and
        similarity_map (Ht, Wt). Zero vectors use an L2 denominator floor
        of 1e-12. rank=0 removes no directions. The SVD slice is not
        truncated by numerical rank; null singular directions are nonunique.

    Raises:
        TypeError: Non-array input, non-real features, or non-integer rank.
        ValueError: Empty/invalid shapes, mismatched channels/mask, nonfinite
            values, nonbinary/empty mask, or negative rank.

    All inputs remain unchanged, including read-only or noncontiguous arrays.
    """
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, Integral):
        raise TypeError("rank must be a nonnegative integer, not a boolean")
    if rank < 0:
        raise ValueError("rank must be nonnegative")

    probe = _feature_grid(probe_features, "probe_features")
    reference = _feature_grid(reference_features, "reference_features")
    target = _feature_grid(target_features, "target_features")
    channels = probe.shape[-1]
    if reference.shape[-1] != channels or target.shape[-1] != channels:
        raise ValueError("all feature grids must have the same channel count")

    if not isinstance(reference_mask, np.ndarray):
        raise TypeError("reference_mask must be a numpy.ndarray")
    if reference_mask.shape != reference.shape[:2]:
        raise ValueError("reference_mask shape must match the reference grid")
    if reference_mask.dtype.kind not in "biuf" or not np.all(
        (reference_mask == 0) | (reference_mask == 1)
    ):
        raise ValueError("reference_mask must contain only binary 0/1 values")
    foreground = reference_mask.astype(bool, copy=True).reshape(-1)
    if not foreground.any():
        raise ValueError("reference_mask must contain at least one foreground patch")

    probe_rows = _normalize(probe.reshape(-1, channels))
    centered_probe = probe_rows - probe_rows.mean(axis=0, keepdims=True)
    effective_rank = min(int(rank), *centered_probe.shape)
    if effective_rank == 0:
        basis = np.empty((channels, 0), dtype=np.float64)
    else:
        # Rows are patches, so the positional directions are RIGHT vectors.
        _, _, right_vectors_t = np.linalg.svd(centered_probe, full_matrices=False)
        basis = right_vectors_t[:effective_rank].T.copy()

    def debias(grid: np.ndarray) -> np.ndarray:
        rows = _normalize(grid.reshape(-1, channels))
        # Same projection for both images; no centering of either image.
        residual = rows - (rows @ basis) @ basis.T
        return _normalize(residual)

    reference_rows = debias(reference)
    target_rows = debias(target)
    prototype = _normalize(reference_rows[foreground].mean(axis=0))
    similarity = (target_rows @ prototype).reshape(target.shape[:2])
    return {
        "basis": basis,
        "reference_prototype": prototype,
        "similarity_map": similarity,
    }
