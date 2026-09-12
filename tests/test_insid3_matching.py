"""Analytic and metamorphic checks of the method, without model features."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from src.insid3_matching import compute_debiased_similarity as match


def positional_probe():
    # After patch normalization and centering: only channel 0 varies.
    # The large constant channel must NOT become the positional direction.
    return np.array([[[1.0, 0.0, 3.0], [-1.0, 0.0, 3.0]]])


def valid_inputs():
    return dict(
        probe_features=positional_probe(),
        reference_features=np.array([[[10.0, 1.0, 0.0]]]),
        reference_mask=np.array([[True]]),
        target_features=np.array([[[-10.0, 1.0, 0.0], [10.0, 0.0, 1.0]]]),
        rank=1,
    )


def test_basis_is_channel_subspace_with_orthogonal_projector():
    result = match(**valid_inputs())
    basis = result["basis"]
    assert basis.shape == (3, 1)
    assert_allclose(basis.T @ basis, np.eye(1), atol=1e-14)
    projector = basis @ basis.T  # Sign-invariant comparison, not raw SVD signs.
    assert_allclose(projector, np.diag([1, 0, 0]), atol=1e-14)
    complement = np.eye(3) - projector
    assert_allclose(complement @ complement, complement, atol=1e-14)
    assert_allclose(complement.T, complement, atol=1e-14)
    assert_allclose(basis.T @ complement, 0, atol=1e-14)
    assert_allclose(basis.T @ result["reference_prototype"], 0, atol=1e-14)


def test_normalize_probe_before_centering_and_choose_top_variance():
    # Unit directions have variance x > y, with a common z offset.
    # Unequal row magnitudes make raw PCA select a different direction.
    probe = np.array([
        [1, 0, 3], [-1, 0, 3], [1, 0, 3], [-1, 0, 3],
        [0, 1, 3], [0, -1, 3],
    ], dtype=float).reshape(2, 3, 3)
    probe *= np.array([1, 2, 3, 4, 100, 200]).reshape(2, 3, 1)
    inputs = valid_inputs()
    inputs["probe_features"] = probe
    basis = match(**inputs)["basis"]
    assert_allclose(basis @ basis.T, np.diag([1, 0, 0]), atol=1e-13)
    inputs["rank"] = 2
    basis = match(**inputs)["basis"]
    assert basis.shape == (3, 2)
    assert_allclose(basis @ basis.T, np.diag([1, 1, 0]), atol=1e-13)


def test_debias_reverses_position_driven_false_match():
    inputs = valid_inputs()
    raw = match(**{**inputs, "rank": 0})["similarity_map"]
    debiased = match(**inputs)["similarity_map"]
    # Column 0: same semantics / opposite position; column 1: same position /
    # unrelated semantics. The raw ranking is wrong, the debiased one is right.
    assert_allclose(raw, [[-99 / 101, 100 / 101]], atol=1e-14)
    assert raw[0, 1] > raw[0, 0]
    assert_allclose(debiased, [[1, 0]], atol=1e-14)
    assert debiased[0, 0] > debiased[0, 1]
    assert abs(debiased[0, 1]) < abs(raw[0, 1])


def test_reference_mask_selects_the_concept_on_different_nonsquare_grids():
    reference = np.array([[[10, 1, 0], [10, 0, 1]]], dtype=float)
    target = np.array([
        [[-10, 1, 0], [10, 0, 1], [5, -1, 0]],
        [[-7, 0, -1], [2, 1, 1], [4, 0, 0]],
    ], dtype=float)
    common = dict(probe_features=positional_probe(), reference_features=reference,
                  target_features=target, rank=1)
    y = match(**common, reference_mask=np.array([[1, 0]]))
    z = match(**common, reference_mask=np.array([[0, 1]]))
    assert_allclose(y["reference_prototype"], [0, 1, 0], atol=1e-14)
    assert_allclose(z["reference_prototype"], [0, 0, 1], atol=1e-14)
    assert y["similarity_map"].shape == (2, 3)
    assert_allclose(y["similarity_map"], [[1, 0, -1], [0, 1 / np.sqrt(2), 0]])
    assert_allclose(z["similarity_map"], [[0, 1, 0], [-1, 1 / np.sqrt(2), 0]])
    assert_array_equal(y["basis"], z["basis"])


def test_normalize_each_debiased_patch_then_mean_then_normalize_prototype():
    inputs = valid_inputs()
    inputs.update(
        reference_features=np.array([[[20.0, 3.0, 0.0], [0.0, 0.0, 7.0]]]),
        reference_mask=np.ones((1, 2)),
        target_features=np.array([[[5.0, 1.0, 0.0], [0.0, 0.0, 100.0]]]),
    )
    result = match(**inputs)
    assert_allclose(result["reference_prototype"], [0, 1 / np.sqrt(2), 1 / np.sqrt(2)])
    assert_allclose(result["similarity_map"], [[1 / np.sqrt(2), 1 / np.sqrt(2)]])


def test_probe_mean_is_not_subtracted_from_reference_or_target():
    inputs = valid_inputs()
    inputs.update(
        reference_features=np.array([[[0.0, 1.0, 0.0]]]),
        target_features=np.array([[[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]]),
    )
    assert_allclose(match(**inputs)["similarity_map"], [[1 / np.sqrt(2), 0]])


def test_rank_zero_is_normalized_masked_cosine_matching():
    inputs = valid_inputs()
    inputs.update(
        rank=0,
        reference_features=np.array([[[3.0, 0, 0], [0, 8.0, 0]]]),
        reference_mask=np.array([[True, True]]),
        target_features=np.array([[[100.0, 0, 0], [0, -4.0, 0], [0, 0, 0]]]),
    )
    result = match(**inputs)
    assert result["basis"].shape == (3, 0)
    assert_allclose(result["reference_prototype"], [1 / np.sqrt(2), 1 / np.sqrt(2), 0])
    assert_allclose(result["similarity_map"], [[1 / np.sqrt(2), -1 / np.sqrt(2), 0]])


@pytest.mark.parametrize("probe_shape,rank,expected", [
    ((1, 2, 5), 100, 2), ((2, 3, 3), 100, 3),
    ((2, 3, 5), 2, 2), ((1, 1, 3), 2, 1),
])
def test_rank_capped_by_reduced_svd_shape(probe_shape, rank, expected):
    channels = probe_shape[-1]
    # All-zero / rank-deficient probe deliberately checks the documented
    # slice convention, without asserting arbitrary null-space orientations.
    result = match(np.zeros(probe_shape), np.ones((1, 1, channels)),
                   np.ones((1, 1)), np.ones((2, 1, channels)), rank)
    assert result["basis"].shape == (channels, expected)
    assert_allclose(result["basis"].T @ result["basis"], np.eye(expected), atol=1e-14)
    assert all(np.isfinite(value).all() for value in result.values())


@pytest.mark.parametrize("reference", [
    [[[0.0, 0, 0], [0, 0, 0]]],        # zero features
    [[[10.0, 0, 0], [-10.0, 0, 0]]],  # fully removed position features
    [[[0.0, 1, 0], [0, -1.0, 0]]],    # cancellation within the reference region
])
def test_zero_or_cancelling_reference_has_zero_scores(reference):
    inputs = valid_inputs()
    inputs.update(reference_features=np.array(reference), reference_mask=np.ones((1, 2)))
    result = match(**inputs)
    assert_allclose(result["reference_prototype"], np.zeros(3), atol=1e-14)
    assert_allclose(result["similarity_map"], np.zeros((1, 2)), atol=1e-14)


def test_rank_covering_all_channels_can_remove_every_feature():
    probe = np.array([[[1.0, 0], [-1.0, 0]], [[0, 1.0], [0, -1.0]]])
    result = match(probe, np.array([[[2.0, 3.0]]]), np.ones((1, 1)),
                   np.array([[[4.0, 5.0]]]), 2)
    assert_allclose(result["basis"] @ result["basis"].T, np.eye(2), atol=1e-14)
    assert_allclose(result["similarity_map"], 0, atol=1e-12)


def test_positive_patch_scaling_and_joint_channel_rotation_preserve_matching():
    rng = np.random.default_rng(19)
    inputs = dict(probe_features=rng.normal(size=(2, 4, 5)),
                  reference_features=rng.normal(size=(2, 3, 5)),
                  reference_mask=np.array([[1, 0, 1], [0, 1, 0]]),
                  target_features=rng.normal(size=(3, 2, 5)), rank=2)
    original = match(**inputs)
    scaled = dict(inputs)
    for name in ("probe_features", "reference_features", "target_features"):
        scaled[name] = inputs[name] * rng.uniform(0.1, 20, size=inputs[name].shape[:2] + (1,))
    rescaled = match(**scaled)
    assert_allclose(rescaled["similarity_map"], original["similarity_map"], atol=1e-12)
    assert_allclose(rescaled["reference_prototype"], original["reference_prototype"], atol=1e-12)
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    rotated = dict(inputs)
    for name in ("probe_features", "reference_features", "target_features"):
        rotated[name] = inputs[name] @ rotation
    rotated_result = match(**rotated)
    assert_allclose(rotated_result["similarity_map"], original["similarity_map"], atol=1e-12)
    assert_allclose(rotated_result["reference_prototype"], original["reference_prototype"] @ rotation,
                    atol=1e-12)
    projector = original["basis"] @ original["basis"].T
    assert_allclose(rotated_result["basis"] @ rotated_result["basis"].T,
                    rotation.T @ projector @ rotation, atol=1e-12)


def test_background_changes_do_not_change_masked_reference():
    inputs = valid_inputs()
    inputs.update(reference_features=np.array([[[10.0, 1, 0], [0, 2, 3]]]),
                  reference_mask=np.array([[1, 0]]))
    before = match(**inputs)
    inputs["reference_features"][0, 1] = [-99, 80, -4]
    after = match(**inputs)
    for key in before:
        assert_array_equal(before[key], after[key])


def test_inputs_unchanged_even_when_readonly_and_noncontiguous():
    rng = np.random.default_rng(5)
    probe = rng.normal(size=(4, 3, 4))[::2, :, ::-1]
    reference = rng.normal(size=(4, 2, 4))[::2, :, ::-1]
    target = rng.normal(size=(2, 6, 4))[:, ::2, ::-1]
    mask = np.array([[1, 0, 0, 1], [0, 0, 1, 0]])[:, ::2]
    arrays = [probe, reference, mask, target]
    saved = [array.copy() for array in arrays]
    for array in arrays:
        array.setflags(write=False)
    result = match(probe, reference, mask, target, 1)
    for actual, expected in zip(arrays, saved):
        assert_array_equal(actual, expected)
    for output in result.values():
        assert all(not np.shares_memory(output, array) for array in arrays)


def test_float32_integer_inputs_and_numpy_integer_rank():
    inputs = valid_inputs()
    inputs["probe_features"] = inputs["probe_features"].astype(np.float32)
    inputs["reference_features"] = inputs["reference_features"].astype(np.int32)
    inputs["rank"] = np.int64(1)
    result = match(**inputs)
    assert all(value.dtype == np.float64 for value in result.values())
    assert_allclose(result["similarity_map"], [[1, 0]], atol=1e-12)


def test_large_finite_features_do_not_overflow_normalization():
    inputs = valid_inputs()
    inputs["reference_features"] = np.array([[[1e300, 1e300, 0]]])
    inputs["target_features"] = np.array([[[-1e300, 1e300, 0], [1e300, 0, 1e300]]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = match(**inputs)
    assert_allclose(result["similarity_map"], [[1, 0]], atol=1e-12)


@pytest.mark.parametrize("name,value,exception", [
    ("reference_mask", np.zeros((1, 1)), ValueError),
    ("reference_mask", np.ones((1, 2)), ValueError),
    ("reference_mask", np.ones((1, 1, 1)), ValueError),
    ("reference_mask", np.array([[0.5]]), ValueError),
    ("reference_mask", np.array([[np.nan]]), ValueError),
    ("reference_mask", np.array([[-1]]), ValueError),
    ("reference_mask", np.array([["1"]]), ValueError),
    ("reference_mask", [[True]], TypeError),
    ("probe_features", np.zeros((2, 3)), ValueError),
    ("probe_features", np.zeros((0, 2, 3)), ValueError),
    ("probe_features", np.zeros((2, 2, 0)), ValueError),
    ("reference_features", np.zeros((1, 1, 4)), ValueError),
    ("target_features", np.zeros((1, 2, 4)), ValueError),
    ("target_features", np.full((1, 2, 3), np.inf), ValueError),
    ("probe_features", np.full((1, 2, 3), np.nan), ValueError),
    ("reference_features", np.ones((1, 1, 3), dtype=complex), TypeError),
    ("reference_features", np.ones((1, 1, 3), dtype=bool), TypeError),
    ("reference_features", [[[1, 2, 3]]], TypeError),
    ("rank", -1, ValueError),
    ("rank", 1.0, TypeError),
    ("rank", True, TypeError),
    ("rank", np.bool_(False), TypeError),
    ("rank", "1", TypeError),
])
def test_invalid_inputs_have_clear_errors(name, value, exception):
    inputs = valid_inputs()
    inputs[name] = value
    with pytest.raises(exception, match="rank|mask|features|channel"):
        match(**inputs)
