"""Black-box cases for the answer requested in bonus_task.md.

The answer is intentionally not bundled. A missing answer is an explicit
failure when this suite is requested, never a silent skip or dummy success.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture(scope="module")
def solve():
    default_path = Path(__file__).resolve().parents[1] / "src" / "insid3_bonus.py"
    answer_path = Path(os.environ.get("INSID3_BONUS_SOLUTION", default_path))
    if not answer_path.is_file():
        pytest.fail(f"Place the model answer at {default_path} before running bonus tests")
    spec = importlib.util.spec_from_file_location("insid3_bonus_answer", answer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_seed_and_merge


def main_case():
    original = np.array([
        [0, 1, 0], [0, 1, 0],
        [0, 0.6, 0.8], [0, 0.6, -0.8],
        [0, 0.6, 0.8], [0, 0.6, -0.8],
        [np.sqrt(3) / 2, 0.5, 0], [np.sqrt(3) / 2, 0.5, 0],
    ]).reshape(2, 4, 3)
    debiased = original.copy().reshape(-1, 3)
    debiased[6:] = [0, 1, 0]
    return dict(
        reference_prototype=np.array([0.0, 1.0, 0.0]),
        target_original=original,
        target_debiased=debiased.reshape(2, 4, 3),
        cluster_labels=np.array([0, 0, 1, 1, 1, 1, 2, 2]).reshape(2, 4),
        candidate_mask=np.array([1, 0, 1, 0, 0, 0, 1, 0], dtype=bool).reshape(2, 4),
        threshold=0.2,
    )


def test_patch_mean_score_two_spaces_area_weight_and_whole_cluster(solve):
    inputs = main_case()
    saved = {key: value.copy() for key, value in inputs.items() if isinstance(value, np.ndarray)}
    out = solve(**inputs)
    assert type(out["seed_id"]) is int
    assert out["seed_id"] == 0
    assert_allclose(out["cross_similarity"], [1, 0.6, 1], atol=1e-12)
    assert_allclose(out["intra_similarity"], [1, 1, 0.5], atol=1e-12)
    assert_allclose(out["area_weights"], [1, 0.25, 0.5], atol=1e-12)
    assert_allclose(out["combined_scores"], [1, 0.15, 0.25], atol=1e-12)
    assert out["final_mask"].dtype == np.bool_
    assert_array_equal(out["final_mask"], np.array(
        [1, 1, 0, 0, 0, 0, 1, 1], dtype=bool).reshape(2, 4))
    for key, value in saved.items():
        assert_array_equal(inputs[key], value)


def test_seed_must_be_a_candidate_and_return_actual_label(solve):
    target = np.array([
        [0, 1, 0], [0, 1, 0],
        [0, 0.8, 0.6], [0, 0.8, 0.6],
        [0, 0.6, 0.8], [0, 0.6, 0.8],
    ]).reshape(2, 3, 3)
    out = solve(
        reference_prototype=np.array([0.0, 1.0, 0.0]),
        target_original=target, target_debiased=target.copy(),
        cluster_labels=np.array([[0, 0, 1], [1, 2, 2]]),
        candidate_mask=np.array([[0, 0, 1], [0, 1, 0]], dtype=bool), threshold=0.3,
    )
    assert out["seed_id"] == 1
    assert_allclose(out["cross_similarity"], [1, 0.8, 0.6], atol=1e-12)
    assert_allclose(out["intra_similarity"], [0.8, 1, 0.96], atol=1e-12)
    assert_allclose(out["area_weights"], [0, 1, 0.5], atol=1e-12)
    assert_allclose(out["combined_scores"], [0, 0.8, 0.288], atol=1e-12)
    assert_array_equal(out["final_mask"], [[False, False, True], [True, False, False]])


@pytest.mark.parametrize("threshold,expected", [
    (0.25, [1, 1, 0, 0, 0, 0, 0, 0]),
    (1.0, [0, 0, 0, 0, 0, 0, 0, 0]),
])
def test_author_code_strict_threshold_without_forced_seed_union(solve, threshold, expected):
    inputs = main_case()
    inputs["threshold"] = threshold
    out = solve(**inputs)
    assert_array_equal(out["final_mask"], np.array(expected, dtype=bool).reshape(2, 4))


def test_empty_candidates_return_explicit_empty_result(solve):
    inputs = main_case()
    inputs["candidate_mask"] = np.zeros((2, 4), dtype=bool)
    out = solve(**inputs)
    assert out["seed_id"] is None
    for key in ("cross_similarity", "intra_similarity", "area_weights", "combined_scores"):
        assert_array_equal(out[key], np.zeros(3))
    assert_array_equal(out["final_mask"], np.zeros((2, 4), dtype=bool))
