import numpy as np
import pytest

from qwen38_exl3_k3_weights import (
    bf16_bytes_to_float32,
    cosine_from_moments,
    cosine_moments,
    source_public_matrix,
    stratified_expert_ids,
)


def test_bf16_bytes_convert_by_zero_extending_the_significand():
    raw = np.array([0x3F80, 0xBF80, 0x3F81], dtype="<u2").tobytes()

    converted = bf16_bytes_to_float32(raw)

    assert converted.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(converted, np.array([1.0, -1.0, 1.0078125], dtype=np.float32))


def test_source_fused_gate_up_is_split_before_transpose():
    source = np.array(
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
            [40, 41, 42],
        ],
        dtype=np.float32,
    )

    gate = source_public_matrix(source, "gate", gate_width=2)
    up = source_public_matrix(source, "up", gate_width=2)

    np.testing.assert_array_equal(gate, np.array([[10, 20], [11, 21], [12, 22]], dtype=np.float32))
    np.testing.assert_array_equal(up, np.array([[30, 40], [31, 41], [32, 42]], dtype=np.float32))


def test_source_down_is_transposed_to_public_in_out_orientation():
    source = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    np.testing.assert_array_equal(
        source_public_matrix(source, "down"),
        np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32),
    )


def test_cosine_uses_float64_moments_and_can_combine_matrices():
    source_a = np.array([1.0, 0.0], dtype=np.float32)
    target_a = np.array([1.0, 1.0], dtype=np.float32)
    source_b = np.array([0.0, 2.0], dtype=np.float32)
    target_b = np.array([0.0, 2.0], dtype=np.float32)

    moments_a = cosine_moments(source_a, target_a)
    moments_b = cosine_moments(source_b, target_b)
    combined = tuple(x + y for x, y in zip(moments_a, moments_b))

    expected = np.concatenate([source_a, source_b])
    expected_target = np.concatenate([target_a, target_b])
    np.testing.assert_allclose(
        cosine_from_moments(*combined),
        cosine_from_moments(*cosine_moments(expected, expected_target)),
        rtol=0.0,
        atol=1e-15,
    )


def test_cosine_rejects_nonfinite_and_zero_norm_inputs():
    with pytest.raises(ValueError, match="non-finite"):
        cosine_moments(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="zero-norm"):
        cosine_moments(np.zeros(2), np.ones(2))
    with pytest.raises(ValueError, match="zero-norm"):
        cosine_from_moments(0.0, 1.0, 0.0)


def test_expert_sample_is_four_quartile_midpoints():
    assert stratified_expert_ids(512, 4) == (63, 191, 319, 447)
