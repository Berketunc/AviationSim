from pathlib import Path

import numpy as np
import pytest

from oa_control.residual_policy import (
    ResidualPolicy,
    ResidualPolicyError,
    build_observation,
)


WEIGHTS = Path(__file__).parents[1] / "models" / "residual_actor_weights.npz"


def test_observation_contract():
    observation = build_observation(
        position_xy=(0.0, 0.0),
        velocity_world_xy=(0.2, -0.1),
        goal_xy=(8.5, 0.0),
        classical_velocity_world_xy=(0.4, 0.0),
        velocity_scale=0.4,
    )

    np.testing.assert_allclose(
        observation[:6],
        [8.5, 0.0, 0.5, -0.25, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        observation[6:18],
        [
            0.0,
            -1.5,
            0.0,
            1.5,
            0.0,
            -4.5,
            0.0,
            4.5,
            -5.0,
            0.0,
            5.0,
            0.0,
        ],
    )
    np.testing.assert_allclose(observation[18:], [10.0, 10.0, 7.0, 7.0])


def test_exported_actor_golden_output():
    policy = ResidualPolicy(WEIGHTS)
    output = policy.predict(np.zeros(22, dtype=np.float32))
    np.testing.assert_allclose(
        output,
        [0.30059314, -0.01672308],
        rtol=0.0,
        atol=1.0e-7,
    )


def test_correction_respects_combined_speed_cap():
    policy = ResidualPolicy(WEIGHTS)
    observation = np.zeros(22, dtype=np.float32)
    corrected, residual, action = policy.correct_velocity(
        observation,
        classical_velocity_world_xy=(0.4, 0.0),
        residual_scale_mps=0.3,
        combined_speed_cap_mps=0.2,
    )

    assert corrected.shape == residual.shape == action.shape == (2,)
    assert np.linalg.norm(corrected) <= 0.200001
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_rejects_invalid_observation_shape():
    policy = ResidualPolicy(WEIGHTS)
    with pytest.raises(ResidualPolicyError):
        policy.predict(np.zeros(21, dtype=np.float32))
