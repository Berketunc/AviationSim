import json

import pytest

from oa_control.trial_recorder import (
    NavigationTrialRecorder,
    obstacle_surface_clearance,
    shield_residual_velocity,
)


def test_signed_obstacle_clearance():
    assert obstacle_surface_clearance((-5.0, 0.0)) == pytest.approx(-0.2)
    assert obstacle_surface_clearance((-4.45, 0.0)) == pytest.approx(0.35)


def test_trial_metrics_and_atomic_json(tmp_path):
    output = tmp_path / "trial.json"
    recorder = NavigationTrialRecorder(
        controller="residual",
        label="paired_01",
        output_path=output,
        timeout_s=10.0,
    )
    recorder.start(2.0, (-8.5, 0.0))
    recorder.set_goal((8.5, 0.0))
    recorder.observe(3.0, (-8.0, 0.0))
    recorder.record_path_update(True)
    recorder.record_path_update(False)
    recorder.record_residual_recovery()
    recorder.record_controller_latency(1_000_000)
    recorder.record_controller_latency(3_000_000)
    recorder.record_residual_inference(500_000, 0.2, shield_blend=0.5)
    result = recorder.finish(
        "success", 4.0, residual_active_at_finish=True
    )

    assert result["schema_version"] == 5
    assert result["landing_outcome"] == "in_progress"
    assert result["duration_s"] == pytest.approx(2.0)
    assert result["path_length_m"] == pytest.approx(0.5)
    assert result["start_position_xy"] == [-8.5, 0.0]
    assert result["final_position_xy"] == [-8.0, 0.0]
    assert result["goal_xy"] == [8.5, 0.0]
    assert result["final_distance_to_goal_m"] == pytest.approx(16.5)
    assert result["controller_latency_ms"]["mean"] == pytest.approx(2.0)
    assert result["residual_inference_latency_ms"]["mean"] == pytest.approx(0.5)
    assert result["mean_residual_mps"] == pytest.approx(0.2)
    assert result["mean_residual_shield_blend"] == pytest.approx(0.5)
    assert result["shield_intervention_count"] == 1
    assert result["residual_goal_handoff_count"] == 0
    assert result["residual_recovery_count"] == 1
    assert result["path_updates"] == {
        "total": 2,
        "nonempty": 1,
        "empty": 1,
    }
    assert json.loads(output.read_text()) == result
    recorder.record_landing_outcome("success")
    assert result["landing_outcome"] == "success"
    assert json.loads(output.read_text()) == result
    assert not output.with_suffix(".json.tmp").exists()


def test_collision_and_timeout_flags():
    recorder = NavigationTrialRecorder(
        controller="classical",
        timeout_s=5.0,
        collision_radius_m=0.35,
    )
    recorder.start(10.0, (-8.5, 0.0))
    recorder.observe(11.0, (-4.45, 0.0))

    assert recorder.collision_detected
    assert not recorder.timed_out(14.99)
    assert recorder.timed_out(15.0)

    result = recorder.finish(
        "collision", 11.0, residual_active_at_finish=False
    )
    assert result["collision"]
    assert not result["success"]
    assert result["landing_outcome"] == "not_started"


def test_residual_shield_preserves_planner_clearance():
    position = (-4.2, 0.0)
    classical = (0.0, 0.2)
    requested_residual = (-0.3, 0.0)

    corrected, applied_residual, blend = shield_residual_velocity(
        position,
        classical,
        requested_residual,
        speed_cap_mps=0.6,
        minimum_clearance_m=0.55,
        lookahead_s=1.0,
    )

    assert 0.0 <= blend < 1.0
    assert corrected == pytest.approx(
        (
            classical[0] + applied_residual[0],
            classical[1] + applied_residual[1],
        )
    )
    projected = (position[0] + corrected[0], position[1] + corrected[1])
    assert obstacle_surface_clearance(projected) >= 0.55 - 1.0e-6


def test_residual_shield_honors_terminal_handoff():
    classical = (0.2, -0.1)
    corrected, applied_residual, blend = shield_residual_velocity(
        (7.5, 0.0),
        classical,
        (0.3, 0.3),
        speed_cap_mps=0.6,
        minimum_clearance_m=0.65,
        lookahead_s=1.0,
        maximum_blend=0.0,
    )

    assert corrected == pytest.approx(classical)
    assert applied_residual == pytest.approx((0.0, 0.0))
    assert blend == 0.0


def test_residual_shield_tapers_at_clearance_boundary():
    classical = (0.0, 0.2)
    corrected, applied_residual, blend = shield_residual_velocity(
        (-4.45, 0.0),
        classical,
        (-0.3, 0.0),
        speed_cap_mps=0.6,
        minimum_clearance_m=0.65,
        lookahead_s=1.0,
    )

    assert corrected == pytest.approx(classical)
    assert applied_residual == pytest.approx((0.0, 0.0))
    assert blend == 0.0
