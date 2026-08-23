import pytest

from oa_control.landing_safety import should_handoff_to_land


def test_marker_distance_can_trigger_handoff():
    assert should_handoff_to_land(0.49, 0.8, 0.5)


def test_odometry_altitude_covers_close_range_marker_loss():
    assert should_handoff_to_land(None, 0.01, 0.5)


def test_handoff_waits_while_both_measurements_are_high():
    assert not should_handoff_to_land(0.75, 0.8, 0.5)


def test_invalid_or_negative_measurements_do_not_trigger():
    assert not should_handoff_to_land(float("nan"), -0.01, 0.5)


def test_threshold_must_be_positive():
    with pytest.raises(ValueError):
        should_handoff_to_land(0.1, 0.1, 0.0)
