import pytest

from pl_perception.camera_geometry import optical_translation_to_body_frd


def test_live_downward_camera_transform():
    assert optical_translation_to_body_frd((0.2, -0.3, 1.4)) == pytest.approx(
        (0.3, 0.2, 1.4)
    )


def test_rejects_invalid_translation():
    with pytest.raises(ValueError):
        optical_translation_to_body_frd((0.0, 1.0))
    with pytest.raises(ValueError):
        optical_translation_to_body_frd((0.0, float("nan"), 1.0))
