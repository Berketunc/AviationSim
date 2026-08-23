from oa_control.waypoint_selection import select_replanned_waypoint_index


PATH = [
    (0.1, 0.1, 1.5),
    (0.3, 0.1, 1.5),
    (0.5, 0.1, 1.5),
    (0.7, 0.1, 1.5),
]


def test_new_path_skips_start_voxel():
    assert select_replanned_waypoint_index(PATH, None) == 1


def test_preserves_active_target_within_safe_prefix():
    assert select_replanned_waypoint_index(PATH, PATH[2]) == 2


def test_does_not_preserve_target_deep_in_replanned_path():
    assert select_replanned_waypoint_index(PATH, PATH[3]) == 1


def test_does_not_steer_back_to_new_start_voxel():
    assert select_replanned_waypoint_index(PATH, PATH[0]) == 1


def test_empty_and_singleton_paths():
    assert select_replanned_waypoint_index([], None) == 0
    assert select_replanned_waypoint_index([PATH[0]], None) == 0
