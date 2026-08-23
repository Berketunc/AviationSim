#!/usr/bin/env python3
"""
MAVSDK trajectory follower for obstacle-avoidance path planning.

Takes off, then continuously walks the latest oa_planning Path in order,
advancing through waypoints as each is reached, and drives body-frame
velocity setpoints toward the current one via the same MavsdkBridge
pl_control uses for the precision landing controller — no ArUco/landing-
specific logic in that bridge, so it's reused as-is rather than duplicated.

Every new Path starts with the voxel containing the current position,
because oa_planning_node always plans from wherever the vehicle currently is.
The follower skips that first voxel-center waypoint. If its active target is
still one of the first two safe waypoints in the replacement path, it preserves
that exact target instead of resetting progress every second. It never searches
the full path for the spatially nearest waypoint: that older behavior could
select a point on the far side of an obstacle and cut through it.

Once the A* goal is reached, this hands off from waypoint-following to
ArUco-marker landing (SEARCH_MARKER -> ALIGN_MARKER -> DESCEND_MARKER),
reusing pl_control.landing_controller_node's PIController and align/descend
approach directly (same airframe, same MavsdkBridge, same marker) — but
embedded in this node's own state machine rather than a second MAVSDK-
connected node, since two nodes can't safely share one offboard control
loop on the same vehicle.
"""

import math
import threading
import time
from enum import Enum, auto
from pathlib import Path as FilesystemPath

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool

from oa_control.landing_safety import should_handoff_to_land
from oa_control.residual_policy import (
    ResidualPolicy,
    ResidualPolicyError,
    build_observation,
)
from pl_control.landing_controller_node import PIController
from oa_control.trial_recorder import (
    NavigationTrialRecorder,
    shield_residual_velocity,
)
from oa_control.waypoint_selection import select_replanned_waypoint_index
from pl_control.mavsdk_bridge import MavsdkBridge


class State(Enum):
    INIT = auto()
    FOLLOW = auto()
    SEARCH_MARKER = auto()
    ALIGN_MARKER = auto()
    DESCEND_MARKER = auto()
    LANDED = auto()
    ABORT = auto()


def _yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PathFollowerNode(Node):

    def __init__(self):
        super().__init__('path_follower_node')

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter('takeoff_altitude_m', 1.5)
        self.declare_parameter('cruise_speed_ms', 0.8)
        self.declare_parameter('waypoint_reached_radius_m', 0.3)
        self.declare_parameter('goal_reached_radius_m', 0.3)
        self.declare_parameter('mavsdk_address', 'udpin://0.0.0.0:14540')
        self.declare_parameter('timer_hz', 20.0)
        self.declare_parameter('path_topic', '/oa/path')
        self.declare_parameter('odom_topic', '/oa/odom')

        self.cruise_speed: float = self.get_parameter('cruise_speed_ms').value
        self.waypoint_radius: float = self.get_parameter('waypoint_reached_radius_m').value
        self.goal_radius: float = self.get_parameter('goal_reached_radius_m').value
        self.takeoff_alt: float = self.get_parameter('takeoff_altitude_m').value
        mavsdk_addr: str = self.get_parameter('mavsdk_address').value
        timer_hz: float = self.get_parameter('timer_hz').value

        self.declare_parameter('hold_position_kp', 0.5)
        self.declare_parameter('hold_position_max_speed_ms', 0.3)
        self.hold_kp: float = self.get_parameter('hold_position_kp').value
        self.hold_max_speed: float = self.get_parameter('hold_position_max_speed_ms').value

        # ── optional residual correction (disabled = original controller) ─────
        self.declare_parameter('residual_enabled', False)
        self.declare_parameter('residual_policy_path', '')
        self.declare_parameter('residual_training_classical_speed_ms', 1.0)
        self.declare_parameter('residual_training_scale_ms', 0.75)
        self.declare_parameter('residual_training_speed_cap_ms', 1.5)
        self.declare_parameter('residual_deployment_authority', 0.5)
        self.declare_parameter('residual_velocity_filter_alpha', 0.35)
        self.declare_parameter('residual_status_log_period_s', 10.0)
        self.declare_parameter('residual_planner_clearance_m', 0.65)
        self.declare_parameter('residual_clearance_lookahead_s', 1.0)
        self.declare_parameter('residual_clearance_release_margin_m', 0.35)
        self.declare_parameter('residual_goal_handoff_radius_m', 1.5)

        residual_requested = bool(self.get_parameter('residual_enabled').value)
        self.residual_enabled = False
        self.residual_policy: ResidualPolicy | None = None
        self.residual_training_classical_speed = float(
            self.get_parameter('residual_training_classical_speed_ms').value)
        self.residual_training_scale = float(
            self.get_parameter('residual_training_scale_ms').value)
        self.residual_training_speed_cap = float(
            self.get_parameter('residual_training_speed_cap_ms').value)
        self.residual_deployment_authority = float(
            self.get_parameter('residual_deployment_authority').value)
        self.residual_velocity_filter_alpha = float(
            self.get_parameter('residual_velocity_filter_alpha').value)
        self.residual_status_log_period_s = float(
            self.get_parameter('residual_status_log_period_s').value)
        self.residual_planner_clearance = float(
            self.get_parameter('residual_planner_clearance_m').value)
        self.residual_clearance_lookahead = float(
            self.get_parameter('residual_clearance_lookahead_s').value)
        self.residual_clearance_release_margin = float(
            self.get_parameter('residual_clearance_release_margin_m').value)
        self.residual_goal_handoff_radius = float(
            self.get_parameter('residual_goal_handoff_radius_m').value)
        self.residual_velocity_scale = 1.0
        self.residual_scale = 0.0
        self.residual_speed_cap = self.cruise_speed

        if residual_requested:
            try:
                if self.cruise_speed <= 0.0:
                    raise ResidualPolicyError('cruise_speed_ms must be positive')
                if self.residual_training_classical_speed <= 0.0:
                    raise ResidualPolicyError(
                        'residual_training_classical_speed_ms must be positive')
                if self.residual_training_scale < 0.0:
                    raise ResidualPolicyError(
                        'residual_training_scale_ms must be non-negative')
                if self.residual_training_speed_cap <= 0.0:
                    raise ResidualPolicyError(
                        'residual_training_speed_cap_ms must be positive')
                if not 0.0 < self.residual_deployment_authority <= 1.0:
                    raise ResidualPolicyError(
                        'residual_deployment_authority must be in (0, 1]')
                if not 0.0 < self.residual_velocity_filter_alpha <= 1.0:
                    raise ResidualPolicyError(
                        'residual_velocity_filter_alpha must be in (0, 1]')
                if self.residual_planner_clearance <= 0.0:
                    raise ResidualPolicyError(
                        'residual_planner_clearance_m must be positive')
                if self.residual_clearance_lookahead <= 0.0:
                    raise ResidualPolicyError(
                        'residual_clearance_lookahead_s must be positive')
                if self.residual_clearance_release_margin <= 0.0:
                    raise ResidualPolicyError(
                        'residual_clearance_release_margin_m must be positive')
                if self.residual_goal_handoff_radius <= self.goal_radius:
                    raise ResidualPolicyError(
                        'residual_goal_handoff_radius_m must exceed '
                        'goal_reached_radius_m')
                configured_path = str(
                    self.get_parameter('residual_policy_path').value).strip()
                policy_path = (
                    FilesystemPath(configured_path).expanduser()
                    if configured_path
                    else FilesystemPath(get_package_share_directory('oa_control'))
                    / 'models'
                    / 'residual_actor_weights.npz'
                )
                self.residual_policy = ResidualPolicy(policy_path)
                self.residual_velocity_scale = (
                    self.cruise_speed / self.residual_training_classical_speed)
                self.residual_scale = (
                    self.residual_training_scale
                    * self.residual_velocity_scale
                    * self.residual_deployment_authority
                )
                self.residual_speed_cap = (
                    self.residual_training_speed_cap * self.residual_velocity_scale)
                self.residual_enabled = True
                self.get_logger().info(
                    'Residual correction enabled: '
                    f'policy={self.residual_policy.weights_path}, '
                    f'authority={self.residual_deployment_authority:.2f}, '
                    f'residual_scale={self.residual_scale:.3f}m/s, '
                    f'combined_cap={self.residual_speed_cap:.3f}m/s, '
                    f'clearance_guard={self.residual_planner_clearance:.3f}m, '
                    f'lookahead={self.residual_clearance_lookahead:.2f}s, '
                    f'goal_handoff={self.residual_goal_handoff_radius:.2f}m')
            except Exception as exc:
                self.residual_policy = None
                self.get_logger().error(
                    f'Residual initialization failed ({exc}); '
                    'continuing with the unchanged classical follower.')

        # ── optional paired-transfer trial recorder ─────────────────────────
        self.declare_parameter('trial_metrics_enabled', False)
        self.declare_parameter('trial_label', '')
        self.declare_parameter('trial_output_path', '')
        self.declare_parameter('trial_timeout_s', 90.0)
        self.declare_parameter('trial_collision_radius_m', 0.35)
        self.trial_recorder: NavigationTrialRecorder | None = None

        if bool(self.get_parameter('trial_metrics_enabled').value):
            try:
                output_path = str(
                    self.get_parameter('trial_output_path').value).strip()
                self.trial_recorder = NavigationTrialRecorder(
                    controller='residual' if self.residual_enabled else 'classical',
                    label=str(self.get_parameter('trial_label').value),
                    output_path=output_path or None,
                    timeout_s=float(self.get_parameter('trial_timeout_s').value),
                    collision_radius_m=float(
                        self.get_parameter('trial_collision_radius_m').value),
                )
                self.get_logger().info(
                    'Navigation trial metrics enabled: '
                    f'controller={self.trial_recorder.controller}, '
                    f'label={self.trial_recorder.label!r}, '
                    f'output={self.trial_recorder.output_path}')
            except (TypeError, ValueError) as exc:
                self.trial_recorder = None
                self.get_logger().error(
                    f'Trial recorder initialization failed ({exc}); '
                    'flight control will continue without trial metrics.')

        # ── marker-landing parameters (see module docstring) ────────────────────
        self.declare_parameter('marker_pose_topic', '/oa/landing/target_pose')
        self.declare_parameter('marker_visible_topic', '/oa/landing/is_visible')
        self.declare_parameter('marker_pi_kp', 0.5)
        self.declare_parameter('marker_pi_ki', 0.05)
        self.declare_parameter('marker_pi_integral_max', 0.5)
        self.declare_parameter('marker_horizontal_vel_max', 1.0)
        self.declare_parameter('marker_descend_vel_ms', 0.3)
        self.declare_parameter('marker_hacc_radius_m', 0.20)
        self.declare_parameter('marker_n_frames_aligned', 10)
        self.declare_parameter('marker_final_land_alt_m', 0.5)
        self.declare_parameter('marker_search_speed_ms', 0.2)
        self.declare_parameter('marker_search_step_m', 0.5)
        self.declare_parameter('marker_search_timeout_s', 20.0)
        self.declare_parameter('marker_lost_timeout_s', 2.0)

        marker_pi_kp = self.get_parameter('marker_pi_kp').value
        marker_pi_ki = self.get_parameter('marker_pi_ki').value
        marker_pi_i_max = self.get_parameter('marker_pi_integral_max').value
        marker_v_max = self.get_parameter('marker_horizontal_vel_max').value
        self.marker_pi_x = PIController(marker_pi_kp, marker_pi_ki, marker_pi_i_max, marker_v_max)
        self.marker_pi_y = PIController(marker_pi_kp, marker_pi_ki, marker_pi_i_max, marker_v_max)

        self.marker_descend_vel: float = self.get_parameter('marker_descend_vel_ms').value
        self.marker_hacc_radius_m: float = self.get_parameter('marker_hacc_radius_m').value
        self.marker_n_frames_aligned: int = self.get_parameter('marker_n_frames_aligned').value
        self.marker_final_land_alt_m: float = self.get_parameter('marker_final_land_alt_m').value
        self.marker_search_speed: float = self.get_parameter('marker_search_speed_ms').value
        self.marker_search_step_m: float = self.get_parameter('marker_search_step_m').value
        self.marker_search_timeout_s: float = self.get_parameter('marker_search_timeout_s').value
        self.marker_lost_timeout_s: float = self.get_parameter('marker_lost_timeout_s').value

        # ── state ─────────────────────────────────────────────────────────────
        self.state = State.INIT
        self.path: Path | None = None
        self.waypoint_idx = 0
        self.current_pos = None   # (x, y, z)
        self.current_velocity_world = None  # filtered finite-difference (vx, vy)
        self._last_odom_xy = None
        self._last_odom_time_s = None
        self.current_yaw = 0.0
        self.hold_pos = None      # anchor point while holding (see _hold_position)
        self.last_tick_time = time.monotonic()
        self._residual_inference_count = 0
        self._residual_inference_total_ns = 0
        self._residual_magnitude_sum = 0.0
        self._residual_shield_blend_sum = 0.0
        self._residual_shield_intervention_count = 0
        self._residual_last_status_time = time.monotonic()

        self.marker_pose: PoseStamped | None = None
        self.marker_visible = False
        self.marker_state_entry_time = time.monotonic()
        self.marker_last_seen_time = time.monotonic()
        self.marker_aligned_count = 0
        self.marker_search_timed_out = False
        self._marker_last_status_time = 0.0

        # ── bridge & subscriptions ────────────────────────────────────────────
        self.bridge = MavsdkBridge(system_address=mavsdk_addr)

        self.create_subscription(
            Path, self.get_parameter('path_topic').value, self._path_cb, 1)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self._odom_cb, 10)
        self.create_subscription(
            PoseStamped, self.get_parameter('marker_pose_topic').value, self._marker_pose_cb, 10)
        self.create_subscription(
            Bool, self.get_parameter('marker_visible_topic').value, self._marker_visible_cb, 10)

        # ── startup on background thread (keeps rclpy spin unblocked) ─────────
        threading.Thread(target=self._startup, daemon=True).start()

        self.create_timer(1.0 / timer_hz, self._tick)

        self.get_logger().info('path_follower_node initialized')

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _path_cb(self, msg: Path):
        previous_path_available = bool(self.path and self.path.poses)
        previous_target = None
        if (
            previous_path_available
            and 0 <= self.waypoint_idx < len(self.path.poses)
        ):
            target = self.path.poses[self.waypoint_idx].pose.position
            previous_target = (target.x, target.y, target.z)
        self.path = msg
        if (
            self.residual_enabled
            and previous_path_available
            and not msg.poses
        ):
            self.residual_enabled = False
            self.residual_policy = None
            if self.trial_recorder is not None:
                self.trial_recorder.mark_residual_fallback()
            self.get_logger().error(
                'Empty replan received while residual correction was active; '
                'residual is permanently disabled and the classical follower '
                'will resume when a valid path returns.')
        if self.trial_recorder is not None:
            self.trial_recorder.record_path_update(bool(msg.poses))
            if msg.poses:
                goal = msg.poses[-1].pose.position
                self.trial_recorder.set_goal((goal.x, goal.y))
        # A* includes the current start voxel as path[0]. Preserve the active
        # target only if it remains within the first two safe steps of the new
        # path. This prevents 1 Hz replans from continually resetting progress,
        # without allowing a nearest-point search to jump across an obstacle.
        waypoint_xyz = [
            (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
            for pose in msg.poses
        ]
        self.waypoint_idx = select_replanned_waypoint_index(
            waypoint_xyz,
            previous_target,
        )

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        position = (float(p.x), float(p.y), float(p.z))
        stamp = msg.header.stamp
        stamp_s = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        if stamp_s <= 0.0:
            stamp_s = self.get_clock().now().nanoseconds * 1.0e-9

        if self._last_odom_xy is not None and self._last_odom_time_s is not None:
            dt = stamp_s - self._last_odom_time_s
            if 1.0e-4 < dt <= 1.0:
                raw_velocity = (
                    (position[0] - self._last_odom_xy[0]) / dt,
                    (position[1] - self._last_odom_xy[1]) / dt,
                )
                if math.hypot(*raw_velocity) <= 10.0:
                    if self.current_velocity_world is None:
                        self.current_velocity_world = raw_velocity
                    else:
                        alpha = self.residual_velocity_filter_alpha
                        self.current_velocity_world = (
                            alpha * raw_velocity[0]
                            + (1.0 - alpha) * self.current_velocity_world[0],
                            alpha * raw_velocity[1]
                            + (1.0 - alpha) * self.current_velocity_world[1],
                        )

        self._last_odom_xy = position[:2]
        self._last_odom_time_s = stamp_s
        self.current_pos = position
        self.current_yaw = _yaw_from_quaternion(msg.pose.pose.orientation)

        if self.trial_recorder is not None and self.state == State.FOLLOW:
            try:
                self.trial_recorder.observe(stamp_s, position[:2])
            except ValueError as exc:
                self.get_logger().error(
                    f'Trial recorder rejected odometry ({exc}); '
                    'disabling metrics without changing flight control.')
                self.trial_recorder = None

    def _marker_pose_cb(self, msg: PoseStamped):
        self.marker_pose = msg

    def _marker_visible_cb(self, msg: Bool):
        self.marker_visible = msg.data

    # ── startup (background thread) ───────────────────────────────────────────

    def _startup(self):
        try:
            self.get_logger().info('Connecting to PX4 via MAVSDK...')
            self.bridge.run(self.bridge.connect(), timeout=30.0)

            self.get_logger().info(f'Connected. Arming and taking off to {self.takeoff_alt}m...')
            self.bridge.run(self.bridge.arm_and_takeoff(self.takeoff_alt), timeout=90.0)

            self.get_logger().info('Altitude reached. Starting offboard velocity loop...')
            self.bridge.start_offboard_loop()
            time.sleep(1.0)  # brief settle before handing off to the follower

            self.state = State.FOLLOW
        except Exception as exc:
            self.get_logger().error(f'Startup failed: {exc}')
            self.state = State.ABORT

    # ── control tick (main thread, ROS timer) ─────────────────────────────────

    def _tick(self):
        if self.state in (State.INIT, State.ABORT, State.LANDED):
            return

        now = time.monotonic()
        dt = min(now - self.last_tick_time, 0.5)
        self.last_tick_time = now

        if self.state == State.FOLLOW:
            if (
                self.trial_recorder is not None
                and self.trial_recorder.started
                and self._last_odom_time_s is not None
            ):
                if self.trial_recorder.collision_detected:
                    self._stop_failed_navigation_trial('collision')
                    return
                if self.trial_recorder.timed_out(self._last_odom_time_s):
                    self._stop_failed_navigation_trial('time_out')
                    return
            self._do_follow()
        elif self.state == State.SEARCH_MARKER:
            self._do_marker_search(now)
        elif self.state == State.ALIGN_MARKER:
            self._do_marker_align(now, dt)
        elif self.state == State.DESCEND_MARKER:
            self._do_marker_descend(now, dt)

    def _do_follow(self):
        if self.path is None or not self.path.poses or self.current_pos is None:
            self._hold_position()
            return

        decision_started_ns = time.perf_counter_ns()
        target = self._advance_to_current_waypoint()
        if target is None:
            self._finish_navigation_trial('success')
            self.get_logger().info('Goal reached — searching for landing marker.')
            self._enter_marker_search()
            return

        self.hold_pos = None  # actively navigating again; drop any stale anchor
        cx, cy, cz = self.current_pos
        dx, dy, dz = target[0] - cx, target[1] - cy, target[2] - cz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < 1e-6:
            self.bridge.send_velocity_body(0.0, 0.0, 0.0, 0.0)
            return

        speed = min(self.cruise_speed, dist)
        wx, wy, wz = (dx / dist) * speed, (dy / dist) * speed, (dz / dist) * speed
        if self.residual_enabled:
            wx, wy = self._apply_residual_correction(wx, wy)

        self._send_world_velocity(wx, wy, wz)
        if self.trial_recorder is not None:
            self.trial_recorder.record_controller_latency(time.perf_counter_ns() - decision_started_ns)

    def _apply_residual_correction(
        self, classical_wx: float, classical_wy: float
    ) -> tuple[float, float]:
        """Apply the learned correction or return the classical command."""

        if (
            self.residual_policy is None
            or self.current_velocity_world is None
            or self.current_pos is None
            or self.path is None
            or not self.path.poses
        ):
            return classical_wx, classical_wy

        goal = self.path.poses[-1].pose.position
        started_ns = time.perf_counter_ns()
        try:
            observation = build_observation(
                self.current_pos[:2],
                self.current_velocity_world,
                (goal.x, goal.y),
                (classical_wx, classical_wy),
                velocity_scale=self.residual_velocity_scale,
            )
            corrected, residual, _ = self.residual_policy.correct_velocity(
                observation,
                (classical_wx, classical_wy),
                residual_scale_mps=self.residual_scale,
                combined_speed_cap_mps=self.residual_speed_cap,
            )
            goal_distance_xy = math.hypot(
                goal.x - self.current_pos[0],
                goal.y - self.current_pos[1],
            )
            goal_handoff = (
                goal_distance_xy <= self.residual_goal_handoff_radius
            )
            corrected, residual, shield_blend = shield_residual_velocity(
                self.current_pos[:2],
                (classical_wx, classical_wy),
                residual,
                speed_cap_mps=self.residual_speed_cap,
                minimum_clearance_m=self.residual_planner_clearance,
                lookahead_s=self.residual_clearance_lookahead,
                clearance_release_margin_m=(
                    self.residual_clearance_release_margin
                ),
                maximum_blend=0.0 if goal_handoff else 1.0,
            )
        except Exception as exc:
            self.residual_enabled = False
            self.residual_policy = None
            if self.trial_recorder is not None:
                self.trial_recorder.mark_residual_fallback()
            self.get_logger().error(
                f'Residual inference failed ({exc}); correction is now disabled '
                'and the unchanged classical follower remains active.')
            return classical_wx, classical_wy

        elapsed_ns = time.perf_counter_ns() - started_ns
        self._residual_inference_count += 1
        self._residual_inference_total_ns += elapsed_ns
        residual_magnitude = math.hypot(
            float(residual[0]), float(residual[1]))
        self._residual_magnitude_sum += residual_magnitude
        self._residual_shield_blend_sum += shield_blend
        if shield_blend < 1.0 - 1.0e-6:
            self._residual_shield_intervention_count += 1
        if self.trial_recorder is not None:
            self.trial_recorder.record_residual_inference(
                elapsed_ns,
                residual_magnitude,
                shield_blend,
                goal_handoff=goal_handoff,
            )

        now = time.monotonic()
        if (
            self.residual_status_log_period_s > 0.0
            and now - self._residual_last_status_time
            >= self.residual_status_log_period_s
        ):
            count = self._residual_inference_count
            mean_latency_ms = self._residual_inference_total_ns / count / 1.0e6
            mean_residual = self._residual_magnitude_sum / count
            mean_shield_blend = self._residual_shield_blend_sum / count
            self.get_logger().info(
                'Residual status: '
                f'samples={count}, mean_latency={mean_latency_ms:.3f}ms, '
                f'mean_correction={mean_residual:.3f}m/s, '
                f'mean_shield_blend={mean_shield_blend:.3f}, '
                f'shield_interventions={self._residual_shield_intervention_count}')
            self._residual_last_status_time = now

        return float(corrected[0]), float(corrected[1])

    # ── marker landing: SEARCH_MARKER -> ALIGN_MARKER -> DESCEND_MARKER ────────
    #
    # A* always plans to a goal placed exactly at the marker's (x, y) (see
    # planner_params.yaml), so on arrival the marker should already be ~underfoot
    # — SEARCH_MARKER mostly just waits out detector/tracking latency, with a
    # bounded expanding-square scan as a fallback for residual position error
    # rather than the wide-open-ground hunt pl_control's Milestone-1 search does.

    # Body-frame unit vectors for the expanding-square fallback scan: forward,
    # right, back, left.
    _MARKER_SCAN_DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    _MARKER_SEARCH_GRACE_S = 2.0

    def _enter_marker_search(self):
        self.state = State.SEARCH_MARKER
        self.marker_state_entry_time = time.monotonic()
        self.marker_last_seen_time = time.monotonic()
        self.marker_search_timed_out = False
        self.hold_pos = self.current_pos

    def _do_marker_search(self, now: float):
        if self.marker_visible and self.marker_pose is not None:
            self.get_logger().info('Landing marker acquired — aligning.')
            self._enter_marker_align()
            return

        elapsed = now - self.marker_state_entry_time
        if elapsed > self.marker_search_timeout_s:
            if not self.marker_search_timed_out:
                self.marker_search_timed_out = True
                self.get_logger().error(
                    'Landing marker not found within '
                    f'{self.marker_search_timeout_s}s of the goal — '
                    'holding position rather than landing blind.')
                if (
                    self.trial_recorder is not None
                    and self.trial_recorder.finished
                ):
                    self.trial_recorder.record_landing_outcome(
                        'marker_not_found')
            self._hold_position()
            return

        if elapsed < self._MARKER_SEARCH_GRACE_S:
            self._hold_position()
            return

        step_s = self.marker_search_step_m / self.marker_search_speed
        scan_elapsed = elapsed - self._MARKER_SEARCH_GRACE_S
        leg, t = 0, 0.0
        while True:
            leg_dur = (leg // 2 + 1) * step_s
            if t + leg_dur > scan_elapsed:
                break
            t += leg_dur
            leg += 1

        dx, dy = self._MARKER_SCAN_DIRS[leg % 4]
        self.bridge.send_velocity_body(
            dx * self.marker_search_speed, dy * self.marker_search_speed,
            self._altitude_hold_vz_down(), 0.0)

    def _enter_marker_align(self):
        self.state = State.ALIGN_MARKER
        self.marker_pi_x.reset()
        self.marker_pi_y.reset()
        self.marker_aligned_count = 0

    def _altitude_hold_vz_down(self) -> float:
        """Active altitude correction toward hold_pos's z (set at
        _enter_marker_search, i.e. the altitude the goal was reached at),
        in send_velocity_body's `down` convention (positive = descend).

        SEARCH_MARKER's scan and ALIGN_MARKER both otherwise command xy
        velocity directly rather than through _hold_position(), and a raw
        vz=0 setpoint isn't held perfectly by PX4 (see _hold_position()'s
        own docstring) — it sinks toward the floor. That was a brief,
        single-tick bug the first time it was found (the GOAL_REACHED tick
        guard fix); here it ran unnoticed for up to ~20s and was confirmed
        live to sink the vehicle all the way to the floor (z=0.003) while
        drifting over a meter off goal in xy — which is also what pushed
        the marker out of frame during a 12s ALIGN_MARKER that should have
        converged in under a second, well before SEARCH_MARKER's own copy
        of the same bug ever ran.
        """
        if self.current_pos is None or self.hold_pos is None:
            return 0.0
        ez = self.hold_pos[2] - self.current_pos[2]
        vz_up = max(-self.hold_max_speed, min(self.hold_max_speed, self.hold_kp * ez))
        return -vz_up

    def _do_marker_align(self, now: float, dt: float):
        if not self._marker_visible_or_revert(now):
            return

        dx = self.marker_pose.pose.position.x
        dy = self.marker_pose.pose.position.y
        error = math.hypot(dx, dy)

        vx = self.marker_pi_x.update(dx, dt)
        vy = self.marker_pi_y.update(dy, dt)
        self.bridge.send_velocity_body(
            vx, vy, self._altitude_hold_vz_down(), 0.0)
        self._log_marker_status(now, 'ALIGN', dx, dy, error)

        if error < self.marker_hacc_radius_m:
            self.marker_aligned_count += 1
        else:
            self.marker_aligned_count = 0

        if self.marker_aligned_count >= self.marker_n_frames_aligned:
            self.state = State.DESCEND_MARKER
            self.marker_pi_x.reset()
            self.marker_pi_y.reset()

    def _do_marker_descend(self, now: float, dt: float):
        odometry_altitude = (
            self.current_pos[2] if self.current_pos is not None else None
        )
        if should_handoff_to_land(
            None,
            odometry_altitude,
            self.marker_final_land_alt_m,
        ):
            self._begin_land(
                "Odometry altitude "
                f"{odometry_altitude:.3f}m is below the "
                f"{self.marker_final_land_alt_m:.3f}m handoff threshold.")
            return

        if not self._marker_visible_or_revert(now):
            return

        dx = self.marker_pose.pose.position.x
        dy = self.marker_pose.pose.position.y
        dz = self.marker_pose.pose.position.z   # AGL distance to marker (optical z)
        error = math.hypot(dx, dy)

        vx = self.marker_pi_x.update(dx, dt)
        vy = self.marker_pi_y.update(dy, dt)

        # Only descend when on-target; pause vz if drifting (do NOT reverse).
        if error < self.marker_hacc_radius_m * 2.0:
            vz = min(self.marker_descend_vel, dz * 0.3)
        else:
            vz = 0.0

        self.bridge.send_velocity_body(vx, vy, vz, 0.0)
        self._log_marker_status(now, 'DESCEND', dx, dy, error, dz)

        if should_handoff_to_land(
            dz,
            odometry_altitude,
            self.marker_final_land_alt_m,
        ):
            self._begin_land(
                "Marker/odometry height reached the "
                f"{self.marker_final_land_alt_m:.3f}m handoff threshold.")

    def _begin_land(self, reason: str):
        if self.state == State.LANDED:
            return
        self.get_logger().info(f"{reason} Handing off to Action.land().")
        self.state = State.LANDED
        threading.Thread(target=self._land_async, daemon=True).start()

    def _log_marker_status(
        self,
        now: float,
        phase: str,
        dx: float,
        dy: float,
        error: float,
        dz: float | None = None,
    ):
        if now - self._marker_last_status_time < 1.0:
            return
        altitude = f', dz={dz:.3f}m' if dz is not None else ''
        self.get_logger().info(
            f'Marker {phase}: dx={dx:+.3f}m, dy={dy:+.3f}m, '
            f'error={error:.3f}m{altitude}, '
            f'aligned_frames={self.marker_aligned_count}')
        self._marker_last_status_time = now

    def _marker_visible_or_revert(self, now: float) -> bool:
        """Return True if the marker is currently visible; otherwise, once
        it's been gone longer than marker_lost_timeout_s, revert to
        SEARCH_MARKER rather than continuing to align/descend on a stale
        pose."""
        if self.marker_visible and self.marker_pose is not None:
            self.marker_last_seen_time = now
            return True
        # Never leave the last alignment velocity latched while visual
        # feedback is absent. The MAVSDK loop repeats its most recent command,
        # so an intermittent detection would otherwise drive the marker out
        # of frame before the lost-marker timeout expires.
        self.bridge.send_velocity_body(
            0.0, 0.0, self._altitude_hold_vz_down(), 0.0)
        lost_for = now - self.marker_last_seen_time
        if lost_for > self.marker_lost_timeout_s:
            self.get_logger().warn(f'Landing marker absent {lost_for:.1f}s — reverting to search.')
            self._enter_marker_search()
        return False

    def _land_async(self):
        try:
            self.bridge.run(self.bridge.land(), timeout=60.0)
            if self.trial_recorder is not None and self.trial_recorder.finished:
                self.trial_recorder.record_landing_outcome('success')
            self.get_logger().info('Landed successfully.')
        except Exception as exc:
            if self.trial_recorder is not None and self.trial_recorder.finished:
                self.trial_recorder.record_landing_outcome('land_failed')
            self.get_logger().error(f'Action.land() failed: {exc}')

    def _hold_position(self):
        """Actively station-keep at a fixed anchor point, rather than just
        sending zero velocity: PX4 doesn't hold altitude perfectly under a
        constant zero-velocity setpoint, and over an extended hold (e.g.
        while the planner repeatedly fails to find a path) that drift is
        enough to settle onto the floor — which then reads as "occupied",
        permanently blocking any further path from being found at all."""
        if self.current_pos is None:
            return
        if self.hold_pos is None:
            self.hold_pos = self.current_pos

        cx, cy, cz = self.current_pos
        hx, hy, hz = self.hold_pos
        ex, ey, ez = hx - cx, hy - cy, hz - cz

        wx = max(-self.hold_max_speed, min(self.hold_max_speed, self.hold_kp * ex))
        wy = max(-self.hold_max_speed, min(self.hold_max_speed, self.hold_kp * ey))
        wz = max(-self.hold_max_speed, min(self.hold_max_speed, self.hold_kp * ez))

        self._send_world_velocity(wx, wy, wz)

    def _finish_navigation_trial(self, outcome: str):
        if self.trial_recorder is None or self.trial_recorder.finished:
            return None
        sim_time_s = (
            self._last_odom_time_s
            if self._last_odom_time_s is not None
            else self.trial_recorder.started_at_s
        )
        if sim_time_s is None:
            return None
        try:
            result = self.trial_recorder.finish(
                outcome,
                sim_time_s,
                residual_active_at_finish=self.residual_enabled,
            )
        except (OSError, ValueError) as exc:
            self.get_logger().error(f'Failed to finalize trial metrics: {exc}')
            return None

        self.get_logger().info(
            'Navigation trial complete: '
            f'outcome={result["outcome"]}, '
            f'duration={result["duration_s"]:.3f}s, '
            f'path={result["path_length_m"]:.3f}m, '
            f'min_clearance={result["min_clearance_m"]}m')
        if self.trial_recorder.output_path is not None:
            self.get_logger().info(
                f'Trial JSON: {self.trial_recorder.output_path}')
        return result

    def _stop_failed_navigation_trial(self, outcome: str):
        self._send_world_velocity(0.0, 0.0, 0.0)
        self._finish_navigation_trial(outcome)
        self.state = State.ABORT
        self.get_logger().error(
            f'Navigation trial ended with {outcome}; zero velocity commanded.')

    def _send_world_velocity(self, wx: float, wy: float, wz: float):
        # World ENU -> body FRD (MAVSDK VelocityBodyYawspeed convention).
        yaw = self.current_yaw
        forward = wx * math.cos(yaw) + wy * math.sin(yaw)
        right = wx * math.sin(yaw) - wy * math.cos(yaw)
        down = -wz
        self.bridge.send_velocity_body(forward, right, down, 0.0)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _advance_to_current_waypoint(self):
        """Return the world-frame (x,y,z) of the waypoint to head toward, or
        None once the last waypoint has been reached."""
        poses = self.path.poses
        cx, cy, cz = self.current_pos

        while self.waypoint_idx < len(poses):
            p = poses[self.waypoint_idx].pose.position
            dist = math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2 + (p.z - cz) ** 2)
            is_last = self.waypoint_idx == len(poses) - 1
            radius = self.goal_radius if is_last else self.waypoint_radius
            if dist > radius:
                return (p.x, p.y, p.z)
            if is_last:
                return None
            self.waypoint_idx += 1

        return None


    def destroy_node(self):
        if self.trial_recorder is not None:
            if (
                self.trial_recorder.started
                and not self.trial_recorder.finished
            ):
                self._finish_navigation_trial('aborted')
            elif (
                self.trial_recorder.finished
                and self.trial_recorder.result["landing_outcome"]
                == "in_progress"
            ):
                self.trial_recorder.record_landing_outcome('aborted')
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PathFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
