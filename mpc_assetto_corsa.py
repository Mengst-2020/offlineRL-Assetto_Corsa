from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import time
import numpy as np

TOP_SPEED_MS = 80.0

DEFAULT_CURVILINEAR_VEHICLE_PARAMS: Dict[str, Dict[str, float]] = {
    "dallara_f317": {
        "m": 760.0,
        "Iz": 690.0,
        "lf": 1.50,
        "lr": 1.10,
        "Cf": 90000.0,
        "Cr": 110000.0,
        "steer_max_deg": 25.0,
        "steer_rate_deg": 180.0,
        "ax_max": 6.0,
        "ax_min": -10.0,
        "ay_max": 18.0,
        "max_speed_kmh": 140.0,
    },
    "bmw_z4_gt3": {
        "m": 1450.0,
        "Iz": 1900.0,
        "lf": 1.40,
        "lr": 1.30,
        "Cf": 80000.0,
        "Cr": 90000.0,
        "steer_max_deg": 25.0,
        "steer_rate_deg": 180.0,
        "ax_max": 4.5,
        "ax_min": -8.0,
        "ay_max": 12.0,
        "max_speed_kmh": 140.0,
    },
}


@dataclass
class CurvilinearMPCConfig:
    horizon: int = 12
    dt: float = 0.08
    vx_min: float = 2.0
    low_speed_threshold: float = 8.0
    low_speed_exit_threshold: float = 10.0
    recovery_ey: float = 1.35
    recovery_ey_exit: float = 0.45
    recovery_epsi: float = 0.20
    recovery_epsi_exit: float = 0.10
    recovery_vy: float = 1.2
    recovery_r: float = 0.45
    lookahead_time: float = 0.35
    lookahead_min: float = 6.0
    lookahead_max: float = 22.0
    preview_curvature_points: int = 14
    max_target_speed: float = 140.0 / 3.6
    speed_scale: float = 0.42
    ay_limit_scale: float = 0.92
    speed_kp: float = 0.9
    brake_kp: float = 2.8
    low_speed_accel: float = 1.8
    tracking_k_ey: float = 0.26
    tracking_k_epsi: float = 0.95
    tracking_k_vy: float = 0.16
    tracking_k_r: float = 0.22
    low_speed_k_ey: float = 0.18
    low_speed_k_epsi: float = 0.65
    recovery_k_pos: float = 0.95
    recovery_k_heading: float = 1.30
    recovery_k_vy: float = 0.18
    recovery_k_r: float = 0.22
    recovery_reorient_epsi: float = 0.35
    recovery_reorient_vy: float = 0.70
    recovery_reorient_r: float = 0.35
    recovery_rejoin_epsi: float = 0.85
    recovery_rejoin_vy: float = 1.00
    recovery_rejoin_r: float = 0.60
    recovery_rejoin_gain: float = 0.75
    recovery_heading_gain: float = 1.50
    recovery_brake_heading_threshold: float = 0.80
    recovery_coast_heading_threshold: float = 0.35
    recovery_high_speed_vx: float = 12.0
    recovery_high_speed_ey: float = 1.40
    recovery_high_speed_epsi: float = 0.12
    recovery_high_speed_vy: float = 1.10
    recovery_high_speed_r: float = 0.75
    recovery_roll_speed: float = 4.0
    recovery_reorient_accel: float = 0.55
    steer_rate_limit_scale: float = 0.85
    brake_turn_speed_margin: float = 2.0
    brake_turn_curvature_threshold: float = 7.0e-4
    brake_turn_steer_scale: float = 0.28
    turn_direction_curvature_threshold: float = 0.006
    turn_direction_epsi_threshold: float = 0.08
    turn_direction_ey_threshold: float = 0.45
    turn_direction_oppose_scale: float = 0.20
    recovery_turn_direction_epsi_threshold: float = 0.30
    recovery_turn_direction_ey_threshold: float = 1.20
    recovery_turn_direction_oppose_scale: float = 0.16
    recovery_overuse_brake_gain: float = 1.8
    recovery_overuse_steer_scale: float = 0.72
    recovery_crossline_ey_release: float = 0.55
    recovery_crossline_heading_scale: float = 0.45
    recovery_crossline_lateral_scale: float = 0.10
    recovery_crossline_steer_limit_scale: float = 0.45
    recovery_speed_error_margin: float = 0.75
    recovery_rejoin_lookahead_min: int = 4
    recovery_rejoin_lookahead_max: int = 14
    recovery_stabilize_friction_threshold: float = 0.95
    recovery_rejoin_heading_gain: float = 1.05
    recovery_rejoin_lateral_gain: float = 0.42
    recovery_stabilize_heading_gain: float = 0.70
    recovery_stabilize_vy_gain: float = 0.22
    recovery_stabilize_r_gain: float = 0.18
    recovery_stabilize_speed_gain: float = 0.32
    recovery_rejoin_speed_gain: float = 0.22
    recovery_rejoin_ey: float = 0.18
    recovery_exit_hold_steps: int = 8
    recovery_curve_kappa_threshold: float = 0.004
    recovery_curve_ey_exit: float = 0.25
    recovery_curve_epsi_exit: float = 0.07
    non_corner_kappa_threshold: float = 0.05
    non_corner_tracking_steer_scale: float = 0.30
    non_corner_tracking_ff_scale: float = 0.20
    non_corner_tracking_gain_scale: float = 0.45
    non_corner_recovery_release_steer_scale: float = 0.22
    non_corner_recovery_heading_scale: float = 0.35
    non_corner_recovery_lateral_damp_scale: float = 0.55
    non_corner_recovery_brake: float = -0.25
    recovery_contain_ey: float = 0.35
    recovery_contain_heading_base: float = 0.18
    recovery_contain_heading_max: float = 0.45
    recovery_contain_lookahead_min: int = 2
    recovery_contain_lookahead_max: int = 8
    recovery_contain_path_blend_min: float = 0.20
    recovery_contain_path_blend_max: float = 0.55
    recovery_contain_steer_gain: float = 1.35
    recovery_contain_speed_gain: float = 0.018
    recovery_contain_brake_gain: float = 0.55
    recovery_contain_brake_max: float = -1.8
    recovery_contain_vy_gain: float = 0.10
    recovery_contain_r_gain: float = 0.10
    recovery_contain_lateral_gain: float = 0.14
    recovery_contain_lateral_max: float = 0.38
    recovery_contain_lateral_sign_ey: float = 0.20
    recovery_contain_lateral_sign_strength: float = 0.70
    recovery_contain_hold_ey: float = 0.35
    recovery_contain_min_mag_ey: float = 0.60
    recovery_contain_min_mag_ratio: float = 0.35
    recovery_contain_brake: float = -0.9
    recovery_contain_steer_scale: float = 0.28
    recovery_contain_steer_scale_max: float = 0.62
    recovery_contain_steer_scale_ey_gain: float = 0.10
    recovery_contain_steer_scale_heading_gain: float = 0.18
    recovery_contain_steer_scale_speed_gain: float = 0.010
    non_corner_recovery_contain_steer_scale_max: float = 0.50
    recovery_rejoin_crossline_ey: float = 0.18
    recovery_rejoin_crossline_heading_scale: float = 0.55
    recovery_rejoin_crossline_lateral_scale: float = 0.35
    recovery_rejoin_crossline_brake: float = -0.30
    recovery_ey_worsen_tol: float = 0.02
    recovery_ey_worsen_steps: int = 3
    recovery_ey_unwind_scale: float = 0.15
    recovery_ey_unwind_brake: float = -0.45
    throttle_steer_center_scale: float = 0.55
    friction_circle_scale: float = 0.90
    tracking_high_speed_gain_min: float = 0.38

    def resolved_weights(self) -> Dict[str, float]:
        return {
            "ey": 10.0,
            "epsi": 18.0,
            "vx": 1.0,
            "vy": 1.0,
            "r": 2.0,
            "delta": 0.4,
            "ax": 0.05,
        }


class CurvilinearSingleTrackModel:
    def __init__(self, vehicle_params: Dict[str, float], dt: float):
        self.params = dict(vehicle_params)
        self.dt = float(dt)
        self.steer_max = np.deg2rad(self.params["steer_max_deg"])
        self.steer_rate = np.deg2rad(self.params["steer_rate_deg"])

    def step(self, x: np.ndarray, u: np.ndarray, kappa_ref: float) -> np.ndarray:
        s, ey, epsi, vx, vy, yaw_rate = np.asarray(x, dtype=np.float64)
        delta, ax = np.asarray(u, dtype=np.float64)

        m = self.params["m"]
        Iz = self.params["Iz"]
        lf = self.params["lf"]
        lr = self.params["lr"]
        Cf = self.params["Cf"]
        Cr = self.params["Cr"]

        vx_safe = max(float(vx), 0.5)
        one_minus_kappa_ey = max(1.0 - float(kappa_ref) * float(ey), 1e-3)

        alpha_f = np.arctan2(vy + lf * yaw_rate, vx_safe) - delta
        alpha_r = np.arctan2(vy - lr * yaw_rate, vx_safe)

        Fyf = -Cf * alpha_f
        Fyr = -Cr * alpha_r

        s_dot = (vx * np.cos(epsi) - vy * np.sin(epsi)) / one_minus_kappa_ey
        ey_dot = vx * np.sin(epsi) + vy * np.cos(epsi)
        epsi_dot = yaw_rate - kappa_ref * s_dot
        vx_dot = ax + vy * yaw_rate
        vy_dot = (Fyf + Fyr) / m - vx * yaw_rate
        yaw_rate_dot = (lf * Fyf - lr * Fyr) / Iz

        return np.array(
            [
                s + self.dt * s_dot,
                ey + self.dt * ey_dot,
                self._wrap_to_pi(epsi + self.dt * epsi_dot),
                max(0.0, vx + self.dt * vx_dot),
                vy + self.dt * vy_dot,
                yaw_rate + self.dt * yaw_rate_dot,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi


class AssettoCorsaMPCController:
    def __init__(
        self,
        env: Any,
        horizon: int = 15,
        dt: Optional[float] = None,
        vehicle_params: Optional[Dict[str, float]] = None,
        config: Optional[CurvilinearMPCConfig] = None,
    ):
        self.env = env
        if config is None:
            config = CurvilinearMPCConfig(horizon=horizon, dt=dt or getattr(env, "dt", 0.04))
        elif dt is not None:
            config.dt = dt
        self.config = config

        params = dict(vehicle_params or {})
        if not params:
            env_params = getattr(env, "vehicle_params", None)
            if env_params:
                params = dict(env_params)
        if not params:
            car_name = getattr(env, "car_name", "dallara_f317")
            params = dict(
                DEFAULT_CURVILINEAR_VEHICLE_PARAMS.get(
                    car_name,
                    DEFAULT_CURVILINEAR_VEHICLE_PARAMS["dallara_f317"],
                )
            )
        self.vehicle_params = params

        max_speed_ms = self.vehicle_params.get("max_speed_ms")
        max_speed_kmh = self.vehicle_params.get("max_speed_kmh")
        if max_speed_ms is not None:
            self.config.max_target_speed = float(max_speed_ms)
        elif max_speed_kmh is not None:
            self.config.max_target_speed = float(max_speed_kmh) / 3.6

        self.model = CurvilinearSingleTrackModel(self.vehicle_params, self.config.dt)
        self.weights = self.config.resolved_weights()
        self.last_u_sequence = np.zeros((self.config.horizon, 2), dtype=np.float64)
        self.last_solution = np.zeros(2, dtype=np.float64)
        self.planning_times = deque(maxlen=100)
        self.reset()

    def reset(self) -> None:
        self.current_mode = "low_speed_fallback"
        self.recovery_converged_steps = 0
        self.last_recovery_abs_ey = None
        self.recovery_ey_worsen_count = 0
        self.last_solution = np.zeros(2, dtype=np.float64)
        self.last_u_sequence = np.zeros((self.config.horizon, 2), dtype=np.float64)

    def solve(self, curvilinear_state: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.time()
        state_dict = curvilinear_state or self.env.get_curvilinear_state()
        self._maybe_reset_mode_state(state_dict)
        ref = self.env.get_reference_trajectory(
            state_dict["s"],
            horizon=self.config.horizon,
            dt=self.config.dt,
            vx_current=state_dict["vx"],
        )
        mode = self._select_mode(state_dict)

        if mode == "low_speed_fallback":
            return self._low_speed_fallback(state_dict, ref, start_time)
        if mode == "recovery_fallback":
            return self._recovery_fallback(state_dict, ref, start_time)
        return self._trajectory_tracking(state_dict, ref, start_time)

    def select_action(self, state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.solve()

    def _maybe_reset_mode_state(self, state_dict: Dict[str, float]) -> None:
        # A fresh episode often starts nearly stationary and close to the reference line.
        # Reset the controller mode here because the env does not call back into the
        # controller on reset.
        if (
            float(state_dict["vx"]) < 0.2
            and abs(float(state_dict["ey"])) < 0.10
            and abs(float(state_dict["epsi"])) < 0.10
            and abs(float(state_dict["vy"])) < 0.20
            and abs(float(state_dict["r"])) < 0.20
        ):
            self.reset()

    def _select_mode(self, state_dict: Dict[str, float]) -> str:
        abs_ey = abs(float(state_dict["ey"]))
        abs_epsi = abs(float(state_dict["epsi"]))
        abs_vy = abs(float(state_dict["vy"]))
        abs_r = abs(float(state_dict["r"]))
        vx = float(state_dict["vx"])
        abs_kappa = abs(float(state_dict.get("kappa_ref", 0.0)))

        if self.current_mode == "recovery_fallback":
            exit_ey_limit = self.config.recovery_ey_exit
            exit_epsi_limit = self.config.recovery_epsi_exit
            if abs_kappa > self.config.recovery_curve_kappa_threshold:
                exit_ey_limit = min(exit_ey_limit, self.config.recovery_curve_ey_exit)
                exit_epsi_limit = min(exit_epsi_limit, self.config.recovery_curve_epsi_exit)

            converged = (
                abs_ey < exit_ey_limit
                and abs_epsi < exit_epsi_limit
                and abs_vy < self.config.recovery_vy * 0.5
                and abs_r < self.config.recovery_r * 0.5
            )
            if converged:
                self.recovery_converged_steps += 1
            else:
                self.recovery_converged_steps = 0

            if self.recovery_converged_steps >= self.config.recovery_exit_hold_steps:
                self.recovery_converged_steps = 0
                self.last_recovery_abs_ey = None
                self.recovery_ey_worsen_count = 0
                self.current_mode = "low_speed_fallback" if vx < self.config.low_speed_exit_threshold else "trajectory_tracking"
            return self.current_mode

        enter_recovery = (
            abs_ey > self.config.recovery_ey
            or abs_epsi > self.config.recovery_epsi
            or abs_vy > self.config.recovery_vy
            or abs_r > self.config.recovery_r
        )

        # Do not jump into recovery too early at speed when the car is only mildly off-line.
        if vx > self.config.recovery_high_speed_vx:
            mild_fast_state = (
                abs_ey < self.config.recovery_high_speed_ey
                and abs_epsi < self.config.recovery_high_speed_epsi
                and abs_vy < self.config.recovery_high_speed_vy
                and abs_r < self.config.recovery_high_speed_r
            )
            if mild_fast_state:
                enter_recovery = False

        if enter_recovery:
            self.recovery_converged_steps = 0
            self.last_recovery_abs_ey = None
            self.recovery_ey_worsen_count = 0
            self.current_mode = "recovery_fallback"
            return self.current_mode

        if self.current_mode == "low_speed_fallback":
            self.recovery_converged_steps = 0
            self.last_recovery_abs_ey = None
            self.recovery_ey_worsen_count = 0
            if vx < self.config.low_speed_exit_threshold:
                return self.current_mode
            self.current_mode = "trajectory_tracking"
            return self.current_mode

        self.recovery_converged_steps = 0
        self.last_recovery_abs_ey = None
        self.recovery_ey_worsen_count = 0
        if vx < self.config.low_speed_threshold:
            self.current_mode = "low_speed_fallback"
        else:
            self.current_mode = "trajectory_tracking"
        return self.current_mode

    def _low_speed_fallback(
        self,
        state_dict: Dict[str, float],
        ref: Dict[str, np.ndarray],
        start_time: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        current_idx = self._nearest_index(state_dict, ref)
        target_idx = self._target_index(state_dict, ref, mode="low_speed_fallback")
        target_speed, speed_debug = self._preview_target_speed(ref, current_idx, target_idx, state_dict)
        steering_debug = self._low_speed_steering_debug(state_dict, ref, current_idx, target_idx)
        steer_limit = self._steer_limit(state_dict, mode="low_speed_fallback")
        steer_cmd = np.clip(steering_debug["steer_cmd"], -steer_limit, steer_limit)
        steer_cmd = self._limit_steer_rate(state_dict, steer_cmd)
        ax_cmd = np.clip(self.config.low_speed_accel, 0.2, min(2.0, self.vehicle_params["ax_max"]))
        steer_cmd, ax_cmd, friction_debug = self._apply_friction_circle_constraints(
            state_dict, ref, current_idx, target_idx, steer_cmd, ax_cmd, target_speed, "low_speed_fallback"
        )
        return self._finalize_result(
            state_dict,
            ref,
            start_time,
            "low_speed_fallback",
            current_idx,
            target_idx,
            target_speed,
            steer_cmd,
            ax_cmd,
            steer_limit,
            steering_debug,
            speed_debug,
            friction_debug,
        )

    def _recovery_fallback(
        self,
        state_dict: Dict[str, float],
        ref: Dict[str, np.ndarray],
        start_time: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        current_idx = self._nearest_index(state_dict, ref)
        target_idx = self._target_index(state_dict, ref, mode="recovery_fallback")
        target_speed, speed_debug = self._preview_target_speed(ref, current_idx, target_idx, state_dict)
        steering_debug = self._recovery_control_debug(state_dict, ref, current_idx, target_idx, target_speed)
        steer_limit = self._steer_limit(state_dict, mode="recovery_fallback")
        steer_limit = min(steer_limit, float(steering_debug["phase_steer_limit"]))
        steer_cmd = np.clip(steering_debug["steer_cmd"], -steer_limit, steer_limit)
        steer_cmd, turn_guard_debug = self._apply_turn_direction_guard(
            state_dict, ref, current_idx, target_idx, steer_cmd, "recovery_fallback"
        )
        steer_cmd = self._limit_steer_rate(state_dict, steer_cmd)
        steering_debug.update({"turn_guard": turn_guard_debug})
        ax_cmd = float(steering_debug["ax_cmd_raw"])

        steer_cmd, ax_cmd, friction_debug = self._apply_friction_circle_constraints(
            state_dict, ref, current_idx, target_idx, steer_cmd, ax_cmd, target_speed, "recovery_fallback"
        )
        return self._finalize_result(
            state_dict,
            ref,
            start_time,
            "recovery_fallback",
            current_idx,
            target_idx,
            target_speed,
            steer_cmd,
            ax_cmd,
            steer_limit,
            steering_debug,
            speed_debug,
            friction_debug,
        )

    def _trajectory_tracking(
        self,
        state_dict: Dict[str, float],
        ref: Dict[str, np.ndarray],
        start_time: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        current_idx = self._nearest_index(state_dict, ref)
        target_idx = self._target_index(state_dict, ref, mode="trajectory_tracking")
        target_speed, speed_debug = self._preview_target_speed(ref, current_idx, target_idx, state_dict)
        steering_debug = self._tracking_steering_debug(state_dict, ref, current_idx, target_idx)
        steer_limit = self._steer_limit(state_dict, mode="trajectory_tracking")
        steer_cmd = np.clip(steering_debug["steer_cmd"], -steer_limit, steer_limit)
        steer_cmd, turn_guard_debug = self._apply_turn_direction_guard(
            state_dict, ref, current_idx, target_idx, steer_cmd, "trajectory_tracking"
        )
        steer_cmd = self._limit_steer_rate(state_dict, steer_cmd)
        steering_debug.update({"turn_guard": turn_guard_debug})
        ax_cmd = self._tracking_accel_command(state_dict, target_speed)
        steer_cmd, ax_cmd, friction_debug = self._apply_friction_circle_constraints(
            state_dict, ref, current_idx, target_idx, steer_cmd, ax_cmd, target_speed, "trajectory_tracking"
        )
        return self._finalize_result(
            state_dict,
            ref,
            start_time,
            "trajectory_tracking",
            current_idx,
            target_idx,
            target_speed,
            steer_cmd,
            ax_cmd,
            steer_limit,
            steering_debug,
            speed_debug,
            friction_debug,
        )

    def _finalize_result(
        self,
        state_dict: Dict[str, float],
        ref: Dict[str, np.ndarray],
        start_time: float,
        mode: str,
        current_idx: int,
        target_idx: int,
        target_speed: float,
        steer_cmd: float,
        ax_cmd: float,
        steer_limit: float,
        steering_debug: Dict[str, Any],
        speed_debug: Dict[str, Any],
        friction_debug: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        u0 = self._clip_single_u(np.array([steer_cmd, ax_cmd], dtype=np.float64))
        self.last_solution = u0.copy()
        self.last_u_sequence[:] = u0

        predicted_states = self.rollout(
            self._state_dict_to_vector(state_dict),
            np.tile(u0, (self.config.horizon, 1)),
            ref["kappa"],
        )
        objective = self._tracking_objective(state_dict, ref, target_idx, target_speed, u0)
        action = self.env.mpc_control_to_env_action(u0[0], u0[1])

        planning_time = time.time() - start_time
        self.planning_times.append(planning_time)
        debug = self._build_debug_payload(
            state_dict=state_dict,
            ref=ref,
            current_idx=current_idx,
            target_idx=target_idx,
            target_speed=target_speed,
            steer_cmd=float(u0[0]),
            ax_cmd=float(u0[1]),
            steer_limit=steer_limit,
            steering_debug=steering_debug,
            speed_debug=speed_debug,
            friction_debug=friction_debug,
            mode=mode,
        )
        info = {
            "success": True,
            "status": 0,
            "message": mode,
            "planning_time": planning_time,
            "objective": float(objective),
            "u0": u0.copy(),
            "u_sequence": self.last_u_sequence.copy(),
            "predicted_states": predicted_states,
            "reference": ref,
            "state": dict(state_dict),
            "debug": debug,
            "selected_mode": mode,
        }
        return action, info

    def _nearest_index(self, state_dict: Dict[str, float], ref: Dict[str, np.ndarray]) -> int:
        return int(np.argmin(np.abs(ref["s"] - state_dict["s"])))

    def _target_index(self, state_dict: Dict[str, float], ref: Dict[str, np.ndarray], mode: str) -> int:
        current_idx = self._nearest_index(state_dict, ref)
        lookahead = np.clip(
            self.config.lookahead_time * max(float(state_dict["vx"]), self.config.vx_min),
            self.config.lookahead_min,
            self.config.lookahead_max,
        )
        if mode == "low_speed_fallback":
            lookahead *= 0.65
        elif mode == "recovery_fallback":
            lookahead *= 0.55
        s_target = float(state_dict["s"]) + lookahead
        return int(np.argmin(np.abs(ref["s"] - s_target)))

    def _tracking_steering_debug(
        self,
        state_dict: Dict[str, float],
        ref: Dict[str, np.ndarray],
        current_idx: int,
        target_idx: int,
    ) -> Dict[str, Any]:
        yaw_ref = float(ref["yaw"][target_idx])
        heading_error = self._wrap_to_pi(yaw_ref - float(state_dict["yaw"]))
        ff_delta = -np.arctan((self.vehicle_params["lf"] + self.vehicle_params["lr"]) * float(ref["kappa_full"][target_idx]))
        ey = float(state_dict["ey"])
        epsi = float(state_dict["epsi"])
        vy = float(state_dict["vy"])
        yaw_rate = float(state_dict["r"])
        vx = float(state_dict["vx"])
        gain_scale = np.clip(
            1.0 - 0.035 * max(vx - self.config.recovery_high_speed_vx, 0.0),
            self.config.tracking_high_speed_gain_min,
            1.0,
        )
        if vx > self.config.recovery_high_speed_vx and abs(ey) < self.config.recovery_high_speed_ey:
            gain_scale *= 0.85
        stability_scale = 1.0
        if vx > 15.0 and abs(ey) < 0.25 and abs(epsi) < 0.06:
            stability_scale = 0.55
        preview_kappa = self._turn_preview_curvature(ref, current_idx, target_idx)
        non_corner_region = abs(preview_kappa) < self.config.non_corner_kappa_threshold
        ff_scale = 1.0
        feedback_scale = 1.0
        if non_corner_region:
            ff_scale = self.config.non_corner_tracking_ff_scale
            feedback_scale = self.config.non_corner_tracking_gain_scale

        fb_delta = (
            -feedback_scale * gain_scale * self.config.tracking_k_ey * ey
            + feedback_scale * gain_scale * self.config.tracking_k_epsi * epsi
            - stability_scale * self.config.tracking_k_vy * vy
            - stability_scale * self.config.tracking_k_r * yaw_rate
        )
        steer_cmd = ff_scale * ff_delta + fb_delta
        return {
            "strategy": "trajectory_tracking",
            "yaw_ref": yaw_ref,
            "heading_error": heading_error,
            "gain_scale": float(gain_scale),
            "preview_kappa": float(preview_kappa),
            "non_corner_region": bool(non_corner_region),
            "non_corner_scale": float(feedback_scale),
            "feedforward_delta": float(ff_scale * ff_delta),
            "feedback_delta": float(fb_delta),
            "crosstrack_term": float(-feedback_scale * gain_scale * self.config.tracking_k_ey * ey),
            "epsi_term": float(feedback_scale * gain_scale * self.config.tracking_k_epsi * epsi),
            "stability_scale": float(stability_scale),
            "vy_term": float(-stability_scale * self.config.tracking_k_vy * vy),
            "r_term": float(-stability_scale * self.config.tracking_k_r * yaw_rate),
            "steer_cmd_raw": float(steer_cmd),
            "steer_cmd": float(steer_cmd),
        }

    def _low_speed_steering_debug(
        self,
        state_dict: Dict[str, float],
        ref: Dict[str, np.ndarray],
        current_idx: int,
        target_idx: int,
    ) -> Dict[str, Any]:
        yaw_ref = float(ref["yaw"][current_idx])
        heading_error = self._wrap_to_pi(yaw_ref - float(state_dict["yaw"]))
        ey = float(state_dict["ey"])
        epsi = float(state_dict["epsi"])
        steer_cmd = -self.config.low_speed_k_ey * ey + self.config.low_speed_k_epsi * epsi
        steer_cmd += 0.15 * heading_error
        return {
            "strategy": "low_speed_fallback",
            "yaw_ref": yaw_ref,
            "heading_error": heading_error,
            "crosstrack_term": float(-self.config.low_speed_k_ey * ey),
            "epsi_term": float(self.config.low_speed_k_epsi * epsi),
            "steer_cmd_raw": float(steer_cmd),
            "steer_cmd": float(steer_cmd),
        }

    def _recovery_control_debug(
        self,
        state_dict: Dict[str, float],
        ref: Dict[str, np.ndarray],
        current_idx: int,
        target_idx: int,
        target_speed: float,
    ) -> Dict[str, Any]:
        yaw_ref = float(ref["yaw"][current_idx])
        path_heading = yaw_ref
        x = float(state_dict["x"])
        y = float(state_dict["y"])
        ey = float(state_dict["ey"])
        abs_ey = abs(ey)
        epsi = float(state_dict["epsi"])
        vy = float(state_dict["vy"])
        yaw_rate = float(state_dict["r"])
        vx = float(state_dict["vx"])
        wheelbase = float(self.vehicle_params["lf"] + self.vehicle_params["lr"])
        ay_limit = float(self.vehicle_params.get("ay_max", 12.0)) * self.config.friction_circle_scale
        kappa_preview = self._turn_preview_curvature(ref, current_idx, target_idx)
        kappa_from_delta = abs(np.tan(float(state_dict["u_prev"][0])) / max(wheelbase, 1e-6))
        friction_usage = np.clip(vx * vx * max(kappa_preview, kappa_from_delta) / max(ay_limit, 1e-6), 0.0, 2.0)
        crossline_release = abs_ey < self.config.recovery_crossline_ey_release
        speed_error = vx - float(target_speed)
        non_corner_region = abs(kappa_preview) < self.config.non_corner_kappa_threshold
        ey_worsening = False
        contain_heading = path_heading
        if self.last_recovery_abs_ey is not None:
            ey_worsening = abs_ey > (self.last_recovery_abs_ey + self.config.recovery_ey_worsen_tol)
        if ey_worsening:
            self.recovery_ey_worsen_count += 1
        else:
            self.recovery_ey_worsen_count = 0
        self.last_recovery_abs_ey = abs_ey
        need_contain = (
            self.recovery_ey_worsen_count >= self.config.recovery_ey_worsen_steps
            or abs_ey > self.config.recovery_contain_ey
        )
        if self.current_mode == "recovery_fallback" and abs_ey > self.config.recovery_contain_hold_ey:
            need_contain = True

        need_stabilize = (
            friction_usage > self.config.recovery_stabilize_friction_threshold
            or abs(epsi) > self.config.recovery_reorient_epsi
            or abs(vy) > self.config.recovery_reorient_vy
            or abs(yaw_rate) > self.config.recovery_reorient_r
        )

        if need_contain:
            phase = "contain"
            contain_lookahead = int(
                np.clip(
                    self.config.recovery_contain_lookahead_min + 1.5 * abs_ey + 0.08 * vx,
                    self.config.recovery_contain_lookahead_min,
                    self.config.recovery_contain_lookahead_max,
                )
            )
            contain_target_idx = min(current_idx + contain_lookahead, len(ref["x"]) - 1)
            x_contain = float(ref["x"][contain_target_idx])
            y_contain = float(ref["y"][contain_target_idx])
            contain_heading = np.arctan2(y_contain - y, x_contain - x)
            contain_path_blend = np.clip(
                0.20 + 0.10 * min(abs_ey, 2.0),
                self.config.recovery_contain_path_blend_min,
                self.config.recovery_contain_path_blend_max,
            )
            desired_heading = self._wrap_to_pi(
                contain_path_blend * path_heading + (1.0 - contain_path_blend) * contain_heading
            )
            heading_error = self._wrap_to_pi(desired_heading - float(state_dict["yaw"]))
            damp_vy = self.config.recovery_contain_vy_gain
            damp_r = self.config.recovery_contain_r_gain
            contain_gain = (
                self.config.recovery_contain_steer_gain
                + self.config.recovery_contain_speed_gain * min(vx, 30.0)
                + 0.18 * min(abs_ey, 2.5)
            )
            contain_lateral_term = np.clip(
                -self.config.recovery_contain_lateral_gain * ey,
                -self.config.recovery_contain_lateral_max * self.model.steer_max,
                self.config.recovery_contain_lateral_max * self.model.steer_max,
            )
            steer_cmd = (
                contain_gain * heading_error
                + contain_lateral_term
                - damp_vy * vy
                - damp_r * yaw_rate
            )
            lateral_sign_guard = False
            if (
                abs_ey > self.config.recovery_contain_lateral_sign_ey
                and abs(contain_lateral_term) > 1e-6
                and np.sign(steer_cmd) != np.sign(contain_lateral_term)
            ):
                lateral_sign_guard = True
                steer_cmd = (
                    (1.0 - self.config.recovery_contain_lateral_sign_strength) * steer_cmd
                    + self.config.recovery_contain_lateral_sign_strength * contain_lateral_term
                )
                if np.sign(steer_cmd) != np.sign(contain_lateral_term):
                    steer_cmd = np.sign(contain_lateral_term) * max(
                        abs(contain_lateral_term),
                        0.55 * abs(steer_cmd),
                    )
            contain_steer_scale = (
                self.config.recovery_contain_steer_scale
                + self.config.recovery_contain_steer_scale_ey_gain * min(abs_ey, 3.0)
                + self.config.recovery_contain_steer_scale_heading_gain * min(abs(heading_error), 0.8)
                + self.config.recovery_contain_steer_scale_speed_gain * min(vx, 25.0)
            )
            contain_steer_scale = min(
                contain_steer_scale,
                self.config.recovery_contain_steer_scale_max,
            )
            if non_corner_region:
                non_corner_scale_cap = (
                    self.config.recovery_contain_steer_scale
                    + 0.08 * min(abs_ey, 3.0)
                    + 0.08 * min(abs(heading_error), 0.8)
                )
                contain_steer_scale = min(
                    contain_steer_scale,
                    max(
                        self.config.recovery_contain_steer_scale,
                        min(
                            non_corner_scale_cap,
                            self.config.non_corner_recovery_contain_steer_scale_max,
                        ),
                    ),
                )
            if crossline_release:
                contain_steer_scale = min(
                    contain_steer_scale,
                    max(
                        self.config.recovery_crossline_steer_limit_scale,
                        self.config.recovery_contain_steer_scale
                        + 0.05 * min(abs_ey, 2.0),
                    ),
                )
            contain_steer_limit = contain_steer_scale * self.model.steer_max
            if abs_ey > self.config.recovery_contain_min_mag_ey and abs(contain_lateral_term) > 1e-6:
                contain_min_mag = self.config.recovery_contain_min_mag_ratio * contain_steer_limit
                if abs(steer_cmd) < contain_min_mag:
                    steer_cmd = np.sign(contain_lateral_term) * contain_min_mag
            steer_cmd = np.clip(steer_cmd, -contain_steer_limit, contain_steer_limit)
            phase_steer_limit = contain_steer_limit
            if speed_error > self.config.recovery_speed_error_margin:
                ax_cmd_raw = max(
                    self.vehicle_params["ax_min"],
                    min(
                        self.config.recovery_contain_brake_max,
                        -(
                            self.config.recovery_contain_brake_gain
                            + 0.30 * speed_error
                            + 0.18 * min(abs_ey, 3.0)
                        ),
                    ),
                )
            elif vx < self.config.recovery_roll_speed:
                ax_cmd_raw = 0.15
            else:
                ax_cmd_raw = self.config.recovery_contain_brake
            rejoin_heading = desired_heading
            rejoin_target_idx = contain_target_idx
            path_blend = float(contain_path_blend)
            lateral_term = float(contain_lateral_term)
        elif need_stabilize:
            phase = "stabilize"
            desired_heading = path_heading
            heading_error = self._wrap_to_pi(desired_heading - float(state_dict["yaw"]))
            heading_gain = self.config.recovery_stabilize_heading_gain
            if crossline_release:
                heading_gain *= self.config.recovery_crossline_heading_scale
            if non_corner_region:
                heading_gain *= self.config.non_corner_recovery_heading_scale
            damp_vy = self.config.recovery_stabilize_vy_gain
            damp_r = self.config.recovery_stabilize_r_gain
            if non_corner_region:
                damp_vy *= self.config.non_corner_recovery_lateral_damp_scale
                damp_r *= self.config.non_corner_recovery_lateral_damp_scale
            steer_cmd = heading_gain * heading_error - damp_vy * vy - damp_r * yaw_rate
            if non_corner_region:
                steer_cmd = np.clip(
                    steer_cmd,
                    -self.config.non_corner_recovery_release_steer_scale * self.model.steer_max,
                    self.config.non_corner_recovery_release_steer_scale * self.model.steer_max,
                )
            if crossline_release:
                steer_cmd = np.clip(
                    steer_cmd,
                    -self.config.recovery_crossline_steer_limit_scale * self.model.steer_max,
                    self.config.recovery_crossline_steer_limit_scale * self.model.steer_max,
                )
            phase_steer_limit = 0.38 * self.model.steer_max if friction_usage > 1.0 else 0.48 * self.model.steer_max
            if non_corner_region:
                phase_steer_limit = min(
                    phase_steer_limit,
                    self.config.non_corner_recovery_release_steer_scale * self.model.steer_max,
                )
            if speed_error > self.config.recovery_speed_error_margin:
                ax_cmd_raw = -min(2.0, 0.55 + self.config.recovery_stabilize_speed_gain * speed_error)
            elif vx < self.config.recovery_roll_speed:
                ax_cmd_raw = self.config.recovery_reorient_accel
            else:
                ax_cmd_raw = -0.10
            if non_corner_region:
                ax_cmd_raw = min(ax_cmd_raw, self.config.non_corner_recovery_brake)
            rejoin_heading = desired_heading
            rejoin_target_idx = current_idx
            path_blend = 0.0
            lateral_term = 0.0
        else:
            phase = "rejoin"
            lookahead_steps = int(np.clip(4 + 2.0 * abs_ey + 0.15 * vx, self.config.recovery_rejoin_lookahead_min, self.config.recovery_rejoin_lookahead_max))
            rejoin_target_idx = min(current_idx + lookahead_steps, len(ref["x"]) - 1)
            x_rejoin = float(ref["x"][rejoin_target_idx])
            y_rejoin = float(ref["y"][rejoin_target_idx])
            rejoin_heading = np.arctan2(y_rejoin - y, x_rejoin - x)
            path_blend = np.clip(abs_ey / 2.0, 0.25, 0.75)
            desired_heading = self._wrap_to_pi(path_blend * rejoin_heading + (1.0 - path_blend) * path_heading)
            heading_error = self._wrap_to_pi(desired_heading - float(state_dict["yaw"]))
            lateral_term = -self.config.recovery_rejoin_lateral_gain * np.sign(ey) * min(abs_ey, 2.0)
            steer_cmd = (
                self.config.recovery_rejoin_heading_gain * heading_error
                + lateral_term
                - 0.18 * vy
                - 0.16 * yaw_rate
            )
            if crossline_release:
                steer_cmd *= self.config.recovery_crossline_heading_scale
            if abs_ey < self.config.recovery_rejoin_crossline_ey:
                steer_cmd = (
                    self.config.recovery_rejoin_crossline_heading_scale * self.config.recovery_rejoin_heading_gain * heading_error
                    + self.config.recovery_rejoin_crossline_lateral_scale * lateral_term
                    - 0.12 * vy
                    - 0.10 * yaw_rate
                )
            phase_steer_limit = 0.55 * self.model.steer_max
            if speed_error > self.config.recovery_speed_error_margin:
                ax_cmd_raw = -min(1.2, 0.30 + self.config.recovery_rejoin_speed_gain * speed_error)
            elif vx < self.config.recovery_roll_speed:
                ax_cmd_raw = 0.35
            elif abs_ey > self.config.recovery_rejoin_ey * 0.5 or abs(epsi) > self.config.recovery_epsi_exit:
                ax_cmd_raw = 0.0
            else:
                ax_cmd_raw = 0.15
            if abs_ey < self.config.recovery_rejoin_crossline_ey:
                ax_cmd_raw = min(ax_cmd_raw, self.config.recovery_rejoin_crossline_brake)

        return {
            "strategy": "recovery_fallback",
            "phase": phase,
            "desired_heading": float(desired_heading),
            "path_heading": float(path_heading),
            "rejoin_heading": float(rejoin_heading),
            "contain_heading": float(contain_heading) if phase == "contain" else float(rejoin_heading),
            "heading_error": float(heading_error),
            "rejoin_blend": 0.0 if phase == "stabilize" else float(path_blend),
            "rejoin_target_idx": int(rejoin_target_idx),
            "contain_target_idx": int(rejoin_target_idx) if phase == "contain" else int(current_idx),
            "crossline_release": bool(crossline_release),
            "ey_worsening": bool(ey_worsening),
            "ey_worsen_count": int(self.recovery_ey_worsen_count),
            "force_unwind": bool(phase == "contain"),
            "phase_steer_limit": float(phase_steer_limit),
            "ax_cmd_raw": float(ax_cmd_raw),
            "friction_usage_est": float(friction_usage),
            "non_corner_region": bool(non_corner_region),
            "lateral_sign_guard": bool(lateral_sign_guard) if phase == "contain" else False,
            "position_term": float(lateral_term),
            "vy_term": float(
                -damp_vy * vy if phase in ("contain", "stabilize") else -0.18 * vy
            ),
            "r_term": float(
                -damp_r * yaw_rate if phase in ("contain", "stabilize") else -0.16 * yaw_rate
            ),
            "steer_cmd_raw": float(steer_cmd),
            "steer_cmd": float(steer_cmd),
        }

    def _preview_target_speed(
        self,
        ref: Dict[str, np.ndarray],
        current_idx: int,
        target_idx: int,
        state_dict: Dict[str, float],
    ) -> Tuple[float, Dict[str, Any]]:
        end_idx = min(target_idx + self.config.preview_curvature_points, len(ref["kappa_full"]))
        preview_kappa = np.abs(ref["kappa_full"][current_idx:end_idx])
        preview_v = ref["vx_full"][current_idx:end_idx]
        if len(preview_kappa) == 0:
            target_speed = self.config.vx_min
        else:
            ay_limit = float(self.vehicle_params.get("ay_max", 12.0)) * self.config.ay_limit_scale
            curvature_speed = np.sqrt(ay_limit / max(float(np.max(preview_kappa)), 1e-4))
            target_speed = min(
                float(np.min(preview_v)) * self.config.speed_scale,
                curvature_speed,
                self.config.max_target_speed,
            )

        if abs(float(state_dict["ey"])) > 0.8:
            target_speed *= 0.82
        if abs(float(state_dict["epsi"])) > 0.12:
            target_speed *= 0.78
        if abs(float(state_dict["r"])) > 0.25:
            target_speed *= 0.84
        target_speed = max(target_speed, self.config.vx_min)

        debug = {
            "preview_start_idx": current_idx,
            "preview_end_idx": end_idx,
            "preview_kappa": preview_kappa.copy(),
            "preview_vx": preview_v.copy(),
            "target_speed": float(target_speed),
        }
        return float(target_speed), debug

    def _tracking_accel_command(self, state_dict: Dict[str, float], target_speed: float) -> float:
        vx = float(state_dict["vx"])
        speed_error = target_speed - vx
        if speed_error >= 0.0:
            ax_cmd = self.config.speed_kp * speed_error
        else:
            ax_cmd = self.config.brake_kp * speed_error
        return float(np.clip(ax_cmd, self.vehicle_params["ax_min"], self.vehicle_params["ax_max"]))

    def _steer_limit(self, state_dict: Dict[str, float], mode: str) -> float:
        steer_max = self.model.steer_max
        if mode == "trajectory_tracking":
            vx = float(state_dict["vx"])
            abs_ey = abs(float(state_dict["ey"]))
            abs_epsi = abs(float(state_dict["epsi"]))
            if vx > 15.0 and abs_ey < 0.25 and abs_epsi < 0.06:
                scale = 0.18
            elif vx > 12.0 and abs_ey < 0.50:
                scale = 0.26
            else:
                scale = 0.40
            if abs(float(state_dict.get("kappa_ref", 0.0))) < self.config.non_corner_kappa_threshold:
                scale = min(scale, self.config.non_corner_tracking_steer_scale)
        elif mode == "recovery_fallback":
            scale = 0.55
        else:
            scale = 0.35
        return float(scale * steer_max)

    def _limit_steer_rate(self, state_dict: Dict[str, float], steer_cmd: float) -> float:
        current_delta = float(state_dict["u_prev"][0])
        max_delta_step = self.model.steer_rate * self.config.dt * self.config.steer_rate_limit_scale
        delta_change = np.clip(steer_cmd - current_delta, -max_delta_step, max_delta_step)
        return float(np.clip(current_delta + delta_change, -self.model.steer_max, self.model.steer_max))

    def _turn_preview_curvature(self, ref: Dict[str, np.ndarray], current_idx: int, target_idx: int) -> float:
        end_idx = min(max(target_idx + 1, current_idx + 6), len(ref["kappa_full"]))
        preview = np.abs(ref["kappa_full"][current_idx:end_idx])
        if len(preview) == 0:
            return 0.0
        return float(np.max(preview))

    def _apply_turn_direction_guard(
        self,
        state_dict: Dict[str, float],
        ref: Dict[str, np.ndarray],
        current_idx: int,
        target_idx: int,
        steer_cmd: float,
        mode: str,
    ) -> Tuple[float, Dict[str, Any]]:
        start_idx = current_idx
        end_idx = min(max(target_idx + 1, current_idx + 6), len(ref["kappa_full"]))
        preview_kappa = ref["kappa_full"][start_idx:end_idx]
        if len(preview_kappa) == 0:
            return float(steer_cmd), {
                "active": False,
                "expected_turn_sign": 0.0,
                "mean_preview_kappa": 0.0,
            }

        mean_preview_kappa = float(np.mean(preview_kappa))
        # In this environment, the steering sign that follows the reference line
        # has the same sign as the local reference curvature.
        expected_turn_sign = float(np.sign(mean_preview_kappa))
        if mode == "trajectory_tracking":
            active = (
                abs(mean_preview_kappa) > self.config.turn_direction_curvature_threshold
                and abs(float(state_dict["epsi"])) < self.config.turn_direction_epsi_threshold
                and abs(float(state_dict["ey"])) < self.config.turn_direction_ey_threshold
                and expected_turn_sign != 0.0
            )
            oppose_scale = self.config.turn_direction_oppose_scale
        elif mode == "recovery_fallback":
            active = (
                abs(mean_preview_kappa) > self.config.turn_direction_curvature_threshold
                and abs(float(state_dict["epsi"])) < self.config.recovery_turn_direction_epsi_threshold
                and abs(float(state_dict["ey"])) < self.config.recovery_turn_direction_ey_threshold
                and expected_turn_sign != 0.0
            )
            oppose_scale = self.config.recovery_turn_direction_oppose_scale
        else:
            active = False
            oppose_scale = self.config.turn_direction_oppose_scale
        opposing = active and (float(steer_cmd) * expected_turn_sign < 0.0)
        if opposing:
            limited_mag = min(abs(float(steer_cmd)), oppose_scale * self.model.steer_max)
            steer_cmd = expected_turn_sign * limited_mag

        return float(steer_cmd), {
            "active": bool(active),
            "opposing": bool(opposing),
            "expected_turn_sign": float(expected_turn_sign),
            "mean_preview_kappa": float(mean_preview_kappa),
        }

    def _apply_friction_circle_constraints(
        self,
        state_dict: Dict[str, float],
        ref: Dict[str, np.ndarray],
        current_idx: int,
        target_idx: int,
        steer_cmd: float,
        ax_cmd: float,
        target_speed: float,
        mode: str,
    ) -> Tuple[float, float, Dict[str, Any]]:
        vx = float(state_dict["vx"])
        wheelbase = float(self.vehicle_params["lf"] + self.vehicle_params["lr"])
        ay_limit = float(self.vehicle_params.get("ay_max", 12.0)) * self.config.friction_circle_scale
        kappa_preview = self._turn_preview_curvature(ref, current_idx, target_idx)
        kappa_from_steer = abs(np.tan(float(steer_cmd)) / max(wheelbase, 1e-6))
        kappa_eff = max(kappa_preview, kappa_from_steer)
        ay_est = vx * vx * kappa_eff
        friction_usage = np.clip(ay_est / max(ay_limit, 1e-6), 0.0, 2.0)

        brake_before_turn = (
            mode == "trajectory_tracking"
            and kappa_preview > self.config.brake_turn_curvature_threshold
            and vx > target_speed + self.config.brake_turn_speed_margin
        )

        if brake_before_turn:
            steer_cmd *= self.config.brake_turn_steer_scale
            ax_cmd = min(ax_cmd, -min(abs(self.vehicle_params["ax_min"]) * 0.45, 1.0 + 0.25 * (vx - target_speed)))

        if mode == "recovery_fallback":
            if friction_usage > 1.0:
                steer_cmd *= self.config.recovery_overuse_steer_scale
                ax_cmd = min(
                    ax_cmd,
                    -min(
                        abs(self.vehicle_params["ax_min"]) * 0.35,
                        0.4 + self.config.recovery_overuse_brake_gain * (friction_usage - 1.0),
                    ),
                )
            ax_allow = np.sqrt(max(0.0, ay_limit * ay_limit - min(ay_est, ay_limit) ** 2))
            recovery_ax_upper = min(ax_allow, 0.6 if friction_usage > 1.0 else 1.2)
            ax_cmd = np.clip(ax_cmd, self.vehicle_params["ax_min"] * 0.20, recovery_ax_upper)
        elif ax_cmd > 0.0:
            ax_allow = np.sqrt(max(0.0, ay_limit * ay_limit - min(ay_est, ay_limit) ** 2))
            ax_cmd = min(ax_cmd, ax_allow)
            steer_cmd *= 1.0 - self.config.throttle_steer_center_scale * np.clip(ax_cmd / max(self.vehicle_params["ax_max"], 1e-6), 0.0, 1.0)

        debug = {
            "ay_limit": float(ay_limit),
            "ay_est": float(ay_est),
            "friction_usage": float(friction_usage),
            "kappa_preview": float(kappa_preview),
            "kappa_from_steer": float(kappa_from_steer),
            "brake_before_turn": bool(brake_before_turn),
            "mode": mode,
        }
        return float(steer_cmd), float(ax_cmd), debug

    def rollout(self, x0: np.ndarray, u_sequence: np.ndarray, kappa_sequence: np.ndarray) -> np.ndarray:
        xs = [x0.copy()]
        x = x0.copy()
        for k in range(self.config.horizon):
            x = self.model.step(x, u_sequence[k], float(kappa_sequence[k]))
            xs.append(x.copy())
        return np.asarray(xs, dtype=np.float64)

    def _tracking_objective(
        self,
        state_dict: Dict[str, float],
        ref: Dict[str, np.ndarray],
        target_idx: int,
        target_speed: float,
        u0: np.ndarray,
    ) -> float:
        yaw_ref = float(ref["yaw"][target_idx])
        heading_error = self._wrap_to_pi(yaw_ref - float(state_dict["yaw"]))
        return float(
            10.0 * float(state_dict["ey"]) ** 2
            + 18.0 * heading_error ** 2
            + 1.0 * (float(state_dict["vx"]) - target_speed) ** 2
            + 0.5 * u0[0] ** 2
            + 0.05 * u0[1] ** 2
        )

    def _build_debug_payload(
        self,
        state_dict: Dict[str, float],
        ref: Dict[str, np.ndarray],
        current_idx: int,
        target_idx: int,
        target_speed: float,
        steer_cmd: float,
        ax_cmd: float,
        steer_limit: float,
        steering_debug: Dict[str, Any],
        speed_debug: Dict[str, Any],
        friction_debug: Dict[str, Any],
        mode: str,
    ) -> Dict[str, Any]:
        preview_end_idx = min(current_idx + self.config.preview_curvature_points, len(ref["kappa_full"]))
        return {
            "mode": mode,
            "current_idx": int(current_idx),
            "target_idx": int(target_idx),
            "state_gap": float(state_dict["ey"]),
            "state_epsi": float(state_dict["epsi"]),
            "state_vx": float(state_dict["vx"]),
            "state_vy": float(state_dict["vy"]),
            "state_r": float(state_dict["r"]),
            "state_s": float(state_dict["s"]),
            "target_speed": float(target_speed),
            "steer_cmd": float(steer_cmd),
            "ax_cmd": float(ax_cmd),
            "current_delta": float(state_dict["u_prev"][0]),
            "steer_limit": float(steer_limit),
            "current_kappa": float(ref["kappa_full"][current_idx]),
            "target_kappa": float(ref["kappa_full"][target_idx]),
            "preview_curvature": ref["kappa_full"][current_idx:preview_end_idx].copy(),
            "preview_ref_speed": ref["vx_full"][current_idx:preview_end_idx].copy(),
            "steering": steering_debug,
            "speed": speed_debug,
            "friction": friction_debug,
            "transition": None,
        }

    def _clip_single_u(self, u: np.ndarray) -> np.ndarray:
        steer_limit = self.model.steer_max
        ax_min = float(self.vehicle_params["ax_min"])
        ax_max = float(self.vehicle_params["ax_max"])
        return np.array(
            [
                np.clip(float(u[0]), -steer_limit, steer_limit),
                np.clip(float(u[1]), ax_min, ax_max),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _state_dict_to_vector(state_dict: Dict[str, float]) -> np.ndarray:
        return np.array(
            [
                state_dict["s"],
                state_dict["ey"],
                state_dict["epsi"],
                state_dict["vx"],
                state_dict["vy"],
                state_dict["r"],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
