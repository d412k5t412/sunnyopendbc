"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Subaru LKAS_ANGLE lateral extension: driver-override hysteresis, MADS-only
guards, engage/disengage shaping, and a jerk-limited motion planner that
bounds both angle rate and angle acceleration on the commanded output.
"""

import numpy as np

DRIVER_OVERRIDE_TORQUE = 120
DRIVER_OVERRIDE_TORQUE_RELEASE = 70
SUSPEND_HOLD_FRAMES = 25                 # ~0.5 s
MADS_ONLY_MAX_STEER_ANGLE = 120          # deg

PRE_ENGAGE_CLEAN_FRAMES = 5              # ~100 ms
REACTIVATION_RAMP_FRAMES = 35            # ~0.7 s
DISENGAGE_TAPER_FRAMES = 8               # ~160 ms; keeps LKAS_Request from edge-falling

# Noise filter on the planner target. Heavy below ~10 mph where EPS angle
# jitter propagates through the planner as low-speed wobble.
PLANNER_ANGLE_LP_ALPHA_BP = [0., 4.5, 13., 18.]    # m/s
PLANNER_ANGLE_LP_ALPHA_V  = [0.05, 0.12, 0.55, 0.80]


class AnglePlanner:
  """Jerk-limited trapezoidal motion planner for the LKAS_ANGLE command.

  Bounds peak rate (deg/frame) and peak acceleration (deg/frame^2). The accel
  bound is what removes the "jerky" feel — without it, an LPF + rate-limit
  pipeline turns step targets into sharp velocity corners. Peak rate stays
  at or below ANGLE_RATE_LIMIT_UP so the safety envelope is unchanged.
  """

  MAX_RATE_BP = [0., 1.5, 5., 15., 35.]              # m/s
  MAX_RATE_V  = [0.70, 0.60, 0.45, 0.28, 0.14]       # deg/frame

  # Tuned so reaching peak rate from rest takes ~0.25-0.30 s at every speed.
  MAX_ACCEL_BP = [0., 5., 15., 35.]                  # m/s
  MAX_ACCEL_V  = [0.050, 0.035, 0.022, 0.012]        # deg/frame^2

  # Scale accel up when error is large (lane changes, recovery) so big
  # maneuvers don't feel sluggish; small corrections keep the smooth profile.
  ERR_SCALE_BP = [1.5, 15.0]                         # deg wheel
  ERR_SCALE_V  = [1.0, 3.0]

  # Low-speed deadband; suppresses planner chasing residual LPF jitter at standstill.
  DEADBAND_BP = [1.5, 3.0]                           # m/s
  DEADBAND_V  = [0.40, 0.0]                          # deg wheel

  def __init__(self):
    self.pos = 0.0
    self.vel = 0.0

  def reset(self, angle: float) -> None:
    self.pos = float(angle)
    self.vel = 0.0

  def update(self, target: float, v_ego: float) -> float:
    max_rate       = float(np.interp(v_ego, self.MAX_RATE_BP,  self.MAX_RATE_V))
    base_max_accel = float(np.interp(v_ego, self.MAX_ACCEL_BP, self.MAX_ACCEL_V))
    deadband       = float(np.interp(v_ego, self.DEADBAND_BP,  self.DEADBAND_V))

    err = float(target) - self.pos

    if abs(err) <= deadband:
      new_vel = float(np.clip(0.0, self.vel - base_max_accel, self.vel + base_max_accel))
      self.pos += new_vel
      self.vel = new_vel
      return self.pos

    accel_scale = float(np.interp(abs(err), self.ERR_SCALE_BP, self.ERR_SCALE_V))
    max_accel = base_max_accel * accel_scale

    # v^2 = 2 a d  ->  brake distance to reach 0 from |vel| at max_accel
    brake_dist = (self.vel * self.vel) / (2.0 * max_accel) if max_accel > 0.0 else 0.0

    if abs(err) > brake_dist:
      desired_vel = np.sign(err) * max_rate
    else:
      desired_vel = np.sign(err) * np.sqrt(max(2.0 * max_accel * abs(err), 0.0))

    new_vel = float(np.clip(desired_vel, self.vel - max_accel, self.vel + max_accel))
    new_vel = float(np.clip(new_vel, -max_rate, max_rate))

    self.pos += new_vel
    self.vel = new_vel
    return self.pos


class LkasAngleStateMachine:
  def __init__(self):
    self.suspended = False
    self.below_release_count = 0
    self.pre_engage_clean_frames = 0
    self.frames_since_resume = REACTIVATION_RAMP_FRAMES
    self.disengage_taper_remaining = 0
    self.active_last = False
    self.planner_angle_filt = 0.0
    self.last_out_angle = 0.0
    self.planner = AnglePlanner()

  def update(self, CC, CS):
    """Returns (commanded_angle, active) — feed to apply_std_steer_angle_limits."""
    torque = abs(CS.out.steeringTorque)
    extreme_angle = abs(CS.out.steeringAngleDeg) > MADS_ONLY_MAX_STEER_ANGLE
    extreme_angle_mads_only = extreme_angle and not CC.enabled

    # pre-engage clean-frame gate
    if torque < DRIVER_OVERRIDE_TORQUE_RELEASE and not extreme_angle_mads_only:
      self.pre_engage_clean_frames = min(self.pre_engage_clean_frames + 1, PRE_ENGAGE_CLEAN_FRAMES)
    else:
      self.pre_engage_clean_frames = 0
    pre_engage_ok = self.pre_engage_clean_frames >= PRE_ENGAGE_CLEAN_FRAMES

    # suspend hysteresis on driver override / extreme angle
    if self.suspended:
      if torque < DRIVER_OVERRIDE_TORQUE_RELEASE and not extreme_angle_mads_only:
        self.below_release_count += 1
        if self.below_release_count >= SUSPEND_HOLD_FRAMES:
          self.suspended = False
          self.below_release_count = 0
          self.frames_since_resume = 0
      else:
        self.below_release_count = 0
    else:
      if torque > DRIVER_OVERRIDE_TORQUE or extreme_angle_mads_only:
        self.suspended = True
        self.below_release_count = 0

    want_active = CC.latActive and not self.suspended
    if want_active and not self.active_last and not pre_engage_ok:
      want_active = False

    if want_active and not self.active_last:
      self.frames_since_resume = 0
      self.planner_angle_filt = CS.out.steeringAngleDeg
      self.last_out_angle = CS.out.steeringAngleDeg
      self.planner.reset(CS.out.steeringAngleDeg)

    # Disengage taper keeps LKAS_Request high for a few frames on clean disengage
    # so the EyeSight watchdog doesn't see a request edge. Bypassed when suspended
    # so the panda's command-vs-measured check can't accumulate dropped frames.
    if want_active:
      self.disengage_taper_remaining = DISENGAGE_TAPER_FRAMES
    elif self.disengage_taper_remaining > 0:
      self.disengage_taper_remaining -= 1

    active = want_active or (self.disengage_taper_remaining > 0 and not self.suspended)

    if want_active and self.frames_since_resume < REACTIVATION_RAMP_FRAMES:
      self.frames_since_resume += 1

    if active:
      # Stage 1: LPF on the planner target (noise reject).
      alpha = np.interp(CS.out.vEgoRaw, PLANNER_ANGLE_LP_ALPHA_BP, PLANNER_ANGLE_LP_ALPHA_V)
      filtered_target = alpha * CC.actuators.steeringAngleDeg + (1.0 - alpha) * self.planner_angle_filt
      self.planner_angle_filt = filtered_target

      # During taper, steer planner toward live EPS angle for a smooth merge
      # into the inactive (command = current angle) path.
      if not want_active and self.disengage_taper_remaining > 0:
        filtered_target = CS.out.steeringAngleDeg

      # Stage 2: jerk-limited trajectory.
      planner_angle = self.planner.update(filtered_target, CS.out.vEgoRaw)

      if want_active:
        ramp_w = min(1.0, self.frames_since_resume / REACTIVATION_RAMP_FRAMES)
      else:
        ramp_w = self.disengage_taper_remaining / DISENGAGE_TAPER_FRAMES

      out_angle = ramp_w * planner_angle + (1.0 - ramp_w) * self.last_out_angle
    else:
      self.planner_angle_filt = CS.out.steeringAngleDeg
      self.planner.reset(CS.out.steeringAngleDeg)
      out_angle = CS.out.steeringAngleDeg

    self.last_out_angle = out_angle
    self.active_last = active
    return out_angle, active
