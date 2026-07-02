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
DRIVER_OVERRIDE_TORQUE_RELEASE = 100     # resting-hand torque logs at p90 ~40-90; release must clear it
WHEEL_SETTLED_RATE = 25.                 # deg/s; torque dips mid-maneuver, wheel motion doesn't
RESUME_MAX_TARGET_ERR = 20.              # deg; don't take over while plan and hand-held angle disagree
SUSPEND_HOLD_FRAMES = 25                 # ~0.5 s
RESUME_CHURN_MAX_LEVEL = 3               # hold doubles per churn level: 0.5 -> 1 -> 2 -> 4 s
CHURN_DECAY_FRAMES = 750                 # ~15 s without override forgives one churn level
MADS_ONLY_MAX_STEER_ANGLE = 120          # deg

PRE_ENGAGE_CLEAN_FRAMES = 5              # ~100 ms
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
  MAX_RATE_V  = [0.70, 0.60, 0.45, 0.37, 0.14]       # deg/frame

  # Tuned so reaching peak rate from rest takes ~0.25-0.30 s at every speed.
  MAX_ACCEL_BP = [0., 5., 15., 35.]                  # m/s
  MAX_ACCEL_V  = [0.050, 0.035, 0.030, 0.012]        # deg/frame^2

  # Scale accel up when error is large (lane changes, recovery) so big
  # maneuvers don't feel sluggish; small corrections keep the smooth profile.
  ERR_SCALE_BP = [1.5, 15.0]                         # deg wheel
  ERR_SCALE_V  = [1.0, 3.0]

  # Speed-graded deadband; catches low-freq model sway through the stop-and-go band.
  DEADBAND_BP = [0.5, 1.5, 3.0, 6.0, 9.0, 13.0]      # m/s
  DEADBAND_V  = [3.0, 4.0, 4.0, 3.0, 2.0, 0.0]       # deg wheel

  # Deadband gate on smoothed *signed* target rate: sway alternates sign (filter ~0), real curves sustain it.
  TARGET_RATE_LP_ALPHA = 0.05                        # ~0.4 s tau at 50 Hz
  DEADBAND_RATE_BP = [0.03, 0.09]                    # deg/frame (1.5-4.5 deg/s)
  DEADBAND_RATE_V  = [1.0, 0.0]

  def __init__(self):
    self.pos = 0.0
    self.vel = 0.0
    self.last_target = 0.0
    self.target_rate_filt = 0.0

  def reset(self, angle: float) -> None:
    self.pos = float(angle)
    self.vel = 0.0
    self.last_target = float(angle)
    self.target_rate_filt = 0.0

  def update(self, target: float, v_ego: float) -> float:
    max_rate       = float(np.interp(v_ego, self.MAX_RATE_BP,  self.MAX_RATE_V))
    base_max_accel = float(np.interp(v_ego, self.MAX_ACCEL_BP, self.MAX_ACCEL_V))
    deadband       = float(np.interp(v_ego, self.DEADBAND_BP,  self.DEADBAND_V))

    self.target_rate_filt += self.TARGET_RATE_LP_ALPHA * ((float(target) - self.last_target) - self.target_rate_filt)
    self.last_target = float(target)
    deadband *= float(np.interp(abs(self.target_rate_filt), self.DEADBAND_RATE_BP, self.DEADBAND_RATE_V))

    err = float(target) - self.pos

    if abs(err) <= deadband:
      new_vel = float(np.clip(0.0, self.vel - base_max_accel, self.vel + base_max_accel))
      self.pos += new_vel
      self.vel = new_vel
      return self.pos

    # Soft-boundary: shrink effective error by deadband so exit ramps up gently.
    eff_err = err - np.sign(err) * deadband
    accel_scale = float(np.interp(abs(eff_err), self.ERR_SCALE_BP, self.ERR_SCALE_V))
    max_accel = base_max_accel * accel_scale
    err = eff_err

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
    self.disengage_taper_remaining = 0
    self.active_last = False
    self.planner_angle_filt = 0.0
    self.churn_level = 0
    self.frames_since_suspend = CHURN_DECAY_FRAMES * (RESUME_CHURN_MAX_LEVEL + 1)
    self.planner = AnglePlanner()

  def update(self, CC, CS):
    """Returns (commanded_angle, active) — feed to apply_std_steer_angle_limits."""
    torque = abs(CS.out.steeringTorque)
    extreme_angle = abs(CS.out.steeringAngleDeg) > MADS_ONLY_MAX_STEER_ANGLE
    extreme_angle_mads_only = extreme_angle and not CC.enabled

    # handoff is clear only when torque, wheel motion, and plan-vs-hand disagreement are all low
    handoff_clear = (torque < DRIVER_OVERRIDE_TORQUE_RELEASE
                     and abs(CS.out.steeringRateDeg) < WHEEL_SETTLED_RATE
                     and abs(CC.actuators.steeringAngleDeg - CS.out.steeringAngleDeg) < RESUME_MAX_TARGET_ERR
                     and not extreme_angle_mads_only)

    # pre-engage clean-frame gate
    if handoff_clear:
      self.pre_engage_clean_frames = min(self.pre_engage_clean_frames + 1, PRE_ENGAGE_CLEAN_FRAMES)
    else:
      self.pre_engage_clean_frames = 0
    pre_engage_ok = self.pre_engage_clean_frames >= PRE_ENGAGE_CLEAN_FRAMES

    self.frames_since_suspend = min(self.frames_since_suspend + 1, CHURN_DECAY_FRAMES * (RESUME_CHURN_MAX_LEVEL + 1))

    # suspend hysteresis on driver override / extreme angle; hold doubles per churn level, decays ~15 s/level
    if self.suspended:
      if handoff_clear:
        self.below_release_count += 1
        if self.below_release_count >= SUSPEND_HOLD_FRAMES << self.churn_level:
          self.suspended = False
          self.below_release_count = 0
      else:
        self.below_release_count = 0
    else:
      if torque > DRIVER_OVERRIDE_TORQUE or extreme_angle_mads_only:
        decayed = max(0, self.churn_level - self.frames_since_suspend // CHURN_DECAY_FRAMES)
        self.churn_level = min(decayed + 1, RESUME_CHURN_MAX_LEVEL)
        self.suspended = True
        self.below_release_count = 0
        self.frames_since_suspend = 0

    want_active = CC.latActive and not self.suspended
    if want_active and not self.active_last and not pre_engage_ok:
      want_active = False

    if want_active and not self.active_last:
      self.planner_angle_filt = CS.out.steeringAngleDeg
      self.planner.reset(CS.out.steeringAngleDeg)

    # Disengage taper keeps LKAS_Request high for a few frames on clean disengage
    # so the EyeSight watchdog doesn't see a request edge. Bypassed when suspended
    # so the panda's command-vs-measured check can't accumulate dropped frames.
    if want_active:
      self.disengage_taper_remaining = DISENGAGE_TAPER_FRAMES
    elif self.disengage_taper_remaining > 0:
      self.disengage_taper_remaining -= 1

    active = want_active or (self.disengage_taper_remaining > 0 and not self.suspended)

    if active:
      # Stage 1: LPF on the planner target (noise reject).
      alpha = np.interp(CS.out.vEgoRaw, PLANNER_ANGLE_LP_ALPHA_BP, PLANNER_ANGLE_LP_ALPHA_V)
      self.planner_angle_filt = alpha * CC.actuators.steeringAngleDeg + (1.0 - alpha) * self.planner_angle_filt

      # During taper, chase the live EPS angle for a smooth merge into the inactive path.
      target = self.planner_angle_filt if want_active else CS.out.steeringAngleDeg

      # Stage 2: jerk-limited trajectory (accel bound also shapes engage pull-in).
      out_angle = self.planner.update(target, CS.out.vEgoRaw)
    else:
      self.planner_angle_filt = CS.out.steeringAngleDeg
      self.planner.reset(CS.out.steeringAngleDeg)
      out_angle = CS.out.steeringAngleDeg

    self.active_last = active
    return out_angle, active
