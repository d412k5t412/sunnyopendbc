"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Subaru LKAS_ANGLE lateral extension: driver-override hysteresis, MADS-only
guards, engage/disengage shaping, and a jerk-limited motion planner that
bounds both angle rate and angle acceleration on the commanded output.
"""

import math
import numpy as np

from opendbc.car.vehicle_model import VehicleModel

DRIVER_OVERRIDE_TORQUE = 120
DRIVER_OVERRIDE_TORQUE_RELEASE = 100     # must clear resting-hand torque so light grip doesn't block resume
WHEEL_SETTLED_RATE = 25.                 # deg/s; torque dips mid-maneuver, wheel motion doesn't
RESUME_MAX_TARGET_ERR = 20.              # deg; don't take over while plan and hand-held angle disagree
SUSPEND_HOLD_FRAMES = 25                 # ~0.5 s
MADS_ONLY_MAX_STEER_ANGLE = 120          # deg

PRE_ENGAGE_CLEAN_FRAMES = 5              # ~100 ms
DISENGAGE_TAPER_FRAMES = 8               # ~160 ms; keeps LKAS_Request from edge-falling

# Roll compensation in actuators.steeringAngleDeg diverges as v -> 0 and cranks the wheel at stops on
# crowned roads; fade to a roll-free target rebuilt from actuators.curvature when approaching a stop.
ROLL_COMP_FADE_BP = [2.0, 8.0]           # m/s
ROLL_COMP_FADE_V  = [0.0, 1.0]

# Noise filter on the planner target. Heavy below ~10 mph where the model target flails
# (its v^2 curvature limits go vacuous at creep speed) and EPS jitter propagates as wobble.
PLANNER_ANGLE_LP_ALPHA_BP = [0., 4.5, 13., 18., 30.]    # m/s
PLANNER_ANGLE_LP_ALPHA_V  = [0.06, 0.13, 0.28, 0.33, 0.30]


class AnglePlanner:
  """Jerk-limited motion planner for the LKAS_ANGLE command: bounds rate and
  acceleration so corrections build and release smoothly instead of stepping."""

  # Asymmetric like ANGLE_RATE_LIMIT_UP/DOWN. Below 4.5 m/s authority is deliberately reduced —
  # the model target flails at creep speed and full envelope rates saw the wheel at parking pace.
  MAX_RATE_BP     = [0., 0.9, 2.2, 3.1, 4.5, 15., 35.]              # m/s
  MAX_RATE_UP_V   = [0.35, 0.35, 0.45, 0.55, 0.72, 0.54, 0.18]      # deg/frame
  MAX_RATE_DOWN_V = [0.45, 0.45, 0.60, 0.75, 1.05, 0.80, 0.22]      # deg/frame

  # Tuned so reaching peak rate from rest takes ~0.25-0.30 s at every speed; gentler below 7 mph.
  MAX_ACCEL_BP = [0., 3.1, 5., 15., 35.]             # m/s
  MAX_ACCEL_V  = [0.025, 0.032, 0.035, 0.030, 0.012] # deg/frame^2

  # Scale accel up with error so big maneuvers (lane changes, recovery) don't feel sluggish.
  ERR_SCALE_BP = [1.5, 15.0]                         # deg wheel
  ERR_SCALE_V  = [1.0, 3.0]

  def __init__(self):
    self.pos = 0.0
    self.vel = 0.0

  def reset(self, angle: float) -> None:
    self.pos = float(angle)
    self.vel = 0.0

  def update(self, target: float, v_ego: float) -> float:
    err = float(target) - self.pos

    # moving away from center uses UP limits, unwinding toward center uses the looser DOWN limits
    winding_up = self.pos * np.sign(err) >= 0.
    rate_v = self.MAX_RATE_UP_V if winding_up else self.MAX_RATE_DOWN_V
    max_rate       = float(np.interp(v_ego, self.MAX_RATE_BP,  rate_v))
    base_max_accel = float(np.interp(v_ego, self.MAX_ACCEL_BP, self.MAX_ACCEL_V))
    max_accel = base_max_accel * float(np.interp(abs(err), self.ERR_SCALE_BP, self.ERR_SCALE_V))

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
  def __init__(self, CP):
    self.VM = VehicleModel(CP)
    self.suspended = False
    self.below_release_count = 0
    self.pre_engage_clean_frames = 0
    self.disengage_taper_remaining = 0
    self.active_last = False
    self.enabled_last = False
    self.planner_angle_filt = 0.0
    self.planner = AnglePlanner()

  def _target_angle(self, CC, CS) -> float:
    """actuators.steeringAngleDeg with roll compensation faded out approaching a stop."""
    w = float(np.interp(CS.out.vEgoRaw, ROLL_COMP_FADE_BP, ROLL_COMP_FADE_V))
    if w >= 1.0:
      return CC.actuators.steeringAngleDeg
    angle_no_roll = math.degrees(self.VM.get_steer_from_curvature(-CC.actuators.curvature, CS.out.vEgoRaw, 0.0))
    return w * CC.actuators.steeringAngleDeg + (1.0 - w) * angle_no_roll

  def update(self, CC, CS):
    """Returns (commanded_angle, active) — feed to apply_std_steer_angle_limits."""
    torque = abs(CS.out.steeringTorque)
    extreme_angle = abs(CS.out.steeringAngleDeg) > MADS_ONLY_MAX_STEER_ANGLE
    extreme_angle_mads_only = extreme_angle and not CC.enabled
    target_angle = self._target_angle(CC, CS)

    # handoff is clear only when torque, wheel motion, and plan-vs-hand disagreement are all low
    handoff_clear = (torque < DRIVER_OVERRIDE_TORQUE_RELEASE
                     and abs(CS.out.steeringRateDeg) < WHEEL_SETTLED_RATE
                     and abs(target_angle - CS.out.steeringAngleDeg) < RESUME_MAX_TARGET_ERR
                     and not extreme_angle_mads_only)

    # pre-engage clean-frame gate
    if handoff_clear:
      self.pre_engage_clean_frames = min(self.pre_engage_clean_frames + 1, PRE_ENGAGE_CLEAN_FRAMES)
    else:
      self.pre_engage_clean_frames = 0
    pre_engage_ok = self.pre_engage_clean_frames >= PRE_ENGAGE_CLEAN_FRAMES

    # Stock lane centering only runs with ACC: the EPS hard-faults if LKAS_Request rides through an
    # ACC-engaged -> off transition, so drop the request at the edge and re-engage via the gates.
    if self.enabled_last and not CC.enabled:
      self.suspended = True
      self.below_release_count = 0
    self.enabled_last = CC.enabled

    # suspend hysteresis on driver override / extreme angle
    if self.suspended:
      if handoff_clear:
        self.below_release_count += 1
        if self.below_release_count >= SUSPEND_HOLD_FRAMES:
          self.suspended = False
          self.below_release_count = 0
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
      self.planner_angle_filt = CS.out.steeringAngleDeg
      self.planner.reset(CS.out.steeringAngleDeg)

    # Taper holds LKAS_Request high briefly on clean disengage so the EyeSight watchdog doesn't
    # see a request edge; bypassed when suspended so command-vs-measured frames can't get dropped.
    if want_active:
      self.disengage_taper_remaining = DISENGAGE_TAPER_FRAMES
    elif self.disengage_taper_remaining > 0:
      self.disengage_taper_remaining -= 1

    active = want_active or (self.disengage_taper_remaining > 0 and not self.suspended)

    if active:
      # Stage 1: LPF on the planner target (noise reject).
      alpha = np.interp(CS.out.vEgoRaw, PLANNER_ANGLE_LP_ALPHA_BP, PLANNER_ANGLE_LP_ALPHA_V)
      self.planner_angle_filt = alpha * target_angle + (1.0 - alpha) * self.planner_angle_filt

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
