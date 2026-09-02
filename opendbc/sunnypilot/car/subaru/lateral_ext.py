import math
import numpy as np

from opendbc.car.vehicle_model import VehicleModel

SUSPEND_HOLD_FRAMES = 25                 # ~0.5 s
MADS_ONLY_MAX_STEER_ANGLE = 120          # deg

PRE_ENGAGE_CLEAN_FRAMES = 5              # ~100 ms
DISENGAGE_TAPER_FRAMES = 8               # ~160 ms; keeps LKAS_Request from edge-falling

# short dash lead before LKAS_Request rises; engage is latched so the request always follows and the dash is never stranded active
ENGAGE_DASH_LEAD_FRAMES = 8

# Angle-space LPF on the planner target; safety-only since the curvature-space LPF took over the noise reject at low speed (2026-09-02).
PLANNER_ANGLE_LP_ALPHA_BP = [0., 2.5, 3.5, 4.5, 9., 18., 30.]    # m/s
PLANNER_ANGLE_LP_ALPHA_V  = [0.30, 0.30, 0.30, 0.25, 0.28, 0.33, 0.30]

# Maneuver gate: the creep weave is zero-mean near center while real turns carry a sustained large target, so a slow trend of the target (plus fast divergence for sharp entries) blends 0=calm..1=full authority
MANEUVER_TREND_ALPHA_BP = [0., 2.5, 4.5]   # m/s
MANEUVER_TREND_ALPHA_V  = [0.01, 0.015, 0.20]
MANEUVER_GATE_BP = [10., 20.]              # deg; gate input -> unlock 0..1
MANEUVER_DIV_OFFSET = 4.0                  # deg; slack subtracted from |filt - pos| so weave-sized divergence stays locked

# Curvature-space filter/deadband/clip; runs BEFORE get_steer_from_curvature where low speed 10x's the noise (2026-09-02).
CURV_LP_ALPHA_BP  = [0.,     2.5,    4.5,    9.,     15.]     # m/s
CURV_LP_ALPHA_V   = [0.05,   0.06,   0.12,   0.30,   1.0]     # slow: kills the near-zero weave. Above 34 mph: bypass (BLEND is 0 there anyway).
# Magnitude-adaptive alpha: real turn commands (|raw_curv| >= 0.005 1/m ~= 11deg wheel at rest) bypass the slow LPF for immediate response.
CURV_MAG_ALPHA_BP = [0.001, 0.005]                            # 1/m
CURV_MAG_ALPHA_V  = [0.0,   0.60]                             # replaces the slow alpha when raw curvature is turn-sized
CURV_DEADBAND_BP  = [0.,     4.5,    9.]                      # m/s
CURV_DEADBAND_V   = [0.0004, 0.0002, 0.0]                     # 1/m; kill zero-mean dead-zone weave
CURV_MAX_BP       = [0.,   2.5,   4.5,   9.,    15.]          # m/s
CURV_MAX_V        = [0.030, 0.028, 0.022, 0.015, 0.010]       # 1/m; cap real turns (0.030 @ rest ~= 68deg wheel; the LPF+deadband still kills sub-turn noise)
# Filtered curvature-derived angle at low speed, raw actuators.steeringAngleDeg (with roll comp) at highway.
CURV_BLEND_BP     = [5.,   15.]
CURV_BLEND_V      = [0.0,  1.0]

# Freeze the maneuver gate at 0 below creep so it can't limit-cycle open/closed (CALM<->OPEN is 4x rate jump).
MANEUVER_SPEED_GATE_BP = [2.0, 3.5]         # m/s
MANEUVER_SPEED_GATE_V  = [0.0, 1.0]


class AnglePlanner:
  """Jerk-limited motion planner for the LKAS_ANGLE command: bounds rate and
  acceleration so corrections build and release smoothly instead of stepping."""

  # Asymmetric like ANGLE_RATE_LIMIT_UP/DOWN; CALM rates slew-limit the near-center weave, OPEN rates restore full maneuver authority, blended by the maneuver gate
  MAX_RATE_BP      = [0., 0.9, 2.2, 3.1, 4.5, 15., 35.]              # m/s
  CALM_RATE_UP_V   = [0.08, 0.08, 0.12, 0.20, 0.45, 0.54, 0.18]      # deg/frame
  CALM_RATE_DOWN_V = [0.11, 0.11, 0.17, 0.28, 0.65, 0.80, 0.22]      # deg/frame
  OPEN_RATE_UP_V   = [0.35, 0.35, 0.45, 0.55, 0.72, 0.54, 0.18]      # deg/frame
  OPEN_RATE_DOWN_V = [0.45, 0.45, 0.60, 0.75, 1.05, 0.80, 0.22]      # deg/frame

  # Tuned so reaching peak rate from rest takes ~0.25-0.30 s at every speed; gentler below 7 mph.
  MAX_ACCEL_BP = [0., 3.1, 5., 15., 35.]             # m/s
  MAX_ACCEL_V  = [0.025, 0.032, 0.035, 0.030, 0.012] # deg/frame^2

  # Scale accel up with error so big maneuvers (lane changes, recovery) don't feel sluggish; 08/xx logs showed steerSaturated on 30mph curve entries where accel spin-up was too slow to track a 25 deg/s desired ramp
  ERR_SCALE_BP = [1.0, 5.0, 15.0]                    # deg wheel
  ERR_SCALE_V  = [1.0, 3.0, 6.0]

  # Scale peak rate up with error so sharp turns slew faster; min() against ANGLE_LIMITS still bounds the product
  RATE_ERR_SCALE_BP = [2.5, 10.0, 25.0]              # deg wheel
  RATE_ERR_SCALE_V  = [1.0, 2.0, 3.0]

  def __init__(self, angle_limits):
    self.pos = 0.0
    self.vel = 0.0
    self.angle_limits = angle_limits   # shared ANGLE_LIMITS (same object apply_std_steer_angle_limits clips to): one source of truth for the rate ceiling

  def reset(self, angle: float) -> None:
    self.pos = float(angle)
    self.vel = 0.0

  def update(self, target: float, v_ego: float, maneuver: float = 1.0) -> float:
    err = float(target) - self.pos

    # moving away from center uses UP limits, unwinding toward center uses the looser DOWN limits
    winding_up = self.pos * np.sign(err) >= 0.
    calm_v = self.CALM_RATE_UP_V if winding_up else self.CALM_RATE_DOWN_V
    open_v = self.OPEN_RATE_UP_V if winding_up else self.OPEN_RATE_DOWN_V
    base_max_rate  = float((1. - maneuver) * np.interp(v_ego, self.MAX_RATE_BP, calm_v) +
                           maneuver * np.interp(v_ego, self.MAX_RATE_BP, open_v))
    rate_boost     = float(np.interp(abs(err), self.RATE_ERR_SCALE_BP, self.RATE_ERR_SCALE_V))
    rate_lim       = self.angle_limits.ANGLE_RATE_LIMIT_UP if winding_up else self.angle_limits.ANGLE_RATE_LIMIT_DOWN
    max_rate       = min(base_max_rate * rate_boost, float(np.interp(v_ego, rate_lim[0], rate_lim[1])))
    base_max_accel = float(np.interp(v_ego, self.MAX_ACCEL_BP, self.MAX_ACCEL_V))
    max_accel = base_max_accel * float(np.interp(abs(err), self.ERR_SCALE_BP, self.ERR_SCALE_V)) * (1. + 5. * maneuver)

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
  def __init__(self, CP, angle_limits):
    self.VM = VehicleModel(CP)
    self.suspended = False
    self.below_release_count = 0
    self.pre_engage_clean_frames = 0
    self.disengage_taper_remaining = 0
    self.active_last = False
    self.dash_active = False
    self.dash_active_frames = 0
    self.engaged = False
    self.enabled_last = False
    self.planner_angle_filt = 0.0
    self.target_trend = 0.0
    self.curvature_filt = 0.0
    self.planner = AnglePlanner(angle_limits)

  def _target_angle(self, CC, CS) -> float:
    """Filter/clip curvature upstream of the low-speed angle blow-up, then speed-blend against raw actuators.steeringAngleDeg."""
    v = CS.out.vEgoRaw
    raw_curv = CC.actuators.curvature

    # LPF in curvature space: slow at near-zero (noise), fast when raw curvature is turn-sized (immediate turn-in).
    alpha_slow = float(np.interp(v, CURV_LP_ALPHA_BP, CURV_LP_ALPHA_V))
    alpha_mag  = float(np.interp(abs(raw_curv), CURV_MAG_ALPHA_BP, CURV_MAG_ALPHA_V))
    alpha = max(alpha_slow, alpha_mag)
    self.curvature_filt = alpha * raw_curv + (1.0 - alpha) * self.curvature_filt

    # Deadband: pull small signed values toward 0 to kill zero-mean weave
    db = float(np.interp(v, CURV_DEADBAND_BP, CURV_DEADBAND_V))
    if abs(self.curvature_filt) <= db:
      c_out = 0.0
    else:
      c_out = self.curvature_filt - math.copysign(db, self.curvature_filt)

    # Clip: cap absolute curvature demand (allows sharp turns up to ~68 deg wheel at rest, ~30 deg at 34 mph).
    c_max = float(np.interp(v, CURV_MAX_BP, CURV_MAX_V))
    c_out = max(-c_max, min(c_max, c_out))

    angle_from_curv = math.degrees(self.VM.get_steer_from_curvature(-c_out, v, 0.0))

    # Speed-blend filtered curvature path (low speed) vs. raw angle carrying roll comp (highway).
    w = float(np.interp(v, CURV_BLEND_BP, CURV_BLEND_V))
    return w * CC.actuators.steeringAngleDeg + (1.0 - w) * angle_from_curv

  def update(self, CC, CS):
    """Returns (commanded_angle, active) — feed to apply_std_steer_angle_limits."""
    extreme_angle = abs(CS.out.steeringAngleDeg) > MADS_ONLY_MAX_STEER_ANGLE
    extreme_angle_mads_only = extreme_angle and not CC.enabled
    target_angle = self._target_angle(CC, CS)

    # only remaining engage/resume gate: driver isn't holding the wheel past the MADS-only extreme-angle guard
    handoff_clear = not extreme_angle_mads_only

    # pre-engage clean-frame gate: require a clean driver handoff before a fresh engage
    if handoff_clear:
      self.pre_engage_clean_frames = min(self.pre_engage_clean_frames + 1, PRE_ENGAGE_CLEAN_FRAMES)
    else:
      self.pre_engage_clean_frames = 0
    pre_engage_ok = self.pre_engage_clean_frames >= PRE_ENGAGE_CLEAN_FRAMES

    # ACC dropping (e.g. brake) once suspended LKAS, but MADS lateral is independent of ACC: only
    # suspend when lateral is actually ending, so LKAS stays engaged through a brake while MADS holds it.
    if self.enabled_last and not CC.enabled and not CC.latActive:
      self.suspended = True
      self.below_release_count = 0
    self.enabled_last = CC.enabled

    # suspend hysteresis; no driver-torque override — only extreme angle (MADS-only) suspends
    if self.suspended:
      if handoff_clear:
        self.below_release_count += 1
        if self.below_release_count >= SUSPEND_HOLD_FRAMES:
          self.suspended = False
          self.below_release_count = 0
      else:
        self.below_release_count = 0
    else:
      if extreme_angle_mads_only:
        self.suspended = True
        self.below_release_count = 0

    # latch the engage: a fresh engage needs a clean handoff, a continued engage rides active_last; loss of latActive or extreme-angle suspension disengages
    raw_want = CC.latActive and not self.suspended
    if raw_want and (self.active_last or pre_engage_ok):
      self.engaged = True
    if self.suspended or not CC.latActive:
      self.engaged = False
    want_active = self.engaged

    if want_active and not self.active_last:
      self.planner_angle_filt = CS.out.steeringAngleDeg
      self.target_trend = CS.out.steeringAngleDeg
      self.curvature_filt = CC.actuators.curvature   # start synced so first frame doesn't step
      self.planner.reset(CS.out.steeringAngleDeg)

    # Taper holds LKAS_Request high briefly on clean disengage so the EyeSight watchdog doesn't
    # see a request edge; bypassed when suspended so command-vs-measured frames can't get dropped.
    if want_active:
      self.disengage_taper_remaining = DISENGAGE_TAPER_FRAMES
    elif self.disengage_taper_remaining > 0:
      self.disengage_taper_remaining -= 1

    # dash advertises intent (ES_LKAS_State); request is held back a lead so the dash reaches the EPS first.
    dash_active = want_active or (self.disengage_taper_remaining > 0 and not self.suspended)

    if dash_active:
      self.dash_active_frames = min(self.dash_active_frames + 1, ENGAGE_DASH_LEAD_FRAMES)
    else:
      self.dash_active_frames = 0

    request_active = dash_active and (self.active_last or self.dash_active_frames >= ENGAGE_DASH_LEAD_FRAMES)

    if request_active:
      # Stage 1: LPF on the planner target (noise reject).
      alpha = np.interp(CS.out.vEgoRaw, PLANNER_ANGLE_LP_ALPHA_BP, PLANNER_ANGLE_LP_ALPHA_V)
      self.planner_angle_filt = alpha * target_angle + (1.0 - alpha) * self.planner_angle_filt

      # slow trend of the target discriminates zero-mean weave from sustained turns; divergence term catches sharp entries
      trend_alpha = np.interp(CS.out.vEgoRaw, MANEUVER_TREND_ALPHA_BP, MANEUVER_TREND_ALPHA_V)
      self.target_trend = trend_alpha * self.planner_angle_filt + (1.0 - trend_alpha) * self.target_trend
      gate_in = max(abs(self.target_trend), abs(self.planner_angle_filt - self.planner.pos) - MANEUVER_DIV_OFFSET)
      maneuver = float(np.interp(gate_in, MANEUVER_GATE_BP, [0., 1.]))
      # Force calm-only under 3.5 m/s so the CALM<->OPEN 4x rate jump can't drive its own limit cycle.
      maneuver *= float(np.interp(CS.out.vEgoRaw, MANEUVER_SPEED_GATE_BP, MANEUVER_SPEED_GATE_V))

      # During taper, chase the live EPS angle for a smooth merge into the inactive path.
      if want_active:
        target = self.planner_angle_filt
      else:
        target = CS.out.steeringAngleDeg
        maneuver = 1.0

      # Stage 2: jerk-limited trajectory (accel bound also shapes engage pull-in).
      out_angle = self.planner.update(target, CS.out.vEgoRaw, maneuver)
    else:
      # inactive or holding for the lead: pin to measured so LKAS_Request rises from zero error, not a step
      self.planner_angle_filt = CS.out.steeringAngleDeg
      self.target_trend = CS.out.steeringAngleDeg
      self.curvature_filt = CC.actuators.curvature
      self.planner.reset(CS.out.steeringAngleDeg)
      out_angle = CS.out.steeringAngleDeg

    self.dash_active = dash_active
    self.active_last = request_active
    return out_angle, request_active
