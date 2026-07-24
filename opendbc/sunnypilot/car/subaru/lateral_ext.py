import math
import numpy as np

from opendbc.car.vehicle_model import VehicleModel

SUSPEND_HOLD_FRAMES = 25                 # ~0.5 s
MADS_ONLY_MAX_STEER_ANGLE = 120          # deg
PRE_ENGAGE_CLEAN_FRAMES = 5              # ~100 ms
DISENGAGE_TAPER_FRAMES = 8               # ~160 ms; keeps LKAS_Request from edge-falling
ENGAGE_DASH_LEAD_FRAMES = 8              # latched engage prevents stranded dash

# Safety-only angle-space LPF; the curvature LPF does the primary noise reject.
PLANNER_ANGLE_LP_ALPHA    = ([0., 2.5, 3.5, 4.5, 9., 18., 30.], [0.30, 0.30, 0.30, 0.25, 0.28, 0.33, 0.30])   # m/s -> alpha

# Maneuver gate: target trend + divergence detector unlock 0=calm..1=full authority.
MANEUVER_TREND_ALPHA    = ([0., 2.5, 4.5], [0.01, 0.015, 0.20])   # m/s -> trend alpha
MANEUVER_GATE = ([10., 20.], [0., 1.])       # deg -> gate 0..1
MANEUVER_DIV_OFFSET = 4.0                  # deg; slack subtracted from |filt - pos| so weave-sized divergence stays locked

# Curvature-space filter/deadband/clip, upstream of the low-speed angle blow-up.
CURV_LP_ALPHA     = ([0., 2.5, 4.5, 9., 15.], [0.05, 0.06, 0.12, 0.30, 1.0])   # slow at low speed kills weave; bypass at highway.
# Magnitude-adaptive alpha: real-turn curvature bypasses the slow LPF.
CURV_MAG_ALPHA    = ([0.001, 0.005], [0.0, 0.60])   # |curv| -> alpha boost (real-turn fast track)
CURV_DEADBAND     = ([0., 4.5, 9.], [0.0004, 0.0002, 0.0])   # m/s -> curvature deadband (1/m)
CURV_MAX          = ([0., 2.5, 4.5, 9., 15.], [0.030, 0.028, 0.022, 0.015, 0.010])   # m/s -> curvature cap (0.030 @ rest ~= 68deg wheel)
# Split MPC angle into curvature + roll comp; filter each so road crown doesn't push the car sideways.
ROLL_COMP_ALPHA   = 0.02                                      # slow LPF on roll comp (~500 ms)
ROLL_COMP_FADE    = ([1.5, 4.0], [0.0, 1.0])   # m/s -> roll-comp fade-in

# Force CALM-only below creep; the CALM<->OPEN 4x jump would limit-cycle.
MANEUVER_SPEED_GATE    = ([2.0, 3.5], [0.0, 1.0])   # m/s -> maneuver-gate authority

class AnglePlanner:
  """Jerk-limited motion planner for the LKAS_ANGLE command: bounds rate and
  acceleration so corrections build and release smoothly instead of stepping."""

  # CALM slew-limits the weave; OPEN gives full authority; blended by the maneuver gate.
  MAX_RATE_BP      = [0., 0.9, 2.2, 3.1, 4.5, 15., 35.]              # m/s
  CALM_RATE_UP_V   = [0.08, 0.08, 0.12, 0.20, 0.45, 0.54, 0.18]      # deg/frame
  CALM_RATE_DOWN_V = [0.11, 0.11, 0.17, 0.28, 0.65, 0.80, 0.22]      # deg/frame
  OPEN_RATE_UP_V   = [0.35, 0.35, 0.45, 0.55, 0.72, 0.54, 0.18]      # deg/frame
  OPEN_RATE_DOWN_V = [0.45, 0.45, 0.60, 0.75, 1.05, 0.80, 0.22]      # deg/frame

  # ~0.25-0.30 s to peak rate from rest at every speed; gentler below 7 mph.
  MAX_ACCEL = ([0., 3.1, 5., 15., 35.], [0.025, 0.032, 0.035, 0.030, 0.012])   # m/s -> deg/frame^2
  # Scale accel with error so lane changes and recovery aren't sluggish.
  ERR_SCALE = ([1.0, 5.0, 15.0], [1.0, 3.0, 6.0])   # deg wheel -> accel boost x
  # Scale rate with error for sharp turns; ANGLE_LIMITS still caps the product.
  RATE_ERR_SCALE = ([2.5, 10.0, 25.0], [1.0, 2.0, 3.0])   # deg wheel -> rate boost x

  def __init__(self, angle_limits):
    self.pos = 0.0
    self.vel = 0.0
    self.angle_limits = angle_limits   # shared with apply_std_steer_angle_limits: one source of truth for the rate ceiling.

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
    rate_boost     = float(np.interp(abs(err), *self.RATE_ERR_SCALE))
    rate_lim       = self.angle_limits.ANGLE_RATE_LIMIT_UP if winding_up else self.angle_limits.ANGLE_RATE_LIMIT_DOWN
    max_rate       = min(base_max_rate * rate_boost, float(np.interp(v_ego, rate_lim[0], rate_lim[1])))
    base_max_accel = float(np.interp(v_ego, *self.MAX_ACCEL))
    max_accel = base_max_accel * float(np.interp(abs(err), *self.ERR_SCALE)) * (1. + 5. * maneuver)

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
    self.roll_comp_filt = 0.0
    self.planner = AnglePlanner(angle_limits)

  def _curv_no_roll_angle(self, curv, v):
    return math.degrees(self.VM.get_steer_from_curvature(-curv, v, 0.0))

  def _reset_state(self, CC, CS):
    """Sync all filter/planner state to the current measured wheel + MPC roll-comp."""
    self.planner_angle_filt = CS.out.steeringAngleDeg
    self.target_trend = CS.out.steeringAngleDeg
    self.curvature_filt = CC.actuators.curvature
    self.roll_comp_filt = CC.actuators.steeringAngleDeg - self._curv_no_roll_angle(CC.actuators.curvature, CS.out.vEgoRaw)
    self.planner.reset(CS.out.steeringAngleDeg)

  def _target_angle(self, CC, CS) -> float:
    """Filter/clip curvature-derived turn intent; preserve roll comp separately so we don't drift with road crown."""
    v = CS.out.vEgoRaw
    raw_curv = CC.actuators.curvature

    # Roll comp = raw MPC angle minus pure-curvature bicycle model.
    raw_ang_no_roll = self._curv_no_roll_angle(raw_curv, v)
    roll_comp_raw = CC.actuators.steeringAngleDeg - raw_ang_no_roll
    self.roll_comp_filt = ROLL_COMP_ALPHA * roll_comp_raw + (1.0 - ROLL_COMP_ALPHA) * self.roll_comp_filt

    # LPF: slow near zero (kills noise), fast on real turns (immediate turn-in).
    alpha_slow = float(np.interp(v, *CURV_LP_ALPHA))
    alpha_mag  = float(np.interp(abs(raw_curv), *CURV_MAG_ALPHA))
    alpha = max(alpha_slow, alpha_mag)
    self.curvature_filt = alpha * raw_curv + (1.0 - alpha) * self.curvature_filt

    # Deadband: kill zero-mean weave via soft-threshold.
    db = float(np.interp(v, *CURV_DEADBAND))
    c_out = math.copysign(max(0.0, abs(self.curvature_filt) - db), self.curvature_filt)

    # Cap curvature demand; roll comp is added on top.
    c_max = float(np.interp(v, *CURV_MAX))
    c_out = max(-c_max, min(c_max, c_out))

    angle_from_curv = self._curv_no_roll_angle(c_out, v)

    # Fade roll comp in above walking pace.
    roll_scale = float(np.interp(v, *ROLL_COMP_FADE))
    return angle_from_curv + roll_scale * self.roll_comp_filt

  def update(self, CC, CS, vp=None):
    """Returns (commanded_angle, active) — feed to apply_std_steer_angle_limits."""
    # Live-track paramsd so our VM matches openpilot's runtime math.
    if vp is not None:
      self.VM.update_params(max(vp.stiffnessFactor, 0.1), max(vp.steerRatio, 0.1))
    extreme_angle_mads_only = abs(CS.out.steeringAngleDeg) > MADS_ONLY_MAX_STEER_ANGLE and not CC.enabled
    target_angle = self._target_angle(CC, CS)

    # only engage gate: not past the MADS-only extreme-angle guard.
    handoff_clear = not extreme_angle_mads_only

    # require a clean driver handoff before a fresh engage.
    self.pre_engage_clean_frames = min(self.pre_engage_clean_frames + 1, PRE_ENGAGE_CLEAN_FRAMES) if handoff_clear else 0
    pre_engage_ok = self.pre_engage_clean_frames >= PRE_ENGAGE_CLEAN_FRAMES

    # ACC drop suspends only when lateral itself ends; MADS keeps LKAS through a brake.
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

    # latch engage: fresh needs clean handoff, continued rides active_last; disengage on latActive drop or suspend.
    raw_want = CC.latActive and not self.suspended
    if raw_want and (self.active_last or pre_engage_ok):
      self.engaged = True
    if self.suspended or not CC.latActive:
      self.engaged = False
    want_active = self.engaged

    if want_active and not self.active_last:
      self._reset_state(CC, CS)

    # Taper holds LKAS_Request briefly on clean disengage (EyeSight watchdog); bypassed when suspended.
    self.disengage_taper_remaining = DISENGAGE_TAPER_FRAMES if want_active else max(0, self.disengage_taper_remaining - 1)

    # dash advertises intent (ES_LKAS_State); request is held back a lead so the dash reaches the EPS first.
    dash_active = want_active or (self.disengage_taper_remaining > 0 and not self.suspended)

    self.dash_active_frames = min(self.dash_active_frames + 1, ENGAGE_DASH_LEAD_FRAMES) if dash_active else 0

    request_active = dash_active and (self.active_last or self.dash_active_frames >= ENGAGE_DASH_LEAD_FRAMES)

    if request_active:
      # Stage 1: LPF on the planner target.
      alpha = np.interp(CS.out.vEgoRaw, *PLANNER_ANGLE_LP_ALPHA)
      self.planner_angle_filt = alpha * target_angle + (1.0 - alpha) * self.planner_angle_filt

      # trend discriminates weave from sustained turns; divergence catches sharp entries.
      trend_alpha = np.interp(CS.out.vEgoRaw, *MANEUVER_TREND_ALPHA)
      self.target_trend = trend_alpha * self.planner_angle_filt + (1.0 - trend_alpha) * self.target_trend
      gate_in = max(abs(self.target_trend), abs(self.planner_angle_filt - self.planner.pos) - MANEUVER_DIV_OFFSET)
      maneuver = float(np.interp(gate_in, *MANEUVER_GATE))
      # Force calm-only under 3.5 m/s so the CALM<->OPEN 4x rate jump can't drive its own limit cycle.
      maneuver *= float(np.interp(CS.out.vEgoRaw, *MANEUVER_SPEED_GATE))

      # During taper, chase the live EPS angle for a smooth merge into the inactive path.
      if want_active:
        target = self.planner_angle_filt
      else:
        target = CS.out.steeringAngleDeg
        maneuver = 1.0

      # Stage 2: jerk-limited trajectory (accel bound also shapes engage pull-in).
      out_angle = self.planner.update(target, CS.out.vEgoRaw, maneuver)
    else:
      # inactive or holding for the lead: pin state to measured so LKAS_Request rises from zero error
      self._reset_state(CC, CS)
      out_angle = CS.out.steeringAngleDeg

    self.dash_active = dash_active
    self.active_last = request_active
    return out_angle, request_active
