import math
import numpy as np

from opendbc.car.vehicle_model import VehicleModel

DRIVER_OVERRIDE_TORQUE = 170     # log 08/02: resting-hand touches peak 120-170 (brief, no wheel motion); deliberate takeovers peak 250+ so they still trigger instantly
DRIVER_OVERRIDE_TORQUE_BLINKER = 300     # raised threshold with blinker on so nudges don't suspend during automatic lane changes
DRIVER_OVERRIDE_TORQUE_RELEASE = 100     # must clear resting-hand torque so light grip doesn't block resume
WHEEL_SETTLED_RATE = 25.                 # deg/s; torque dips mid-maneuver, wheel motion doesn't
RESUME_MAX_TARGET_ERR = 20.              # deg; don't take over while plan and hand-held angle disagree
SUSPEND_HOLD_FRAMES = 25                 # ~0.5 s
MADS_ONLY_MAX_STEER_ANGLE = 120          # deg

PRE_ENGAGE_CLEAN_FRAMES = 5              # ~100 ms
DISENGAGE_TAPER_FRAMES = 8               # ~160 ms; keeps LKAS_Request from edge-falling

# short dash lead before LKAS_Request rises; engage is latched so the request always follows and the dash is never stranded active
ENGAGE_DASH_LEAD_FRAMES = 8

# Roll compensation in actuators.steeringAngleDeg diverges as v -> 0 and cranks the wheel at stops on
# crowned roads; fade to a roll-free target rebuilt from actuators.curvature when approaching a stop.
ROLL_COMP_FADE_BP = [2.0, 8.0]           # m/s
ROLL_COMP_FADE_V  = [0.0, 1.0]

# Noise filter on the planner target; heavy at creep to kill jitter, lighter through 8-29 mph to cut LPF lag and damp the weave the highway-learned delay under-compensates
PLANNER_ANGLE_LP_ALPHA_BP = [0., 2.5, 3.5, 4.5, 9., 18., 30.]    # m/s
PLANNER_ANGLE_LP_ALPHA_V  = [0.06, 0.08, 0.12, 0.20, 0.28, 0.33, 0.30]

# Soft deadzone at creep, applied to deviation from the target trend: kills zero-mean weave while sustained alignment corrections pass through; faded by the maneuver gate, gone by 10 mph
TARGET_DEADZONE_BP = [0., 1.5, 3.0, 4.5]   # m/s
TARGET_DEADZONE_V  = [6.0, 6.0, 3.0, 0.0]  # deg

# Maneuver gate: the creep weave is zero-mean near center while real turns carry a sustained large target, so a slow trend of the target (plus fast divergence for sharp entries) blends 0=calm..1=full authority
MANEUVER_TREND_ALPHA_BP = [0., 2.5, 4.5]   # m/s
MANEUVER_TREND_ALPHA_V  = [0.01, 0.015, 0.20]
MANEUVER_GATE_BP = [10., 20.]              # deg; gate input -> unlock 0..1
MANEUVER_DIV_OFFSET = 4.0                  # deg; slack subtracted from |filt - pos| so weave-sized divergence stays locked


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

  # Scale accel up with error so big maneuvers (lane changes, recovery) don't feel sluggish.
  ERR_SCALE_BP = [1.5, 15.0]                         # deg wheel
  ERR_SCALE_V  = [1.0, 3.0]

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
    max_accel = base_max_accel * float(np.interp(abs(err), self.ERR_SCALE_BP, self.ERR_SCALE_V)) * (1. + 2. * maneuver)

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
    self.planner = AnglePlanner(angle_limits)

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
    blinker_on = CS.out.leftBlinker or CS.out.rightBlinker
    override_torque = DRIVER_OVERRIDE_TORQUE_BLINKER if blinker_on else DRIVER_OVERRIDE_TORQUE
    extreme_angle = abs(CS.out.steeringAngleDeg) > MADS_ONLY_MAX_STEER_ANGLE
    extreme_angle_mads_only = extreme_angle and not CC.enabled
    target_angle = self._target_angle(CC, CS)

    # handoff is clear only when torque, wheel motion, and plan-vs-hand disagreement are all low
    handoff_clear = (torque < DRIVER_OVERRIDE_TORQUE_RELEASE
                     and abs(CS.out.steeringRateDeg) < WHEEL_SETTLED_RATE
                     and abs(target_angle - CS.out.steeringAngleDeg) < RESUME_MAX_TARGET_ERR
                     and not extreme_angle_mads_only)

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
      if torque > override_torque or extreme_angle_mads_only:
        self.suspended = True
        self.below_release_count = 0

    # latch the engage: a fresh engage needs a clean handoff, a continued engage rides active_last; driver override or loss of latActive disengages
    raw_want = CC.latActive and not self.suspended
    if raw_want and (self.active_last or pre_engage_ok):
      self.engaged = True
    if self.suspended or not CC.latActive:
      self.engaged = False
    want_active = self.engaged

    if want_active and not self.active_last:
      self.planner_angle_filt = CS.out.steeringAngleDeg
      self.target_trend = CS.out.steeringAngleDeg
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

      # During taper, chase the live EPS angle for a smooth merge into the inactive path.
      if want_active:
        # deadzone the deviation from the slow trend (not from pos): sustained corrections ride the trend through so the car stays aligned, only zero-mean jitter is ignored
        dz = float(np.interp(CS.out.vEgoRaw, TARGET_DEADZONE_BP, TARGET_DEADZONE_V)) * (1.0 - maneuver)
        dev = self.planner_angle_filt - self.target_trend
        target = self.target_trend + float(np.sign(dev)) * max(0.0, abs(dev) - dz)
      else:
        target = CS.out.steeringAngleDeg
        maneuver = 1.0

      # Stage 2: jerk-limited trajectory (accel bound also shapes engage pull-in).
      out_angle = self.planner.update(target, CS.out.vEgoRaw, maneuver)
    else:
      # inactive or holding for the lead: pin to measured so LKAS_Request rises from zero error, not a step
      self.planner_angle_filt = CS.out.steeringAngleDeg
      self.target_trend = CS.out.steeringAngleDeg
      self.planner.reset(CS.out.steeringAngleDeg)
      out_angle = CS.out.steeringAngleDeg

    self.dash_active = dash_active
    self.active_last = request_active
    return out_angle, request_active
