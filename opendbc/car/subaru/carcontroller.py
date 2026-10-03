import math
import numpy as np
from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL, make_tester_present_msg
from opendbc.car.common.filter_simple import FirstOrderFilter
from opendbc.car.lateral import apply_driver_steer_torque_limits, apply_std_steer_angle_limits, common_fault_avoidance
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.subaru import subarucan
from opendbc.car.subaru.values import DBC, GLOBAL_ES_ADDR, CanBus, CarControllerParams, SubaruFlags

from opendbc.sunnypilot.car.subaru.stop_and_go import SnGCarController

# torque-path EPS rate guard (Impreza/Forester); unused on LKAS_ANGLE cars
MAX_STEER_RATE = 25  # deg/s
MAX_STEER_RATE_FRAMES = 7  # tx control frames needed before torque can be cut

# LPF in pipeline (MPC -> LPF -> panda rate limit); ticks every 10 ms, CAN sends every 20 ms
PLANNER_ANGLE_LP_TAU = ([5., 10., 20.], [0.3, 0.1, 0.0])  # m/s -> tau (s); identity above 20 m/s
MAX_LATERAL_ACCEL = 3.6  # m/s^2; ISO 11270 3.0 + 6% bank tolerance
STEER_STIFFNESS_K = 0.0015  # per (m/s)^2; matches VehicleModel default

class LkasAngleStateMachine:
  def __init__(self, CP):
    self.wheelbase = CP.wheelbase  # for MAX_LATERAL_ACCEL cap
    self.steer_ratio = CP.steerRatio  # static; paramsd live SR is applied by MPC upstream
    self.suspended = False
    self.below_release_count = 0
    self.disengage_taper_remaining = 0
    self.active_last = False
    self.dash_active = False
    self.dash_active_frames = 0
    self.engaged = False
    self.enabled_last = False
    self.planner_angle_lpf = FirstOrderFilter(0.0, PLANNER_ANGLE_LP_TAU[1][0], DT_CTRL)

  def step_filter(self, CC, CS):
    # advance LPF at 100 Hz (independent of STEER_STEP); physics-cap target to MAX_LATERAL_ACCEL; update_alpha() takes tau
    if self.engaged:
      v = max(CS.out.vEgoRaw, 1.0)
      max_angle = math.degrees((MAX_LATERAL_ACCEL / (v * v)) * self.wheelbase * self.steer_ratio * (1.0 + STEER_STIFFNESS_K * v * v))
      self.planner_angle_lpf.update_alpha(float(np.interp(CS.out.vEgo, *PLANNER_ANGLE_LP_TAU)))
      self.planner_angle_lpf.update(max(-max_angle, min(max_angle, CC.actuators.steeringAngleDeg)))
    else:
      self.planner_angle_lpf.x = CS.out.steeringAngleDeg  # pin to measured so re-engage starts at zero error

  def update(self, CC, CS):
    """State machine, called on STEER_STEP ticks. Returns (commanded_angle, active). Filter is advanced separately."""
    # suspend LKAS (held 25 frames ~0.5s) on ACC drop or MADS-only extreme angle — both would fault EPS on re-arm
    if not CC.enabled and ((self.enabled_last and not CC.latActive) or abs(CS.out.steeringAngleDeg) > 190):
      self.suspended = True
      self.below_release_count = 0
    self.enabled_last = CC.enabled

    if self.suspended:
      self.below_release_count += 1
      if self.below_release_count >= 25:
        self.suspended = False
        self.below_release_count = 0

    self.engaged = CC.latActive and not self.suspended

    # pin filter to measured until LKAS_Request can rise (zero error on first active command)
    if self.engaged and not self.active_last:
      self.planner_angle_lpf.x = CS.out.steeringAngleDeg

    # 8-frame disengage taper (EyeSight watchdog) and 8-frame dash lead so ES_LKAS_State precedes LKAS_Request
    self.disengage_taper_remaining = 8 if self.engaged else max(0, self.disengage_taper_remaining - 1)
    self.dash_active = self.engaged or (self.disengage_taper_remaining > 0 and not self.suspended)
    self.dash_active_frames = min(self.dash_active_frames + 1, 8) if self.dash_active else 0
    request_active = self.dash_active and (self.active_last or self.dash_active_frames >= 8)

    # during taper, chase measured for a smooth merge back into the inactive path
    out_angle = self.planner_angle_lpf.x if (request_active and self.engaged) else CS.out.steeringAngleDeg
    self.active_last = request_active
    return out_angle, request_active

class CarController(CarControllerBase, SnGCarController):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    SnGCarController.__init__(self, CP, CP_SP)
    self.apply_torque_last = 0
    self.apply_angle_last = 0.0
    self.angle_sm = LkasAngleStateMachine(CP)

    self.cruise_button_prev = 0
    self.steer_rate_counter = 0

    self.p = CarControllerParams(CP)
    self.packer = CANPacker(DBC[CP.carFingerprint][Bus.pt])

  def handle_torque_lateral(self, CC, CS):
    apply_torque = int(round(CC.actuators.torque * self.p.STEER_MAX))

    # limits due to driver torque
    apply_torque = apply_driver_steer_torque_limits(apply_torque, self.apply_torque_last, CS.out.steeringTorque, self.p)

    if not CC.latActive:
      apply_torque = 0

    if self.CP.flags & SubaruFlags.PREGLOBAL:
      msg = subarucan.create_preglobal_steering_control(self.packer, self.frame // self.p.STEER_STEP, apply_torque, CC.latActive)
    else:
      apply_steer_req = CC.latActive
      if self.CP.flags & SubaruFlags.STEER_RATE_LIMITED:
        # Steering rate fault prevention
        self.steer_rate_counter, apply_steer_req = \
          common_fault_avoidance(abs(CS.out.steeringRateDeg) > MAX_STEER_RATE, apply_steer_req,
                                 self.steer_rate_counter, MAX_STEER_RATE_FRAMES)
      msg = subarucan.create_steering_control(self.packer, apply_torque, apply_steer_req)

    self.apply_torque_last = apply_torque
    return msg

  def handle_angle_lateral(self, CC, CS):
    planner_angle, active = self.angle_sm.update(CC, CS)
    self.apply_angle_last = apply_std_steer_angle_limits(planner_angle, self.apply_angle_last, CS.out.vEgoRaw,
                                                        CS.out.steeringAngleDeg, active, self.p.ANGLE_LIMITS)
    return subarucan.create_steering_control_angle(self.packer, self.apply_angle_last, active)

  def update(self, CC, CC_SP, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    pcm_cancel_cmd = CC.cruiseControl.cancel

    can_sends = []

    # *** steering ***
    if self.CP.flags & SubaruFlags.LKAS_ANGLE:
      # Advance the LPF at the full 100 Hz control rate so its dynamics don't depend on STEER_STEP.
      self.angle_sm.step_filter(CC, CS)
    if (self.frame % self.p.STEER_STEP) == 0:
      if self.CP.flags & SubaruFlags.LKAS_ANGLE:
        can_sends.append(self.handle_angle_lateral(CC, CS))
      else:
        can_sends.append(self.handle_torque_lateral(CC, CS))

    # *** longitudinal ***

    if CC.longActive:
      apply_throttle = int(round(np.interp(actuators.accel, CarControllerParams.THROTTLE_LOOKUP_BP, CarControllerParams.THROTTLE_LOOKUP_V)))
      apply_rpm = int(round(np.interp(actuators.accel, CarControllerParams.RPM_LOOKUP_BP, CarControllerParams.RPM_LOOKUP_V)))
      apply_brake = int(round(np.interp(actuators.accel, CarControllerParams.BRAKE_LOOKUP_BP, CarControllerParams.BRAKE_LOOKUP_V)))

      # limit min and max values
      cruise_throttle = np.clip(apply_throttle, CarControllerParams.THROTTLE_MIN, CarControllerParams.THROTTLE_MAX)
      cruise_rpm = np.clip(apply_rpm, CarControllerParams.RPM_MIN, CarControllerParams.RPM_MAX)
      cruise_brake = np.clip(apply_brake, CarControllerParams.BRAKE_MIN, CarControllerParams.BRAKE_MAX)
    else:
      cruise_throttle = CarControllerParams.THROTTLE_INACTIVE
      cruise_rpm = CarControllerParams.RPM_MIN
      cruise_brake = CarControllerParams.BRAKE_MIN

    # *** alerts and pcm cancel ***
    if self.CP.flags & SubaruFlags.PREGLOBAL:
      if self.frame % 5 == 0:
        # 1 = main, 2 = set shallow, 3 = set deep, 4 = resume shallow, 5 = resume deep
        # disengage ACC when OP is disengaged
        if pcm_cancel_cmd:
          cruise_button = 1
        # turn main on if off and past start-up state
        elif not CS.out.cruiseState.available and CS.ready:
          cruise_button = 1
        else:
          cruise_button = CS.cruise_button

        # unstick previous mocked button press
        if cruise_button == 1 and self.cruise_button_prev == 1:
          cruise_button = 0
        self.cruise_button_prev = cruise_button

        can_sends.append(subarucan.create_preglobal_es_distance(self.packer, cruise_button, CS.es_distance_msg))

    else:
      if self.CP.flags & SubaruFlags.LKAS_ANGLE:
        # dash leads the request so an active-dash frame reaches the EPS before LKAS_Request rises
        lkas_dash_active = self.angle_sm.dash_active and not CS.out.steerFaultPermanent
      else:
        lkas_dash_active = CC.latActive

      if self.frame % 10 == 0:
        can_sends.append(subarucan.create_es_dashstatus(self.packer, self.frame // 10, CS.es_dashstatus_msg, CC.enabled,
                                                        self.CP.openpilotLongitudinalControl, CC.longActive, hud_control.leadVisible))

        can_sends.append(subarucan.create_es_lkas_state(self.packer, self.frame // 10, CS.es_lkas_state_msg, lkas_dash_active, hud_control.visualAlert,
                                                        hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                                                        hud_control.leftLaneDepart, hud_control.rightLaneDepart))

        if self.CP.flags & SubaruFlags.SEND_INFOTAINMENT:
          can_sends.append(subarucan.create_es_infotainment(self.packer, self.frame // 10, CS.es_infotainment_msg, hud_control.visualAlert))

      if self.CP.openpilotLongitudinalControl:
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_status(self.packer, self.frame // 5, CS.es_status_msg,
                                                      self.CP.openpilotLongitudinalControl, CC.longActive, cruise_rpm))

          can_sends.append(subarucan.create_es_brake(self.packer, self.frame // 5, CS.es_brake_msg,
                                                     self.CP.openpilotLongitudinalControl, CC.longActive, cruise_brake))

          can_sends.append(subarucan.create_es_distance(self.packer, self.frame // 5, CS.es_distance_msg, 0, pcm_cancel_cmd,
                                                        self.CP.openpilotLongitudinalControl, cruise_brake > 0, cruise_throttle))
      else:
        # skip while braking: car cancels ACC itself; forged-counter frame vs camera's live ES_Distance can fault EyeSight/EPS.
        if pcm_cancel_cmd and not CS.out.brakePressed:
          if not (self.CP.flags & SubaruFlags.HYBRID):
            bus = CanBus.alt if self.CP.flags & SubaruFlags.GLOBAL_GEN2 else CanBus.main
            can_sends.append(subarucan.create_es_distance(self.packer, CS.es_distance_msg["COUNTER"] + 1, CS.es_distance_msg, bus, pcm_cancel_cmd))

      if self.CP.flags & SubaruFlags.DISABLE_EYESIGHT:
        # Tester present (keeps eyesight disabled)
        if self.frame % 100 == 0:
          can_sends.append(make_tester_present_msg(GLOBAL_ES_ADDR, CanBus.camera, suppress_response=True))

        # Create all of the other eyesight messages to keep the rest of the car happy when eyesight is disabled
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_highbeamassist(self.packer))

        if self.frame % 10 == 0:
          can_sends.append(subarucan.create_es_static_1(self.packer))

        if self.frame % 2 == 0:
          can_sends.append(subarucan.create_es_static_2(self.packer))

    can_sends.extend(SnGCarController.create_stop_and_go(self, self.packer, CC, CS, self.frame))

    new_actuators = actuators.as_builder()
    if self.CP.flags & SubaruFlags.LKAS_ANGLE:
      new_actuators.steeringAngleDeg = self.apply_angle_last
    new_actuators.torque = self.apply_torque_last / self.p.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last

    self.frame += 1
    return new_actuators, can_sends
