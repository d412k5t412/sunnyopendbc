import numpy as np
from opendbc.can import CANPacker
from opendbc.car import Bus, make_tester_present_msg
from opendbc.car.lateral import (apply_driver_steer_torque_limits, apply_std_steer_angle_limits,
                                 apply_steer_angle_limits_vm, common_fault_avoidance)
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.subaru import subarucan
from opendbc.car.subaru.values import DBC, GLOBAL_ES_ADDR, CanBus, CarControllerParams, SubaruFlags
from opendbc.car.vehicle_model import VehicleModel

from opendbc.sunnypilot.car.subaru.stop_and_go import SnGCarController

# FIXME: These limits aren't exact. The real limit is more than likely over a larger time period and
# involves the total steering angle change rather than rate, but these limits work well for now
MAX_STEER_RATE = 25  # deg/s
MAX_STEER_RATE_FRAMES = 7  # tx control frames needed before torque can be cut

LOW_SPEED_HANDOFF = 1.0   # m/s (~2 mph)   below: command measured (LKAS off)
LOW_SPEED_BLEND   = 5.0   # m/s (~11 mph)  above: full planner authority
DRIVER_OVERRIDE_TORQUE        = 85    # raw Steering_Torque sensor units (engage)
DRIVER_OVERRIDE_TORQUE_RELEASE = 50   # below this for SUSPEND_HOLD_FRAMES = release
MADS_ONLY_MAX_STEER_ANGLE = 60.0   # degrees - bound MADS-only authority well under runaway range
SUSPEND_HOLD_FRAMES = 25           # ~0.5 s at 50 Hz STEER_STEP
REACTIVATION_RAMP_FRAMES = 35      # ~0.7 s at 50 Hz STEER_STEP
PLANNER_ANGLE_LP_ALPHA = 0.4

class CarController(CarControllerBase, SnGCarController):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    SnGCarController.__init__(self, CP, CP_SP)
    self.apply_torque_last = 0
    self.apply_angle_last = 0
    self.planner_angle_filt = 0.0
    self.suspended = False
    self.below_release_count = 0
    self.frames_since_resume = REACTIVATION_RAMP_FRAMES  # start with no ramp

    self.cruise_button_prev = 0
    self.steer_rate_counter = 0
    self.es_disengage_frames = 1000

    self.p = CarControllerParams(CP)
    self.packer = CANPacker(DBC[CP.carFingerprint][Bus.pt])

    self.VM = VehicleModel(CP)

  def handle_angle_lateral(self, CC, CS):
    torque = abs(CS.out.steeringTorque)
    extreme_angle = abs(CS.out.steeringAngleDeg) > MADS_ONLY_MAX_STEER_ANGLE
    extreme_angle_mads_only = extreme_angle and not CC.enabled

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

    active = CC.latActive and not self.suspended

    if active and self.frames_since_resume < REACTIVATION_RAMP_FRAMES:
      self.frames_since_resume += 1

    if active:
      planner_angle = (PLANNER_ANGLE_LP_ALPHA * CC.actuators.steeringAngleDeg +
                       (1.0 - PLANNER_ANGLE_LP_ALPHA) * self.planner_angle_filt)
      self.planner_angle_filt = planner_angle

      v = CS.out.vEgoRaw
      if v < LOW_SPEED_HANDOFF:
        speed_w = 0.0
      elif v < LOW_SPEED_BLEND:
        speed_w = (v - LOW_SPEED_HANDOFF) / (LOW_SPEED_BLEND - LOW_SPEED_HANDOFF)
      else:
        speed_w = 1.0
      ramp_w = min(1.0, self.frames_since_resume / REACTIVATION_RAMP_FRAMES)
      w = speed_w * ramp_w
      apply_angle = w * planner_angle + (1.0 - w) * CS.out.steeringAngleDeg
    else:
      self.planner_angle_filt = CS.out.steeringAngleDeg
      apply_angle = CS.out.steeringAngleDeg

    apply_steer = apply_steer_angle_limits_vm(apply_angle, self.apply_angle_last, CS.out.vEgoRaw,
                                              CS.out.steeringAngleDeg, active, self.p, self.VM)

    apply_steer = apply_std_steer_angle_limits(apply_steer, self.apply_angle_last, CS.out.vEgoRaw,
                                               CS.out.steeringAngleDeg, active, self.p.ANGLE_LIMITS)

    self.apply_angle_last = apply_steer

    return subarucan.create_steering_control_angle(self.packer, apply_steer, active)

  def handle_torque_lateral(self, CC, CS):
    apply_torque = int(round(CC.actuators.torque * self.p.STEER_MAX))

    new_torque = int(round(apply_torque))
    apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorque, self.p)

    if not CC.latActive:
      apply_torque = 0

    msg = None
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

  def update(self, CC, CC_SP, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    pcm_cancel_cmd = CC.cruiseControl.cancel

    can_sends = []

    # *** steering ***
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
      if CC.enabled:
        self.es_disengage_frames = 0
      else:
        self.es_disengage_frames = min(self.es_disengage_frames + 1, 1000)
      es_enabled = self.es_disengage_frames < 50 or (CS.out.brakePressed and self.es_disengage_frames < 500)

      if self.CP.flags & SubaruFlags.LKAS_ANGLE:
        lkas_dash_active = not CS.out.steerFaultPermanent and CC.latActive
      else:
        lkas_dash_active = es_enabled

      if self.frame % 10 == 0:
        can_sends.append(subarucan.create_es_dashstatus(self.packer, self.frame // 10, CS.es_dashstatus_msg, es_enabled,
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
        if pcm_cancel_cmd:
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
