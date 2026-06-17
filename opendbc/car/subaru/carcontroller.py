import numpy as np
from opendbc.can import CANPacker
from opendbc.car import Bus, make_tester_present_msg
from opendbc.car.lateral import apply_driver_steer_torque_limits, apply_std_steer_angle_limits, common_fault_avoidance
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.subaru import subarucan
from opendbc.car.subaru.values import DBC, GLOBAL_ES_ADDR, CanBus, CarControllerParams, SubaruFlags

from opendbc.sunnypilot.car.subaru.stop_and_go import SnGCarController

# FIXME: These limits aren't exact. The real limit is more than likely over a larger time period and
# involves the total steering angle change rather than rate, but these limits work well for now
MAX_STEER_RATE = 25  # deg/s
MAX_STEER_RATE_FRAMES = 7  # tx control frames needed before torque can be cut

ANGLE_ENGAGE_MAX_STEER_RATE = 2.0      # deg/s
ANGLE_ENGAGE_RATE_SETTLE_FRAMES = 30   # 0.3 s at 100 Hz
ANGLE_ENGAGE_MAX_ANGLE_DELTA = 3.0     # deg

LOW_SPEED_ANGLE_HOLD_SPEED = 2.24  # m/s (5 mph)
LOW_SPEED_MIN_ANGLE_DELTA = 0.3    # deg/cmd at standstill
LOW_SPEED_MAX_ANGLE_DELTA = 3.0    # deg/cmd at threshold

RELEASE_MAX_ANGLE_DELTA = 0.5
RELEASE_MAX_FRAMES = 50

MADS_ONLY_MAX_STEER_ANGLE = 120.0

LOW_SPEED_FILTER_ALPHA = 0.1  # EMA weight on new sample; lower = more smoothing

class CarController(CarControllerBase, SnGCarController):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    SnGCarController.__init__(self, CP, CP_SP)
    self.apply_torque_last = 0
    self.apply_angle_last = 0
    self.lat_active_prev = False
    self.lkas_request_last = False
    self.last_high_steer_rate_frame = -ANGLE_ENGAGE_RATE_SETTLE_FRAMES
    self.release_frame_count = 0
    self.filtered_angle_cmd = None

    self.cruise_button_prev = 0
    self.steer_rate_counter = 0
    self.es_disengage_frames = 1000

    self.p = CarControllerParams(CP)
    self.packer = CANPacker(DBC[CP.carFingerprint][Bus.pt])

  def handle_angle_lateral(self, CC, CS):
    rising_edge = CC.latActive and not self.lat_active_prev
    if rising_edge:
      self.apply_angle_last = CS.out.steeringAngleDeg

    if abs(CS.out.steeringRateDeg) > ANGLE_ENGAGE_MAX_STEER_RATE:
      self.last_high_steer_rate_frame = self.frame

    if CC.latActive:
      mads_only_ok = CC.enabled or abs(CS.out.steeringAngleDeg) < MADS_ONLY_MAX_STEER_ANGLE
      if self.lkas_request_last:
        lkas_request_desired = mads_only_ok
      else:
        rate_settled = (self.frame - self.last_high_steer_rate_frame) >= ANGLE_ENGAGE_RATE_SETTLE_FRAMES
        angle_aligned = abs(CC.actuators.steeringAngleDeg - CS.out.steeringAngleDeg) < ANGLE_ENGAGE_MAX_ANGLE_DELTA
        lkas_request_desired = rate_settled and angle_aligned and mads_only_ok
    else:
      lkas_request_desired = False

    over_mads_only_angle = not CC.enabled and abs(CS.out.steeringAngleDeg) >= MADS_ONLY_MAX_STEER_ANGLE

    releasing = False
    if lkas_request_desired:
      lkas_request = True
      self.release_frame_count = 0
    elif self.lkas_request_last and not over_mads_only_angle:
      releasing = True
      self.release_frame_count += 1
      angle_error = abs(self.apply_angle_last - CS.out.steeringAngleDeg)
      lkas_request = angle_error >= RELEASE_MAX_ANGLE_DELTA and self.release_frame_count < RELEASE_MAX_FRAMES
    else:
      lkas_request = False

    self.lkas_request_last = lkas_request

    if CC.latActive and not rising_edge and not releasing:
      apply_angle = CC.actuators.steeringAngleDeg
    else:
      apply_angle = CS.out.steeringAngleDeg
      self.filtered_angle_cmd = None

    if CC.latActive and lkas_request and not releasing and CS.out.vEgoRaw < LOW_SPEED_ANGLE_HOLD_SPEED:
      if self.filtered_angle_cmd is None:
        self.filtered_angle_cmd = apply_angle
      else:
        self.filtered_angle_cmd += LOW_SPEED_FILTER_ALPHA * (apply_angle - self.filtered_angle_cmd)
      apply_angle = self.filtered_angle_cmd

      low_speed_delta = float(np.interp(CS.out.vEgoRaw, [0., LOW_SPEED_ANGLE_HOLD_SPEED],
                                        [LOW_SPEED_MIN_ANGLE_DELTA, LOW_SPEED_MAX_ANGLE_DELTA]))
      apply_angle = float(np.clip(apply_angle, self.apply_angle_last - low_speed_delta,
                                  self.apply_angle_last + low_speed_delta))
    else:
      self.filtered_angle_cmd = None

    if lkas_request:
      apply_steer = apply_std_steer_angle_limits(apply_angle, self.apply_angle_last, CS.out.vEgoRaw,
                                                 CS.out.steeringAngleDeg, True, self.p.ANGLE_LIMITS)
    else:
      apply_steer = CS.out.steeringAngleDeg

    self.apply_angle_last = apply_steer
    self.lat_active_prev = CC.latActive
    return subarucan.create_steering_control_angle(self.packer, apply_steer, lkas_request)

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
    self.lat_active_prev = CC.latActive
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
        lkas_dash_active = self.lkas_request_last
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
    new_actuators.steeringAngleDeg = self.apply_angle_last
    new_actuators.torque = self.apply_torque_last / self.p.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last

    self.frame += 1
    return new_actuators, can_sends
