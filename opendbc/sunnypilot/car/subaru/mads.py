"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from enum import StrEnum
from opendbc.car import Bus, structs

from opendbc.car.subaru.values import SubaruFlags
from opendbc.sunnypilot.mads_base import MadsCarStateBase
from opendbc.can.parser import CANParser

ButtonType = structs.CarState.ButtonEvent.Type

LKAS_DASH_STATE_ACTIVE_MIN = 1
LKAS_DASH_STATE_ACTIVE_MAX = 3

LKAS_DASH_STATE_DEBOUNCE_FRAMES = 3

class MadsCarState(MadsCarStateBase):
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    super().__init__(CP, CP_SP)
    self._lkas_button_debounce = 0   # frames the raw value has differed from stable
    self._lkas_button_stable = 0     # last debounce-accepted LKAS_Dash_State value

  @staticmethod
  def create_lkas_button_events(cur_btn: int, prev_btn: int,
                                buttons_dict: dict[int, structs.CarState.ButtonEvent.Type]) -> list[structs.CarState.ButtonEvent]:
    events: list[structs.CarState.ButtonEvent] = []

    if cur_btn == prev_btn:
      return events

    prev_active = LKAS_DASH_STATE_ACTIVE_MIN <= prev_btn <= LKAS_DASH_STATE_ACTIVE_MAX
    cur_active = LKAS_DASH_STATE_ACTIVE_MIN <= cur_btn <= LKAS_DASH_STATE_ACTIVE_MAX

    if prev_active != cur_active:
      events.append(structs.CarState.ButtonEvent(
        pressed=True,
        type=ButtonType.lkas,
      ))

    return events

  def update_mads(self, ret: structs.CarState, can_parsers: dict[StrEnum, CANParser]) -> None:
    cp_cam = can_parsers[Bus.cam]

    self.prev_lkas_button = self.lkas_button
    if not self.CP.flags & SubaruFlags.PREGLOBAL:
      raw_btn = int(cp_cam.vl["ES_LKAS_State"]["LKAS_Dash_State"])

      self._lkas_button_debounce = self._lkas_button_debounce + 1 if raw_btn != self._lkas_button_stable else 0
      if self._lkas_button_debounce >= LKAS_DASH_STATE_DEBOUNCE_FRAMES:
        self._lkas_button_stable = raw_btn
        self._lkas_button_debounce = 0

      self.lkas_button = self._lkas_button_stable

    ret.buttonEvents = self.create_lkas_button_events(self.lkas_button, self.prev_lkas_button, {1: ButtonType.lkas})
