import unittest
from types import SimpleNamespace

from opendbc.car.subaru.carcontroller import CarController
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR, SubaruFlags


class TestSubaruCarController(unittest.TestCase):
  def test_lkas_angle_rising_edge_uses_live_steering_angle(self):
    STALE_APPLY_ANGLE_LAST = 50.0
    LIVE_ANGLE = 2.61
    PLANNER_ANGLE = 1.11
    V_EGO_RAW = 8.791

    for car in CAR.with_flags(SubaruFlags.LKAS_ANGLE):
      with self.subTest(car=car):
        CP = CarInterface.get_non_essential_params(car)
        CP_SP = CarInterface.get_non_essential_params_sp(CP, car)
        controller = CarController({}, CP, CP_SP)
        controller.apply_angle_last = STALE_APPLY_ANGLE_LAST

        cs = SimpleNamespace(out=SimpleNamespace(
          vEgoRaw=V_EGO_RAW,
          steeringAngleDeg=LIVE_ANGLE,
        ))
        cc = SimpleNamespace(
          latActive=True,
          enabled=False,
          actuators=SimpleNamespace(steeringAngleDeg=PLANNER_ANGLE),
        )

        controller.handle_angle_lateral(cc, cs)

        self.assertAlmostEqual(controller.apply_angle_last, LIVE_ANGLE)


if __name__ == "__main__":
  unittest.main()
