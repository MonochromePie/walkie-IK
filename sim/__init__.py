"""Simulation-only code: MuJoCo backend, target GUI and the loop that ties them to the IK."""
from .mujoco_sim import MujocoSim
from .target_gui import TargetGUI
from .control_loop import run_control_loop

__all__ = ["MujocoSim", "TargetGUI", "run_control_loop"]
