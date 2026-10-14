"""
Walkie bimanual IK (OpenArm x2 on the Walkie base).

This file has three parts:

    PART 1  ROBOT MODEL  - DH tables, joint names, joint limits. Pure kinematics.
    PART 2  IK           - BimanualIK.step(joint_state, targets) -> joint_command.
                           No simulator code: this is the part to reuse on the real robot.
    PART 3  SIM DEMO     - MuJoCo sim + target-pose GUI (code lives in sim/). Only main() touches it.

Conventions used by the IK (PART 1 + 2):
    - joint_state / joint_command: dict {joint name (str): value}, names as in JOINT_NAMES,
      radians for revolute joints, metres for prismatic (x, y, lift).
    - Targets are in the world frame (= base frame when x = y = yaw = 0).
      Orientation is roll/pitch/yaw, ZYX convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), radians.
    - The end-effector frame is the last DH frame. It equals MuJoCo's openarm_*_hand_tcp frame
      rotated by +90 deg about z.
"""
from dataclasses import dataclass

import numpy as np
import sympy as sp

from IK import DHParameter, ForwardKinematics, InverseKinematics


# =====================================================================================
# PART 1: ROBOT MODEL
# =====================================================================================

x, y, yaw, lift = sp.symbols('x y yaw lift')
q_l1, q_l2, q_l3, q_l4, q_l5, q_l6, q_l7 = sp.symbols('q_l1 q_l2 q_l3 q_l4 q_l5 q_l6 q_l7')
q_r1, q_r2, q_r3, q_r4, q_r5, q_r6, q_r7 = sp.symbols('q_r1 q_r2 q_r3 q_r4 q_r5 q_r6 q_r7')

BASE_SYMBOLS = [x, y, yaw, lift]
# base joints the IK may move, toggled per group: planar position, heading, torso height
BASE_GROUPS = {"xy": [x, y], "yaw": [yaw], "lift": [lift]}
LEFT_ARM_SYMBOLS = [q_l1, q_l2, q_l3, q_l4, q_l5, q_l6, q_l7]
RIGHT_ARM_SYMBOLS = [q_r1, q_r2, q_r3, q_r4, q_r5, q_r6, q_r7]
ALL_SYMBOLS = BASE_SYMBOLS + LEFT_ARM_SYMBOLS + RIGHT_ARM_SYMBOLS
JOINT_NAMES = [s.name for s in ALL_SYMBOLS]

# (lo, hi) per joint. Arm limits from the OpenArm model; base limits from the Walkie controllers.
JOINT_LIMITS = {
    'x': (-2.0, 2.0), 'y': (-2.0, 2.0), 'yaw': (-np.pi, np.pi), 'lift': (0.1, 2.0),
    'q_l1': (-3.490659, 1.396263), 'q_l2': (-3.316125, 0.174533), 'q_l3': (-1.570796, 1.570796),
    'q_l4': (0.0, 2.443461), 'q_l5': (-1.570796, 1.570796), 'q_l6': (-0.785398, 0.785398),
    'q_l7': (-1.570796, 1.570796),
    'q_r1': (-1.396263, 3.490659), 'q_r2': (-0.174533, 3.316125), 'q_r3': (-1.570796, 1.570796),
    'q_r4': (0.0, 2.443461), 'q_r5': (-1.570796, 1.570796), 'q_r6': (-0.785398, 0.785398),
    'q_r7': (-1.570796, 1.570796),
}

p2 = sp.pi/2
m_p2 = -1.0 * p2
h = 0.697 - 0.1 # ADJUST THIS TO MIN WALKIE BASE ARM OFFSET
l01 = 0.0915
l12 = 0.061131
l23 = 0.063
l34 = 0.157
l45 = 0.0965
l56 = 0.1195
l_eff = 0.114501 #+ 0.055

# World -> base: translate by (x, y) with no net rotation
base_xy_dh_params = [
        DHParameter(d=0.0, theta=m_p2, a=0.0, alpha=m_p2),
        DHParameter(d=x, theta=0.0, a=0.0, alpha=0.0),
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2),
        DHParameter(d=0.0, theta=p2, a=0.0, alpha=m_p2),
        DHParameter(d=y, theta=0.0, a=0.0, alpha=0.0),
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2),
    ]

left_dh_params = base_xy_dh_params + [
        DHParameter(d=lift, theta=yaw, a=0.0, alpha=0.0),
        DHParameter(d=h, theta=0.0, a=0.0, alpha=0.0),  #BASE AND LIFT
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha= m_p2),
        DHParameter(d=l01, theta=q_l1 + p2, a=0.0, alpha= 0.0), #DOF1
        DHParameter(d=l12, theta=0.0, a=0.0, alpha= m_p2), #DOF2
        DHParameter(d=0.0, theta=q_l2, a=l23, alpha= 0.0),
        DHParameter(d=0.0, theta=m_p2, a=0.0, alpha= m_p2),
        DHParameter(d=l34, theta=q_l3 + p2, a=0.0, alpha= 0.0), #DOF3
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha= m_p2),
        DHParameter(d=0.0, theta=q_l4 + m_p2, a=0.0, alpha= 0.0), #DOF4
        DHParameter(d=0.0, theta=p2, a=0.0, alpha= p2),
        DHParameter(d=l45, theta=q_l5, a=0.0, alpha= 0.0), #DOF5
        DHParameter(d=l56, theta=m_p2, a=0.0, alpha= m_p2),
        DHParameter(d=0.0, theta=q_l6 + m_p2, a=0.0, alpha= 0.0), #DOF6
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha= m_p2),
        DHParameter(d=0.0, theta=q_l7, a=l_eff, alpha= 0.0), #DOF7
        DHParameter(d=0.0, theta=m_p2, a=0.0, alpha=m_p2),
        DHParameter(d=0.0, theta=m_p2, a=0.0, alpha=0.0)  # End-effector Adjustment to convention
    ]

right_dh_params = base_xy_dh_params + [
        DHParameter(d=lift, theta=yaw, a=0.0, alpha=0.0),
        DHParameter(d=h, theta=0.0, a=0.0, alpha=0.0),  #BASE AND LIFT
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2),
        DHParameter(d=l01, theta=q_r1 + m_p2, a=0.0, alpha=0.0),#DOF1
        DHParameter(d=l12, theta=0.0, a=0.0, alpha=p2),
        DHParameter(d=0.0, theta=q_r2, a=l23, alpha=0.0), #DOF2
        DHParameter(d=0.0, theta=m_p2, a=0.0, alpha=m_p2),
        DHParameter(d=l34, theta=q_r3 + p2, a=0.0, alpha=0.0), #DOF3
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=m_p2),
        DHParameter(d=0.0, theta=q_r4 + m_p2, a=0.0, alpha=0.0), #DOF4
        DHParameter(d=0.0, theta=p2, a=0.0, alpha=p2),
        DHParameter(d=l45, theta=q_r5, a=0.0, alpha=0.0), #DOF5
        DHParameter(d=l56, theta=m_p2, a=0.0, alpha=m_p2),
        DHParameter(d=0.0, theta=q_r6 + m_p2, a=0.0, alpha=0.0), #DOF6
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2),
        DHParameter(d=0.0, theta=q_r7, a=l_eff, alpha=0.0), #DOF7
        DHParameter(d=0.0, theta=p2, a=0.0, alpha=p2),
        DHParameter(d=0.0, theta=m_p2, a=0.0, alpha=0.0)  # End-effector Adjustment to convention
    ]


# =====================================================================================
# PART 2: IK  (hardware independent)
# =====================================================================================

@dataclass
class Target:
    """End-effector target in the world frame."""
    pos: np.ndarray                    # [x, y, z] in metres
    rpy: np.ndarray | None = None      # [roll, pitch, yaw] in radians (ZYX). None = position only


class BimanualIK:
    """
    Usage, once per control cycle:
        joint_command = ik.step(joint_state, {"left": Target(...), "right": Target(...)})
    Each call takes ONE damped-least-squares step from the current state towards the targets,
    so call it repeatedly (e.g. 100 Hz) and send each command to the joint position controllers.
    An arm left out of `targets` is held still.
    """

    ARM_DAMPING = 0.2           # DLS damping for arm joints (higher = slower but safer near singularities)
    BASE_DAMPING = 0.6          # higher damping => the arms do the work, the base only helps when needed
    ORIENTATION_WEIGHT = 0.5    # metres of position error that 1 rad of orientation error is worth
    MAX_JOINT_STEP = 0.05       # rad (or m) per call; caps speed when a target jumps far away

    def __init__(self):
        self.arms = {
            "left": (ForwardKinematics(left_dh_params, BASE_SYMBOLS + LEFT_ARM_SYMBOLS), LEFT_ARM_SYMBOLS),
            "right": (ForwardKinematics(right_dh_params, BASE_SYMBOLS + RIGHT_ARM_SYMBOLS), RIGHT_ARM_SYMBOLS),
        }
        self.solver = InverseKinematics()

    def step(self, joint_state: dict, targets: dict, base_groups=()) -> dict:
        """
        joint_state: {joint name: value} for every name in JOINT_NAMES.
        targets:     {"left": Target, "right": Target}; either may be missing (that arm is held
                     still and left out of the IK entirely).
        base_groups: which base joints may move, any subset of BASE_GROUPS ("xy", "yaw", "lift").
                     Empty = only the arm joints move.
        Returns the next joint position command, {joint name: value} for every name in JOINT_NAMES.
        """
        q = {s: joint_state[s.name] for s in ALL_SYMBOLS}

        # one task ("chain") per arm that has a target
        chains = []
        solve_symbols = [s for group, symbols in BASE_GROUPS.items() if group in base_groups for s in symbols]
        for side, target in targets.items():
            fk, arm_symbols = self.arms[side]
            target_rot = None if target.rpy is None else ForwardKinematics.euler_ZYX_to_rotmat(*target.rpy)
            chains.append((fk, BASE_SYMBOLS + arm_symbols, target.pos, target_rot))
            solve_symbols += arm_symbols
        if not chains:
            return dict(joint_state)

        damping = [self.BASE_DAMPING if s in BASE_SYMBOLS else self.ARM_DAMPING for s in solve_symbols]
        dq = self.solver.solve_ik_stacked(
            current_joint_state=q,
            chains=chains,
            all_symbols=solve_symbols,
            joint_limits={s: JOINT_LIMITS[s.name] for s in solve_symbols},
            damped_factor=damping,
            desired_pos={x: 0.0, y: 0.0, yaw: 0.0},  # base drifts back home when the arms allow it
            desired_gain=0.05,
            orientation_weight=self.ORIENTATION_WEIGHT,
        )

        # cap the step so a far-away target gives a smooth motion instead of a jump
        biggest = np.max(np.abs(dq))
        if biggest > self.MAX_JOINT_STEP:
            dq *= self.MAX_JOINT_STEP / biggest

        command = dict(joint_state)
        for s, d in zip(solve_symbols, dq):
            command[s.name] = joint_state[s.name] + d
        return command

    def forward(self, joint_state: dict) -> dict:
        """End-effector poses: {"left": (pos [3], R [3x3]), "right": (...)} in the world frame."""
        poses = {}
        for side, (fk, arm_symbols) in self.arms.items():
            sub = {s: joint_state[s.name] for s in BASE_SYMBOLS + arm_symbols}
            T = fk.evaluate("transformation_matrix", sub)
            poses[side] = (T[:3, 3], T[:3, :3])
        return poses

    def errors(self, joint_state: dict, targets: dict) -> dict:
        """{side: (position error [m], orientation error [rad] or None)} of the IK model."""
        out = {}
        for side, (pos, R) in self.forward(joint_state).items():
            if side not in targets:
                continue
            t = targets[side]
            rot_err = None
            if t.rpy is not None:
                R_t = ForwardKinematics.euler_ZYX_to_rotmat(*t.rpy)
                rot_err = float(np.linalg.norm(ForwardKinematics.rotation_error(R_t, R)))
            out[side] = (float(np.linalg.norm(t.pos - pos)), rot_err)
        return out


# =====================================================================================
# PART 3: SIM DEMO  (MuJoCo + GUI; replace this part with the real robot's I/O)
# =====================================================================================

def main():
    from sim import MujocoSim, TargetGUI, run_control_loop

    print("Building IK model (symbolic, takes a few seconds)...")
    ik = BimanualIK()

    sim = MujocoSim("mujoco/walker_scene.xml", JOINT_NAMES)
    # all joints at 0, clipped into the joint limits (lift's minimum is 0.1); also the GUI's reset pose
    home_state = {name: float(np.clip(0.0, *JOINT_LIMITS[name])) for name in JOINT_NAMES}
    sim.set_joint_state(home_state)
    start_poses = ik.forward(home_state)
    gui = TargetGUI({side: Target(pos, ForwardKinematics.rotmat_to_euler_ZYX(R))
                     for side, (pos, R) in start_poses.items()})

    run_control_loop(ik, sim, gui, home_state)   # control thread in the background, GUI in the main thread


if __name__ == "__main__":
    main()
