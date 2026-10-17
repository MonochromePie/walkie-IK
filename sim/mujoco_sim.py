import mujoco
import mujoco.viewer
import numpy as np

# IK joint name -> MuJoCo joint name
JOINT_MAP = {"x": "x", "y": "y", "yaw": "yaw", "lift": "lift"}
JOINT_MAP.update({f"q_l{i}": f"openarm_left_joint{i}" for i in range(1, 8)})
JOINT_MAP.update({f"q_r{i}": f"openarm_right_joint{i}" for i in range(1, 8)})

TCP_BODIES = {"left": "openarm_left_hand_tcp", "right": "openarm_right_hand_tcp"}


class MujocoSim:
    """
    Stand-in for the real robot: read joint positions, send joint position commands, step.
    Every joint is driven by a PID position actuator, so a command is a target position.
    """

    def __init__(self, xml_path: str, joint_names: list[str]):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.timestep = self.model.opt.timestep

        self._qpos_adr, self._actuator, self._joint_id = {}, {}, {}
        for name in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_MAP[name])
            self._joint_id[name] = jid
            self._qpos_adr[name] = self.model.jnt_qposadr[jid]
            # the actuator whose transmission is this joint
            self._actuator[name] = next(a for a in range(self.model.nu)
                                        if self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                                        and self.model.actuator_trnid[a, 0] == jid)
        self._tcp_body = {side: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
                          for side, body in TCP_BODIES.items()}
        mujoco.mj_forward(self.model, self.data)

    # ---- robot-like interface ----

    def read_joint_state(self) -> dict:
        return {name: float(self.data.qpos[adr]) for name, adr in self._qpos_adr.items()}

    def send_joint_command(self, command: dict) -> None:
        for name, value in command.items():
            self.data.ctrl[self._actuator[name]] = value

    def set_joint_state(self, state: dict) -> None:
        """Teleport the robot (and its command) to `state`, at rest; used at start-up and on reset."""
        for name, value in state.items():
            self.data.qpos[self._qpos_adr[name]] = value
        self.data.qvel[:] = 0.0
        self.data.act[:] = 0.0          # PID integrators
        self.send_joint_command(state)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=n)

    def tcp_pose(self, side: str):
        """(pos [3], R [3x3]) of the gripper TCP body as simulated."""
        b = self._tcp_body[side]
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def joint_frame(self, name: str):
        """(pos [3], R [3x3]) of a joint as simulated: its anchor point, with the orientation
        of the body it moves."""
        jid = self._joint_id[name]
        body = self.model.jnt_bodyid[jid]
        return self.data.xanchor[jid].copy(), self.data.xmat[body].reshape(3, 3).copy()

    # ---- visualisation ----

    def launch_viewer(self):
        return mujoco.viewer.launch_passive(self.model, self.data)

    @staticmethod
    def draw_frames(viewer, frames) -> None:
        """
        Draw coordinate frames as RGB (= x, y, z) axes into the viewer's user scene.
        frames: list of (pos, R or None, axis_length, alpha). R = None draws only a sphere.
        Call inside `with viewer.lock():`.
        """
        scn = viewer.user_scn
        scn.ngeom = 0
        for pos, R, length, alpha in frames:
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.12 * length, 0, 0]),
                                np.asarray(pos, float), np.eye(3).flatten(),
                                np.array([1.0, 0.8, 0.0, alpha], np.float32))
            scn.ngeom += 1
            if R is None:
                continue
            for axis in range(3):
                if scn.ngeom >= scn.maxgeom:
                    break
                g = scn.geoms[scn.ngeom]
                rgba = np.zeros(4, np.float32)
                rgba[axis], rgba[3] = 1.0, alpha
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                    np.eye(3).flatten(), rgba)
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.06 * length,
                                     np.asarray(pos, float), np.asarray(pos, float) + length * R[:, axis])
                scn.ngeom += 1
