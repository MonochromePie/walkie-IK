import threading
import time

import numpy as np

from IK import ForwardKinematics
from reference_frame import WORLD, convert_pose, targets_to_world

# base joint -> base group (as in BimanualIK.step) and top speed when following the manual slider
BASE_JOINT_GROUPS = {"x": "xy", "y": "xy", "yaw": "yaw", "lift": "lift"}
MANUAL_BASE_SPEED = {"x": 0.5, "y": 0.5, "yaw": 1.0, "lift": 0.2}     # m/s, rad/s
# joints whose frames are drawn for each controlled arm; the base joints are drawn if any arm is
ARM_JOINTS = {side: [f"q_{side[0]}{i}" for i in range(1, 8)] for side in ("left", "right")}


def run_control_loop(ik, sim, gui, home_state: dict, ik_rate: float = 100.0) -> None:
    """
    Control thread: read targets from the GUI -> ik.step -> send command -> step physics.
    The GUI owns the main thread; closing either window stops both.
    The GUI's reset button teleports the robot to `home_state` and moves the targets there.

    The GUI's targets are in its selected frame (world or robot base); they are converted to
    world for the IK every cycle, using the base pose of the same command the IK starts from.
    A frame switch re-expresses the stored targets with that same base pose, so the world
    targets the IK sees do not change at the switch.

    Base joints the IK does not move follow the GUI's manual base sliders (speed limited);
    while the IK moves them, the sliders are kept on the IK's base pose so handing a joint
    back to manual control does not make it jump.

    The IK integrates from its own previous command rather than from the measured joint
    state, as usual for position-controlled joints (the measured state lags and sags under
    gravity, which would make the IK chase the controller).
    """
    substeps = max(1, round(1.0 / (ik_rate * sim.timestep)))
    period = substeps * sim.timestep

    def worker():
        command = sim.read_joint_state()
        gui.set_base_values({joint: command[joint] for joint in BASE_JOINT_GROUPS})
        tick = 0
        with sim.launch_viewer() as viewer:
            while viewer.is_running():
                t0 = time.perf_counter()

                # ---- reset ----
                if gui.take_reset():
                    sim.set_joint_state(home_state)
                    command = dict(home_state)
                    gui.set_base_values({joint: command[joint] for joint in BASE_JOINT_GROUPS})
                    frame = gui.get_request()[2]
                    gui.set_targets({side: convert_pose(pos, ForwardKinematics.rotmat_to_euler_ZYX(R),
                                                        WORLD, frame, command)
                                     for side, (pos, R) in ik.forward(command).items()})

                # ---- target frame switch ----
                new_frame = gui.take_frame_request()
                if new_frame is not None:
                    gui.switch_frame(new_frame, lambda pos, rpy, src, dst:
                                     convert_pose(pos, rpy, src, dst, command))

                # ---- manual base: joints the IK does not move head for their slider value ----
                targets, base_groups, frame, base_manual = gui.get_request()
                command = dict(command)
                for joint, group in BASE_JOINT_GROUPS.items():
                    if group not in base_groups:
                        max_step = MANUAL_BASE_SPEED[joint] * period
                        command[joint] += float(np.clip(base_manual[joint] - command[joint], -max_step, max_step))

                # ---- IK ----
                targets = targets_to_world(targets, frame, command)
                command = ik.step(command, targets, base_groups)
                ik_base = {joint: command[joint] for joint, group in BASE_JOINT_GROUPS.items() if group in base_groups}
                if ik_base:
                    gui.set_base_values(ik_base)

                # ---- sim ----
                sim.send_joint_command(command)
                sim.step(substeps)

                # ---- visuals + status (10 Hz) ----
                if tick % max(1, int(ik_rate // 10)) == 0:
                    _update_display(ik, sim, gui, viewer, command, targets, frame)
                viewer.sync()
                tick += 1

                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        gui.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    gui.run()           # blocks until the window is closed
    thread.join(timeout=1.0)


def _update_display(ik, sim, gui, viewer, command, targets, frame):
    """targets: world frame. The pose shown in the GUI is in `frame`, like its sliders."""
    model_poses = ik.forward(command)
    errors = ik.errors(command, targets)

    frames = []
    for side, target in targets.items():
        R_target = None if target.rpy is None else ForwardKinematics.euler_ZYX_to_rotmat(*target.rpy)
        frames.append((target.pos, R_target, 0.10, 0.9))      # target: big frame
        pos, R = model_poses[side]
        frames.append((pos, R, 0.06, 0.4))                    # IK model's end effector: small, faint

    if gui.show_joint_frames() and targets:
        joints = list(BASE_JOINT_GROUPS) + [j for side in targets for j in ARM_JOINTS[side]]
        for joint in joints:
            pos, R = sim.joint_frame(joint)
            frames.append((pos, R, 0.05, 0.8))                # joint frames: smallest

    with viewer.lock():
        sim.draw_frames(viewer, frames)

    for side, (pos, R) in model_poses.items():
        shown_pos, shown_rpy = convert_pose(pos, ForwardKinematics.rotmat_to_euler_ZYX(R), WORLD, frame, command)
        lines = [f"current pose ({frame} frame)",
                 f"pos  {shown_pos[0]:+.3f} {shown_pos[1]:+.3f} {shown_pos[2]:+.3f} m",
                 "rpy  " + " ".join(f"{a:+6.1f}" for a in np.degrees(shown_rpy)) + " deg"]
        if side in errors:
            pos_err, rot_err = errors[side]
            lines.append(f"IK err  {pos_err * 1000:6.1f} mm" +
                         ("" if rot_err is None else f"  {np.degrees(rot_err):5.1f} deg"))
        sim_pos, _ = sim.tcp_pose(side)
        lines.append(f"sim TCP vs model {np.linalg.norm(sim_pos - pos) * 1000:6.1f} mm")
        gui.set_status(side, "\n".join(lines), shown_pos, shown_rpy)
