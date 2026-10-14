import threading
import time

import numpy as np

from IK import ForwardKinematics


def run_control_loop(ik, sim, gui, home_state: dict, ik_rate: float = 100.0) -> None:
    """
    Control thread: read targets from the GUI -> ik.step -> send command -> step physics.
    The GUI owns the main thread; closing either window stops both.
    The GUI's reset button teleports the robot to `home_state` and moves the targets there.

    The IK integrates from its own previous command rather than from the measured joint
    state, as usual for position-controlled joints (the measured state lags and sags under
    gravity, which would make the IK chase the controller).
    """
    substeps = max(1, round(1.0 / (ik_rate * sim.timestep)))
    period = substeps * sim.timestep

    def worker():
        command = sim.read_joint_state()
        tick = 0
        with sim.launch_viewer() as viewer:
            while viewer.is_running():
                t0 = time.perf_counter()

                # ---- reset ----
                if gui.take_reset():
                    sim.set_joint_state(home_state)
                    command = dict(home_state)
                    gui.set_targets({side: (pos, ForwardKinematics.rotmat_to_euler_ZYX(R))
                                     for side, (pos, R) in ik.forward(command).items()})

                # ---- IK ----
                targets, base_groups = gui.get_request()
                command = ik.step(command, targets, base_groups)

                # ---- sim ----
                sim.send_joint_command(command)
                sim.step(substeps)

                # ---- visuals + status (10 Hz) ----
                if tick % max(1, int(ik_rate // 10)) == 0:
                    _update_display(ik, sim, gui, viewer, command, targets)
                viewer.sync()
                tick += 1

                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        gui.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    gui.run()           # blocks until the window is closed
    thread.join(timeout=1.0)


def _update_display(ik, sim, gui, viewer, command, targets):
    model_poses = ik.forward(command)
    errors = ik.errors(command, targets)

    frames = []
    for side, target in targets.items():
        R_target = None if target.rpy is None else ForwardKinematics.euler_ZYX_to_rotmat(*target.rpy)
        frames.append((target.pos, R_target, 0.10, 0.9))      # target: big frame
        pos, R = model_poses[side]
        frames.append((pos, R, 0.06, 0.4))                    # IK model's end effector: small, faint

    with viewer.lock():
        sim.draw_frames(viewer, frames)

    for side, (pos, R) in model_poses.items():
        lines = [f"pos  {pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f} m",
                 "rpy  " + " ".join(f"{a:+6.1f}" for a in np.degrees(ForwardKinematics.rotmat_to_euler_ZYX(R))) + " deg"]
        if side in errors:
            pos_err, rot_err = errors[side]
            lines.append(f"IK err  {pos_err * 1000:6.1f} mm" +
                         ("" if rot_err is None else f"  {np.degrees(rot_err):5.1f} deg"))
        sim_pos, _ = sim.tcp_pose(side)
        lines.append(f"sim TCP vs model {np.linalg.norm(sim_pos - pos) * 1000:6.1f} mm")
        gui.set_status(side, "\n".join(lines), pos, ForwardKinematics.rotmat_to_euler_ZYX(R))
