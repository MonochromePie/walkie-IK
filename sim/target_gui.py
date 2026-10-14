import dataclasses
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

# (name, unit, slider min, slider max, slider resolution); positions are world frame
FIELDS = [("x", "m", -2.5, 2.5, 0.005), ("y", "m", -2.5, 2.5, 0.005), ("z", "m", 0.0, 2.5, 0.005),
          ("roll", "deg", -180.0, 180.0, 1.0), ("pitch", "deg", -180.0, 180.0, 1.0),
          ("yaw", "deg", -180.0, 180.0, 1.0)]
SLIDER_LENGTH = 280     # pixels
# base group (as in BimanualIK.step) -> checkbox label
BASE_TOGGLES = [("xy", "Base movement (x, y)"), ("yaw", "Orientation (yaw)"), ("lift", "Lift")]


class TargetGUI:
    """
    Pop-up window with sliders for the end-effector targets (xyz + roll/pitch/yaw, ZYX) for each arm.
    Runs in the main thread (run()); the control thread talks to it only through
    get_request(), set_status() and close(), which are thread-safe.
    """

    def __init__(self, initial_targets: dict):
        self._lock = threading.Lock()
        self._targets = dict(initial_targets)         # side -> Target, what the IK tracks
        self._template = dict(initial_targets)        # used to build new Target objects
        self._status = {}                             # side -> (text, current pos, current rpy)
        self._closing = False
        self._reset_requested = False
        self._new_fields = None                       # side -> (pos, rpy) to show in the fields

        self.root = tk.Tk()
        self.root.title("Walkie IK targets")
        self._vars, self._use_rot, self._status_labels = {}, {}, {}
        self._active, self._active_value, self._arm_widgets = {}, {}, {}

        for col, side in enumerate(initial_targets):
            box = ttk.LabelFrame(self.root, text=f"{side.capitalize()} arm", padding=8)
            box.grid(row=0, column=col, padx=8, pady=8, sticky="n")
            t = initial_targets[side]
            values = list(t.pos) + list(np.degrees(t.rpy if t.rpy is not None else np.zeros(3)))
            self._vars[side], self._arm_widgets[side] = [], []

            # arm on/off: an inactive arm gets no target, so the IK holds it still and ignores it
            self._active[side] = tk.BooleanVar(value=True)
            self._active_value[side] = True
            ttk.Checkbutton(box, text="Control this arm", variable=self._active[side],
                            command=lambda s=side: self._on_active(s)).grid(row=0, column=0, columnspan=2, sticky="w")

            for row, ((name, unit, lo, hi, res), value) in enumerate(zip(FIELDS, values), start=1):
                ttk.Label(box, text=f"{name} [{unit}]").grid(row=row, column=0, sticky="sw", pady=(0, 4))
                var = tk.DoubleVar(value=value)
                slider = tk.Scale(box, variable=var, from_=lo, to=hi, resolution=res, orient="horizontal",
                                  length=SLIDER_LENGTH, digits=4 if unit == "m" else 0)
                slider.grid(row=row, column=1, padx=4)
                var.trace_add("write", lambda *_, s=side: self._on_edit(s))
                self._vars[side].append(var)
                self._arm_widgets[side].append(slider)

            self._use_rot[side] = tk.BooleanVar(value=t.rpy is not None)
            rot_check = ttk.Checkbutton(box, text="Track orientation", variable=self._use_rot[side],
                                        command=lambda s=side: self._on_edit(s))
            rot_check.grid(row=7, column=0, columnspan=2, sticky="w")
            copy_button = ttk.Button(box, text="Set to current pose", command=lambda s=side: self._copy_current(s))
            copy_button.grid(row=8, column=0, columnspan=2, pady=4)
            self._arm_widgets[side] += [rot_check, copy_button]
            self._status_labels[side] = ttk.Label(box, text="", font=("TkFixedFont", 9), justify="left")
            self._status_labels[side].grid(row=9, column=0, columnspan=2, sticky="w")

        # base: each group of base joints can be handed to the IK separately
        base_box = ttk.LabelFrame(self.root, text="Base (moved by the IK)", padding=8)
        base_box.grid(row=1, column=0, columnspan=len(initial_targets), padx=8, pady=(0, 8), sticky="ew")
        self._base_vars = {}
        self._base_groups_value = frozenset()
        for col, (group, label) in enumerate(BASE_TOGGLES):
            self._base_vars[group] = tk.BooleanVar(value=False)
            ttk.Checkbutton(base_box, text=label, variable=self._base_vars[group],
                            command=self._on_base).grid(row=0, column=col, padx=6, sticky="w")

        ttk.Button(self.root, text="Reset all joints to 0", command=self._on_reset).grid(
            row=2, column=0, columnspan=len(initial_targets), pady=(0, 8))

        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)
        self.root.after(100, self._poll)

    # ---- called from the control thread ----

    def get_request(self):
        """(targets {side: Target} of the active arms only, base groups the IK may move)"""
        with self._lock:
            targets = {side: t for side, t in self._targets.items() if self._active_value[side]}
            return targets, self._base_groups_value

    def set_status(self, side: str, text: str, pos, rpy) -> None:
        with self._lock:
            self._status[side] = (text, np.array(pos), np.array(rpy))

    def take_reset(self) -> bool:
        """True once per press of the reset button."""
        with self._lock:
            requested, self._reset_requested = self._reset_requested, False
            return requested

    def set_targets(self, poses: dict) -> None:
        """Replace the targets with {side: (pos, rpy)}, e.g. after a reset."""
        with self._lock:
            for side, (pos, rpy) in poses.items():
                t = self._targets[side]
                self._targets[side] = dataclasses.replace(
                    t, pos=np.array(pos), rpy=None if t.rpy is None else np.array(rpy))
            self._new_fields = dict(poses)

    def close(self) -> None:
        with self._lock:
            self._closing = True

    # ---- GUI thread ----

    def run(self) -> None:
        self.root.mainloop()
        self.root.destroy()

    def _on_edit(self, side: str) -> None:
        values = [v.get() for v in self._vars[side]]
        rpy = np.radians(values[3:]) if self._use_rot[side].get() else None
        target = dataclasses.replace(self._template[side], pos=np.array(values[:3]), rpy=rpy)
        with self._lock:
            self._targets[side] = target

    def _on_base(self) -> None:
        groups = frozenset(g for g, var in self._base_vars.items() if var.get())
        with self._lock:
            self._base_groups_value = groups

    def _on_active(self, side: str) -> None:
        active = bool(self._active[side].get())
        for widget in self._arm_widgets[side]:
            if isinstance(widget, tk.Scale):
                widget.config(state="normal" if active else "disabled")
            else:
                widget.state(["!disabled"] if active else ["disabled"])
        with self._lock:
            self._active_value[side] = active

    def _on_reset(self) -> None:
        with self._lock:
            self._reset_requested = True

    def _copy_current(self, side: str) -> None:
        with self._lock:
            status = self._status.get(side)
        if status is None:
            return
        _, pos, rpy = status
        self._fill_fields(side, pos, rpy)

    def _fill_fields(self, side: str, pos, rpy) -> None:
        for var, value in zip(self._vars[side], list(pos) + list(np.degrees(rpy))):
            var.set(round(float(value), 3))

    def _poll(self) -> None:
        with self._lock:
            closing = self._closing
            texts = {side: s[0] for side, s in self._status.items()}
            new_fields, self._new_fields = self._new_fields, None
        if closing:
            self.root.quit()
            return
        for side, (pos, rpy) in (new_fields or {}).items():
            self._fill_fields(side, pos, rpy)
        for side, text in texts.items():
            self._status_labels[side].config(text=text)
        self.root.after(100, self._poll)
