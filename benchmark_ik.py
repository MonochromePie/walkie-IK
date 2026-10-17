"""
IK benchmark: how the solver evolved, measured.

Each stage is one IK method that was tried. Stages 1-4 are the older methods, rebuilt here for the
comparison; stage 5 is the current solver (IK/inverse_kinematics.py, solve_ik_stacked) with the
BimanualIK settings from main.py.

    STAGE 1  Jacobian pseudoinverse     left arm, position only, base locked
             dq = J^T (J J^T)^-1 e
    STAGE 2  Damped least squares       left arm, position only, base locked
             dq = (J^T J + lam^2 I)^-1 J^T e
    STAGE 3  DLS + rest weight          left arm + base, position + orientation
             dq = (J^T J + diag(lam^2 w) + b S)^-1 (J^T e + b S (q_rest - q))
             (w = joint-limit weights, S = diagonal selector of the joints that have a rest value)
    STAGE 4  same as stage 3 for both arms stacked, sharing the base joints
    STAGE 5  current solver: stacked DLS + joint-limit weights, rest pose in the NULL SPACE of J

Experiments:
    A  Singularity demo (stage 1 vs 2), from a nearly straight left arm (elbow q_l4 = 0.05 rad):
         A1 raise the hand 10 cm (reachable, but the start is close to a singularity)
         A2 push the hand 10 cm further down (out of reach: the arm stretches into the singularity)
    B  Rest weight (stage 3 vs 4 vs 5) on one bimanual target that needs the base and lift
    C  Random reachable targets for every stage: convergence rate, iterations, solve time, accuracy

Every method is iterated q <- clip(q + dq, limits) from a fresh start until the error is under the
tolerance or max_iters is reached. There is no per-step cap (BimanualIK.MAX_JOINT_STEP) here, so the
numbers describe the solver itself rather than the speed limit of the controller.

Run:  uv run benchmark_ik.py                    (all experiments)
      uv run benchmark_ik.py --only C --n 500   (one experiment, more targets)
Outputs: tables on stdout; CSV + PNG plots in benchmark_results/ (plots need matplotlib).
"""
import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from IK import ForwardKinematics, InverseKinematics
from main import (
    BASE_SYMBOLS, LEFT_ARM_SYMBOLS, RIGHT_ARM_SYMBOLS, JOINT_LIMITS, BimanualIK,
    left_dh_params, right_dh_params,
    x, y, yaw, lift, q_l1, q_l2, q_l3, q_l4, q_l5, q_l6, q_l7, q_r1, q_r2, q_r3, q_r4, q_r5, q_r6, q_r7,
)

OUT_DIR = Path("benchmark_results")

# ---- old-solver settings (stage 3/4), as in the earlier rest-weight version ----
OLD_DAMPING = 0.3
OLD_REST_POS = {q_l1: 0.0, q_l2: 0.0, q_l4: 1.2, lift: 0.2}
OLD_REST_POS_RIGHT = {q_r1: 0.0, q_r2: 0.0, q_r4: 1.2}       # mirrored for the right arm in stage 4
OLD_REST_WEIGHT = 0.002
LIMIT_MARGIN, LIMIT_GAIN = 0.15, 50.0

# ---- stage 1/2 damping ----
DLS_DAMPING = 0.2           # same as BimanualIK.ARM_DAMPING

LIMITS = {s: JOINT_LIMITS[s.name] for s in BASE_SYMBOLS + LEFT_ARM_SYMBOLS + RIGHT_ARM_SYMBOLS}
BASE_ALL = [x, y, yaw, lift]

# start pose for experiments B and C: elbows bent (the all-zero home pose is exactly singular)
READY = {s: float(np.clip(0.0, *LIMITS[s])) for s in LIMITS}
READY.update({lift: 0.2, q_l4: 1.2, q_r4: 1.2})


# =====================================================================================
# Kinematic model
# =====================================================================================

class Arms:
    """Left/right FK and a helper that stacks task Jacobians/errors over a set of solved joints."""

    def __init__(self):
        print("Building FK model (symbolic, takes a few seconds)...")
        self.fk = {
            "left": ForwardKinematics(left_dh_params, BASE_SYMBOLS + LEFT_ARM_SYMBOLS),
            "right": ForwardKinematics(right_dh_params, BASE_SYMBOLS + RIGHT_ARM_SYMBOLS),
        }
        self.arm_symbols = {"left": LEFT_ARM_SYMBOLS, "right": RIGHT_ARM_SYMBOLS}
        # lambdify every function once now, so its one-off compile time is not counted as solve time
        for side, fk in self.fk.items():
            sub = {s: READY[s] for s in self.chain_symbols(side)}
            fk.pos_np(sub), fk.jacobian_np(sub), fk.angular_jacobian_np(sub)

    def chain_symbols(self, side):
        return BASE_SYMBOLS + self.arm_symbols[side]

    def pose(self, side, q):
        sub = {s: q[s] for s in self.chain_symbols(side)}
        fk = self.fk[side]
        return fk.pos_np(sub), fk.orientation_matrix_np(sub)

    def task(self, q, targets, solve, orientation_weight=0.5):
        """
        Stacked J (rows: 3 position [+3 weighted orientation] per arm, cols: `solve`) and error e.
        targets: {side: (pos, R or None)}.
        """
        col = {s: i for i, s in enumerate(solve)}
        Js, es = [], []
        for side, (tp, tR) in targets.items():
            syms = self.chain_symbols(side)
            sub = {s: q[s] for s in syms}
            fk = self.fk[side]
            Jc = fk.jacobian_np(sub)
            ec = tp - fk.pos_np(sub)
            if tR is not None:
                Jc = np.vstack([Jc, orientation_weight * fk.angular_jacobian_np(sub)])
                ec = np.concatenate([ec, orientation_weight * fk.rotation_error(tR, fk.orientation_matrix_np(sub))])
            Jf = np.zeros((Jc.shape[0], len(solve)))
            for k, s in enumerate(syms):
                if s in col:
                    Jf[:, col[s]] = Jc[:, k]
            Js.append(Jf)
            es.append(ec)
        return np.vstack(Js), np.concatenate(es)

    def errors(self, q, targets):
        """Worst position error [m] and worst orientation error [rad] (0 if none) over the arms."""
        pe, re = 0.0, 0.0
        for side, (tp, tR) in targets.items():
            p, R = self.pose(side, q)
            pe = max(pe, float(np.linalg.norm(tp - p)))
            if tR is not None:
                re = max(re, float(np.linalg.norm(ForwardKinematics.rotation_error(tR, R))))
        return pe, re


# =====================================================================================
# The methods. Each returns dq (in `solve` order) for the current q.
# =====================================================================================

def vec(q, solve):
    return np.array([q[s] for s in solve], dtype=float)


def limits_of(solve):
    return np.array([LIMITS[s] for s in solve], dtype=float)


def step_pinv(arms, q, targets, solve):
    """Stage 1: textbook pseudoinverse dq = J^T (J J^T)^-1 e. Blows up as J J^T becomes singular."""
    J, e = arms.task(q, targets, solve)
    return J.T @ np.linalg.solve(J @ J.T, e)


def step_dls(arms, q, targets, solve):
    """Stage 2: damped least squares dq = (J^T J + lam^2 I)^-1 J^T e."""
    J, e = arms.task(q, targets, solve)
    return np.linalg.solve(J.T @ J + DLS_DAMPING ** 2 * np.eye(len(solve)), J.T @ e)


def step_rest(arms, q, targets, solve, rest_pos):
    """
    Stages 3/4: limit-aware DLS with the rest pose as an extra weighted cost
        min ||J dq - e||^2 + dq^T diag(lam^2 w) dq + b ||S (q + dq - q_rest)||^2
    At a fixed point (dq = 0): J^T e = -b S (q_rest - q), so e != 0 unless the solution happens to
    sit at the rest pose. The rest pull and the task trade off, which is the steady-state error.
    """
    J, e = arms.task(q, targets, solve)
    v, lim, n = vec(q, solve), limits_of(solve), len(solve)
    sel = np.array([s in rest_pos for s in solve], dtype=float)
    rest = np.array([rest_pos.get(s, 0.0) for s in solve], dtype=float)
    A0 = J.T @ J + OLD_REST_WEIGHT * np.diag(sel)
    b = J.T @ e + OLD_REST_WEIGHT * sel * (rest - v)
    lam2 = OLD_DAMPING ** 2
    dq = np.linalg.solve(A0 + lam2 * np.eye(n), b)
    for _ in range(2):
        w = InverseKinematics._limit_weights(v, dq, lim, LIMIT_MARGIN, LIMIT_GAIN)
        dq = np.linalg.solve(A0 + np.diag(lam2 * w), b)
    return dq


_SOLVER = InverseKinematics()


def step_current(arms, q, targets, solve):
    """Stage 5: the current solver with BimanualIK's settings (without its MAX_JOINT_STEP cap)."""
    chains = []
    for side, (tp, tR) in targets.items():
        chains.append((arms.fk[side], arms.chain_symbols(side), tp, tR))
    damping = [BimanualIK.BASE_DAMPING if s in BASE_SYMBOLS else BimanualIK.ARM_DAMPING for s in solve]
    return _SOLVER.solve_ik_stacked(
        current_joint_state=q, chains=chains, all_symbols=solve,
        joint_limits={s: LIMITS[s] for s in solve},
        damped_factor=damping,
        desired_pos={x: 0.0, y: 0.0, yaw: 0.0},
        desired_gain=0.05,
        orientation_weight=BimanualIK.ORIENTATION_WEIGHT,
    )


@dataclass
class Method:
    key: str
    label: str
    step: callable
    sides: tuple            # arms with a target
    solve: list             # joints being solved for
    orientation: bool       # 6-D targets (else position only)
    color: str


def build_methods():
    left_base = BASE_ALL + LEFT_ARM_SYMBOLS
    both_base = BASE_ALL + LEFT_ARM_SYMBOLS + RIGHT_ARM_SYMBOLS
    return {
        "pinv": Method("pinv", "1. Pseudoinverse (L, pos)", step_pinv,
                       ("left",), LEFT_ARM_SYMBOLS, False, "#eb6834"),
        "dls": Method("dls", "2. DLS (L, pos)", step_dls,
                      ("left",), LEFT_ARM_SYMBOLS, False, "#1baf7a"),
        "rest_left": Method("rest_left", "3. DLS + rest weight (L+base, pose)",
                            lambda a, q, t, s: step_rest(a, q, t, s, OLD_REST_POS),
                            ("left",), left_base, True, "#2a78d6"),
        "rest_both": Method("rest_both", "4. DLS + rest weight (L+R+base, pose)",
                            lambda a, q, t, s: step_rest(a, q, t, s, OLD_REST_POS | OLD_REST_POS_RIGHT),
                            ("left", "right"), both_base, True, "#6250d6"),
        "current": Method("current", "5. Current: null-space rest (L+R+base, pose)", step_current,
                          ("left", "right"), both_base, True, "#e87ba4"),
    }


# =====================================================================================
# Iteration loop
# =====================================================================================

@dataclass
class Run:
    converged: bool
    iters: int
    pos_err: float
    rot_err: float
    time_s: float                        # total wall time (FK + Jacobian + solve), all iterations
    max_step: float                      # largest |dq_i| seen in any iteration (before clipping)
    pos_hist: list = field(default_factory=list)
    rot_hist: list = field(default_factory=list)
    step_hist: list = field(default_factory=list)
    failed: str = ""                     # e.g. "singular matrix"


def solve(arms, method: Method, q0: dict, targets: dict, pos_tol, rot_tol, max_iters, record=False):
    q = dict(q0)
    lim = limits_of(method.solve)
    pos_hist, rot_hist, step_hist = [], [], []
    pe, re = arms.errors(q, targets)
    if record:
        pos_hist.append(pe)
        rot_hist.append(re)
    max_step, failed, it = 0.0, "", 0
    t0 = time.perf_counter()
    for it in range(1, max_iters + 1):
        if pe < pos_tol and re < rot_tol:
            it -= 1
            break
        try:
            dq = method.step(arms, q, targets, method.solve)
        except np.linalg.LinAlgError:
            failed = "singular matrix"
            break
        if not np.all(np.isfinite(dq)):
            failed = "non-finite step"
            break
        biggest = float(np.max(np.abs(dq)))
        max_step = max(max_step, biggest)
        v = np.clip(vec(q, method.solve) + dq, lim[:, 0], lim[:, 1])
        q.update(zip(method.solve, v))
        pe, re = arms.errors(q, targets)
        if record:
            pos_hist.append(pe)
            rot_hist.append(re)
            step_hist.append(biggest)
    elapsed = time.perf_counter() - t0
    converged = not failed and pe < pos_tol and re < rot_tol
    return Run(converged, it, pe, re, elapsed, max_step, pos_hist, rot_hist, step_hist, failed)


def make_targets(arms, method: Method, q_target: dict):
    out = {}
    for side in method.sides:
        p, R = arms.pose(side, q_target)
        out[side] = (p, R if method.orientation else None)
    return out


# =====================================================================================
# Experiment A: singularity
# =====================================================================================

def experiment_singularity(arms, methods, args):
    print("\n" + "=" * 90)
    print("A. SINGULARITY: pseudoinverse vs damped least squares (left arm, position only, base locked)")
    print("=" * 90)
    q0 = dict(READY)
    q0.update({lift: 0.2, q_l4: 0.05, q_r4: 0.0})    # left arm nearly straight, hanging down
    J, _ = arms.task(q0, {"left": (np.zeros(3), None)}, LEFT_ARM_SYMBOLS)
    s = np.linalg.svd(J, compute_uv=False)
    p0, _ = arms.pose("left", q0)
    print(f"start: q_l4 = 0.05 rad, singular values of J = {np.array2string(s, precision=4)}, "
          f"condition number = {s[0] / s[-1]:.0f}")

    scenarios = {
        "A1_raise_10cm": ("A1 raise hand 10 cm (reachable)", p0 + [0.0, 0.0, 0.10]),
        "A2_out_of_reach": ("A2 push hand 10 cm down (out of reach)", p0 + [0.0, 0.0, -0.10]),
    }
    results = {}
    rows = []
    print(f"\n{'scenario':<42}{'method':<30}{'conv':>5}{'iters':>7}{'pos err [mm]':>14}"
          f"{'err after 1st [mm]':>20}{'1st |dq| [rad]':>16}{'joint travel [rad]':>20}")
    for key, (title, tp) in scenarios.items():
        for mk in ("pinv", "dls"):
            m = methods[mk]
            r = solve(arms, m, q0, {"left": (tp, None)}, args.pos_tol, args.rot_tol, args.sing_iters, record=True)
            results[(key, mk)] = r
            first = r.step_hist[0] if r.step_hist else float("nan")
            q_end = replay_final(arms, m, q0, {"left": (tp, None)}, r.iters)
            travel = np.linalg.norm(vec(q_end, m.solve) - vec(q0, m.solve))
            print(f"{title:<42}{m.label:<30}{'yes' if r.converged else 'NO':>5}{r.iters:>7}"
                  f"{r.pos_err * 1e3:>14.3f}{r.pos_hist[1] * 1e3:>20.1f}{first:>16.3f}{travel:>20.3f}")
            for i, (pe, st) in enumerate(zip(r.pos_hist, [np.nan] + r.step_hist)):
                rows.append([key, mk, i, pe, st])
    print("\nNotes: A1 has a solution, A2's best answer is 10 cm away (the closest reachable point).")
    print("       |dq| is the largest joint step of the first iteration, before clipping to the joint limits.")
    print("       joint travel = ||q_final - q_start||: how far the solution ended from where the arm started.")
    write_csv("A_singularity.csv", ["scenario", "method", "iter", "pos_err_m", "max_abs_dq"], rows)
    if args.plots:
        plot_singularity(results, scenarios, methods)


# =====================================================================================
# Experiment B: rest weight vs null space (one bimanual target)
# =====================================================================================

def experiment_rest(arms, methods, args):
    print("\n" + "=" * 90)
    print("B. REST WEIGHT: one arm vs two arms vs current null-space solver")
    print("=" * 90)
    # the two arms are posed differently (left reaching high and bent, right low and to the side), so the
    # shared base/lift joints cannot sit where both arms AND the rest pose want them
    qt = dict(READY)
    qt.update({x: 0.1, yaw: 0.3, lift: 0.4,
               q_l1: -1.2, q_l2: -0.2, q_l4: 2.0, q_l7: 0.6,
               q_r1: 0.3, q_r2: 1.0, q_r3: 0.5, q_r4: 1.0, q_r6: 0.3})
    print("target: asymmetric pose of both hands, from a configuration with base x = 0.1 m, yaw = 0.3 rad, "
          "lift = 0.4 m")
    print("start: elbows at 1.2 rad, lift 0.2 m (= the rest pose)")
    print(f"tolerance: position < {args.pos_tol * 1e3:g} mm, orientation < {np.degrees(args.rot_tol):.2f} deg")
    results = {}
    rows = []
    print(f"\n{'method':<48}{'conv':>5}{'iters':>7}{'pos err [mm]':>14}{'rot err [deg]':>15}"
          f"{'lift [m]':>10}{'time [ms]':>11}")
    for mk in ("rest_left", "rest_both", "current"):
        m = methods[mk]
        targets = make_targets(arms, m, qt)
        r = solve(arms, m, READY, targets, args.pos_tol, args.rot_tol, args.rest_iters, record=True)
        results[mk] = r
        q_final = replay_final(arms, m, READY, targets, r.iters)
        print(f"{m.label:<48}{'yes' if r.converged else 'NO':>5}{r.iters:>7}{r.pos_err * 1e3:>14.3f}"
              f"{np.degrees(r.rot_err):>15.3f}{q_final[lift]:>10.3f}{r.time_s * 1e3:>11.1f}")
        for i, (pe, re) in enumerate(zip(r.pos_hist, r.rot_hist)):
            rows.append([mk, i, pe, re])
    print("\nThe rest weight is a cost the solver trades against the target error. With one arm the")
    print("leftover error is small; with two arms the shared base/lift is pulled by both tasks and the")
    print("rest pose at once, and the error settles above the tolerance.")
    print("The current solver puts the rest pull in the null space of J, where it cannot move the hands.")
    write_csv("B_rest_weight.csv", ["method", "iter", "pos_err_m", "rot_err_rad"], rows)
    if args.plots:
        plot_rest(results, methods, args)


def replay_final(arms, method, q0, targets, iters):
    """Re-run a solve for `iters` iterations and return the final joint state."""
    q = dict(q0)
    lim = limits_of(method.solve)
    for _ in range(iters):
        dq = method.step(arms, q, targets, method.solve)
        q.update(zip(method.solve, np.clip(vec(q, method.solve) + dq, lim[:, 0], lim[:, 1])))
    return q


# =====================================================================================
# Experiment C: random targets
# =====================================================================================

def random_config(rng):
    """Random joint configuration, 10% inside each limit; base kept near home."""
    q = dict(READY)
    for s in LEFT_ARM_SYMBOLS + RIGHT_ARM_SYMBOLS:
        lo, hi = LIMITS[s]
        pad = 0.1 * (hi - lo)
        q[s] = rng.uniform(lo + pad, hi - pad)
    q[x], q[y] = rng.uniform(-0.2, 0.2, size=2)
    q[yaw] = rng.uniform(-0.3, 0.3)
    q[lift] = rng.uniform(0.1, 0.6)
    return q


def experiment_random(arms, methods, args):
    print("\n" + "=" * 90)
    print(f"C. RANDOM TARGETS: {args.n} reachable targets (FK of random configurations), seed {args.seed}")
    print(f"   tolerance: position < {args.pos_tol * 1e3:g} mm, orientation < {np.degrees(args.rot_tol):.2f} deg, "
          f"max {args.max_iters} iterations")
    print("=" * 90)
    print("Stages 1-2 lock the base, so their targets come from random ARM joints only (base at the start pose).")
    rng = np.random.default_rng(args.seed)
    configs = [random_config(rng) for _ in range(args.n)]

    runs = {}
    rows = []
    for mk, m in methods.items():
        rs = []
        for i, qc in enumerate(configs):
            if not any(s in m.solve for s in BASE_ALL):
                qc = {**qc, **{s: READY[s] for s in BASE_ALL}}    # base locked: target must not need it
            r = solve(arms, m, READY, make_targets(arms, m, qc), args.pos_tol, args.rot_tol, args.max_iters,
                      record=args.plots)
            rs.append(r)
            rows.append([mk, i, int(r.converged), r.iters, r.pos_err, r.rot_err, r.time_s, r.max_step, r.failed])
        runs[mk] = rs
        print(f"  {m.label}: done")

    print(f"\n{'method':<48}{'conv %':>7}{'iters':>13}{'ms/iter':>9}{'solve ms':>15}"
          f"{'pos err mm':>17}{'rot err deg':>17}{'max|dq|':>9}")
    print(f"{'':<48}{'':>7}{'med (p95)':>13}{'med':>9}{'med (p95)':>15}{'med (p95)':>17}{'med (p95)':>17}"
          f"{'p95':>9}")
    summary = []
    for mk, rs in runs.items():
        m = methods[mk]
        conv = [r for r in rs if r.converged]
        rate = 100.0 * len(conv) / len(rs)
        it = np.array([r.iters for r in conv]) if conv else np.array([np.nan])
        t = np.array([r.time_s for r in conv]) * 1e3 if conv else np.array([np.nan])
        per_iter = np.array([r.time_s / r.iters for r in rs if r.iters > 0]) * 1e3
        pe = np.array([r.pos_err for r in rs]) * 1e3
        re = np.degrees(np.array([r.rot_err for r in rs]))
        ms = np.array([r.max_step for r in rs])
        rot_txt = f"{np.median(re):.3f} ({np.percentile(re, 95):.3f})" if m.orientation else "-"
        print(f"{m.label:<48}{rate:>7.1f}{fmt_mp(it, '.0f'):>13}{np.median(per_iter):>9.3f}{fmt_mp(t, '.1f'):>15}"
              f"{fmt_mp(pe, '.3f'):>17}{rot_txt:>17}{np.percentile(ms, 95):>9.2f}")
        summary.append([mk, m.label, rate, np.median(it), np.percentile(it, 95), np.median(per_iter),
                        np.median(t), np.percentile(t, 95), np.median(pe), np.percentile(pe, 95),
                        np.median(re) if m.orientation else "", np.percentile(re, 95) if m.orientation else "",
                        np.percentile(ms, 95)])
        fails = [r.failed for r in rs if r.failed]
        if fails:
            print(f"{'':<48}  {len(fails)} runs aborted: {', '.join(sorted(set(fails)))}")
    print("\niters / solve ms: over converged runs only. pos/rot err: final error of every run.")
    print("ms/iter includes evaluating FK and the Jacobian (lambdified sympy), which dominates.")

    write_csv("C_random_runs.csv", ["method", "target", "converged", "iters", "pos_err_m", "rot_err_rad",
                                    "time_s", "max_abs_dq", "failed"], rows)
    write_csv("C_random_summary.csv", ["method", "label", "conv_rate_pct", "iters_median", "iters_p95",
                                       "ms_per_iter_median", "solve_ms_median", "solve_ms_p95",
                                       "pos_err_mm_median", "pos_err_mm_p95", "rot_err_deg_median",
                                       "rot_err_deg_p95", "max_abs_dq_p95"], summary)
    if args.plots:
        plot_random(runs, methods, args)


def fmt_mp(a, f):
    if np.all(np.isnan(a)):
        return "-"
    return f"{np.median(a):{f}} ({np.percentile(a, 95):{f}})"


def write_csv(name, header, rows):
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> {OUT_DIR / name}")


# =====================================================================================
# Plots
# =====================================================================================

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
        "legend.frameon": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    })


def _save(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    print(f"  -> {OUT_DIR / name}")


def plot_singularity(results, scenarios, methods):
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for col, (key, (title, _)) in enumerate(scenarios.items()):
        for mk in ("pinv", "dls"):
            r, m = results[(key, mk)], methods[mk]
            axes[0, col].semilogy(range(len(r.pos_hist)), np.maximum(r.pos_hist, 1e-12), color=m.color, label=m.label)
            axes[1, col].semilogy(range(1, len(r.step_hist) + 1), np.maximum(r.step_hist, 1e-12),
                                  color=m.color, label=m.label)
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("position error [m]")
        axes[1, col].set_ylabel("largest joint step |dq| [rad]")
        axes[1, col].set_xlabel("iteration")
    axes[0, 0].legend()
    fig.suptitle("Pseudoinverse vs DLS from a nearly straight arm (q_l4 = 0.05 rad)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "A_singularity.png")
    plt.close(fig)


def plot_rest(results, methods, args):
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mk, r in results.items():
        m = methods[mk]
        axes[0].semilogy(np.maximum(r.pos_hist, 1e-12), color=m.color, label=m.label)
        axes[1].semilogy(np.maximum(np.degrees(r.rot_hist), 1e-12), color=m.color, label=m.label)
    axes[0].axhline(args.pos_tol, color=INK2, linestyle="--", linewidth=1)
    axes[0].text(1, args.pos_tol * 1.15, "tolerance", color=INK2, fontsize=9)
    axes[1].axhline(np.degrees(args.rot_tol), color=INK2, linestyle="--", linewidth=1)
    axes[0].set_ylabel("worst-arm position error [m]")
    axes[1].set_ylabel("worst-arm orientation error [deg]")
    for ax in axes:
        ax.set_xlabel("iteration")
    axes[0].legend()
    fig.suptitle("Rest weight leaves a steady-state error; the null-space rest pull does not", fontweight="bold")
    fig.tight_layout()
    _save(fig, "B_rest_weight.png")
    plt.close(fig)


def plot_random(runs, methods, args):
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # convergence: fraction of targets solved by iteration k
    for mk, rs in runs.items():
        m = methods[mk]
        its = np.sort([r.iters for r in rs if r.converged])
        k = np.arange(0, args.max_iters + 1)
        axes[0].plot(k, 100.0 * np.searchsorted(its, k, side="right") / len(rs), color=m.color, label=m.label)
    axes[0].set_xscale("symlog", linthresh=10)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("targets solved [%]")
    axes[0].set_title("Convergence rate")
    axes[0].legend(fontsize=8, loc="upper left")

    # median error vs iteration (runs that stopped early hold their final error)
    for mk, rs in runs.items():
        m = methods[mk]
        L = args.max_iters + 1
        H = np.array([np.pad(r.pos_hist, (0, L - len(r.pos_hist)), mode="edge") for r in rs])
        axes[1].semilogy(np.maximum(np.median(H, axis=0), 1e-12), color=m.color, label=m.label)
    axes[1].axhline(args.pos_tol, color=INK2, linestyle="--", linewidth=1)
    axes[1].set_xscale("symlog", linthresh=10)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("median position error [m]")
    axes[1].set_title("Error vs iteration")

    # final accuracy: CDF of final position error
    for mk, rs in runs.items():
        m = methods[mk]
        pe = np.sort([max(r.pos_err, 1e-9) for r in rs])
        axes[2].plot(pe, 100.0 * np.arange(1, len(pe) + 1) / len(pe), color=m.color, label=m.label)
    axes[2].axvline(args.pos_tol, color=INK2, linestyle="--", linewidth=1)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("final position error [m]")
    axes[2].set_ylabel("targets [%]")
    axes[2].set_title("Final accuracy (CDF)")

    fig.tight_layout()
    _save(fig, "C_random.png")
    plt.close(fig)


# =====================================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["A", "B", "C"], nargs="+", default=["A", "B", "C"])
    ap.add_argument("--n", type=int, default=200, help="number of random targets (experiment C)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-tol", type=float, default=1e-3, help="position tolerance [m]")
    ap.add_argument("--rot-tol", type=float, default=np.radians(0.5), help="orientation tolerance [rad]")
    ap.add_argument("--max-iters", type=int, default=300, help="iteration limit (experiment C)")
    ap.add_argument("--sing-iters", type=int, default=100, help="iteration limit (experiment A)")
    ap.add_argument("--rest-iters", type=int, default=300, help="iteration limit (experiment B)")
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    args = ap.parse_args()

    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            print("matplotlib not installed: skipping plots (uv add --dev matplotlib)")
            args.plots = False

    arms = Arms()
    methods = build_methods()
    if "A" in args.only:
        experiment_singularity(arms, methods, args)
    if "B" in args.only:
        experiment_rest(arms, methods, args)
    if "C" in args.only:
        experiment_random(arms, methods, args)


if __name__ == "__main__":
    main()
