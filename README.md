# walkie-IK

Inverse kinematics (IK) for two **OpenArm** 7-DoF arms mounted on the **Walkie** mobile base (EIC Chula).
One solver moves both arms, and can also move the base (x, y, yaw) and the lift when the arms
need more reach.

Most of this README is about **how the IK works**, so you can understand it, tune it, or rewrite it
for your own robot. The MuJoCo simulation and GUI are only a test bench; they get a short section at the end.

---

## 1. Setup

You need **Python 3.14** and [uv](https://docs.astral.sh/uv/). The GUI also needs **Tkinter**
(on Ubuntu/Debian: `sudo apt install python3-tk`, or use a Python build that includes it).

```bash
git clone <this repo> walkie-IK
cd walkie-IK
uv sync              # creates .venv and installs mujoco, numpy, sympy
uv run main.py       # builds the IK model, then opens the MuJoCo viewer + target GUI
```

On start-up you will see `Building IK model (symbolic, takes a few seconds)...`. This is sympy
deriving the forward kinematics and Jacobians; it only happens once per run.

To view the robot model without the IK: `uv run mujoco/launch.py`.

### Project layout

| Path | What it is |
|---|---|
| [IK/dh_parameter.py](IK/dh_parameter.py) | One DH row → 4×4 transform (symbolic) |
| [IK/forward_kinematics.py](IK/forward_kinematics.py) | Chains DH rows, derives pose + Jacobians with sympy, evaluates them fast with numpy |
| [IK/inverse_kinematics.py](IK/inverse_kinematics.py) | **The solver**: damped pseudoinverse + joint-limit damping + null-space projection |
| [main.py](main.py) | Part 1: robot model (DH tables, limits). Part 2: `BimanualIK` (the API you call). Part 3: sim demo |
| [reference_frame.py](reference_frame.py) | Convert targets between world frame and robot-base frame |
| [sim/](sim/) | MuJoCo wrapper, control loop, Tkinter slider GUI |
| [mujoco/](mujoco/) | Robot XML + meshes |

---

## 2. Using the IK in your own code

The IK has no simulator code in it. On the real robot you only need `IK/` and parts 1–2 of `main.py`.

```python
from main import BimanualIK, Target
import numpy as np

ik = BimanualIK()   # slow (symbolic), do it once

# every control cycle (e.g. 100 Hz):
targets = {
    "left":  Target(pos=np.array([0.5,  0.2, 1.0]), rpy=np.array([0.0, 0.0, 0.0])),
    "right": Target(pos=np.array([0.5, -0.2, 1.0])),          # rpy=None -> position only
}
command = ik.step(joint_state, targets, base_groups=("lift",))   # allow arms + lift to move
send_to_robot(command)
```

- `joint_state` / `command`: `{joint name: value}` for every name in `JOINT_NAMES`
  (`x, y, yaw, lift, q_l1..q_l7, q_r1..q_r7`). Radians for revolute joints, metres for x, y, lift.
- `targets`: world frame, orientation as roll/pitch/yaw (ZYX: `R = Rz(yaw) Ry(pitch) Rx(roll)`).
  Leave an arm out of the dict to hold it still.
- `base_groups`: any of `"xy"`, `"yaw"`, `"lift"`. Empty = only the arms move.
- **`step()` does one small step, not a full solve.** Call it every cycle and send each result to
  the joint position controllers. The robot moves smoothly towards the target over many cycles.
- Feed the previous **command** back in as `joint_state` (not the measured joints). Measured
  positions lag and sag under gravity, and the IK would end up chasing the controller.

---

## 3. How the IK works

### 3.1 The problem

Joint angles `q` decide where the hand is. That direction is easy (**forward kinematics**, FK):

```
q  ──FK──►  hand pose (position p, rotation R)
```

IK is the other direction: given a desired hand pose, find `q`. For a 7-DoF arm on a moving base
there is no neat closed-form answer, and usually there are infinitely many answers. So we solve it
**iteratively with the Jacobian**: at each step, find a small joint change `Δq` that moves the hand
a little closer to the target, apply it, and repeat.

### 3.2 Forward kinematics from DH parameters

Each joint/link is described by one Denavit–Hartenberg row `(d, θ, a, α)`, which gives a 4×4 transform:

```
        ┌ cosθ  -sinθ·cosα   sinθ·sinα   a·cosθ ┐
T_i  =  │ sinθ   cosθ·cosα  -cosθ·sinα   a·sinθ │
        │  0       sinα         cosα        d    │
        └  0        0            0          1    ┘
```

Multiply them all to get base → hand: `T = T_1 · T_2 · … · T_n`. The top-left 3×3 of `T` is the hand
rotation `R`, the right column is the hand position `p`.

In this repo the joint variables are **sympy symbols**, so `T` is an exact formula in `q`
([IK/forward_kinematics.py](IK/forward_kinematics.py)). The mobile base is modelled as extra joints in
the same chain: two prismatic joints for x and y, a revolute joint for yaw and a prismatic joint for the lift
(see `left_dh_params` / `right_dh_params` in [main.py](main.py)). That way the base is
"just more joints" and the solver needs no special case for it.

### 3.3 The Jacobian

The Jacobian `J` says how the hand moves when the joints move a tiny bit:

```
Δx ≈ J · Δq          (Δx = hand motion, Δq = joint motion)
```

Each **column** of `J` is "how the hand moves if only joint i moves". Each **row** is one direction the
hand can move. We use 6 rows per arm:

- **Position rows (3)**: `J_v = ∂p/∂q`. Because `p` is a sympy expression, this is exact
  symbolic differentiation (`eff_pos.jacobian(variables)`).
- **Orientation rows (3)**: the *geometric angular* Jacobian `J_ω`, where `ω = J_ω · Δq` is the
  angular velocity of the hand. Column i is the vector form of the skew matrix `(∂R/∂q_i) · Rᵀ`
  (`angular_jacobian_np`). We use this instead of differentiating Euler angles because Euler-angle
  Jacobians break down at gimbal lock; this one never does.

The symbolic formulas are turned into fast numpy functions once with `sp.lambdify` and cached, so
each control step only evaluates numbers.

### 3.4 The error

For each arm with a target:

```
e_pos = p_target − p_current                         (metres)
e_rot = log(R_target · R_currentᵀ)                   (rotation vector = axis × angle, radians)
```

`e_rot` is "the rotation that takes the current hand orientation onto the target", written as one
3-vector whose direction is the axis and whose length is the angle (`rotation_error`). It always takes
the short way round. It is in the same frame as `J_ω`, so the two can be used together.

Position and rotation have different units, so the orientation rows of both `J` and `e` are multiplied by
`ORIENTATION_WEIGHT` (0.5 = "1 rad of rotation error matters as much as 0.5 m of position error").
Lower it if you want orientation to give way first when both can't be met.

### 3.5 Solving for Δq: the (damped) pseudoinverse

We want `Δq` such that `J · Δq = e`. `J` is usually **not square**: with both arms and the whole base
enabled, `J` has 12 rows (6 per arm) and 18 columns (4 base + 7 + 7 arm joints). There are more
unknowns than equations, so there are infinitely many solutions.

**Pseudoinverse.** The Moore–Penrose pseudoinverse `J⁺` picks the solution with the **smallest joint
motion** `‖Δq‖`:

```
Δq = J⁺ e,        J⁺ = Jᵀ (J Jᵀ)⁻¹
```

**The problem: singularities.** When the arm is fully stretched, or two joint axes line up, `J`
loses a direction (it can't move the hand that way). `J Jᵀ` becomes nearly singular, `J⁺` blows up, and
the joints get huge, violent commands.

**Damped least squares (DLS).** Instead of solving `J Δq = e` exactly, we minimise

```
‖J Δq − e‖²  +  λ² ‖Δq‖²
```

i.e. "reach the target, but don't move the joints too much". The solution is

```
Δq = (Jᵀ J + λ² I)⁻¹ Jᵀ e
```

Far from singularities this is almost exactly `J⁺ e`. Near a singularity, the `λ²` term keeps the
matrix invertible, so the robot slows down in the bad direction instead of exploding. Larger `λ`
means safer but slower and less accurate.

> You can see this in the demo: the home pose (all joints at 0, arms hanging straight) is close to a
> singularity, so the first moves away from it are slow. From a bent-elbow pose the same solver
> reaches a 7 cm target to sub-millimetre accuracy within a few hundred steps.

To see why, use the SVD `J = U Σ Vᵀ` with singular values `sᵢ`. The pseudoinverse divides by `sᵢ`;
DLS replaces that with `sᵢ / (sᵢ² + λ²)`:

| | `sᵢ` large (healthy) | `sᵢ → 0` (singular) |
|---|---|---|
| Pseudoinverse `1/sᵢ` | ≈ `1/sᵢ` | → ∞ (explodes) |
| DLS `sᵢ/(sᵢ²+λ²)` | ≈ `1/sᵢ` | → 0 (gives up that direction smoothly) |

### 3.6 Per-joint damping: making the base "lazy"

`λ` does not have to be the same for every joint. We use a diagonal damping matrix:

```
Δq = (Jᵀ J + diag(λᵢ² · wᵢ))⁻¹ Jᵀ e
```

Each joint's `λᵢ²` is the price of moving that joint. In `BimanualIK`:

- `ARM_DAMPING = 0.2`: arm joints are cheap.
- `BASE_DAMPING = 0.6`: base joints are expensive, so **the arms do the work and the base only
  helps when the arms can't reach**.

### 3.7 Joint limits as damping

The `wᵢ ≥ 1` in the formula above comes from the joint limits (`_limit_weights`):

- `wᵢ = 1` if joint i is far from its limits, or moving **away** from the nearest one.
- Inside the last `margin` (15%) of its range and moving **towards** the limit, `wᵢ` grows quickly
  and becomes huge right at the limit.

So a joint near a limit becomes very "expensive" and the solver uses the other joints instead. Since
`wᵢ` depends on the direction of `Δq`, the solve is repeated `n_refine = 2` times, each time using the
weights from the previous `Δq`. A final `clip` makes sure no joint ever steps past a hard limit.

### 3.8 Two arms, one system (stacking)

Both arms share the base joints (x, y, yaw, lift). If each arm was solved on its own, the two would
fight over the base. Instead both tasks are put into **one big system**:

```
          base   left arm   right arm
        ┌──────┬──────────┬───────────┐
J  =    │ J_Lb │  J_La    │     0     │   ← 6 rows: left hand
        ├──────┼──────────┼───────────┤
        │ J_Rb │    0     │   J_Ra    │   ← 6 rows: right hand
        └──────┴──────────┴───────────┘

e  =  [ e_left ; e_right ]
```

A shared joint has one column, which gets contributions from both arms, so one `Δq` satisfies both
hands at once. The left arm's joints don't affect the right hand, so those blocks are zero.

Joints that should **not** move (e.g. base groups turned off) are simply left out of the columns.
They stay at their current value.

### 3.9 Null-space projection: a secondary goal for free

Remember: 18 joints, 12 task rows, so there are **6 directions** in joint space that do not move
either hand at all. (Even with the base locked: 14 joints, 12 rows, leaving one "elbow swivel" per
arm.) This set of directions is the **null space** of `J`:

```
null space = { Δq  such that  J · Δq = 0 }
```

Motion in the null space is "free": it changes the robot's posture without moving the hands. We use
it for a **secondary task**: pull the base back towards a desired pose (`x = y = yaw = 0`), but only as
far as the hands allow.

**The projector.** `J⁺ J` keeps the part of a joint motion that *does* move the hands. So

```
N = I − J⁺ J
```

keeps only the part that does **not** move the hands. Any vector multiplied by `N` lands in the null
space. The full update is:

```
Δq  =  Δq_primary  +  N · k · (q_desired − q)
        ─────────     ─────────────────────────
        reach the      drift toward the desired pose,
        targets        with every hand-moving part removed
```

with `k = desired_gain` (0.05 = close 5% of the gap per step). Joints without a desired value get 0.

Because `J · N = 0`, the secondary task **cannot disturb the primary task**. That is the important
property: the base drifts home whenever the arms can cover for it, and it stays put when the arms need it.
Unlike adding the desired pose as an extra weighted cost, it causes no steady-state error at the hands.

**Soft projector near singularities.** Building `N` directly from `J⁺` has the same
singularity problem as before. The code builds it from the SVD `J = U Σ Vᵀ` instead:

```
P = V · diag( sᵢ² / (sᵢ² + ε²) ) · Vᵀ          N = I − P
```

For healthy directions `sᵢ²/(sᵢ²+ε²) ≈ 1` (fully blocked from the secondary task); for directions
where `sᵢ ≈ 0` it fades to 0 (free for the secondary task) smoothly instead of switching on and
off. `ε = null_damping = 0.01`.

### 3.10 One full step

This is what `BimanualIK.step()` → `InverseKinematics.solve_ik_stacked()` does every cycle:

```
1. for each arm with a target:
       p, R      = FK(q)
       e_pos     = p_target − p
       e_rot     = log(R_target Rᵀ)
       J_arm     = [ J_v ; w_rot · J_ω ]          (6 × n)
       e_arm     = [ e_pos ; w_rot · e_rot ]
2. stack the arms:  J (12 × n), e (12)            shared joints share columns
3. primary:  Δq = (JᵀJ + diag(λ²))⁻¹ Jᵀ e
             repeat 2×:  w = limit_weights(q, Δq)
                         Δq = (JᵀJ + diag(λ² w))⁻¹ Jᵀ e
4. secondary: U,S,Vᵀ = svd(J)
              P  = V diag(s²/(s²+ε²)) Vᵀ
              Δq += (I − P) · k · (q_desired − q)
5. clip q + Δq into the joint limits
6. if max|Δq| > MAX_JOINT_STEP: scale Δq down    (smooth motion for far targets)
7. return q + Δq
```

### 3.11 Porting to another robot

1. Write the DH table for your robot (one `DHParameter` per row; joint variables as sympy symbols).
   If you have a mobile base, put it at the start of the chain as extra joints.
2. `fk = ForwardKinematics(dh_params, variables=[...])` per end effector.
3. Call `InverseKinematics().solve_ik_stacked(...)` with one chain per end effector, the joint limits,
   per-joint damping and an optional desired pose. See its docstring for every argument.
4. Check the FK against your simulator/robot before testing the IK (the sim GUI shows
   `sim TCP vs model` in mm for exactly this reason).

### 3.12 Tuning cheat sheet

| Parameter | Where | Effect |
|---|---|---|
| `ARM_DAMPING` (λ arm) | `BimanualIK` | Higher = smoother near singularities, slower convergence |
| `BASE_DAMPING` (λ base) | `BimanualIK` | Higher = base moves less, arms do more |
| `ORIENTATION_WEIGHT` | `BimanualIK` | Lower = orientation gives way to position first |
| `MAX_JOINT_STEP` | `BimanualIK` | Max change per call (rad or m); caps speed |
| `margin`, `gain` | `solve_ik_stacked` | How early / how strongly joints are kept off their limits |
| `desired_pos`, `desired_gain` | `BimanualIK.step` | Secondary (null-space) posture and how fast it's pulled |
| `null_damping` (ε) | `solve_ik_stacked` | Larger = smoother projector, but lets a little secondary motion leak into the hands |

---

## 4. Simulation demo (MuJoCo + GUI)

`uv run main.py` starts:

- a **MuJoCo viewer** with the robot. Every joint has a PID position actuator, so the sim behaves like
  the real robot's position controllers. Big axes = target, small faint axes = IK model's hand pose.
- a **Tkinter window** with sliders for each arm's target (x, y, z, roll, pitch, yaw), checkboxes to let
  the IK move the base (x/y, yaw, lift), manual base sliders for the joints the IK is not moving, a
  reset button, and a choice of target frame:
  - **World (absolute)**: targets are fixed in the room.
  - **Robot base (relative)**: targets move with the base ([reference_frame.py](reference_frame.py)).
    Base x/y/yaw are locked for the IK in this mode, since moving the base would also move the target.

The control loop ([sim/control_loop.py](sim/control_loop.py)) runs at 100 Hz:
read GUI → convert targets to world → `ik.step()` → send command → step physics. To run on the real
robot, replace `MujocoSim` with your robot's I/O (`read_joint_state`, `send_joint_command`) and keep
everything else.
