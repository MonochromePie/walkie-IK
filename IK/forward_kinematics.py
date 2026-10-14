from .dh_parameter import DHParameter
import sympy as sp
import numpy as np

class ForwardKinematics:
    
    def __init__(self, dh_parameters: list[DHParameter], variables: list[sp.Symbol] = None):
        self.dh_parameters = dh_parameters
        self.variables: list[sp.Symbol] | None = variables if variables is not None else []
        self.transformation_matrix: sp.MutableDenseMatrix | None = self.get_transformation_matrix()
        self.eff_pos: sp.MutableDenseMatrix | None = self.transformation_matrix[:3, 3] 
        self.eff_orientation_matrix = self.transformation_matrix[:3, :3]  

        self._orientation_ZYX = self.get_euler_angles_ZYX()
        self.jacobian: sp.MutableDenseMatrix | None = self.eff_pos.jacobian(self.variables) if self.variables else None

        self._pose = sp.Matrix([self.eff_pos, self._orientation_ZYX])
        self.pose_jacobian: sp.MutableDenseMatrix | None = self._pose.jacobian(self.variables) if self.variables else None

        self._lambdas: dict[str, callable] = {}
        
    def get_transformation_matrix(self) -> sp.MutableDenseMatrix | None:
        """
        Calculate the overall transformation matrix from the base frame to the end-effector frame.
        Returns:
            A 4x4 sympy Matrix representing the overall transformation matrix.
        """
        T_total = sp.eye(4)  

        for dh_param in self.dh_parameters:
            T = dh_param.get_transformation_matrix()
            T_total = T_total * T 

        return sp.Matrix(T_total)

    def get_euler_angles_ZYX(self) -> sp.Matrix:
       
        R = self.eff_orientation_matrix
        roll = sp.atan2(R[2, 1], R[2, 2])
        pitch = sp.atan2(-R[2, 0], sp.sqrt(R[0,0]**2 + R[1,0]**2))
        yaw = sp.atan2(R[1, 0], R[0, 0])

        return sp.Matrix([[roll], [pitch], [yaw]])  

    def get_euler_angles_XYZ(self) -> sp.Matrix:

        R = self.eff_orientation_matrix
        roll = sp.atan2(-R[1, 2], R[2, 2])
        pitch = sp.atan2(R[0, 2], sp.sqrt(R[0,0]**2 + R[0,1]**2))
        yaw = sp.atan2(-R[0, 1], R[0, 0])

        return sp.Matrix([[roll], [pitch], [yaw]])

    def get_quaternion(self) -> sp.Matrix:
        """
        Convert the end-effector rotation matrix to a unit quaternion [qw, qx, qy, qz]
        using Shepperd's method (stable for all rotations, including 180° ones).
        The branch is a sp.Piecewise, so it resolves once numeric values are substituted.
        """
        R = self.eff_orientation_matrix
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        # Branch 1: trace is the largest -> qw dominant
        s1 = 2 * sp.sqrt(1 + trace)
        q1 = [s1 / 4,
              (R[2, 1] - R[1, 2]) / s1,
              (R[0, 2] - R[2, 0]) / s1,
              (R[1, 0] - R[0, 1]) / s1]

        # Branch 2: R00 is the largest -> qx dominant
        s2 = 2 * sp.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q2 = [(R[2, 1] - R[1, 2]) / s2,
              s2 / 4,
              (R[0, 1] + R[1, 0]) / s2,
              (R[0, 2] + R[2, 0]) / s2]

        # Branch 3: R11 is the largest -> qy dominant
        s3 = 2 * sp.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q3 = [(R[0, 2] - R[2, 0]) / s3,
              (R[0, 1] + R[1, 0]) / s3,
              s3 / 4,
              (R[1, 2] + R[2, 1]) / s3]

        # Branch 4: R22 is the largest -> qz dominant
        s4 = 2 * sp.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q4 = [(R[1, 0] - R[0, 1]) / s4,
              (R[0, 2] + R[2, 0]) / s4,
              (R[1, 2] + R[2, 1]) / s4,
              s4 / 4]

        cond1 = trace >= R[0, 0]
        cond1 = sp.And(trace >= R[0, 0], trace >= R[1, 1], trace >= R[2, 2])
        cond2 = sp.And(R[0, 0] >= R[1, 1], R[0, 0] >= R[2, 2])
        cond3 = R[1, 1] >= R[2, 2]

        comps = []
        for i in range(4):
            comps.append(sp.Piecewise(
                (q1[i], cond1),
                (q2[i], cond2),
                (q3[i], cond3),
                (q4[i], True),
            ))

        return sp.Matrix(comps)  # [qw, qx, qy, qz]

    def quaternions_to_euler_XYZ(self, quarternions: sp.Matrix) -> sp.Matrix:
        """
        Convert a unit quaternion [qw, qx, qy, qz] to Euler angles (roll, pitch, yaw) in XYZ convention.
        """
        qw, qx, qy, qz = quarternions

        # Roll (X-axis rotation)
        roll = sp.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

        # Pitch (Y-axis rotation)
        pitch = sp.asin(2 * (qw * qy - qz * qx))

        # Yaw (Z-axis rotation)
        yaw = sp.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        return sp.Matrix([[roll], [pitch], [yaw]])


        # ---- numeric evaluation (lambdify-backed, dict interface) ----
    
    @staticmethod
    def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
        """Shepperd's method. Returns [qw, qx, qy, qz]."""
        t = R[0, 0] + R[1, 1] + R[2, 2]
        k = int(np.argmax([t, R[0, 0], R[1, 1], R[2, 2]]))  # which term is largest

        if k == 0:
            s = 2.0 * np.sqrt(1.0 + t)
            q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        elif k == 1:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif k == 2:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]

        q = np.array(q, dtype=float)
        return q / np.linalg.norm(q)

    

    def _values_from_dict(self, values: dict) -> list[float]:
        """Order a {symbol_or_name: value} dict by self.variables and cast to float."""
        by_name = {(k.name if isinstance(k, sp.Symbol) else str(k)): v
                   for k, v in values.items()}
        missing = [s.name for s in self.variables if s.name not in by_name]
        if missing:
            raise KeyError(f"Missing values for variables: {missing}")
        return [float(by_name[s.name]) for s in self.variables]  # float() handles -sp.pi/2 etc.

    def _expressions(self) -> dict:
        # callables so expensive ones (quaternion) are only built if requested
        return {
            "transformation_matrix": lambda: self.transformation_matrix,
            "eff_pos": lambda: self.eff_pos,
            "orientation_matrix": lambda: self.eff_orientation_matrix,
            "euler_ZYX": lambda: self._orientation_ZYX,
            "euler_XYZ": self.get_euler_angles_XYZ,
            "quaternion": self.get_quaternion,
            "jacobian": lambda: self.jacobian,
            "pose_jacobian": lambda: self.pose_jacobian,
            # d(R)/dq, R flattened row-major -> (9, n); used for the geometric angular Jacobian
            "rotation_jacobian": lambda: self.eff_orientation_matrix.reshape(9, 1).jacobian(self.variables),
        }

    def evaluate(self, name: str, values: dict) -> np.ndarray:
        """
        Numeric equivalent of expr.subs(values), using a cached sp.lambdify function.
        name: one of transformation_matrix, eff_pos, orientation_matrix, euler_ZYX,
              euler_XYZ, quaternion, jacobian, pose_jacobian, rotation_jacobian
        """
        fn = self._lambdas.get(name)
        if fn is None:
            expr = self._expressions()[name]()
            fn = sp.lambdify(self.variables, expr, modules="numpy", cse=True)
            self._lambdas[name] = fn
        return np.asarray(fn(*self._values_from_dict(values)), dtype=float)

    # convenience wrappers
    def pos_np(self, values: dict) -> np.ndarray:          # (3,)
        return self.evaluate("eff_pos", values).ravel()

    def quaternion_np(self, values: dict) -> np.ndarray:   # (4,) [qw, qx, qy, qz]
            R = self.evaluate("orientation_matrix", values)
            return self.rotmat_to_quat(R)

    def euler_ZYX_np(self, values: dict) -> np.ndarray:    # (3,)
        return self.evaluate("euler_ZYX", values).ravel()

    def jacobian_np(self, values: dict) -> np.ndarray:     # (3, n)
        return self.evaluate("jacobian", values)

    def pose_jacobian_np(self, values: dict) -> np.ndarray:  # (6, n)
        return self.evaluate("pose_jacobian", values)

    def pose_jacobian_pinv(self, values: dict) -> np.ndarray:  # (n, 6)
        return np.linalg.pinv(self.pose_jacobian_np(values))

    def orientation_matrix_np(self, values: dict) -> np.ndarray:  # (3, 3)
        return self.evaluate("orientation_matrix", values)

    def angular_jacobian_np(self, values: dict) -> np.ndarray:  # (3, n)
        """
        Geometric angular-velocity Jacobian (base frame): omega = Jw @ dq.
        Column i is vee(dR/dq_i @ R^T); prismatic joints give a zero column.
        Unlike pose_jacobian (Euler-angle rates) it has no gimbal-lock singularity.
        """
        R = self.evaluate("orientation_matrix", values)
        dR = self.evaluate("rotation_jacobian", values).reshape(3, 3, -1)
        W = np.einsum("ijk,lj->ilk", dR, R)  # W[:, :, k] = dR_k @ R^T (skew-symmetric)
        return np.stack([W[2, 1], W[0, 2], W[1, 0]])

    def geometric_jacobian_np(self, values: dict) -> np.ndarray:  # (6, n) [linear; angular]
        return np.vstack([self.jacobian_np(values), self.angular_jacobian_np(values)])

    # ---- rotation helpers (numpy) ----

    @staticmethod
    def euler_ZYX_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), angles in radians (inverse of get_euler_angles_ZYX)."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp_ = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy * cp, cy * sp_ * sr - sy * cr, cy * sp_ * cr + sy * sr],
            [sy * cp, sy * sp_ * sr + cy * cr, sy * sp_ * cr - cy * sr],
            [-sp_,    cp * sr,                 cp * cr],
        ])

    @staticmethod
    def rotmat_to_euler_ZYX(R: np.ndarray) -> np.ndarray:
        """Numeric twin of get_euler_angles_ZYX. Returns [roll, pitch, yaw] in radians."""
        return np.array([
            np.arctan2(R[2, 1], R[2, 2]),
            np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])),
            np.arctan2(R[1, 0], R[0, 0]),
        ])

    @staticmethod
    def rotation_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
        """
        Rotation vector (axis * angle, base frame) that rotates R_current onto R_target,
        i.e. log(R_target @ R_current^T). Always takes the short way round (angle <= pi).
        Pairs with angular_jacobian_np: driving omega along this vector closes the error.
        """
        q = ForwardKinematics.rotmat_to_quat(R_target @ R_current.T)
        if q[0] < 0.0:  # q and -q are the same rotation; pick the one with angle <= pi
            q = -q
        s = np.linalg.norm(q[1:])
        if s < 1e-9:
            return 2.0 * q[1:]  # small-angle limit
        return (2.0 * np.arctan2(s, q[0]) / s) * q[1:]
    
if __name__ == "__main__":
    
    x,y, yaw, lift, q_l1, q_l2, q_l3, q_l4, q_l5, q_l6, q_l7 = sp.symbols('x y yaw lift q_l1 q_l2 q_l3 q_l4 q_l5 q_l6 q_l7')
    p2 = sp.pi/2
    m_p2 = -1.0 * p2
    h = 0.697 + 0.1
    l01 = 0.0915
    l12 = 0.061131
    l23 = 0.063
    l34 = 0.157
    l45 = 0.0965
    l56 = 0.1195
    l_eff = 0.114501 + 0.055

    dh_params = [
        DHParameter(d=0.0, theta= m_p2, a= 0.0, alpha= m_p2),
        DHParameter(d=x, theta=0.0, a=0.0, alpha=0.0),
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2), 
        DHParameter(d=0.0, theta=p2, a=0.0, alpha=m_p2),
        DHParameter(d=y, theta=0.0, a=0.0, alpha=0.0),
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2),
        DHParameter(d=lift, theta=yaw, a=0.0, alpha=0.0),
        DHParameter(d=h, theta=0.0, a=0.0, alpha=0.0),  #BASE
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
    
    symbols = [x, y, yaw, lift, q_l1, q_l2, q_l3, q_l4, q_l5, q_l6, q_l7]

    fk = ForwardKinematics(dh_params, variables=symbols)

    subs = {x: 0.0, y: 0.0, yaw: 0.0, lift: 0.0, q_l1: 0.0, q_l2: 0.0, q_l3: 0.0, q_l4: 0.0, q_l5: 0.0, q_l6: 0.0, q_l7: 0.0}  # Substitute some values for demonstration
    # print("End-Effector Position:")
    # print(fk.eff_pos.subs(subs))  

    # print("End-Effector Orientation (Quaternion):")
    # print(fk.get_quaternion().subs(subs))

    # print("End-Effector Orientation (Euler Angles ZYX):")
    # print(fk.get_euler_angles_ZYX().subs(subs))  

    print(fk.pos_np(subs))
    print(fk.quaternion_np(subs))
    print(fk.euler_ZYX_np(subs))
    # print(fk.pose_jacobian_np(subs))
    # print(fk.pose_jacobian_pinv(subs))