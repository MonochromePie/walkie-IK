import sympy as sp

class DHParameter:
    def __init__(self, d: float | sp.Symbol , theta: float | sp.Symbol , a: float | sp.Symbol , alpha: float | sp.Symbol ):
        self.a = a
        self.alpha = alpha
        self.d = d
        self.theta = theta

    def get_transformation_matrix(self) -> sp.Matrix:
        """
        Calculate the transformation matrix using the DH parameters.
        Returns:
            A 4x4 numpy array representing the transformation matrix.
        """
        a, alpha, d, theta = self.a, self.alpha, self.d, self.theta
        
        # Create the transformation matrix using sympy for symbolic computation
        T = sp.Matrix([
            [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
            [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
            [0, sp.sin(alpha), sp.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        
        return sp.Matrix(T)  


if __name__ == "__main__":

    x = sp.symbols('x')
    print(x)
    dh_param = DHParameter(d=0.5, theta=x, a=1.0, alpha=sp.pi/2)
    transformation_matrix = dh_param.get_transformation_matrix()
    print("Transformation Matrix:")
    print(transformation_matrix)