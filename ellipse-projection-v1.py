import numpy as np
import matplotlib.pyplot as plt


class EllipseCalculator:

    def __init__(self, num_attrs: int) -> None:
        self.attrs = num_attrs
        self.original_ellipsoid = None
        self.ellipsoid = None
        self.axis_bias = None
        self.rotations = None

    def set_ellipsoid(self, new_ellipsoid: np.ndarray[np.ndarray[float]] | None = None) -> None:
        if new_ellipsoid is None:
            if self.original_ellipsoid is None:
                raise AttributeError("No ellipsoid set")
        else:
            self.original_ellipsoid = new_ellipsoid.copy()

        self.ellipsoid = self.original_ellipsoid.copy()
        self.rotations = np.identity(self.attrs)
        self.update_axis_bias()


    def ellipsoid_from_axes(self, axes: np.ndarray[np.ndarray[float]]) -> np.ndarray[np.ndarray[float]]:
        A_inv = axes.T @ axes  # is symmetric, positive definite
        return np.linalg.inv(A_inv)
    
    def in_ellipsoid(self, point: np.ndarray[float]) -> bool:
        return point @ self.ellipsoid @ point.T <= 1


    def _get_rotation_matrix(self, axes: tuple[int], angle: float) -> np.ndarray[np.ndarray[float]]:
        S = np.identity(self.attrs)

        S[[0, axes[0]]] = S[[axes[0], 0]]
        S[[1, axes[1]]] = S[[axes[1], 1]]

        c, s = np.cos(angle), np.sin(angle)
        R_XY = np.identity(self.attrs)
        R_XY[0, 0], R_XY[0, 1], R_XY[1, 0], R_XY[1, 1] = c, -s, s, c

        R = R_XY @ S
        R[[1, axes[1]]] = R[[axes[1], 1]]
        R[[0, axes[0]]] = R[[axes[0], 0]]
        
        return R
    
    def apply_new_rotation(self, axes: tuple[int], angle: float) -> None:
        if self.ellipsoid is None:
            raise AttributeError("Cannot rotate before setting ellipsoid")
        
        R = self._get_rotation_matrix(axes, angle)
        
        self.ellipsoid = R.T @ self.ellipsoid @ R
        self.rotations = self.rotations @ R

    def update_axis_bias(self) -> None:

        x_bias = np.square(self.rotations[:, 0])
        y_bias = np.square(self.rotations[:, 1])
        self.axis_bias = (x_bias, y_bias)

    def set_axis_bias(self, bias: tuple[np.ndarray[float]]) -> None:
        self.axis_bias = bias
        print("helo")
        
        B = np.identity(self.attrs)
        print(B)
        B[:, 0] = np.sqrt(bias[0])
        B[:, 1] = np.sqrt(bias[1])

        

        #for i in range(2, self.attrs):
            #new_v = -np.linalg.inv(B[:i, :i].T) @ B[]


    def project_onto_plane(self) -> np.ndarray[np.ndarray[float]]:
        J = self.ellipsoid[:2, :2]
        L = self.ellipsoid[2:, :2]
        K = self.ellipsoid[2:, 2:]

        P = np.subtract(J, (L.T @ np.linalg.inv(K)) @ L)
        A, B, C = P[0, 0], P[0, 1] + P[1, 0], P[1, 1] 
        return A, B, C
    
    def plot_on_plane(self, x_range: tuple[float] = None, y_range: tuple[float] = None) -> None:

        A, B, C = self.project_onto_plane()

        x_min, x_max = (-2/np.sqrt(A), 2/np.sqrt(A)) if x_range is None else (x_range[0], x_range[1])
        y_min, y_max = (-2/np.sqrt(C), 2/np.sqrt(C)) if y_range is None else (y_range[0], y_range[1])

        delta = 0.025
        x = np.arange(x_min, x_max, delta)
        y = np.arange(y_min, y_max, delta)
        X, Y = np.meshgrid(x, y)
        Z = A*X**2 + B*X*Y + C*Y**2

        fig, ax = plt.subplots()
        CS = ax.contourf(X, Y, Z, [0, 1, 2, 3, 4])
        #ax.clabel(CS, inline=True, fontsize=10)
        ax.set_title('Ellipse')

        plt.show()
        

    
    def __str__(self) -> str:
        msg = f"*** Ellipse Calculator at {id(self)} *** \n"
        if self.original_ellipsoid is None:
            msg += "Empty. Set an ellipsoid to use."
            return msg
        msg += "Original ellipsoid: \n"
        msg += str(np.round(self.original_ellipsoid, decimals = 2)) + "\n\n"
        msg += "Rotations applied, matrix form: \n"
        msg += str(np.round(self.rotations, decimals = 2)) + "\n\n"
        msg += "Resulting ellipsoid: \n"
        msg += str(np.round(self.ellipsoid, decimals = 2)) + "\n\n"
        
        msg += "Axis biases: \n"
        msg += " attribute | x-bias | y-bias \n"
        self.update_axis_bias()
        for i in range(self.attrs):
            msg += "   attr" + str(i)
            msg += "   " if i < 10 else "  "
            for ax in [0, 1]:
                msg += "|"
                x = int(round(100*self.axis_bias[ax][i]))
                if x == 100:
                    msg += f"  100%  "
                elif x >= 10:
                    msg += f"  {x}%   "
                else:
                    msg += f"   {x}%   "
            msg += "\n"
        
        msg += "\nEllipse projected onto these axes: \n"
        A, B, C = self.project_onto_plane()

        msg += f"{round(A, 2)}x^2 + {round(B, 2)}xy + {round(C, 2)}y^2 = 1\n"

        return msg
        

            


        



# TESTING

#each row is an axis of the ellipsoid
EC = EllipseCalculator(6)

test_axes = np.array(
    [
        [6, 0, 2, 1, 0, 7],
        [1, 5, 0, 3, 0, 3],
        [0, 1, 9, 4, 2, 2],
        [1, 2, 0, 6, 1, 0],
        [6, 0, 0, 0, 0, 0],
        [0, 0, 7, 0, 8, 3]
    ]
)

A = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 6]
    ]
)

bias = (np.ndarray([20, 30, 5, 15, 0, 30]), np.ndarray([0, 9, 11, 75, 2, 3]))
EC.set_ellipsoid(A)
print("hello?!?!")
#EC.apply_new_rotation((1, 3), np.pi/4)
#EC.apply_new_rotation((0, 5), np.pi/6)
#EC.apply_new_rotation((0, 1), np.pi/3)
#print(EC)
#EC.plot_on_plane()
EC.set_axis_bias(bias)
print("hi")

    

