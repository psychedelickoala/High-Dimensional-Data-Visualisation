import numpy as np
import matplotlib.pyplot as plt
from data_analysis_v1 import StatsCalculator


class EllipseCalculator:

    """
        Performs operations on a hyperellipsoid in matrix form.
    """

    def __init__(self, data: StatsCalculator | np.ndarray[np.ndarray[float]]) -> None:
        """
        Initialises.

        :param data: The ellipsoid to analyse, as a matrix, OR a stats calculator object.
        """
        self.__N: int = None
        self.__original_ellipsoid: np.ndarray[np.ndarray[float]] = None
        self.__ellipsoid: np.ndarray[np.ndarray[float]] = None
        self.__axis_bias: tuple[dict] = None
        self.__rotations: np.ndarray[np.ndarray[float]] = None
        self.__attribute_names: list[str] = None

        if type(data) == StatsCalculator:
            self.__attribute_names = data.get_attribute_names()
            self.set_ellipsoid(data.get_covariance())

        else:
            self.__attribute_names = ["attr" + str(i) for i in range(data.shape()[0])]
            self.set_ellipsoid(data)


    def set_ellipsoid(self, new_ellipsoid: np.ndarray[np.ndarray[float]] | None = None) -> None:
        """
        Sets, or resets, self.__ellipsoid. Resets all other variables accordingly.

        :param new_ellipsoid: Hyperellipsoid, as an NxN matrix.
        :raise ValueError: If no ellipsoid passed, and no original ellipsoid.
        :raise ValueError: If new_ellipsoid is not an NxN matrix.
        """
        if new_ellipsoid is None:
            if self.__original_ellipsoid is None:
                raise ValueError("No ellipsoid previously set")
        elif new_ellipsoid.shape[0] != new_ellipsoid.shape[1]:
            raise ValueError(f"Argument has wrong shape. Required square, got shape {new_ellipsoid.shape}")
        else:
            self.__N = new_ellipsoid.shape[0]
            self.__original_ellipsoid = new_ellipsoid.copy()

        self.__ellipsoid = self.__original_ellipsoid.copy()
        self.__rotations = np.identity(self.__N)
        self.__update_axis_bias()


    def ellipsoid_from_axes(self, axes: np.ndarray[np.ndarray[float]]) -> np.ndarray[np.ndarray[float]]:
        """
        Constructs an ellipsoid in matrix form from a collection of points on its boundary.

        :param axes: An NxN array, each row representing a point on the ellipsoid.
        :return: The NxN matrix form of the ellipsoid.
        """
        A_inv = axes.T @ axes  # is symmetric, positive definite
        return np.linalg.inv(A_inv)
    

    def in_ellipsoid(self, point: np.ndarray[float]) -> bool:
        """
        Determines if a point is inside the calculator's ellipsoid.

        :param point: An array of length N, representing a point in N-dimensional space.
        :return: True if the point is inside the ellipsoid, otherwise False.
        """
        return point @ self.__ellipsoid @ point.T <= 1


    def __get_rotation_matrix(self, axes: tuple[int], angle: float) -> np.ndarray[np.ndarray[float]]:
        """
        Forms a rotation matrix given two basis vectors to rotate and an angle to rotate them by.

        :param axes: Tuple of integer values between 0 and N-1, indicating which two of the N basis vectors to move.
        :param angle: Angle in radians to rotate by.

        :return: An NxN rotation matrix.
        """
        S = np.identity(self.__N)

        S[[0, axes[0]]] = S[[axes[0], 0]]
        S[[1, axes[1]]] = S[[axes[1], 1]]

        c, s = np.cos(angle), np.sin(angle)
        R_XY = np.identity(self.__N)
        R_XY[0, 0], R_XY[0, 1], R_XY[1, 0], R_XY[1, 1] = c, -s, s, c

        R = R_XY @ S
        R[[1, axes[1]]] = R[[axes[1], 1]]
        R[[0, axes[0]]] = R[[axes[0], 0]]
        
        return R
    

    def apply_new_rotation(self, axes: tuple[int], angle: float) -> None:
        """
        Rotates our perspective of the ellipsoid, modifying __ellipsoid, __rotations and __axis_bias

        :param axes: Tuple of integer values between 0 and N-1, indicating which two of the N basis vectors to move.
        :param angle: Angle in radians to rotate by.

        :raises AttributeError: If __ellipsoid has not been set.
        """
        if self.__ellipsoid is None:
            raise AttributeError("Cannot rotate before setting ellipsoid")
        
        R = self.__get_rotation_matrix(axes, angle)
        
        self.__ellipsoid = R.T @ self.__ellipsoid @ R
        self.__rotations = self.__rotations @ R

        self.__update_axis_bias()


    def __update_axis_bias(self) -> None:
        """
        Updates axis biases based on __rotations.
        """
        x_bias = np.square(self.__rotations[:, 0])
        y_bias = np.square(self.__rotations[:, 1])
        x_dict = {self.__attribute_names[i]: x_bias[i]*100 for i in range(self.__N) if x_bias[i] > 0.001}
        y_dict = {self.__attribute_names[i]: y_bias[i]*100 for i in range(self.__N) if y_bias[i] > 0.001}
        self.__axis_bias = (x_dict, y_dict)


    def __new_transformation(self, new_T: np.ndarray[np.ndarray[float]]) -> None:
        """
        Scrubs previous transformations and applies a new one.

        :param new_T: An NxN matrix representing the new transformation to apply.
        :raises AttributeError: If __ellipsoid has not been set.
        """
        if self.__ellipsoid is None:
            raise AttributeError("Cannot transform before setting ellipsoid")
        
        self.set_ellipsoid()
        self.__rotations = new_T
        self.__ellipsoid = new_T.T @ self.__ellipsoid @ new_T
        self.__update_axis_bias()


    def set_axis_bias(self, x_bias: dict, y_bias: dict) -> None:
        """
        Rotates our perspective according to requested axis biases towards set attributes.
        Constructs an orthonormal basis, finding orthogonal verctors by solving systems of equations and then normalising.

        :param x_bias, y_bias: Lists of percentage weightings for attributes belonging to the x and y axes.
        """

        assert set(x_bias.keys()).issubset(set(self.__attribute_names))
        assert set(y_bias.keys()).issubset(set(self.__attribute_names))

        # want first attribute in x_bias to be the first row/column, first attribute in y_bias to be second row/column
        x_ind, y_ind = None, None
        for i, attr in enumerate(self.__attribute_names):
            if attr in x_bias.keys() and x_ind is None:
                x_ind = i
            if attr in y_bias.keys() and y_ind is None:
                y_ind = i
            if attr in x_bias.keys() and attr in y_bias.keys():
                raise ValueError("x attributes and y attributes should not overlap.")
            
        if {x_ind, y_ind} != {0, 1}:
            order = ((0, x_ind), (1, y_ind)) if x_ind < y_ind else ((1, y_ind), (0, x_ind))
            for i1, i2 in order:
                self.__attribute_names[i1], self.__attribute_names[i2] = self.__attribute_names[i2], self.__attribute_names[i1]
                self.__original_ellipsoid[[i1, i2]] = self.__original_ellipsoid[[i2, i1]]
                self.__original_ellipsoid[:, [i1, i2]] = self.__original_ellipsoid[:, [i2, i1]]

        x_vec = np.array(
            [(x_bias[attr] if attr in x_bias.keys() else 0) for attr in self.__attribute_names]
        )
        y_vec = np.array(
            [(y_bias[attr] if attr in y_bias.keys() else 0) for attr in self.__attribute_names]
        )

        print(x_vec)
        print(y_vec)
        
        B = np.identity(self.__N)
        B[:, 0] = np.sqrt(x_vec*0.01).T
        B[:, 1] = np.sqrt(y_vec*0.01).T

        for i in range(2, self.__N):
            v = B[:, i].T
            v[:i] = - B[i, :i] @ np.linalg.inv(B[:i, :i])
            v /= np.linalg.norm(v)

        self.__new_transformation(B)


    def project_onto_plane(self) -> tuple[float]:
        """
        Orthogonally projects original ellipsoid onto plane from our perspective.

        :raises AttributeError: If __ellipse has not been set.
        :return: 3-Tuple of A, B, C; coefficents in ellipse equation Ax^2 + Bxy + Cy^2 = 1.
        """
        if self.__ellipsoid is None:
            raise AttributeError("Cannot project before setting ellipsoid")

        J = self.__ellipsoid[:2, :2]
        L = self.__ellipsoid[2:, :2]
        K = self.__ellipsoid[2:, 2:]

        P = np.subtract(J, (L.T @ np.linalg.inv(K)) @ L)
        A, B, C = P[0, 0], P[0, 1] + P[1, 0], P[1, 1] 
        return A, B, C
    
    
    def plot_on_plane(self, x_range: tuple[float] = None, y_range: tuple[float] = None, res: float = 0.025) -> None:
        """
        Generates and displays a static plot of the 2-D projection of the ellipsoid.

        :optional params x_range, y_range: Tuples containing the min and max values of x and y to plot.
        :optional param res: The resolution of the plot, smaller value -> higher resolution.
        """
        A, B, C = self.project_onto_plane()

        x_min, x_max = (-5/np.sqrt(A), 5/np.sqrt(A)) if x_range is None else x_range
        y_min, y_max = (-5/np.sqrt(C), 5/np.sqrt(C)) if y_range is None else y_range

        x = np.arange(x_min, x_max, res)
        y = np.arange(y_min, y_max, res)
        X, Y = np.meshgrid(x, y)
        Z = A*X**2 + B*X*Y + C*Y**2

        fig, ax = plt.subplots()
        CS = ax.contourf(X, Y, Z, [0, 1, 4, 9])
        #ax.clabel(CS, inline=True, fontsize=10)
        ax.set_title(f'Ellipse Calculator at {id(self)}')

        plt.show()
        
    
    def __str__(self) -> str:
        """
        Describes the current state of the calculator.

        :return: String detailing the attributes of the calculator.
        """
        msg = f"*** Ellipse Calculator at {id(self)} *** \n"
        if self.__original_ellipsoid is None:
            msg += "Empty. Set an ellipsoid to use."
            return msg
        msg += "Original ellipsoid: \n"
        msg += str(np.round(self.__original_ellipsoid, decimals = 2)) + "\n\n"
        msg += "Rotations applied, matrix form: \n"
        msg += str(np.round(self.__rotations, decimals = 2)) + "\n\n"
        msg += "Resulting ellipsoid: \n"
        msg += str(np.round(self.__ellipsoid, decimals = 2)) + "\n\n"
        
        self.__update_axis_bias()
        msg += "x-axis biases: \n" + str(self.__axis_bias[0]) + "\n"
        msg += "y-axis biases: \n" + str(self.__axis_bias[1]) + "\n"
        
        """
        msg += " attribute | x-bias | y-bias \n"
        self.__update_axis_bias()
        for i in range(self.__N):
            msg += "   attr" + str(i)
            msg += "   " if i < 10 else "  "
            for ax in [0, 1]:
                msg += "|"
                x = int(round(100*self.__axis_bias[ax][i]))
                if x == 100:
                    msg += f"  100%  "
                elif x >= 10:
                    msg += f"  {x}%   "
                else:
                    msg += f"   {x}%   "
            msg += "\n"
        """
        
        msg += "\nEllipse projected onto these axes: \n"
        A, B, C = self.project_onto_plane()

        msg += f"{round(A, 2)}x^2 + {round(B, 2)}xy + {round(C, 2)}y^2 = 1\n"

        return msg
    