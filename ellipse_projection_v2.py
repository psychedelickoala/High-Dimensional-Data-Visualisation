import numpy as np
import matplotlib.pyplot as plt
from data_analysis_v2 import StatsCalculator


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

    

    def in_ellipsoid(self, point: np.ndarray[float]) -> bool:
        """
        Determines if a point is inside the calculator's ellipsoid.

        :param point: An array of length N, representing a point in N-dimensional space.
        :return: True if the point is inside the ellipsoid, otherwise False.
        """
        return point @ self.__ellipsoid @ point.T <= 1


    def project_onto_plane(self, vec1: np.ndarray[float], vec2: np.ndarray[float]) -> tuple[float]:
        """
        Orthogonally projects original ellipsoid onto plane from our perspective.

        :raises AttributeError: If __ellipse has not been set.
        :return: 3-Tuple of A, B, C; coefficents in ellipse equation Ax^2 + Bxy + Cy^2 = 1.
        """
        # orthonormalise <u, v>
        u = vec1 / np.linalg.norm(vec1)
        v = vec2 - np.dot(vec2, u)
        v /= np.linalg.norm(v)

        P = np.vstack([u, v])
        M = P @ np.linalg.inv(self.__ellipsoid) @ P.T

        J, K, L = M[0, 0], M[0, 1] + M[1, 0], M[1, 1]

        a = (J + L + np.sqrt((J-L)**2 + K**2))/2
        b = J+L-a
        theta = np.arcsin(K/(a-b))/2

        T = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) \
            @ np.array([[np.sqrt(a), 0],[0, np.sqrt(b)]])

        return T
    
    
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
        
    