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
        self.__ellipsoid: np.ndarray[np.ndarray[float]] = None

        if type(data) == StatsCalculator:
            self.set_ellipsoid(data.get_covariance())
        else:
            self.set_ellipsoid(data)


    def set_ellipsoid(self, new_ellipsoid: np.ndarray[np.ndarray[float]] | None = None) -> None:
        """
        Sets, or resets, self.__ellipsoid. Resets all other variables accordingly.

        :param new_ellipsoid: Hyperellipsoid, as an NxN matrix.
        :raise ValueError: If no ellipsoid passed, and no original ellipsoid.
        :raise ValueError: If new_ellipsoid is not an NxN matrix.
        """
        self.__N = new_ellipsoid.shape[0]
        self.__ellipsoid = new_ellipsoid.copy()

    

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

        :return: 3-Tuple of A, B, C; coefficents in ellipse equation Ax^2 + Bxy + Cy^2 = 1.
        """
        # orthonormalise <u, v>
        u = vec1 / np.linalg.norm(vec1)
        v = vec2 - np.dot(vec2, u)
        if np.linalg.norm(v) == 0:
            raise ValueError("u and v must span a plane")
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
    
    def transformation_to_points(self, T: np.ndarray[float], num_points: int = 50) -> np.ndarray[float]:
        X = np.linspace(0, 2*np.pi, num=num_points)
        Y = np.linspace(0, 2*np.pi, num=num_points)
        circle = np.vstack([np.cos(X), np.sin(Y)])
        return T @ circle


    def plot_on_plane(self, points: np.ndarray) -> None:
        """
        Generates and displays a static plot of the 2-D projection of the ellipsoid.

        :optional params x_range, y_range: Tuples containing the min and max values of x and y to plot.
        :optional param res: The resolution of the plot, smaller value -> higher resolution.
        """
        plt.plot(points[0], points[1])
        plt.show()
        
    