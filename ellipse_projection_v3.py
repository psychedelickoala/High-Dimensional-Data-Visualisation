import numpy as np
import matplotlib.pyplot as plt


class EllipseCalculator:

    """
        Performs operations on a hyperellipsoid in matrix form.
    """

    def __init__(self, data: np.ndarray, num_points: int = 50) -> None:
        """
        Initialises.

        :param data: The ellipsoid to analyse, as a matrix.
        :param num_points: The number of points to plot when projecting the ellipse.
        """
        self.__ellipsoid: np.ndarray = data
        self.__N: int = self.__ellipsoid.shape[0]
        
        # constructing a circle to transform - done ONCE (on initialisation)
        X = np.linspace(0, 2*np.pi, num=num_points)
        Y = np.linspace(0, 2*np.pi, num=num_points)
        
        self.__circle: np.ndarray = np.vstack([np.cos(X), np.sin(Y)])
    
    
    def get_projection_matrix(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Constructs an orthonormalised projection matrix from two vectors.

        :params vec1, vec2: 1 x n vectors spanning a plane.
        :return: a 2 x n matrix which, post-multiplied by an n x 1 vector, projects that vector onto the plane.

        :raises ValueError: if vec1 and vec2 are linearly dependent.
        """
        # orthonormalise u, v
        u = vec1 / np.linalg.norm(vec1)
        v = vec2 - np.dot(vec2, u)*u
        
        # checking linear independence
        if np.linalg.norm(v) == 0:
            raise ValueError("Input vectors must span a plane")
        
        v /= np.linalg.norm(v)

        # return 2 x n array
        return np.vstack([u, v])
    
    
    def points_onto_plane(self, P: np.ndarray, points: np.ndarray, ) -> np.ndarray:
        """
        Projects a collection of points onto the plane spanned by two vectors.

        :param P: Orthonormal 2 x n projection matrix onto a plane.
        :param points: An n x num_points array, with num_points n-dimensional points to project.

        :return: A 2 x num_points array of num_points 2-d points on the plane
        """
        return P @ points
    

    def axes_onto_plane(self, P: np.ndarray) -> np.ndarray:
        """
        Projects the basis vectors onto a plane.

        :param P: Orthonormal 2 x n projection matrix onto a plane.
        :return: n x 2 matrix with each row containing the x, y coordinates of the projected axis
        """
        I = np.identity(self.__N)
        return (P @ I).T


    def ellipsoid_onto_plane(self, P: np.ndarray, mean: np.ndarray = None, m_dists = [1]) -> list[np.ndarray]:
        """
        Orthogonally projects original ellipsoid onto plane.

        :param P: Orthonormalised 2 x n projection matrix onto a plane.
        :optional param mean: An n x 1 vector; the centre of the hyperellipsoid.
        :optional param m_dists: List of Mahalanobis distances to draw ellipses
        
        :return: A list of 2 x num_points arrays of points for each specified Mahalnobis distances.
        """
        # M is the INVERSE of the ellipse equation matrix, so M = BB^T for change of basis B
        M = P @ np.linalg.inv(self.__ellipsoid) @ P.T
        J, K, L = M[0, 0], M[0, 1] + M[1, 0], M[1, 1]

        # see algebra in documentation
        a = (J + L + np.sqrt((J-L)**2 + K**2))/2
        b = J+L-a
        theta = np.arcsin(K/(a-b))/2

        # a matrix to transform the circle into the required ellipse: stretching then rotation
        T = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) \
            @ np.array([[np.sqrt(a), 0],[0, np.sqrt(b)]])

        # E is the projected ellipse
        ellipses = []
        for dist in m_dists:
            E = dist * T @ self.__circle
        
            if mean is not None:
                proj_mean = P @ mean
                E[0] += proj_mean[0]
                E[1] += proj_mean[1]

            ellipses.append(E)

        return ellipses


    def plot_on_plane(self, ellipses: list[np.ndarray], data_points: np.ndarray = None, axes: np.ndarray = None) -> None:
        """
        Generates and displays a static plot of a collection of points.

        :param ellipses: list of 2 x num_points ordered arrays of points to plot as ellipses
        :optional param data_points: A 2 x num_points array of num_points 2-d points on the plane
        :optional param axes: An n x 2 matrix with each row containing the x, y coordinates of the projected axis
        """
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ["#fecb3e", "#fc8370", "#c2549d", "#7e549e"])

        if axes is not None:
            for i in range(self.__N):
                plt.plot([0, axes[i][0]], [0, axes[i][1]], c = "grey", linewidth = 1)

        for i in range(len(ellipses)):
            plt.plot(ellipses[i][0], ellipses[i][1])

        if data_points is not None:
            plt.scatter(data_points[0], data_points[1], c = "#000000", marker = ".")

        plt.show()