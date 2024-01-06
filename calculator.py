import numpy as np
import csv
import matplotlib.pyplot as plt


class Calculator:

    """
    An object to store and process information about a dataset passed on initialisation.

    :attribute __circle: 2 x num_points array; points around the unit circle, anticlockwise.
    :attribute __covariance: dim x dim array; covariance matrix of data.
    :attribute __data: dim x num_points array; each column is a datapoint, sorted in order of increasing Mahalanobis distance.
    :attribute __dim: integer greater than one; number of dimensions.
    :attribute __ellipsoid: dim x dim array; inverse of covariance matrix hence matrix of ellipse of best fit.
    :attribute __mahal_dists: num_points length array; sorted Mahalanobis distances of each data point from the mean.
    :attribute __mean: dim length array; centroid of the data set.
    """

    def __init__(self, data: str | np.ndarray, ellipse_res: int = 30) -> None:
        """
        Initialises calculator.
        Reads and sorts data and constructs circle. These can be reset later.

        :param data: filename of a csv file containing data, or dim x num_points array of data points.
        :optional param ellipse_res: integer number of points to draw of the projected ellipses.
        """

        self.set_data(data)
        self.set_ellipse_res(ellipse_res)

    def __len__(self) -> int:
        """Returns the number of dimensions"""
        return len(self.__mahal_dists)
    
    def get_covariance(self) -> np.ndarray[np.ndarray[float]]:
        """Returns covariance matrix, dim x dim array"""
        return self.__covariance
    
    def get_data(self) -> np.ndarray:
        """Returns sorted data, dim x num_points array"""
        return self.__data
    
    def get_mean(self) -> np.ndarray:
        """Returns mean of data, dim length array"""
        return self.__mean
    
    def get_random_plane(self) -> np.ndarray:
        u = np.random.rand(self.__dim) - 0.5
        v = np.random.rand(self.__dim) - 0.5
        u, v = self.__orthonormalise(u, v)
        return np.vstack([u, v])
    
    def get_max_cutoff(self) -> float:
        return self.__mahal_dists[-2]
    
    def get_max_norm(self) -> float:
        return max(np.linalg.norm(self.__data, axis = 0))

    def partition_data(self, cutoff: float) -> tuple[np.ndarray]:
        ind = np.searchsorted(self.__mahal_dists, cutoff)
        return ind

    def get_outliers(self, cutoff: float) -> np.ndarray:
        """
        Gets data points with a certain Mahalanobis distance from the mean.

        :param cutoff: float representing the minimum Mahalanobis distance that constitutes an 'outlier'
        :return: dim x num_outliers array of points past this Mahalanobis distance.
        """
        ind = np.searchsorted(self.__mahal_dists, cutoff)
        return self.__data[:, ind:]

    def set_data(self, data: str | np.ndarray) -> None:
        """
        Set data for calculator to analyse.

        :param data: filename of a csv file containing data, or dim x num_points array of data points.
        """
        # gather raw data
        if type(data) == str:
            unsorted_data = np.loadtxt(fname = data, dtype=float, delimiter=",", skiprows=0).T
        else:
            unsorted_data = data
        
        # calculate mean (can add/subtract from matrices)
        self.__mean = np.mean(unsorted_data, axis = 1, keepdims=True)
        
        #sorting the data based on increasing mahalanobis distance from mean
        temp_covariance = np.cov(unsorted_data)
        temp_basis = np.linalg.cholesky(temp_covariance)

        t_data = np.linalg.inv(temp_basis) @ (unsorted_data - self.__mean)
        print("we sortin...")
        indexlist = np.argsort(np.linalg.norm(t_data, axis=0))
        sorted_t_data = t_data[:, indexlist]

        self.__mahal_dists = np.linalg.norm(sorted_t_data, axis=0)
        self.__data = (temp_basis @ sorted_t_data) + self.__mean
        
        # set covariance, dimensions based on SORTED data
        self.__dim = self.__data.shape[0]
        self.__covariance = np.cov(self.__data)

        # ellipsoid matrix
        self.__ellipsoid: np.ndarray = np.linalg.inv(self.__covariance)

    def set_ellipse_res(self, ellipse_res: float) -> None:
        """
        Sets resolution of ellipse.

        :param ellipse_res: integer number of points to draw of the projected ellipses.
        """
        # constructing a circle to transform - done ONCE (on initialisation)
        X = np.linspace(0, 2*np.pi, num=ellipse_res)
        Y = np.linspace(0, 2*np.pi, num=ellipse_res)
        
        self.__circle: np.ndarray = np.vstack([np.cos(X), np.sin(Y)])

    @staticmethod
    def __unit(u: np.ndarray) -> np.ndarray:
        """Normalises a vector u."""
        return u/np.linalg.norm(u)
    
    @staticmethod
    def __orthonormalise(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray]:
        """
        Orthonormalises two vectors u and v, shifting them both evenly.

        :params u, v: arrays of equal dimensions
        :return: tuple of two orthonormal vectors spanning the same plane as u and v
        """
        u = Calculator.__unit(u)
        v = Calculator.__unit(v)
        m = Calculator.__unit(u+v)

        u = (m/np.sqrt(2)) + Calculator.__unit(u - np.dot(u, m)*m)/np.sqrt(2)
        v = (m/np.sqrt(2)) + Calculator.__unit(v - np.dot(v, m)*m)/np.sqrt(2)

        return u, v
    
    def get_proj_ellipses(self, P: np.ndarray, m_dists: list[float] = [1, 2, 3]) -> list[np.ndarray]:
        M = P @ self.__covariance @ P.T
        T = np.linalg.cholesky(M)
        proj_mean = P @ self.__mean
        return [(dist * T @ self.__circle) + proj_mean for dist in m_dists]

    
    def plot_on_plane(self, plane: tuple[np.ndarray] | str = "optimised 0", m_dists: list[float] = [1, 2, 3], show_axes: bool = True,\
        points_only: bool = False, opt_step: float = 0.005, opt_tol: float = 0.0001, verbose: bool = False) -> list[np.ndarray] | None:
        """
        Plots data onto a plane.

        :optional param plane: tuple of dim x 1 vectors spanning a plane, else "optimised [cutoff]" or "random"; plane to project onto. Default "optimised 0"
        :optional param m_dists: list of floats; Mahalanobis distances to draw ellipse. Default [1, 2, 3]
        :optional param show_axes: boolean; whether or not to plot axes in grey. Default True.
        :optional param points_only: boolean; whether or not to return an array of ellipse points rather than plotting. Default False.
        :optional param opt_step: small float; sensitivity of steps in optimisation method. Smaller values cost computation time. Default 0.005
        :optional param opt_tol: small float; tolerance of optimisation method. Default 0.0001
        :optional param verbose: whether or not to print progress. Default False.

        :return: list of 2 x ellipse_res arrays of points of an ellipse if points_only is True, else None.
        """
        # set colours
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ["#fecb3e", "#fc8370", "#c2549d", "#7e549e"])

        # set plane
        cutoff = None
        if type(plane) is str and "optimise" in plane:
            cutoff = float(plane.split()[-1])
            u, v = self.optimise_plane(cutoff=cutoff, step=opt_step, tol = opt_tol, verbose=verbose)
            P = np.vstack([u, v])
        elif plane == "random":
            u = np.random.rand(self.__dim) - 0.5
            v = np.random.rand(self.__dim) - 0.5
            u, v = self.__orthonormalise(u, v)
            P = np.vstack([u, v])
        else:
            u, v = self.__orthonormalise(plane[0], plane[1])
            P = np.vstack([u, v])

        # get ellipses
        M = P @ self.__covariance @ P.T
        T = np.linalg.cholesky(M)
        proj_mean = P @ self.__mean
        ellipses = [(dist * T @ self.__circle) + proj_mean for dist in m_dists]

        # return points
        if points_only:
            return ellipses

        # plot axes
        if show_axes:
            for i in range(self.__dim):
                plt.plot([0, P.T[i][0]], [0, P.T[i][1]], c = "grey", linewidth = 1)

        # plot points
        proj_data = P @ self.__data
        if cutoff is not None:
            ind = np.searchsorted(self.__mahal_dists, cutoff)
            plt.scatter(proj_data[0, :ind], proj_data[1, :ind], c = "#808080", marker = ".")
            plt.scatter(proj_data[0, ind:], proj_data[1, ind:], c = "#000000", marker = ".")
        else:
            plt.scatter(proj_data[0], proj_data[1], c = "#000000", marker = ".")

        # plot ellipses
        for i in range(len(ellipses)):
            plt.plot(ellipses[i][0], ellipses[i][1])
        
        plt.show()
        return P[0], P[1]
    
    def plot_comparison(self, optimise_cutoffs: list[float] = [0, 3], across: int = 2, down: int = 2, m_dists:list[float] = [1, 2, 3],\
        show_axes: bool = True, opt_step: float = 0.005, opt_tol: float = 0.0001, verbose: bool = False) -> None:
        """
        Plots an across x down collection of viewpoints, some optimised and some random.

        :optional param optimise_cutoffs: list of floats; cutoffs for different optimised viewpoints
        :optional param across: integer number of plots to put across
        :optional param down: integer number of plots to put down
        :optional param m_dists: list of floats; Mahalanobis distances to draw ellipses. Default [1, 2, 3]
        :optional param show_axes: boolean; whether or not to plot axes in grey. Default True.
        :optional param opt_step: small float; sensitivity of steps in optimisation method. Smaller values cost computation time. Default 0.005
        :optional param opt_tol: small float; tolerance of optimisation method. Default 0.0001
        :optional param verbose: whether or not to print progress. Default False.
        """

        # for testing !
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ["#fecb3e", "#fc8370", "#c2549d", "#7e549e"])
        planes = []

        factor = 1.01
        mod = np.log(factor)

        # get optimised plane
        for i in range(down):
            row = []
            for _ in range(across):
                u = np.random.rand(self.__dim) - 0.5
                v = np.random.rand(self.__dim) - 0.5
                u, v = self.__orthonormalise(u, v)
                print(f"random dist = {self.total_m_dist(u, v, self.__data - self.__mean, mod)}")
                P = np.vstack([u, v])
                row.append(P)
            planes.append(row)
        
        from_plane = None
        for i, cutoff in enumerate(optimise_cutoffs):
            u, v = self.optimise_plane(cutoff=cutoff, from_plane=from_plane, step=opt_step, tol=opt_tol, factor=factor, verbose=verbose)
            P = np.vstack([u, v])
            planes[i//across][i%across] = P
            from_plane = (u, v)
        
        fig, axs = plt.subplots(down, across)

        for j in range(down):
            for i in range(across):
                P = planes[j][i]
                this_cutoff = 0
                if j*across + i < len(optimise_cutoffs):
                    if verbose:
                        print(f"plotting optimised {optimise_cutoffs[j*across + i]}")
                    axs[j, i].set_title(f"optimised {optimise_cutoffs[j*across + i]}")
                    this_cutoff = optimise_cutoffs[j*across + i]
                else:
                    if verbose:
                        print("plotting random")
                    axs[j, i].set_title(f"random")
                #M = np.linalg.inv(P @ self.__ellipsoid @ P.T)#
                M= P @ self.__covariance @ P.T
                T = np.linalg.cholesky(M)
                proj_mean = P @ self.__mean
                ellipses = [(dist * T @ self.__circle) + proj_mean for dist in m_dists]

                # plot axes
                if show_axes:
                    for k in range(self.__dim):
                        axs[j, i].plot([0, P.T[k][0]], [0, P.T[k][1]], c = "#505050", linewidth = 1)

                # plot points
                proj_data = P @ self.__data
                ind = np.searchsorted(self.__mahal_dists, this_cutoff)
                axs[j, i].scatter(proj_data[0][:ind], proj_data[1][:ind], c = "#aaaaaa", marker = ".")
                axs[j, i].scatter(proj_data[0][ind:], proj_data[1][ind:], c = "#000000", marker = ".")

                # plot ellipses
                for k in range(len(ellipses)):
                    axs[j, i].plot(ellipses[k][0], ellipses[k][1])

        plt.show()

    @staticmethod
    def __num(u: np.ndarray, v: np.ndarray, C: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Numerator of function on u and v to maximise"""
        return ((u @ C @ u.T) * ((v @ W)**2)) - (2 * (u @ C @ v.T) * (u @ W) * (v @ W)) + ((v @ C @ v.T) * ((u @ W)**2))

    @staticmethod
    def __den(u: np.ndarray, v: np.ndarray, C:np.ndarray) -> float:
        """Denominator of function on u and v to maximise"""
        return (u @ C @ u.T) * (v @ C @ v.T) - ((u @ C @ v.T)**2)


    def total_m_dist(self, u: np.ndarray, v: np.ndarray, W: np.ndarray, mod: float | None = None) -> float:
        """
        Function to maximise = total squared Mahalanobis distances of each data point from mean
        using projection onto plane spanned by u, v
        """
        if mod is not None:
            return np.sum(np.exp(mod*self.__num(u, v, self.__covariance, W)/self.__den(u, v, self.__covariance)))
        else:
            return np.sum(self.__num(u, v, self.__covariance, W)/self.__den(u, v, self.__covariance))

    @staticmethod
    def __d_num(m: np.ndarray, a: np.ndarray, C: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Partial derivative of numerator with respect to vector m"""
        return 2 * (a @ C @ a.T) * (m @ W) * W - 2*np.outer(C @ a, (m @ W)*(a @ W)) \
        - 2*(m @ C @ a.T) * (a @ W) * W + 2 * np.outer(C @ m, (a @ W)**2)

    @staticmethod
    def __d_den(m: np.ndarray, a: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Partial derivative of denominator with respect to vector m"""
        return 2*(a @ C @ a.T)*(C @ m.T) - 2*(m @ C @ a.T)*(C @ a.T)

    def __d_total_m_dist(self, u: np.ndarray, v: np.ndarray, W: np.ndarray, mod: float | None) -> tuple[np.ndarray]:
        """Gradient of total_m_dist function, normalised"""
        n = self.__num(u, v, self.__covariance, W)
        d = self.__den(u, v, self.__covariance)
        exps = np.exp(mod*n/d) if mod is not None else 1
        mod = mod if mod is not None else 1
 
        du_mat = mod*(d*self.__d_num(u, v, self.__covariance, W) - np.outer(self.__d_den(u, v, self.__covariance), n))*exps/(d**2)
        dv_mat = mod*(d*self.__d_num(v, u, self.__covariance, W) - np.outer(self.__d_den(v, u, self.__covariance), n))*exps/(d**2)

        du, dv = np.sum(du_mat, axis=1), np.sum(dv_mat, axis=1)
        size = np.linalg.norm(np.concatenate([du, dv]))
        return du/size, dv/size

    def __optimise_slice_plane(self, W: np.ndarray) -> tuple[np.ndarray]:
        """
        Gives the plane that maximises squared Mahalanobis distances using SLICE ellipse.
        Starting point for projection optimisation.
        
        :param W: dim x num_points array of points to maximise slice plane through
        :return: tuple of arrays u, v; orthonormal vectors spanning the best fit plane.
        """
        B = np.linalg.cholesky(self.__ellipsoid)
        W -= np.mean(W, axis = 1, keepdims=True)
        T = np.linalg.inv(B) @ W
        u, D, Vh = np.linalg.svd(T.T, full_matrices=False)
        K = Vh[:2]
        P = K @ np.linalg.inv(B)
        return self.__orthonormalise(P[0], P[1])

    def optimise_plane(self, cutoff: float | None = None, points: np.ndarray | None = None, factor: float | None = None, from_plane: np.ndarray | None = None, \
        step: float = 0.01, tol: float = 0.00001, verbose: bool = False) -> tuple[np.ndarray]:
        """
        Numerically searches for the plane that will maximise total_m_dist.

        :optional param cutoff: float representing the minimum Mahalanobis distance that constitutes an 'outlier'
        :optional param factor: the factor by which points with a greater Mahalanobis distance are prioritised
        :optional param step: small float; size of the step taken in n-dimensional space on each iteration
        :optional param tol: small float; tolerance, will accept an answer once it is increasing by less than tol
        :optional param verbose: boolean; whether or not to print progress

        :return: tuple of arrays; orthonormal vectors spanning optimal plane.
        """
        if cutoff is None:
            if points is None:
                W = self.__data - self.__mean
            else:
                W = points - self.__mean
        else:
            W = self.get_outliers(cutoff) - self.__mean
        if from_plane is None:
            u, v = self.__optimise_slice_plane(W)
        else:
            u, v = from_plane[0], from_plane[1]
        mod = None if factor is None else np.log(factor)

        d = self.total_m_dist(u, v, W, mod)
        if verbose:
            print(f"Beginning optimisation with total distance of {d}")

        prev_d = d - (tol + 1)
        
        while d - prev_d > tol:
            if verbose:
                print(f"new total dist: {d}. increment: {d - prev_d}")
            du, dv = self.__d_total_m_dist(u, v, W, mod)
            u, v = self.__orthonormalise(u + step*du, v + step*dv)
            prev_d = d
            d = self.total_m_dist(u, v, W, mod)
        if verbose:
            print(f"Process completed with distance {d}. Projected onto plane: ")
            print(u)
            print(v)
        return u, v
    