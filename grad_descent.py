import numpy as np
import csv
import matplotlib.pyplot as plt

class Calculator:
    def __init__(self, data: str | np.ndarray, ellipse_res: int = 50) -> None:

        # gather raw data
        if type(data) == str:
            unsorted_data = self.read_from_file(data)
        else:
            unsorted_data = data
        
        # calculate mean (column vector)
        self.__mean = np.mean(unsorted_data, axis = 1, keepdims=True)
        
        #sorting the data based on increasing mahalanobis distance from mean
        temp_covariance = np.cov(unsorted_data)
        temp_basis = np.linalg.cholesky(temp_covariance)

        t_data = np.linalg.inv(temp_basis) @ (unsorted_data - self.__mean)
        indexlist = np.argsort(np.linalg.norm(t_data, axis=0))
        sorted_t_data = t_data[:, indexlist]

        self.__mahal_dists = np.linalg.norm(sorted_t_data, axis=0)
        self.__data = (temp_basis @ sorted_t_data) + self.__mean
        
        # set covariance, dimensions based on SORTED data
        self.__dim = self.__data.shape[0]
        self.__covariance = np.cov(self.__data)

        # ellipsoid matrix
        self.__ellipsoid: np.ndarray = np.linalg.inv(self.__covariance)
        
        # constructing a circle to transform - done ONCE (on initialisation)
        X = np.linspace(0, 2*np.pi, num=ellipse_res)
        Y = np.linspace(0, 2*np.pi, num=ellipse_res)
        
        self.__circle: np.ndarray = np.vstack([np.cos(X), np.sin(Y)])

    def __len__(self) -> int:
        return self.__dim
    
    def get_covariance(self) -> np.ndarray[np.ndarray[float]]:
        return self.__covariance
    
    def get_data(self) -> np.ndarray:
        return self.__data
    
    def get_mean(self) -> np.ndarray:
        return self.__mean
    
    def m_dist(self, point: np.ndarray) -> float:
        return np.sqrt((point - self.__mean.T) @ np.linalg.inv(self.__covariance) @ (point.T - self.__mean))
    
    def get_outliers(self, cutoff: float) -> np.ndarray:
        ind = np.searchsorted(self.__mahal_dists, cutoff)
        return self.__data[:, ind:]

    def read_from_file(self, filename) -> np.ndarray:
        with open(filename) as data_file:
            csv_reader = csv.reader(data_file, delimiter=',')
            line_number = 0
            for row in csv_reader:
                if line_number == 1:
                    data = np.array(row).astype(float)
                else:
                    data = np.vstack([data, np.array(row).astype(float)])
                line_number += 1
        return data.T
    
    def __unit(self, u):
        return u/np.linalg.norm(u)
    
    def __orthonormalise(self, u, v):
        u = self.__unit(u)
        v = self.__unit(v)
        m = self.__unit(u+v)

        u = (m/np.sqrt(2)) + self.__unit(u - np.dot(u, m)*m)/np.sqrt(2)
        v = (m/np.sqrt(2)) + self.__unit(v - np.dot(v, m)*m)/np.sqrt(2)

        return u, v
    
    def plot_on_plane(self, plane: tuple[np.ndarray] | str = "optimised 0", m_dists = [1, 2, 3], show_axes = True, points_only = False, verbose = False):
        # set colours
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ["#fecb3e", "#fc8370", "#c2549d", "#7e549e"])

        # set plane
        if type(plane) is str and "optimised" in plane:
            cutoff = float(plane.split()[-1])
            u, v = self.optimise_plane(cutoff=cutoff, verbose=verbose)
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
        M = P @ np.linalg.inv(self.__ellipsoid) @ P.T
        J, K, L = M[0, 0], M[0, 1] + M[1, 0], M[1, 1]
        a = (J + L + np.sqrt((J-L)**2 + K**2))/2
        b = J+L-a
        theta = np.arcsin(K/(a-b))/2
        T = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) \
            @ np.array([[np.sqrt(a), 0],[0, np.sqrt(b)]])
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
        plt.scatter(proj_data[0], proj_data[1], c = "#000000", marker = ".")

        # plot ellipses
        for i in range(len(ellipses)):
            plt.plot(ellipses[i][0], ellipses[i][1])

        plt.show()

    def __num(self, u, v, W):
        C = self.__covariance
        return ((u @ C @ u.T) * ((v @ W)**2)) - (2 * (u @ C @ v.T) * (u @ W) * (v @ W)) + ((v @ C @ v.T) * ((u @ W)**2))

    def __den(self, u, v):
        C = self.__covariance
        return (u @ C @ u.T) * (v @ C @ v.T) - ((u @ C @ v.T)**2)

    def __total_m_dist(self, u, v, W):
        C = self.__covariance
        return np.sum(self.__num(u, v, W))/self.__den(u, v)

    def __d_num(self, m, a, W):
        C = self.__covariance
        return 2 * (a @ C @ a.T) * (m @ W) * W - 2*np.outer(C @ a, (m @ W)*(a @ W)) \
        - 2*(m @ C @ a.T) * (a @ W) * W + 2 * np.outer(C @ m, (a @ W)**2)

    def __d_den(self, m, a):
        C = self.__covariance
        return 2*(a @ C @ a.T)*(C @ m.T) - 2*(m @ C @ a.T)*(C @ a.T)

    def __d_total_m_dist(self, u, v, W):
        C = self.__covariance
        n = np.sum(self.__num(u, v, W))
        d = self.__den(u, v)

        du = (d*np.sum(self.__d_num(u, v, W), axis=1) - n*self.__d_den(u, v))/(d**2)
        dv = (d*np.sum(self.__d_num(v, u, W), axis=1) - n*self.__d_den(v, u))/(d**2)

        size = np.linalg.norm(np.concatenate([du, dv]))
        return du/size, dv/size

    def optimise_plane(self, cutoff = 0, step = 0.001, verbose = False):
        W = self.get_outliers(cutoff)
        u, v = np.identity(self.__dim)[0], np.identity(self.__dim)[1]

        d = self.__total_m_dist(u, v, W)
        prev_d = d - 1

        while d - prev_d > 0:
            if verbose:
                print(f"new total dist: {d}. increment: {d - prev_d}")
            du, dv = self.__d_total_m_dist(u, v, W)
            u, v = self.__orthonormalise(u + step*du, v + step*dv)
            prev_d = d
            d = self.__total_m_dist(u, v, W)

        return u, v



#C = np.array([
#    [1, 0.2, 0.1],
#    [0.2, 1, 0.3],
#    [0.1, 0.3, 1]
#])
u = np.array([0, 1, 0])
v = np.array([0, 0, 1])
P = np.vstack([u, v])
W = np.array([
    [3, 2, .1],
    [4, -7, .3],
    [1, 0, -.2],
    [8, 2, -.1],
    [-6, -1, -.2]
]).T  # each COLUMN is a data point

calc = Calculator(W)
calc.plot_on_plane(plane= "optimised 1.5", verbose=True)



