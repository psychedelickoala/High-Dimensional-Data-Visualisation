import numpy as np
from scipy.stats import norm, chi2
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from tqdm import tqdm as tqbar


class Calculator:

    """
    A class to store and process information about a dataset passed on initialisation.
    """

    def __init__(self, data: str | np.ndarray, ellipse_res: int = 60, cov = None, cov_mean = None, sort = None) -> None:
        """
        Initialises calculator.
        Reads and sorts data and constructs circle. These can be reset later.
        """

        self.set_data(data, cov, cov_mean, sort)
        self.set_ellipse_res(ellipse_res)


    def __len__(self) -> int:
        """Returns the number of data points"""
        return len(self.__sds)
    

    def get_dim(self) -> int:
        """Returns the number of dimensions"""
        return self.__dim


    def get_sort(self) -> np.ndarray:
        """Returns an array specifying how the data was sorted."""
        return self.__sort


    def get_covariance(self) -> np.ndarray:
        """Returns covariance matrix, dim x dim array"""
        return self.__covariance
    

    def get_data(self) -> np.ndarray:
        """Returns sorted data, dim x num_points array"""
        return self.__data
    
    
    def get_mean(self) -> np.ndarray:
        """Returns raw mean of data, dim length array"""
        return self.__mean
    

    def get_attrs(self) -> np.ndarray[str]:
        """Returns names of attributes, either default or read from csv"""
        return self.__attribute_names


    def get_random_plane(self) -> np.ndarray:
        """Returns the projection matrix of a random plane"""
        u = np.random.rand(self.__dim) - 0.5
        v = np.random.rand(self.__dim) - 0.5
        u, v = self.orthonormalise(u, v)
        return np.vstack([u, v])
    

    def get_max_sds(self) -> float:
        """Returns the maximum standard deviations of the data, for use in constructing slider and preplots."""
        return self.__sds[-2]
    

    def get_max_norm(self) -> float:
        """Returns max norm of the data, for use in bounding plots."""
        return max(np.linalg.norm(self.__data, axis = 0))


    def partition_data(self, sd: float) -> tuple[np.ndarray]:
        """Returns the index at which data is a certain number of sds from the mean"""
        ind = np.searchsorted(self.__sds, sd)
        return ind


    def set_data(self, data: str | np.ndarray, cov, cov_mean, sort) -> None:
        """
        Set data, covariance matrix and ellipse centroid for calculator to analyse.
        """
        # gather raw data
        if type(data) == str:
            try:
                unsorted_data = np.loadtxt(fname = data, dtype=float, delimiter=",", skiprows=0).T
            except ValueError:
                unsorted_data = np.loadtxt(fname = data, dtype=float, delimiter=",", skiprows=1).T
                self.__attribute_names = np.loadtxt(fname = data, dtype=str, delimiter=",", skiprows=0, max_rows=1)
                self.csv_dist = 2
            else:
                self.__attribute_names = np.array(["e" + str(i) for i in range(unsorted_data.shape[0])])
                self.csv_dist = 1
        else:
            unsorted_data = data

        if type(cov) is str:
            cov = np.loadtxt(fname = cov, dtype=float, delimiter=",", skiprows=0)
        if type(cov_mean) is str:
            cov_mean = np.loadtxt(fname = cov_mean, dtype=float, delimiter=",", skiprows=0)
        
        # calculate mean (can add/subtract from matrices)
        self.__mean = np.mean(unsorted_data, axis = 1, keepdims=True)
        self.__cov_mean = cov_mean[:, np.newaxis] if cov_mean is not None else self.__mean
        
        #sorting the data based on increasing mahalanobis distance from mean
        self.__covariance = np.cov(unsorted_data) if cov is None else cov
        temp_basis = np.linalg.cholesky(self.__covariance)

        t_data = np.linalg.inv(temp_basis) @ (unsorted_data - self.__cov_mean)

        self.__sort = np.argsort(np.linalg.norm(t_data, axis=0)) if sort is None else sort
        sorted_t_data = t_data[:, self.__sort]
        self.__data = unsorted_data[:, self.__sort]
        
        # set dimensions and sds array
        self.__dim = self.__data.shape[0]
        m_dists = np.linalg.norm(sorted_t_data, axis=0)
        self.__sds = norm.isf((chi2.sf(m_dists**2, self.__dim))/2)

        # ellipsoid matrix
        self.__ellipsoid: np.ndarray = np.linalg.inv(self.__covariance)


    def get_clusters(self, inds: np.ndarray | int) -> np.ndarray:
        """Sort data at given indices into up to 9 clusters."""

        if type(inds) is not np.ndarray:
            inds = np.array(range(inds, len(self)))

        # Uncomment below to use elliptical, rather than euclidean, distance
        #B = np.linalg.cholesky(self.__ellipsoid).T
        
        points = self.__data[:, inds]
        num_points = points.shape[1]

        dist_mat = np.zeros((num_points, num_points))
        for i in tqbar(range(num_points), desc = "Building distances matrix..."):
            for j in range(i, num_points):
                dist_mat[i, j] = dist_mat[j, i] = np.linalg.norm((points[:, i] - points[:, j]))

        print("Finding minimum spanning tree...")
        X = minimum_spanning_tree(csr_array(dist_mat))

        # Find cutoff to make a new cluster
        std_dev = np.std(X.data)
        mean = np.mean(X.data)
        upper_bound = mean + 3*std_dev

        # disconnect graph at outlying distances
        X.data = np.array([edge if edge < upper_bound else 0 for edge in X.data])
        X.eliminate_zeros()

        # get 9 biggest clusters
        num_clusters, labels = connected_components(X, directed=False)
        num_clusters = 9 if num_clusters > 9 else num_clusters

        new_labels = np.array([0]*num_points)
        counts = np.unique(labels, return_counts=True)[1]
        top = np.argsort(counts)[-num_clusters:] # 9 biggest clusters

        for i in range(num_clusters):
            new_labels[np.where(labels == top[i])[0]] = num_clusters-i

        return num_clusters, new_labels


    def get_cov_mean(self) -> np.ndarray:
        """Return centroid of ellipses."""
        return self.__cov_mean


    def set_ellipse_res(self, ellipse_res: float) -> None:
        """
        Sets resolution of ellipse.
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
    def orthonormalise(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray]:
        """
        Orthonormalises two vectors u and v, shifting them both evenly.
        """
        u = Calculator.__unit(u)
        v = Calculator.__unit(v)
        m = Calculator.__unit(u+v)
        p = Calculator.__unit(u-v)

        u = m/np.sqrt(2) + p/np.sqrt(2)
        v = m/np.sqrt(2) - p/np.sqrt(2)

        return u, v
    

    @staticmethod
    def move_axis(old_proj: np.ndarray, axis: int, new_pos: tuple[float], precision = 0) -> np.ndarray:
        """Return new projection matrix where axis projects onto new_pos"""
        # set up
        u, v = old_proj[0], old_proj[1]
        a = np.delete(u, axis)
        b = np.delete(v, axis)
        A = new_pos[0] if np.linalg.norm(new_pos) < (1-precision) else (1 - precision)*(new_pos[0])/np.linalg.norm(new_pos)
        B = new_pos[1] if np.linalg.norm(new_pos) < (1-precision) else (1 - precision)*(new_pos[1])/np.linalg.norm(new_pos)

        au = Calculator.__unit(a)
        bu = Calculator.__unit(b)
        m = au + bu
        p = au - bu

        try:
            m = Calculator.__unit(m)
            p = Calculator.__unit(p)
        
        except RuntimeWarning: # all other vectors in a line
            u[axis] = new_pos[0]
            v[axis] = new_pos[1]
            u, v = Calculator.orthonormalise(u, v)
        
        else:
            # Find cos, sin of angle between a and b
            c2 = -A*B/np.sqrt((1 - A**2)*(1 - B**2))
            c, s = np.sqrt((1 + c2)/2), np.sqrt((1 - c2)/2)
            
            # Resize a and b
            au, bu = c*m + s*p, c*m - s*p
            a, b = au*np.sqrt(1-A**2), bu*np.sqrt(1-B**2)

            u = np.insert(a, axis, A)
            v = np.insert(b, axis, B)

        return np.vstack([u, v])


    def get_proj_ellipses(self, P: np.ndarray, m_dists: list[float] = [1, 2, 3]) -> list[np.ndarray]:
        """Return a list of sequences of points, for each ellipse on the plane with projection matrix P."""
        M = P @ self.__covariance @ P.T
        T = np.linalg.cholesky(M)
        return [(dist * T @ self.__circle) for dist in m_dists]


    # ~~~ OPTIMISER ~~~
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
        return self.orthonormalise(P[0], P[1])


    def optimise_plane(self, ind = None, factor: float | None = None, from_plane: np.ndarray | None = None, \
                       step: float = 0.01, tol: float = 0.00001, verbose: bool = False) -> np.ndarray:
        """
        Numerically searches for the plane that will maximise total_m_dist.
        """
        
        # set W
        if ind is None:
            W = self.__data - self.__cov_mean
        elif type(ind) is np.ndarray:
            W = self.__data[:, ind] - self.__cov_mean
        else:
            W = self.__data[:, ind:] - self.__cov_mean

        # set initial u, v
        if from_plane is None:
            u, v = self.__optimise_slice_plane(W)
        else:
            u, v = from_plane[0], from_plane[1]
        
        # set factor (not used)
        mod = None if factor is None else np.log(factor)

        # set d, prev_d
        d = self.total_m_dist(u, v, W, mod)
        if verbose:
            print(f"Beginning optimisation with total distance of {d}")
        prev_d = d - (tol + 1)
        
        # gradient descent
        while d - prev_d > tol:
            if verbose:
                print(f"new total dist: {d}. increment: {d - prev_d}")
            du, dv = self.__d_total_m_dist(u, v, W, mod)
            u, v = self.orthonormalise(u + step*du, v + step*dv)
            prev_d = d
            d = self.total_m_dist(u, v, W, mod)

        # print results
        if verbose:
            print(f"Process completed with distance {d}. Projected onto plane: ")
            print(u)
            print(v)

        return np.vstack([u, v])
    
