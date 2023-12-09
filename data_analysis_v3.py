import csv
import numpy as np

class StatsCalculator:
    def __init__(self, data: str | np.ndarray) -> None:
        print("initialising")
        self.__filename = None
        self.__attribute_names = []
        self.__mahal_dists = None

        if type(data) == str:
            self.read_from_file(data)
        else:
            self.__data = data
        self.__dim = self.__data.shape[0]
        self.sort_data()
        
        self.__covariance = np.cov(self.__data)
        self.__covariance_inv = np.linalg.inv(self.__covariance)
        self.__from_base = np.linalg.cholesky(self.__covariance)  # B @ B.T = covariance
        self.__to_base = np.linalg.inv(self.__from_base)

        self.__mean = np.mean(data, axis = 1)


    def __len__(self) -> int:
        return len(self.__attribute_names)
    
    def get_covariance(self) -> np.ndarray[np.ndarray[float]]:
        return self.__covariance
    
    def get_data(self) -> np.ndarray:
        return self.__data
    
    def get_mean(self) -> np.ndarray:
        return self.__mean
    
    def get_attribute_names(self) -> list[str]:
        return self.__attribute_names
    
    def m_dist(self, point: np.ndarray) -> float:
        return np.sqrt((point - self.__mean.T) @ self.__covariance_inv @ (point.T - self.__mean))
    
    def sort_data(self) -> None:
        covariance = np.cov(self.__data)
        basis = np.linalg.cholesky(covariance)
        mean = np.mean(self.__data, axis = 1, keepdims=True)

        t_data = np.linalg.inv(basis) @ (self.__data - mean)
        indexlist = np.argsort(np.linalg.norm(t_data, axis=0))
        sorted_t_data = t_data[:, indexlist]

        self.__mahal_dists = np.linalg.norm(sorted_t_data, axis=0)
        self.__data = (basis @ sorted_t_data) + mean
    
    def get_outlier_plane(self, cutoff: float) -> tuple[np.ndarray]:
        ind = np.searchsorted(self.__mahal_dists, cutoff)
        print(f"ind: {ind}")
        t_outliers = (self.__to_base @ self.__data).T[ind:]
        if t_outliers.shape[0] == 0:
            raise ValueError("No data points outside confidence interval")
        elif t_outliers.shape[0] == 1:
            u = (self.__from_base @ t_outliers[0]) - self.__mean
            e1 = np.zeros(self.__dim); e1[0] = 1.0
            e2 = np.zeros(self.__dim); e2[1] = 1.0
            v = u + (e2 if np.allclose(u/np.linalg.norm(u), e1) else e1)
            return u, v
        t_outliers -= np.mean(t_outliers, axis = 0, keepdims=True)
        t_proj = np.linalg.svd(t_outliers, full_matrices=True).Vh
        return self.__from_base @ t_proj[0], self.__from_base @ t_proj[1]


    def read_from_file(self, filename) -> None:
        self.__filename = filename
        with open(self.__filename) as data_file:
            csv_reader = csv.reader(data_file, delimiter=',')
            line_number = 0
            self.__data = np.array(csv_reader)
            for row in csv_reader:
                if line_number == 0:
                    self.__attribute_names = list(row)
                    line_number = 1
                elif line_number == 1:
                    self.__data = np.array(row).astype(float)
                    line_number = 2
                else:
                    self.__data = np.vstack([self.__data, np.array(row).astype(float)])

        self.__data = self.__data.T



    