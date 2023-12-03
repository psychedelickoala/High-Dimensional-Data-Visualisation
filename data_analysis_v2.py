import csv
import numpy as np

class StatsCalculator:
    def __init__(self, data: str | np.ndarray) -> None:
        self.__filename = None
        self.__attribute_names = []
        self.__data = None
        self.__covariance = None
        if type(data) == str:
            self.reset(data)
        else:
            self.__data = data
        self.__covariance = np.cov(self.__data)

        avging_vec = np.ones((self.__data.shape[1], 1))/self.__data.shape[1]
        self.__mean = self.__data @ avging_vec


    def __len__(self) -> int:
        return len(self.__attribute_names)
    
    def get_covariance(self) -> np.ndarray[np.ndarray[float]]:
        return self.__covariance
    
    def get_attribute_names(self) -> list[str]:
        return self.__attribute_names

    def reset(self, filename) -> None:
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

    def get_outliers(self, cutoff: float) -> list[np.ndarray]:
        outliers = []
        for point in self.__data.T:
            m_dist = np.sqrt((point - self.__mean.T) @ np.linalg.inv(self.__covariance) @ (point.T - self.__mean))
            if m_dist > cutoff:
                outliers.append(point)
        return outliers

    