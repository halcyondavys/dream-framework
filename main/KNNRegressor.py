import numpy as np

class KNNRegressor:
    def __init__(self, n, l, d):
        self.neighborhood = n
        self.labels = l
        self.data = d

    def calculate(self, points):
        rows, _ = self.data.shape
        rows_points, _ = points.shape

        result = np.zeros(rows_points)

        if rows < self.neighborhood:
            raise ValueError('You specified more neighbors than the number of existing points.')

        for j in range(rows_points):
            point = points[j, :]

            dist = np.sqrt(np.sum((self.data - point) ** 2, axis=1))

            index = np.argsort(dist)

            values = self.labels[index[:self.neighborhood]]

            result[j] = np.mean(values)

        return result
