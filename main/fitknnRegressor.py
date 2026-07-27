from sklearn.neighbors import KNeighborsRegressor

def fit_knn_regressor(data, labels, K):
    regressor = KNeighborsRegressor(n_neighbors=K)
    regressor.fit(data, labels)
    return regressor
