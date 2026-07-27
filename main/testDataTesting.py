

def test_svr(data, regressor):
    results = regressor.predict(data)
    return results


def test_cart(data, regressor):
    results = regressor.predict(data)
    return results


def test_linear(data, regressor):
    results = regressor.predict(data)
    return results


def test_knn(data, regressor):
    results = regressor.predict(data)
    return results


def test_mlp(data, regressor):
    results = regressor.predict(data)
    return results


def test_rbf(data, regressor):
    results = regressor.predict(data)
    return results


def test_data_testing(data, regressor, tipo):
    if tipo == 'cart':
        results = test_cart(data, regressor)
    elif tipo == 'linear':
        results = test_linear(data, regressor)
    elif tipo == 'knn':
        results = test_knn(data, regressor)
    elif tipo == 'mlp':
        results = test_mlp(data, regressor)
    elif tipo == 'svr-linear' or tipo == 'svr-rbf' or tipo == 'svr-poly1' or tipo == 'svr-poly3':
        results = test_svr(data, regressor)
    elif tipo == 'rbf':
        results = test_rbf(data, regressor)
    else:
        raise ValueError(f"Unsupported regressor type: {tipo}")

    return results
