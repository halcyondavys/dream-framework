import json
import os
from main.trainDataTraining  import get_model_instance
from main.configs            import param_grid_models
from main.elm                import ELMRegressor
from sklearn.tree            import DecisionTreeRegressor
from sklearn.linear_model    import LinearRegression, Ridge, HuberRegressor
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.svm             import SVR
from sklearn.kernel_ridge    import KernelRidge
from sklearn.model_selection import RandomizedSearchCV

BEST_PARAMS_DIR = "../results/best_params/"
os.makedirs(BEST_PARAMS_DIR, exist_ok=True)

def get_model_instance(model_name):
    if model_name == 'linear':
        return LinearRegression()
    elif model_name == 'ridge':
        return Ridge()
    elif model_name == 'huber':
        return HuberRegressor()
    elif model_name == 'cart':
        return DecisionTreeRegressor(random_state=42)
    elif model_name == 'knn':
        return KNeighborsRegressor(n_jobs=1)
    elif model_name == 'mlp':
        return MLPRegressor(random_state=42)
    elif model_name == 'svr':
        return SVR()
    elif model_name == 'rbf':
        return KernelRidge()
    elif model_name == 'elm':
        return ELMRegressor(
            regressor=Ridge(),
            random_state=42
        )
    else:
        raise ValueError(f"Modelo não suportado: {model_name}")

def get_model_name(m):
    if isinstance(m, tuple):
        return m[0]
    return m

def _path(dataset_name, model_name):
    ds_dir = os.path.join(BEST_PARAMS_DIR, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)
    return os.path.join(ds_dir, f"{model_name}.json")


def load_best_params(dataset_name, model_name):
    path = _path(dataset_name, model_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def save_best_params(dataset_name, model_name, params):
    path = _path(dataset_name, model_name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(params, f, indent=4)
    os.replace(tmp, path)

def search_best_params(model_name, X, y):
    model = get_model_instance(model_name)
    param_dist = param_grid_models().get(model_name, None)

    # Modelos sem hiperparâmetros
    if param_dist is None or len(param_dist) == 0:
        return {}

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=4,
        random_state=42,
        verbose=0
    )

    search.fit(X, y)
    return search.best_params_

def get_or_create_best_params(dataset_name, model_name, X, y):
    params = load_best_params(dataset_name, model_name)
    if params is not None:
        return params
    params = search_best_params(model_name, X, y)
    save_best_params(dataset_name, model_name, params)
    return params