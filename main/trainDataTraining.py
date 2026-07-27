from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

from main.elm import ELMRegressor
from main.configs import param_grid_models
from sklearn.model_selection import RandomizedSearchCV

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
        return ELMRegressor()
    else:
        raise ValueError(f"Modelo não suportado: {model_name}")

def train_data_training(data, labels, model_name):
    #print(f"Treinando modelo: {model_name}")

    # Obtemos a instância do modelo base
    model = get_model_instance(model_name)

    # Obtemos o grid de hiperparâmetros
    param_grid = param_grid_models().get(model_name, None)

    if param_grid:

        # Parâmetros do RandomSearch
        n_iter = 20  # Pode ajustar dinamicamente por modelo

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=4,  # melhor balanceamento para seu notebook
            random_state=None,  # garante reprodutibilidade
            verbose=0
        )

        search.fit(data, labels)
        return search.best_estimator_
    else:
        # Caso o modelo não tenha hiperparâmetros (ex: LinearRegression)
        model.fit(data, labels)
        return model
