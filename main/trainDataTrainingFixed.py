from sklearn.tree           import DecisionTreeRegressor
from sklearn.linear_model   import LinearRegression
from sklearn.neighbors      import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm            import SVR
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler


def train_mlp(data, labels, neurons):
    # Criando o modelo com configurações similares ao MATLAB
    mlp = MLPRegressor(
        hidden_layer_sizes=neurons,  # Configuração de neurônios por camada
        max_iter=1000,              # Limite de iterações
        tol=1e-6,                   # Tolerância ao erro (ajuste fino pode ser necessário)
        solver='lbfgs',             # Otimizador mais próximo do padrão MATLAB
        activation='logistic',       # Função de ativação sigmoidal (padrão no MATLAB)
        random_state=42
    )
    return mlp.fit(data, labels)

def train_svr_linear(data, labels):
    # Normalização para evitar instabilidade numérica (MATLAB faz isso automaticamente)
    np.random.seed(42)
    scaler = StandardScaler()

    # Criando o modelo SVR similar ao MATLAB
    svr = SVR(kernel='linear', C=1.0, epsilon=0.1)

    # Criando pipeline para normalização + SVR
    model = Pipeline([("scaler", scaler), ("svr", svr)])

    return model.fit(data, labels)

def train_svr_rbf(data, labels):
    np.random.seed(42)
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.05, gamma='auto')  # Ajustado para imitar o MATLAB
    return svr.fit(data, labels)


def train_svr_poly1(data, labels):
    np.random.seed(42)
    svr = SVR(kernel='poly', degree=1, C=1.0, epsilon=0.1, gamma='auto', coef0=1)
    return svr.fit(data, labels)

def train_svr_poly3(data, labels):
    np.random.seed(42)
    svr = SVR(kernel='poly', degree=3, C=1.0, epsilon=0.1, gamma='auto', coef0=1)
    return svr.fit(data, labels)


import numpy as np
from sklearn.kernel_ridge import KernelRidge


def train_rbf(data, labels, neurons):
    # Ajustes refinados para melhorar a precisão
    gamma_value = 2.0 / neurons  # Testar variações aqui (1.0 / neurons, 2.0 / neurons)
    alpha_value = 0.1  # Testar variações aqui (1.0, 0.5, 0.1)

    #rbf = KernelRidge(alpha=alpha_value, kernel='rbf', gamma=gamma_value)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(alpha=alpha_value, kernel='rbf', gamma=gamma_value))
    ])
    return model.fit(data, labels)


def train_cart(data, labels):
    tree = DecisionTreeRegressor(
        criterion='squared_error',  # Equivalente ao MSE no MATLAB
        max_depth=None,            # Sem restrição de profundidade (como MATLAB)
        min_samples_split=2,       # Padrão no MATLAB
        min_samples_leaf=1,         # Padrão no MATLAB
        random_state=None
    )
    return tree.fit(data, labels)


def train_linear(data, labels):
    lm = LinearRegression(fit_intercept=True, n_jobs=-1)
    return lm.fit(data, labels)

def train_knn(data, labels, K):
    knn = KNeighborsRegressor(n_neighbors=K)
    return knn.fit(data, labels)

def train_data_training_fixed(data, labels, tipo, K, neurons):
    #print(f"Treinando modelo: {tipo}")
    if tipo == 'cart':
        regressor = train_cart(data, labels)
    elif tipo == 'linear':
        regressor = train_linear(data, labels)
    elif tipo == 'knn3' or tipo == 'knn5':
        regressor = train_knn(data, labels, K)
    elif tipo == 'mlp1' or tipo == 'mlp2':
        regressor = train_mlp(data, labels, neurons)
    elif tipo == 'svr-linear':
        regressor = train_svr_linear(data, labels)
    elif tipo == 'svr-rbf':
        regressor = train_svr_rbf(data, labels)
    elif tipo == 'svr-poly1':
        regressor = train_svr_poly1(data, labels)
    elif tipo == 'svr-poly3':
        regressor = train_svr_poly3(data, labels)
    elif tipo == 'rbf':
        regressor = train_rbf(data, labels, neurons)
    else:
        raise ValueError(f"Unsupported regressor type: {tipo}")

    return regressor
