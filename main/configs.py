import numpy as np

# pastas de referências
def path_DREAM():
    return {
        'results_path_phase1': '../results/phase1/',
        'results_path_phase2': '../results/phase2/',
        'results_path_phase3': '../results/phase3/',
        'output_csv_path_phase1': '../results/phase1/csv/',
        'output_csv_path_phase2': '../results/phase2/csv/',
        'model_save_root': '../results/models_trained/',
        'best_params_root': '../results/best_params/',
    }


# Lista de conjuntos de dados
def datasets_used():
    datasetsused = [
        'abalone',
        'airfoil_self_noise',
        'albrecht',
        'bank32nh',
        'bank8FM',
        'wiscoinBreastCancer',
        'ccpp',
        'china',
        'cocomonasa60',
        'cocomo81',
        'concrete',
        'cpu_act',
        'cpu_small',
        'delta_ailerons',
        'delta_elevators',
        'desharnais',
        'energy_efficiency',
        'friedman',
        'housing',
        'kin8nm',
        'machine',
        'maxwell',
        'nasa93',
        'parkinsons_updrs',
        'puma32H',
        'puma8NH',
        'stocks',
        'triazines',
        'wineq-red',
        'wineq-white'
    ]
    return datasetsused

# Descomente e altere o nome do conjunto de dados se quiser testar apenas um
def datasets_used_reduzido():
    datasetsused = [
        'desharnais',
        'cocomonasa60',
        'machine'
    ]
    return datasetsused

def datasets_synthetic_8Models():
    """
    Datasets sintéticos para validação controlada do framework DREAM.
    Cada cenário isola uma propriedade estrutural específica.
    Arquivos gerados por data/generate_synthetic.py em data/synthetic/.
    """
    return [
        'SYN-01-NonLinear-Smooth',
        'SYN-02-Piecewise-2R',
        'SYN-03-Piecewise-3R',
        'SYN-04-Soft-Gating',
        'SYN-05-Heteroscedastic-Local',
        'SYN-06-Local-Outliers',
        'SYN-07-HighDim-Irrelevant',
        'SYN-08-Cluster-Specialists',
    ]

def datasets_synthetic():
    """
    Datasets sintéticos para validação controlada do framework DREAM.
    Cada cenário isola uma propriedade estrutural específica.
    Arquivos gerados por data/generate_synthetic.py em data/synthetic/.
    """
    return [
        'C01-continuo-baixo-baixo',
        'C02-continuo-baixo-f3a',
        'C03-continuo-baixo-f3b',
        'C04-continuo-alto-baixo',
        'C05-continuo-alto-f3a',
        'C06-continuo-alto-f3b',
        'C07-discreto-baixo-baixo',
        'C08-discreto-baixo-f3a',
        'C09-discreto-baixo-f3b',
        'C10-discreto-alto-baixo',
        'C11-discreto-alto-f3a',
        'C12-discreto-alto-f3b'
    ]

# Lista dos Modelos
def models_used():
    modelsused = [
        'cart',
        'linear',
        'mlp',
        'svr',
        'knn',
        'rbf',
        'elm',
        'ridge',
        'huber'
    ]
    return modelsused

def models_fixed_used():
    modelsused = [
        ('cart', 0, 0),
        ('linear', 0, 0),
        ('mlp1', 0, 10),
        ('mlp2', 0, [5, 10]),
        ('svr-rbf', 0, 0),
        ('svr-poly1', 0, 0),
        ('svr-poly3', 0, 0),
        ('rbf', 0, 10),
        ('knn3', 3, 0),
        ('knn5', 5, 0)
    ]
    return modelsused

# Número de execuções
def num_execucoes():
    Numexecutions = 20
    return Numexecutions

# Número de folds
def num_folds():
    folds = 10
    return folds

def param_grid_models():
    return {
        'mlp': {
            # Arquiteturas leves e eficientes
            'hidden_layer_sizes': [
                (10,), (20,), (50,),
                (10,10), (20,10), (50,20)
            ],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': np.logspace(-5, -3, 10),
            'learning_rate_init': np.logspace(-4, -2, 10),
            'max_iter': [300],
            'early_stopping': [True],
            'validation_fraction': [0.15]
        },
        'svr': {
            'kernel': ['rbf', 'linear'],
            'C': np.logspace(-2, 2, 20),
            'gamma': np.logspace(-3, 1, 20),
            'epsilon': np.linspace(0.001, 0.1, 20)
        },
        'rbf': {
            'alpha': [0.1, 1.0],
            'gamma': [0.01, 0.1],
            'kernel': ['rbf']
        },
        'knn': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        'cart': {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'linear': {},
        'elm': {
            'n_hidden': [10, 20, 50, 100],              # mais capacidade
            'activation_func': ['tanh', 'sigmoid', 'gaussian'],
            'alpha': [0.0, 0.3, 0.5, 0.7, 1.0],          # cobre MLP puro (alpha=1) e RBF puro (alpha=0)
            'rbf_width': [0.1, 0.3, 0.5, 1.0],
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0],  # regularização Ridge de saída
            'random_state': [42]                         # fixar para reprodutibilidade
        },
        'ridge': {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        },
        'huber': {
            'epsilon': [1.35, 1.5, 1.75],
            'alpha': [0.0001, 0.001],
            'max_iter': [1000, 2000]
        }
    }
