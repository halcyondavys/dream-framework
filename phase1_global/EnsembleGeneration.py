from main.trainDataTrainingFixed import train_data_training_fixed
from main.best_params            import get_model_instance

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

def ensemble_generation(data_train, modelsUsed, GridSearch, best_params_dict=None):
    """
    Cria o ensemble.
    - Se GridSearch=True → usa RandomSearch e best_params_dict
    - Se GridSearch=False → usa modelos fixos (models_fixed_used)
    """
    ensemble_set = []
    y = data_train[:, 0]
    X = data_train[:, 1:]

    if GridSearch:
        # MODO TUNING GLOBAL
        for model_name in modelsUsed:
            model = get_model_instance(model_name)

            # hiperparâmetros ajustados globalmente
            params = best_params_dict.get(model_name, {})
            if params:
                model.set_params(**params)

            model.fit(X, y)
            ensemble_set.append(model)

    else:
        # MODO FIXO
        for (model_name, p1, p2) in modelsUsed:
            model = train_data_training_fixed(X, y, model_name, p1, p2)
            ensemble_set.append(model)

    return ensemble_set



