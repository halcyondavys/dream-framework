from pathlib import Path
import numpy as np
import joblib
from EnsembleGeneration import ensemble_generation
from calculateDiversity import calculate_all_diversity_metrics
from normalizeMatrix import normalize_by_fold

def process_fold(
    dataset_name,
    execution,
    fold,
    data,
    labels,
    train_index,
    valid_index,
    test_index,
    numberRegressors,
    model_save_root,
    method='minmax'
):
    """
    Executa um fold completo da Fase 1 do DREAM: normalização, treino, avaliação e salvamento dos modelos.

    Retorna:
    - errors_local: matriz (n_amostras_validação, n_modelos)
    - div_corr: vetor (n_modelos,)
    - div_var: vetor (n_modelos,)
    - div_df: vetor (n_modelos,)
    """

    # 🔹 Máscaras do fold
    train = train_index[:, fold].astype(bool)
    valid = valid_index[:, fold].astype(bool)
    test  = test_index[:, fold].astype(bool)

    # 🔹 Dados brutos por partição
    X_train_raw = data[train]
    y_train_raw = labels[train]
    X_valid_raw = data[valid]
    y_valid_raw = labels[valid]
    X_test_raw  = data[test]
    y_test_raw  = labels[test]

    # 🔹 Normalização por fold
    data_train, data_valid, data_test = normalize_all_by_fold(
        X_train_raw, y_train_raw, X_valid_raw, y_valid_raw, X_test_raw, y_test_raw, method
    )

    # 🔹 Geração dos modelos
    model_pool = ensemble_generation(numberRegressors, data_train, 0)

    # 🔹 Salvamento dos modelos treinados
    fold_model_dir = Path(f"{model_save_root}/{dataset_name}/exec_{execution}/fold_{fold}")
    fold_model_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    for model_idx, model in enumerate(model_pool):
        pred = model.predict(data_valid[:, 1:])
        predictions.append(pred)

        model_path = fold_model_dir / f"model_{model_idx}.pkl"
        joblib.dump(model, model_path)

    predictions = np.column_stack(predictions)

    # 🔹 Erro local (MSE por instância e por modelo)
    errors_local = (predictions - data_valid[:, 0].reshape(-1, 1)) ** 2

    # 🔹 Métricas de diversidade
    div_corr, div_var, div_df = calculate_all_diversity_metrics(predictions, data_valid[:, 0])

    return errors_local, div_corr, div_var, div_df
