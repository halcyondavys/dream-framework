# arquivo: runPhase2_Competence.py
"""
DREAM Phase 2
A matriz de erro local é baseada no RMSE ponderado por distância no espaço KNN.
Esta adaptação foi incorporada ao framework DREAM como mecanismo de estimativa de competência local dos regressores.
"""
import json
import numpy as np
from pathlib                 import Path
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing   import StandardScaler
from sklearn.base            import clone

from main.configs      import num_folds
from competence_scores import calculate_dream_rmse_matrix
from main.best_params  import get_model_instance, load_best_params

def resolve_k(n_samples_ref, n_features=None, strategy="fixed", k_params=None):
    """
    Resolve o valor de K conforme a estratégia escolhida.

    Estratégias:
    - "fixed"                -> usa k_fixed
    - "adaptive_heuristic"   -> sqrt(N) se N<=threshold, ln(N) se N>threshold
    - "adaptive_theoretical" -> N^(2/(d+2))

    Parâmetros:
    - n_samples_ref : int
        Número de instâncias do conjunto de referência.
    - n_features : int or None
        Número de atributos (dimensionalidade).
        Obrigatório para a estratégia adaptive_theoretical.
    - strategy : str
        Estratégia de escolha do K.
    - k_params : dict
        Dicionário com parâmetros auxiliares.

    Retorna:
    - k_value : int
        Valor final de K.
    - k_rule : str
        Regra usada para gerar K.
    """
    if k_params is None:
        k_params = {
            "k_fixed": 10,
            "k_min": 3,
            "k_max": 20,
            "n_threshold": 1000
        }

    k_fixed     = int(k_params.get("k_fixed", 10))
    k_min       = int(k_params.get("k_min", 3))
    k_max       = int(k_params.get("k_max", 20))
    n_threshold = int(k_params.get("n_threshold", 1000))

    if strategy == "fixed":
        k_value = k_fixed
        k_rule = f"fixed({k_fixed})"

    elif strategy == "adaptive_heuristic":
        if n_samples_ref <= n_threshold:
            k_value = int(np.round(np.sqrt(n_samples_ref)))
            k_rule = "sqrt(N)"
        else:
            k_value = int(np.round(np.log(n_samples_ref)))
            k_rule = "ln(N)"
    elif strategy == "adaptive_theoretical":
        if n_features is None:
            raise ValueError("n_features deve ser informado para strategy='adaptive_theoretical'.")

        exponent = 2.0 / (n_features + 2.0)
        k_value = int(np.round(n_samples_ref ** exponent))
        k_rule = f"N^(2/(d+2)) [d={n_features}]"
    else:
        raise ValueError(f"Estratégia de K inválida: {strategy}")

    # Limites de segurança
    k_value = max(k_min, k_value)
    k_value = min(k_max, k_value)

    # Nunca pode ser maior que o número disponível de vizinhos
    k_value = min(k_value, max(1, n_samples_ref - 1))

    return k_value, k_rule

def runPhase2_Competence(dataset, execution, json_final_file, json_consolidated_file, k_strategy="fixed", k_params=None):
    results_path_phase2 = Path("../results/PHASE2/")
    results_path_phase2.mkdir(parents=True, exist_ok=True)

    exec_folder = results_path_phase2 / dataset / f"exec_{execution}"
    exec_folder.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # CONFIGURAÇÃO DE VIZINHANÇA
    # -----------------------------
    if k_params is None:
        k_params = {
            "k_fixed": 10,
            "k_min": 3,
            "k_max": 20,
            "n_threshold": 1000
        }

    # -----------------------------
    # 1. Carregar JSONs
    # -----------------------------
    with open(json_final_file, "r") as f:
        final_data = json.load(f)  # É um Dict, não lista

    with open(json_consolidated_file, "r") as f:
        consolidated_data = json.load(f)  # É uma List de execuções

    # A. Recuperar dados da execução específica
    # Procura o dicionário dentro da lista que tem "Execution" == execution atual
    execution_data = next(
        item for item in consolidated_data
        if item["Execution"] == execution
    )

    # Extrair vetores brutos
    data = np.array(execution_data["Data"])
    train_index = np.array(execution_data["TrainIndex"])
    valid_index = np.array(execution_data["ValidIndex"])
    test_index  = np.array(execution_data["TestIndex"])

    # -------------------------------------------------
    # 2. Recuperar modelos selecionados
    # -------------------------------------------------
    all_models_info = final_data["Models"]

    # Filtra apenas os selecionados
    selected_models_info = [
        m for m in all_models_info if m.get("SelectedMedian", True)
    ]

    if not selected_models_info:
        selected_models_info = all_models_info

    # Precisamos dos ÍNDICES e dos NOMES para re-treinar
    selected_indices = [m["Index"] for m in selected_models_info]
    selected_names   = [m["ModelName"] for m in selected_models_info]

    # -----------------------------
    # 3. Loop por Folds (Geração da Competência Honesta com CV)
    # -----------------------------
    folds = num_folds()

    for fold in range(folds):

        # Máscaras booleanas (Lógica original)
        train_mask = train_index[:, fold].astype(bool)
        valid_mask = valid_index[:, fold].astype(bool)
        test_mask = test_index[:, fold].astype(bool)

        # Dados brutos das partições
        data_train = data[train_mask, :]
        data_valid = data[valid_mask, :]
        data_test = data[test_mask, :]

        # Train + Validation = Reference set
        train_plus = np.concatenate([data_train, data_valid], axis=0)

        X_ref = train_plus[:, 1:]
        y_ref = train_plus[:, 0]

        X_test = data_test[:, 1:]
        y_test = data_test[:, 0]

        # ---------------------------------------------------------
        # Resolve K do fold atual
        # ---------------------------------------------------------
        n_samples_ref = X_ref.shape[0]
        n_features    = X_ref.shape[1]

        k_value, k_rule = resolve_k(
            n_samples_ref=n_samples_ref,
            n_features=n_features,
            strategy=k_strategy,
            k_params=k_params
        )

        '''print(
            f"[{dataset}] Exec={execution} Fold={fold} | "
            f"N_ref={n_samples_ref} d={n_features} | "
            f"K={k_value} | Strategy={k_strategy} | Rule={k_rule}"
        )'''

        # ---------------------------------------------------------
        # KNN SPACE
        # ---------------------------------------------------------
        scaler_knn = StandardScaler()
        X_ref_knn  = scaler_knn.fit_transform(X_ref)
        X_test_knn = scaler_knn.transform(X_test)

        n_samples_test = X_test.shape[0]
        n_models       = len(selected_indices)

        # Matrizes para guardar predições
        base_preds_ref_cv = np.zeros((n_samples_ref, n_models))
        base_preds_test   = np.zeros((n_samples_test, n_models))

        # -------------------------------------------------
        # 4. Predições dos modelos base
        # -------------------------------------------------
        for i, (model_idx, model_name) in enumerate(zip(selected_indices, selected_names)):

            # Instancia um modelo novo e limpo (Scikit-Learn)
            model_instance = get_model_instance(model_name)

            best_params = load_best_params(dataset, model_name)

            if best_params is None:
                raise FileNotFoundError(
                    f"Hiperparâmetros não encontrados: "
                    f"dataset={dataset}, modelo={model_name}"
                )

            if best_params:
                model_instance.set_params(**best_params)

            # --- PASSO A: Predição Honesta no Reference Set (Para Matriz de Competência) ---
            # Usamos Cross-Validation para simular erro de teste dentro do Reference Set
            model_cv = clone(model_instance)

            try:
                # n_jobs=1 para evitar conflito de threads
                preds_cv = cross_val_predict(
                    model_cv,
                    X_ref,
                    y_ref,
                    cv=5,
                    n_jobs=1
                )
            except ValueError:
                # Fallback para datasets minúsculos (ex: cv=3 ou até LOO)
                n_splits = min(3, n_samples_ref)
                preds_cv = cross_val_predict(
                    model_cv,
                    X_ref,
                    y_ref,
                    cv=n_splits,
                    n_jobs=1
                )

            base_preds_ref_cv[:, i] = preds_cv

            # --- PASSO B: Predição Final no Test Set (Para a Phase 3) ---
            # Treinamos o modelo "Final" em TODO o Reference Set
            model_final = clone(model_instance)
            model_final.fit(X_ref, y_ref)
            base_preds_test[:, i] = model_final.predict(X_test)

        # -------------------------------------------------
        # 5. Cálculo da competência DREAM
        # -------------------------------------------------

        dream_error_test, knn_idx_test, knn_dist_test = calculate_dream_rmse_matrix(
            X_ref_knn,
            X_test_knn,
            base_preds_ref_cv,
            y_ref,
            k=k_value,
            return_neighbors=True
        )

        dream_error_train = calculate_dream_rmse_matrix(
            X_ref_knn,
            X_ref_knn,
            base_preds_ref_cv,
            y_ref,
            k=k_value
        )

        dream_stats = {
            "dream_error_test_median_per_model": np.median(dream_error_test, axis=0),
            "dream_error_test_iqr_per_model": (
                np.percentile(dream_error_test, 75, axis=0)
                - np.percentile(dream_error_test, 25, axis=0)
            )
        }

        # Métricas de registro (MSE real)
        mse_ref_models  = np.mean((y_ref[:, None] - base_preds_ref_cv) ** 2, axis=0)
        mse_test_models = np.mean((y_test[:, None] - base_preds_test) ** 2, axis=0)

        # -------------------------------------------------
        # 5.1 Métricas de auditoria do KNN
        # -------------------------------------------------
        knn_dist_valid = knn_dist_test[np.isfinite(knn_dist_test)]

        if knn_dist_valid.size > 0:
            knn_dist_mean   = float(np.mean(knn_dist_valid))
            knn_dist_median = float(np.median(knn_dist_valid))
            knn_dist_std    = float(np.std(knn_dist_valid))
            knn_dist_min    = float(np.min(knn_dist_valid))
            knn_dist_max    = float(np.max(knn_dist_valid))
        else:
            knn_dist_mean   = np.nan
            knn_dist_median = np.nan
            knn_dist_std    = np.nan
            knn_dist_min    = np.nan
            knn_dist_max    = np.nan

        # -------------------------------------------------
        # 6. Salvar artefatos
        # -------------------------------------------------
        np.savez(
            exec_folder / f"fold_{fold}.npz",

            X_train=X_ref,
            y_train=y_ref,

            X_test=X_test,
            y_test=y_test,

            knn_idx_test=knn_idx_test,
            knn_dist_test=knn_dist_test,

            base_predictions_train=base_preds_ref_cv,
            base_predictions_test=base_preds_test,

            dream_error_train=dream_error_train,
            dream_error_test=dream_error_test,

            dream_error_test_median_per_model=dream_stats["dream_error_test_median_per_model"],
            dream_error_test_iqr_per_model=dream_stats["dream_error_test_iqr_per_model"],

            mse_train_models=mse_ref_models,
            mse_test_models=mse_test_models,

            regressor_indices=np.array(selected_indices),

            # ============================
            # Auditoria do K
            # ============================
            k_strategy    = np.array([k_strategy]),
            k_rule        = np.array([k_rule]),
            k_value       = np.array([k_value]),
            n_samples_ref = np.array([n_samples_ref]),
            n_features    = np.array([n_features]),
            k_fixed       = np.array([k_params["k_fixed"]]),
            k_min         = np.array([k_params["k_min"]]),
            k_max         = np.array([k_params["k_max"]]),
            n_threshold   = np.array([k_params["n_threshold"]]),

                # Métricas agregadas da vizinhança
            knn_dist_mean   = np.array([knn_dist_mean]),
            knn_dist_median = np.array([knn_dist_median]),
            knn_dist_std    = np.array([knn_dist_std]),
            knn_dist_min    = np.array([knn_dist_min]),
            knn_dist_max    = np.array([knn_dist_max])
        )
