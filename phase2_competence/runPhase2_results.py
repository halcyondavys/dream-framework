import numpy as np
import pandas as pd
from pathlib import Path
from main.configs import datasets_used_reduzido, datasets_used, datasets_synthetic, num_execucoes, num_folds

# =========================================================
# CONFIGURAÇÕES
# =========================================================
datasets = datasets_used()
executions = num_execucoes()
folds = num_folds()

phase2_path = Path("../results/PHASE2/")
output_path = Path("../results/PHASE2/parquet/phase2_competence_audit.parquet")
#output_path = Path("../main/DREAM/sequencial/results/PHASE2/csv/phase2_competence_audit.csv")

# Se quiser restringir execuções para teste rápido, use por exemplo:
# EXECUTIONS_TO_USE = [0]
EXECUTIONS_TO_USE = list(range(executions))

rows = []
eps = 1e-12

for dataset in datasets:

    print(f'\n==============================')
    print(f'Dataset: {dataset}')
    print(f'==============================')
    for exec_id in EXECUTIONS_TO_USE:
        exec_folder = phase2_path / dataset / f"exec_{exec_id}"

        print(f'>> Dataset={dataset} | Execution: {exec_id}')

        if not exec_folder.exists():
            print(f"[WARN] Pasta não encontrada: {exec_folder}")
            continue

        for fold_file in sorted(exec_folder.glob("fold_*.npz")):
            fold = int(fold_file.stem.split("_")[1])

            try:
                data = np.load(fold_file, allow_pickle=True)
            except Exception as e:
                print(f"[ERRO] Falha ao carregar {fold_file}: {e}")
                continue

            # -----------------------------------------------------
            # CAMPOS OBRIGATÓRIOS
            # -----------------------------------------------------
            y_test = data["y_test"]
            preds_test = data["base_predictions_test"]
            dream_error_test = data["dream_error_test"]
            knn_dist_test = data["knn_dist_test"]

            mse_train_models = data["mse_train_models"]
            mse_test_models = data["mse_test_models"]
            dream_med = data["dream_error_test_median_per_model"]
            dream_iqr = data["dream_error_test_iqr_per_model"]
            regressor_indices = data["regressor_indices"]

            n_test, n_models = preds_test.shape

            # -----------------------------------------------------
            # CAMPOS DE AUDITORIA DO K (COM FALLBACK)
            # -----------------------------------------------------
            def safe_scalar(field_name, default=np.nan):
                if field_name not in data:
                    return default
                value = data[field_name]
                if isinstance(value, np.ndarray):
                    if value.size == 0:
                        return default
                    value = value.flatten()[0]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                return value

            k_strategy = safe_scalar("k_strategy", default="fixed")
            k_rule = safe_scalar("k_rule", default="fixed(10)")
            k_value = safe_scalar("k_value", default=np.nan)
            n_samples_ref = safe_scalar("n_samples_ref", default=np.nan)
            n_features = safe_scalar("n_features", default=np.nan)

            k_fixed = safe_scalar("k_fixed", default=np.nan)
            k_min = safe_scalar("k_min", default=np.nan)
            k_max = safe_scalar("k_max", default=np.nan)
            n_threshold = safe_scalar("n_threshold", default=np.nan)

            knn_dist_mean_fold = safe_scalar("knn_dist_mean", default=np.nan)
            knn_dist_median_fold = safe_scalar("knn_dist_median", default=np.nan)
            knn_dist_std_fold = safe_scalar("knn_dist_std", default=np.nan)
            knn_dist_min_fold = safe_scalar("knn_dist_min", default=np.nan)
            knn_dist_max_fold = safe_scalar("knn_dist_max", default=np.nan)

            # -----------------------------------------------------
            # LOOP POR INSTÂNCIA DE TESTE
            # -----------------------------------------------------
            for i in range(n_test):
                knn_dists = knn_dist_test[i]

                # remove NaN, padding inválido ou distâncias <= 0
                knn_dists = knn_dists[np.isfinite(knn_dists)]
                knn_dists = knn_dists[knn_dists > 0]

                if len(knn_dists) == 0:
                    mean_knn = np.nan
                    median_knn = np.nan
                    iqr_knn = np.nan
                    std_knn = np.nan
                    min_knn = np.nan
                    max_knn = np.nan
                    k_effective = 0
                else:
                    mean_knn = float(np.mean(knn_dists))
                    median_knn = float(np.median(knn_dists))
                    iqr_knn = float(np.percentile(knn_dists, 75) - np.percentile(knn_dists, 25))
                    std_knn = float(np.std(knn_dists))
                    min_knn = float(np.min(knn_dists))
                    max_knn = float(np.max(knn_dists))
                    k_effective = int(len(knn_dists))

                # -------------------------------------------------
                # LOOP POR MODELO
                # -------------------------------------------------
                for j in range(n_models):
                    abs_error_test = float(abs(y_test[i] - preds_test[i, j]))
                    dream_error_local = float(dream_error_test[i, j])

                    if np.isnan(dream_error_local):
                        dream_competence = np.nan
                    else:
                        dream_competence = float(1.0 / (dream_error_local + eps))

                    rows.append({
                        # Identificação
                        "Dataset": dataset,
                        "Exec": exec_id,
                        "Fold": fold,
                        "TestIdx": i,
                        "RegressorIndex": int(regressor_indices[j]),
                        "ModelPos": j,

                        # Configuração do K
                        "KStrategy": str(k_strategy),
                        "KRule": str(k_rule),
                        "KValue": int(k_value) if pd.notna(k_value) else np.nan,
                        "KEffective": k_effective,
                        "KFixed": int(k_fixed) if pd.notna(k_fixed) else np.nan,
                        "KMin": int(k_min) if pd.notna(k_min) else np.nan,
                        "KMax": int(k_max) if pd.notna(k_max) else np.nan,
                        "NThreshold": int(n_threshold) if pd.notna(n_threshold) else np.nan,
                        "NRef": int(n_samples_ref) if pd.notna(n_samples_ref) else np.nan,
                        "NFeatures": int(n_features) if pd.notna(n_features) else np.nan,

                        # Ground truth e predição
                        "YTrue": float(y_test[i]),
                        "PredTest": float(preds_test[i, j]),
                        "AbsErrorTest": abs_error_test,

                        # Competência local DREAM
                        "DreamErrorLocal": dream_error_local,
                        "DreamCompetence": dream_competence,

                        # Estatísticas da vizinhança por instância
                        "MeanKnnDist": mean_knn,
                        "MedianKnnDist": median_knn,
                        "IQRKnnDist": iqr_knn,
                        "StdKnnDist": std_knn,
                        "MinKnnDist": min_knn,
                        "MaxKnnDist": max_knn,

                        # Estatísticas da vizinhança por fold
                        "FoldMeanKnnDist": float(knn_dist_mean_fold) if pd.notna(knn_dist_mean_fold) else np.nan,
                        "FoldMedianKnnDist": float(knn_dist_median_fold) if pd.notna(knn_dist_median_fold) else np.nan,
                        "FoldStdKnnDist": float(knn_dist_std_fold) if pd.notna(knn_dist_std_fold) else np.nan,
                        "FoldMinKnnDist": float(knn_dist_min_fold) if pd.notna(knn_dist_min_fold) else np.nan,
                        "FoldMaxKnnDist": float(knn_dist_max_fold) if pd.notna(knn_dist_max_fold) else np.nan,

                        # Métricas globais do modelo
                        "MSE_Train_Model": float(mse_train_models[j]),
                        "MSE_Test_Model": float(mse_test_models[j]),
                        "Gap_Overfitting_Model": float(mse_test_models[j] - mse_train_models[j]),

                        # Resumo do erro local do modelo
                        "DreamErrorMedian_Model": float(dream_med[j]),
                        "DreamErrorIQR_Model": float(dream_iqr[j]),
                    })

#df_audit = pd.DataFrame(rows)
#output_path.parent.mkdir(parents=True, exist_ok=True)
#df_audit.to_csv(output_path, index=False)

# Parquet
df_audit = pd.DataFrame(rows)

# Garante que a pasta de destino exista
output_path.parent.mkdir(parents=True, exist_ok=True)

df_audit.to_parquet(
    output_path,
    index=False,
    engine="pyarrow",
    compression="snappy"
)

print(f"CSV consolidado salvo em: {output_path}")
print(f"Total de linhas: {len(df_audit)}")
print(f"Total de colunas: {len(df_audit.columns)}")