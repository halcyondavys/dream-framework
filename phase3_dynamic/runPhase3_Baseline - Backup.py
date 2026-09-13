# arquivo: runPhase3_Baseline.py
# -*- coding: utf-8 -*-
"""
Fase 3 - Baselines do DREAM
---------------------------
Executa as técnicas de comparação:

- STACKING_RF
- STACKING_XGB
- STACKING_ET
- TOP3_ERROR_AVG
- TOP5_ERROR_AVG
- HET_ALL_AVG

Resultados por:
Dataset, Exec, Fold, Method, MSE, MAE, RMSE, MAPE, R2, MedAE, Time, N_Models
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from main.configs import datasets_used, datasets_synthetic, num_execucoes, num_folds


# ---------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))
    r2 = r2_score(y_true, y_pred)
    med_ae = np.median(np.abs(y_true - y_pred))

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "MedAE": med_ae
    }


def build_result_row(dataset, exec_id, fold, method, y_true, y_pred, elapsed_time, n_models):
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        "Dataset": dataset,
        "Exec": exec_id,
        "Fold": fold,
        "Method": method,
        "Time": elapsed_time,
        "N_Models": int(n_models)
    })
    return metrics


def ensemble_avg(pred_matrix, model_indices):
    return np.mean(pred_matrix[:, model_indices], axis=1)


def stacking_models(base_predictions_train, y_train, base_predictions_test):
    """
    Meta-modelos de stacking usando as meta-features geradas na Phase 2.
    """
    results = {}

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(base_predictions_train, y_train)
    results["STACKING_RF"] = rf.predict(base_predictions_test)

    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.015,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(base_predictions_train, y_train)
    results["STACKING_XGB"] = xgb.predict(base_predictions_test)

    et = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    et.fit(base_predictions_train, y_train)
    results["STACKING_ET"] = et.predict(base_predictions_test)

    return results


# ---------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------
datasets = datasets_used()
executions = num_execucoes()
folds = num_folds()

phase1_path = Path("../main/DREAM/sequencial/results/PHASE1")
phase2_path = Path("../main/DREAM/sequencial/results/PHASE2")
output_path = Path("../main/DREAM/sequencial/results/PHASE3")
output_path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------
print("\n==============================")
print("Executando Fase 3 - BASELINES")
print("==============================\n")

dataset_times = []
all_results = []

for dataset in datasets:
    print(f"Dataset: {dataset}")
    dataset_start_time = time.time()

    final_json_path = phase1_path / f"{dataset}-FINAL.json"
    if not final_json_path.exists():
        print(f"[ERRO] FINAL.json não encontrado: {final_json_path}")
        continue

    with open(final_json_path, "r", encoding="utf-8") as f:
        final_data = json.load(f)

    errors_median = np.array(final_data["Metrics"]["ErrorsMedian"], dtype=float)

    top3_error_idx = np.argsort(errors_median)[:3]
    top5_error_idx = np.argsort(errors_median)[:5]

    for exec_id in range(executions):
        print(f">> Dataset={dataset} | Exec={exec_id}")
        exec_phase2_path = phase2_path / dataset / f"exec_{exec_id}"

        for fold in range(folds):
            npz_file = exec_phase2_path / f"fold_{fold}.npz"
            if not npz_file.exists():
                print(f"[AVISO] Arquivo não encontrado: {npz_file}")
                continue

            data = np.load(npz_file, allow_pickle=True)

            y_train = data["y_train"]
            y_test = data["y_test"]

            base_predictions_train = data["base_predictions_train"]
            base_predictions_test = data["base_predictions_test"]

            n_models = base_predictions_test.shape[1]
            all_idx = np.arange(n_models)

            # -------------------------------------------------------------
            # TOP3_ERROR_AVG
            # -------------------------------------------------------------
            valid_top3_idx = top3_error_idx[top3_error_idx < n_models]
            if len(valid_top3_idx) > 0:
                t0 = time.perf_counter()
                y_top3 = ensemble_avg(base_predictions_test, valid_top3_idx)
                elapsed = time.perf_counter() - t0
                all_results.append(
                    build_result_row(
                        dataset,
                        exec_id,
                        fold,
                        "TOP3_ERROR_AVG",
                        y_test,
                        y_top3,
                        elapsed,
                        len(valid_top3_idx)
                    )
                )

            # -------------------------------------------------------------
            # TOP5_ERROR_AVG
            # -------------------------------------------------------------
            valid_top5_idx = top5_error_idx[top5_error_idx < n_models]
            if len(valid_top5_idx) > 0:
                t0 = time.perf_counter()
                y_top5 = ensemble_avg(base_predictions_test, valid_top5_idx)
                elapsed = time.perf_counter() - t0
                all_results.append(
                    build_result_row(
                        dataset,
                        exec_id,
                        fold,
                        "TOP5_ERROR_AVG",
                        y_test,
                        y_top5,
                        elapsed,
                        len(valid_top5_idx)
                    )
                )

            # -------------------------------------------------------------
            # HET_ALL_AVG
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            y_all = ensemble_avg(base_predictions_test, all_idx)
            elapsed = time.perf_counter() - t0
            all_results.append(
                build_result_row(
                    dataset,
                    exec_id,
                    fold,
                    "HET_ALL_AVG",
                    y_test,
                    y_all,
                    elapsed,
                    n_models
                )
            )

            # -------------------------------------------------------------
            # STACKING
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            stacking_preds = stacking_models(
                base_predictions_train,
                y_train,
                base_predictions_test
            )
            elapsed_total_stacking = time.perf_counter() - t0

            # distribuição simples do tempo total entre os 3 meta-modelos
            # se quiser, depois podemos medir separadamente dentro da função
            elapsed_each = elapsed_total_stacking / 3.0

            for method_name, y_stack in stacking_preds.items():
                all_results.append(
                    build_result_row(
                        dataset,
                        exec_id,
                        fold,
                        method_name,
                        y_test,
                        y_stack,
                        elapsed_each,
                        n_models
                    )
                )

    dataset_elapsed = time.time() - dataset_start_time
    dataset_times.append({
        "Dataset": dataset,
        "Tempo_Total_Segundos": dataset_elapsed,
        "Num_Execucoes": executions,
        "Num_Folds": folds
    })

    print(f"Tempo total para o dataset {dataset}: {dataset_elapsed:.2f} segundos")

# ---------------------------------------------------------------------
# Salvar resultados
# ---------------------------------------------------------------------
df_results = pd.DataFrame(all_results)
df_times = pd.DataFrame(dataset_times)

df_results.to_csv(output_path / "dream_phase3_baselines_results.csv", index=False)
df_times.to_csv(output_path / "dream_phase3_baselines_execution_times.csv", index=False)

print("\n==============================")
print("Fase 3 - BASELINES finalizada com sucesso")
print(f"Resultados salvos em: {output_path}")
print("==============================\n")