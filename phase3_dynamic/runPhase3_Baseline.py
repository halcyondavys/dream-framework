# arquivo: runPhase3_Baseline.py
# -*- coding: utf-8 -*-
"""
Fase 3 - Baselines do DREAM (VERSÃO CORRIGIDA — sem vazamento em TOP3/TOP5)
----------------------------------------------------------------------------
Métodos calculados sobre o mesmo pool de predições da Fase 2 (NPZ):

  TOP3_AVG     : média simples dos 3 modelos com menor MSE no conjunto de
                 REFERÊNCIA (mse_train_models), avaliados no teste do fold
  TOP5_AVG     : idem, com 5 modelos (apenas se n_models >= 5)

  STACKING_RF  : meta-modelo Random Forest sobre o pool do ensemble
  STACKING_XGB : meta-modelo XGBoost sobre o pool do ensemble
  STACKING_ET  : meta-modelo Extra Trees sobre o pool do ensemble

Ranking dos modelos:
  O ranking de TOP3/TOP5 é calculado a partir de mse_train_models
  (MSE honesto no conjunto de referência, já salvo no NPZ pela Fase 2 via
  cross_val_predict). Elimina o vazamento de informação do y_test na escolha
  de quais modelos compõem o TOP3/TOP5.

Saída:
  dream_phase3_baselines_results.csv
  dream_phase3_baselines_execution_times.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from main.configs import datasets_used, datasets_used_reduzido, datasets_synthetic, num_execucoes, num_folds, path_DREAM

# ─────────────────────────────────────────────────────────────────────────────
# Configurações
# ─────────────────────────────────────────────────────────────────────────────
paths = path_DREAM()

SCRIPT_DIR = Path(__file__).resolve().parent

phase2_path = (SCRIPT_DIR / paths["results_path_phase2"]).resolve()
output_path = (SCRIPT_DIR / paths["results_path_phase3"]).resolve()

output_path.mkdir(parents=True, exist_ok=True)

# RODADA 1 (atual): 12 células do estudo fatorial (Apêndice B) — NPZ da Fase 2
# já disponível. Validar aqui antes de reprocessar os dados reais.
#datasets   = datasets_used()
#output_suffix = "synthetic"

# RODADA 2 (próxima, após validação): descomentar as duas linhas abaixo e
# comentar as duas linhas acima para reprocessar as 30 bases reais do Cap. 5.
datasets = datasets_used()
output_suffix = "real"
executions = num_execucoes()
folds      = num_folds()

# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    mse    = mean_squared_error(y_true, y_pred)
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mse)
    eps    = 1e-8
    mape   = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))
    r2     = r2_score(y_true, y_pred)
    med_ae = np.median(np.abs(y_true - y_pred))
    return {"MSE": mse, "MAE": mae, "RMSE": rmse,
            "MAPE": mape, "R2": r2, "MedAE": med_ae}


def build_result_row(dataset, exec_id, fold, method, y_true, y_pred, elapsed, n_models):
    row = compute_metrics(y_true, y_pred)
    row.update({
        "Dataset":  dataset,
        "Exec":     exec_id,
        "Fold":     fold,
        "Method":   method,
        "Time":     elapsed,
        "N_Models": int(n_models)
    })
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Stacking (treina em base_predictions_train/y_train, que vêm de
# cross_val_predict no conjunto de referência, e só prediz no teste)
# ─────────────────────────────────────────────────────────────────────────────
def stacking_models(base_predictions_train, y_train, base_predictions_test):
    results = {}

    rf = RandomForestRegressor(
        n_estimators=500, max_depth=15,
        min_samples_split=3, random_state=42, n_jobs=-1
    )
    rf.fit(base_predictions_train, y_train)
    results["STACKING_RF"] = rf.predict(base_predictions_test)

    xgb = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.015,
        colsample_bytree=0.8, reg_alpha=0.3,
        random_state=42, n_jobs=-1
    )
    xgb.fit(base_predictions_train, y_train)
    results["STACKING_XGB"] = xgb.predict(base_predictions_test)

    et = ExtraTreesRegressor(
        n_estimators=500, max_depth=8,
        min_samples_split=5, random_state=42, n_jobs=-1
    )
    et.fit(base_predictions_train, y_train)
    results["STACKING_ET"] = et.predict(base_predictions_test)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TOP-K local ao fold
# ─────────────────────────────────────────────────────────────────────────────
def topk_local(mse_train_models, base_predictions_test, k):
    """
    Seleciona os k modelos com menor MSE no conjunto de REFERÊNCIA
    (mse_train_models), já alinhados por posição com as colunas de
    base_predictions_test. Retorna None se n_models < k.
    """
    n_models = len(mse_train_models)
    if n_models < k:
        return None, None

    top_positions = np.argsort(mse_train_models)[:k]
    y_pred = np.mean(base_predictions_test[:, top_positions], axis=1)
    return y_pred, top_positions


# ─────────────────────────────────────────────────────────────────────────────
# Execução principal
# ─────────────────────────────────────────────────────────────────────────────
print("\n==============================")
print("Executando Fase 3 - BASELINES")
print("==============================\n")

all_results   = []
dataset_times = []

for dataset in datasets:
    print(f"Dataset: {dataset}")
    dataset_start_time = time.time()

    for exec_id in range(executions):
        print(f">> Dataset={dataset} | Exec={exec_id}")
        exec_phase2_path = phase2_path / dataset / f"exec_{exec_id}"

        for fold in range(folds):
            npz_file = exec_phase2_path / f"fold_{fold}.npz"
            if not npz_file.exists():
                print(f"[AVISO] Arquivo não encontrado: {npz_file}")
                continue

            data = np.load(npz_file, allow_pickle=True)

            y_train                = data["y_train"]
            y_test                 = data["y_test"]
            base_predictions_train = data["base_predictions_train"]
            base_predictions_test  = data["base_predictions_test"]
            mse_train_models       = data["mse_train_models"]   # MSE honesto (referência)

            n_models = base_predictions_test.shape[1]

            # -----------------------------------------------------------------
            # TOP3_AVG — 3 modelos com menor MSE de referência, LOCAL ao fold
            # -----------------------------------------------------------------
            y_top3, pos3 = topk_local(mse_train_models, base_predictions_test, k=3)
            if y_top3 is not None:
                t0 = time.perf_counter()
                elapsed = time.perf_counter() - t0
                all_results.append(build_result_row(
                    dataset, exec_id, fold, "TOP3_AVG",
                    y_test, y_top3, elapsed, len(pos3)
                ))
            else:
                print(f"[AVISO] n_models < 3 no fold {fold} "
                      f"(exec={exec_id}, dataset={dataset}). TOP3_AVG ignorado.")

            # -----------------------------------------------------------------
            # TOP5_AVG — apenas se n_models >= 5
            # -----------------------------------------------------------------
            if n_models >= 5:
                y_top5, pos5 = topk_local(mse_train_models, base_predictions_test, k=5)
                if y_top5 is not None:
                    t0 = time.perf_counter()
                    elapsed = time.perf_counter() - t0
                    all_results.append(build_result_row(
                        dataset, exec_id, fold, "TOP5_AVG",
                        y_test, y_top5, elapsed, len(pos5)
                    ))

            # -----------------------------------------------------------------
            # STACKING
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            stacking_preds = stacking_models(
                base_predictions_train, y_train, base_predictions_test
            )
            elapsed_total = time.perf_counter() - t0
            elapsed_each  = elapsed_total / 3.0

            for method_name, y_stack in stacking_preds.items():
                all_results.append(build_result_row(
                    dataset, exec_id, fold, method_name,
                    y_test, y_stack, elapsed_each, n_models
                ))

    dataset_elapsed = time.time() - dataset_start_time
    dataset_times.append({
        "Dataset":             dataset,
        "Tempo_Total_Segundos": dataset_elapsed,
        "Num_Execucoes":       executions,
        "Num_Folds":           folds
    })
    print(f"Tempo total para {dataset}: {dataset_elapsed:.2f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Salvar resultados
# ─────────────────────────────────────────────────────────────────────────────
pd.DataFrame(all_results).to_csv(
    output_path / f"dream_phase3_baselines_results_{output_suffix}.csv", index=False
)
pd.DataFrame(dataset_times).to_csv(
    output_path / f"dream_phase3_baselines_execution_times_{output_suffix}.csv", index=False
)

print("\n==============================")
print(f"Fase 3 - BASELINES finalizada — lote: {output_suffix}")
print(f"Resultados em: {output_path}")
print("==============================\n")