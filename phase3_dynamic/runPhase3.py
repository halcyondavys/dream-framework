# arquivo: runPhase3.py
# -*- coding: utf-8 -*-
"""
Fase 3 - DREAM
--------------
Lê os artefatos da Fase 2 (PHASE2.npz) e executa:

- DREAM-DS
- DREAM-DW
- DREAM-DWS
- HET_DREAM_AVG
- ORACLE_BASE (nível modelo base)
- KMEN_ORACLE (nível técnica de combinação)

Gera um CSV consolidado com as métricas:
Dataset, Exec, Fold, Method, MSE, MAE, RMSE, MAPE, R2, MedAE, Time, N_Models
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from main.configs import datasets_used, datasets_used_reduzido, datasets_synthetic, num_execucoes, num_folds, path_DREAM

# Lista para armazenar resultados
paths = path_DREAM()

SCRIPT_DIR = Path(__file__).resolve().parent

phase2_path = (SCRIPT_DIR / paths["results_path_phase2"]).resolve()
phase3_path = (SCRIPT_DIR / paths["results_path_phase3"]).resolve()

phase3_path.mkdir(parents=True, exist_ok=True)

datasets   = datasets_used()
executions = num_execucoes()
folds      = num_folds()

# ✅ Suas funções finais de DS / DW / DWS
from DREAM_DS  import DREAM_DS
from DREAM_DW  import DREAM_DW
from DREAM_DWS import DREAM_DWS

# -------------------------
# Funções auxiliares
# -------------------------
def oracle_base(predictions_test, y_test):
    """
    Oracle em nível de modelos base.
    Para cada instância, escolhe o modelo com menor erro quadrático.
    """
    errors = (predictions_test - y_test.reshape(-1, 1)) ** 2
    best_idx = np.argmin(errors, axis=1)
    y_pred = predictions_test[np.arange(len(y_test)), best_idx]
    return y_pred

def kmen_oracle(preds_dict, y_test):
    """
    KMEN-Oracle em nível de técnica de combinação.
    Para cada instância, escolhe a técnica com menor erro quadrático.
    """
    methods = list(preds_dict.keys())
    preds_matrix = np.column_stack([preds_dict[mname] for mname in methods])
    errors = (preds_matrix - y_test.reshape(-1, 1)) ** 2
    best_method_idx = np.argmin(errors, axis=1)
    y_pred_kmen = preds_matrix[np.arange(len(y_test)), best_method_idx]

    chosen_counts = {
        methods[i]: int(np.sum(best_method_idx == i))
        for i in range(len(methods))}
    return y_pred_kmen, chosen_counts

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

def heterogeneous_simple_ensemble(predictions_test):
    """
    Média simples das previsões dos modelos selecionados pelo DREAM.
    """
    return np.mean(predictions_test, axis=1)

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

# -------------------------
# MAIN FASE 3
# -------------------------
print("Executando Fase 3 - DREAM")

all_results       = []
kmen_choices_rows = []
dataset_times     = []
audit_phase3_rows = []

for dataset in datasets:
    print(f"\n==============================")
    print(f"Dataset: {dataset}")
    print(f"==============================")

    dataset_start_time = time.time()

    for exec_id in range(executions):
        # Pasta da execução
        exec_folder = phase2_path / dataset / f"exec_{exec_id}"
        print(f">> Dataset={dataset} | Exec={exec_id}")

        for fold in range(folds):
            npz_file = exec_folder / f"fold_{fold}.npz"

            if not npz_file.exists():
                print(f"[AVISO] Arquivo não encontrado: {npz_file}")
                continue

            data = np.load(npz_file, allow_pickle=True)

            y_train = data["y_train"]
            y_test  = data["y_test"]

            base_predictions_test  = data["base_predictions_test"]
            dream_error_test       = data["dream_error_test"]
            mse_train_models       = data["mse_train_models"]
            regressor_indices      = data["regressor_indices"]

            n_models = len(regressor_indices)

            # Converter a Matriz de Erro do DREAM (RMSE) para Competência (Bondade)
            # Nota: Adicionamos epsilon para evitar divisão por zero.
            # Quanto MENOR o erro, MAIOR a competência.
            competence_dream = 1.0 / (dream_error_test + 1e-12)

            # -----------------------
            # Ensemble Heterogêneo Simples (AVG)
            # -----------------------
            t0 = time.perf_counter()
            y_avg = heterogeneous_simple_ensemble(base_predictions_test)
            elapsed = time.perf_counter() - t0
            all_results.append(
                build_result_row(dataset, exec_id, fold, "HET_DREAM_AVG", y_test, y_avg, elapsed, n_models)
            )

            # -----------------------
            # DREAM-DS
            # -----------------------
            t0 = time.perf_counter()
            y_ds, dbg_ds = DREAM_DS(
                base_predictions_test,
                competence_dream.copy(),
                mse_train_models,
                dynamic_threshold = 0.3,
                return_debug = True
            )
            elapsed = time.perf_counter() - t0
            all_results.append(
                build_result_row(dataset, exec_id, fold, "DREAM_DS", y_test, y_ds, elapsed, n_models)
            )

            # -----------------------
            # DREAM-DW
            # -----------------------
            t0 = time.perf_counter()
            y_dw, dbg_dw = DREAM_DW(
                base_predictions_test,
                competence_dream.copy(),
                return_debug=True
            )
            elapsed = time.perf_counter() - t0
            all_results.append(
                build_result_row(dataset, exec_id, fold, "DREAM_DW", y_test, y_dw, elapsed, n_models)
            )

            # -----------------------
            # DREAM-DWS
            # -----------------------
            t0 = time.perf_counter()
            y_dws, dbg_dws = DREAM_DWS(
                base_predictions_test,
                competence_dream.copy(),
                min_relative_threshold=0.5,
                return_debug=True
            )
            elapsed = time.perf_counter() - t0
            all_results.append(
                build_result_row(dataset, exec_id, fold, "DREAM_DWS", y_test, y_dws, elapsed, n_models)
            )

            # -----------------------
            # ORACLE_BASE
            # -----------------------
            t0 = time.perf_counter()
            y_oracle = oracle_base(base_predictions_test, y_test)
            elapsed = time.perf_counter() - t0
            all_results.append(
                build_result_row(dataset, exec_id, fold, "ORACLE_BASE", y_test, y_oracle, elapsed, n_models)
            )

            # -----------------------
            # KMEN_ORACLE
            # -----------------------
            preds_dict = {
                "HET_DREAM_AVG": y_avg,
                "DREAM_DS": y_ds,
                "DREAM_DW": y_dw,
                "DREAM_DWS": y_dws,
            }

            t0 = time.perf_counter()
            y_kmen, chosen_counts = kmen_oracle(preds_dict, y_test)
            elapsed = time.perf_counter() - t0
            all_results.append(
                build_result_row(dataset, exec_id, fold, "KMEN_ORACLE", y_test, y_kmen, elapsed, n_models)
            )

            for method_name, count in chosen_counts.items():
                kmen_choices_rows.append({
                    "Dataset": dataset,
                    "Exec": exec_id,
                    "Fold": fold,
                    "Chosen_Method": method_name,
                    "Count": count,
                    "Total": len(y_test)
                })

            if exec_id == 0:
                # =========================================================
                # AUDITORIA FASE 3 (por instância)
                # Objetivo: registrar o que a SDE "viu" e o que ela "decidiu"
                # =========================================================
                # Captura a competência "limpa" para a auditoria
                competence_mat_pure = np.asarray(competence_dream, dtype=float).copy()
                competence_mat_pure = np.nan_to_num(competence_mat_pure, nan=0.0)

                # Use 'competence_mat_pure' para calcular as margens (top_by_comp_pos, c1, c2, etc.)
                # Isso garantirá que o seu cálculo (17.778 - 17.672) resulte nos ~0.106 esperados.

                # Pesos vindos do debug (compatível com seu retorno atual)
                weights_ds  = dbg_ds["weights"]  # (n_test, n_models)
                weights_dw  = dbg_dw["weights"]  # (n_test, n_models)
                weights_dws = dbg_dws["weights"]  # (n_test, n_models)

                # Índice do "melhor por competência" (argmax da competência) — disponível em DS/DWS
                bestidx_ds  = dbg_ds.get("best_idx", None)  # (n_test,) ou None
                bestidx_dws = dbg_dws.get("best_idx", None)  # (n_test,) ou None

                # Quantos modelos foram selecionados (DWS tem n_selected; DS também tem, se você retornou)
                n_selected_ds  = dbg_ds.get("n_selected", None)  # (n_test,) ou None
                n_selected_dws = dbg_dws.get("n_selected", None)  # (n_test,) ou None

                # Top-1 escolhido por pesos (o mais influente no output final)
                top_modelpos_ds  = np.argmax(weights_ds, axis=1)  # (n_test,)
                top_modelpos_dw  = np.argmax(weights_dw, axis=1)  # (n_test,)
                top_modelpos_dws = dbg_dws["best_idx"]  # modelo de maior competência na janela

                # Se você quiser um "n_selected" do DWS mesmo sem retornar n_selected:
                # (mantive como fallback)
                if n_selected_dws is None:
                    n_selected_dws = np.sum(weights_dws > 0, axis=1)

                if n_selected_ds is None:
                    n_selected_ds = np.sum(weights_ds > 0, axis=1)

                # -------------------------
                # Diagnósticos de confiança
                # -------------------------
                # 1) Top-1 por competência (calculado direto da matriz)
                #    (mesmo que best_idx exista, isso garante consistência)
                competence_mat  = np.asarray(competence_mat_pure, dtype=float)
                competence_mat  = np.nan_to_num(competence_mat, nan=0.0, posinf=0.0, neginf=0.0)
                top_by_comp_pos = np.argmax(competence_mat, axis=1)  # (n_test,)

                # 2) Margem da competência: (c1 - c2) e (c1 / (c2+eps))
                #    Ajuda a separar: empate (margem baixa) vs convicção errada (margem alta)
                eps = 1e-12
                # pega os 2 maiores valores de cada linha sem ordenar tudo (eficiente)
                top2_comp = np.partition(competence_mat, kth=-2, axis=1)[:, -2:]  # (n_test, 2) mas não garante ordem
                c1 = np.max(top2_comp, axis=1)  # maior
                c2 = np.min(top2_comp, axis=1)  # segundo maior
                margin_comp_diff = c1 - c2
                margin_comp_ratio = c1 / (c2 + eps)


                # 3) Margem dos pesos: (w1 - w2) e (w1 / (w2+eps))
                def top2_margins(weights):
                    w = np.asarray(weights, dtype=float)
                    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                    top2 = np.partition(w, kth=-2, axis=1)[:, -2:]
                    w1 = np.max(top2, axis=1)
                    w2 = np.min(top2, axis=1)
                    return (w1 - w2), (w1 / (w2 + eps))

                margin_w_ds_diff, margin_w_ds_ratio = top2_margins(weights_ds)
                margin_w_dw_diff, margin_w_dw_ratio = top2_margins(weights_dw)
                margin_w_dws_diff, margin_w_dws_ratio = top2_margins(weights_dws)

                n_test = len(y_test)

                top_weight_dws = np.max(weights_dws, axis=1)
                #print("Peso do Top-1 (DWS):", top_weight_dws[:10])

                for i in range(n_test):

                    # Para um i específico
                    '''print("Competences:", competence_mat_pure[i])
                    print("Mask DWS thr=0.3:", (competence_mat_pure[i] >= (
                            np.max(competence_mat_pure[i]) - (np.ptp(competence_mat_pure[i]) * 0.3))).astype(int))
                    print("Mask DWS thr=0.5:", (competence_mat_pure[i] >= (
                            np.max(competence_mat_pure[i]) - (np.ptp(competence_mat_pure[i]) * 0.5))).astype(int))
                    print("Mask DWS thr=0.8:", (competence_mat_pure[i] >= (
                            np.max(competence_mat_pure[i]) - (np.ptp(competence_mat_pure[i]) * 0.8))).astype(int))'''

                    # Erro real absoluto por modelo base naquela instância (verdade no teste)
                    abs_errors_models = np.abs(y_test[i] - base_predictions_test[i, :])  # (n_models,)

                    # Melhor modelo real (oracle local, por instância)
                    best_real_modelpos = int(np.argmin(abs_errors_models))
                    best_real_reg_idx = int(regressor_indices[best_real_modelpos])
                    best_real_error = float(abs_errors_models[best_real_modelpos])

                    # ----------------------------
                    # DREAM-DS (top weight)
                    # ----------------------------
                    selected_modelpos_ds = int(top_modelpos_ds[i])
                    selected_reg_idx_ds = int(regressor_indices[selected_modelpos_ds])
                    selected_error_ds = float(abs_errors_models[selected_modelpos_ds])

                    # Best por competência (pré-mask), usando:
                    # - best_idx do debug se existir
                    # - senão, argmax direto em competence_mat
                    if bestidx_ds is not None:
                        best_by_comp_ds_pos = int(bestidx_ds[i])
                    else:
                        best_by_comp_ds_pos = int(top_by_comp_pos[i])

                    best_by_comp_ds_reg = int(regressor_indices[best_by_comp_ds_pos])
                    best_by_comp_ds_err = float(abs_errors_models[best_by_comp_ds_pos])

                    # ----------------------------
                    # DREAM-DW (top weight)
                    # ----------------------------
                    selected_modelpos_dw = int(top_modelpos_dw[i])
                    selected_reg_idx_dw = int(regressor_indices[selected_modelpos_dw])
                    selected_error_dw = float(abs_errors_models[selected_modelpos_dw])

                    # ----------------------------
                    # DREAM-DWS (top weight)
                    # ----------------------------
                    selected_modelpos_dws = int(top_modelpos_dws[i])
                    selected_reg_idx_dws = int(regressor_indices[selected_modelpos_dws])
                    selected_error_dws = float(abs_errors_models[selected_modelpos_dws])

                    if bestidx_dws is not None:
                        best_by_comp_dws_pos = int(bestidx_dws[i])
                    else:
                        best_by_comp_dws_pos = int(top_by_comp_pos[i])

                    best_by_comp_dws_reg = int(regressor_indices[best_by_comp_dws_pos])
                    best_by_comp_dws_err = float(abs_errors_models[best_by_comp_dws_pos])

                    # Diagnóstico do sinal (competência)
                    c_row = competence_mat[i, :]
                    c_max = float(np.max(c_row))
                    c_min = float(np.min(c_row))
                    c_spread = float(c_max - c_min)

                    audit_phase3_rows.append({
                        "Dataset": dataset,
                        "Exec": exec_id,
                        "Fold": fold,
                        "TestIdx": i,

                        # --- Melhor real (oracle local) ---
                        "BestRealModelPos": best_real_modelpos,
                        "BestRealRegressorIndex": best_real_reg_idx,
                        "Error_BestRealModel": best_real_error,

                        # --- DS (top weight) ---
                        "SelectedModelPos_DS": selected_modelpos_ds,
                        "SelectedRegressorIndex_DS": selected_reg_idx_ds,
                        "Error_SelectedModel_DS": selected_error_ds,
                        "Hit_DS": int(selected_modelpos_ds == best_real_modelpos),
                        "DeltaError_DS": float(selected_error_ds - best_real_error),
                        "N_Selected_DS": int(n_selected_ds[i]),

                        # --- DS (best por competência) ---
                        "BestByCompetencePos_DS": best_by_comp_ds_pos,
                        "BestByCompetenceRegressorIndex_DS": best_by_comp_ds_reg,
                        "Error_BestByCompetence_DS": best_by_comp_ds_err,

                        # --- DW (top weight) ---
                        "SelectedModelPos_DW": selected_modelpos_dw,
                        "SelectedRegressorIndex_DW": selected_reg_idx_dw,
                        "Error_SelectedModel_DW": selected_error_dw,
                        "Hit_DW": int(selected_modelpos_dw == best_real_modelpos),
                        "DeltaError_DW": float(selected_error_dw - best_real_error),

                        # --- DWS (top weight) ---
                        "SelectedModelPos_DWS": selected_modelpos_dws,
                        "SelectedRegressorIndex_DWS": selected_reg_idx_dws,
                        "Error_SelectedModel_DWS": selected_error_dws,
                        "Hit_DWS": int(selected_modelpos_dws == best_real_modelpos),
                        "DeltaError_DWS": float(selected_error_dws - best_real_error),
                        "N_Selected_DWS": int(n_selected_dws[i]),

                        # --- DWS (best por competência) ---
                        "BestByCompetencePos_DWS": best_by_comp_dws_pos,
                        "BestByCompetenceRegressorIndex_DWS": best_by_comp_dws_reg,
                        "Error_BestByCompetence_DWS": best_by_comp_dws_err,

                        # --- Diagnóstico do sinal (competência) ---
                        "Competence_Max": c_max,
                        "Competence_Min": c_min,
                        "Competence_Spread": c_spread,

                        # --- NOVO: confiança do sinal (competência) ---
                        "TopByCompetencePos": int(top_by_comp_pos[i]),
                        "MarginCompetence_Diff": float(margin_comp_diff[i]),
                        "MarginCompetence_Ratio": float(margin_comp_ratio[i]),

                        # --- NOVO: confiança da decisão (pesos) ---
                        "MarginWeight_DS_Diff": float(margin_w_ds_diff[i]),
                        "MarginWeight_DS_Ratio": float(margin_w_ds_ratio[i]),
                        "MarginWeight_DW_Diff": float(margin_w_dw_diff[i]),
                        "MarginWeight_DW_Ratio": float(margin_w_dw_ratio[i]),
                        "MarginWeight_DWS_Diff": float(margin_w_dws_diff[i]),
                        "MarginWeight_DWS_Ratio": float(margin_w_dws_ratio[i]),
                    })

    dataset_elapsed = time.time() - dataset_start_time
    dataset_end_time = time.time()

    dataset_times.append({
        "Dataset": dataset,
        "Tempo_Total_Segundos": dataset_elapsed,
        "Num_Execucoes": executions,
        "Num_Folds": folds
    })

    print(f"Tempo total para o dataset {dataset}: {dataset_elapsed:.2f} segundos")

df_results = pd.DataFrame(all_results)
df_kmen = pd.DataFrame(kmen_choices_rows)
df_audit_phase3 = pd.DataFrame(audit_phase3_rows)
df_times = pd.DataFrame(dataset_times)

#df_results.to_csv(phase3_path / "dream_phase3_results.csv", index=False)
#df_kmen.to_csv(phase3_path / "dream_phase3_kmen_choices.csv", index=False)
#df_audit_phase3.to_csv(phase3_path / "dream_phase3_audit_decisions.csv", index=False)
#df_times.to_csv(phase3_path / "dream_execution_times_phase3.csv", index=False)

outputs = {
    "dream_phase3_results": df_results,
    "dream_phase3_kmen_choices": df_kmen,
    "dream_phase3_audit_decisions": df_audit_phase3,
    "dream_execution_times_phase3": df_times,
}

for filename, df in outputs.items():
    # CSV
    df.to_csv(
        phase3_path / f"{filename}.csv",
        index=False
    )

    # Parquet
    #df.to_parquet(
    #    phase3_path / f"{filename}.parquet",
    #    index=False,
    #    engine="pyarrow",
    #    compression="snappy"
    #)

print(f"Resultados Fase 3 salvos em: {phase3_path}")