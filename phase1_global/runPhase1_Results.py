"""
runPhase1 - Results.py
---------------------------------
Converte os arquivos FINAL.json da Phase 1 do DREAM em CSVs consolidados para análise estatística, gráficos
e escrita da tese/artigos.

Fonte ÚNICA: <dataset>-FINAL.json
"""
from main.configs import path_DREAM
import json
import pandas as pd
from pathlib import Path

# =========================================================
# CONFIGURAÇÕES
# =========================================================
paths = path_DREAM()
BASE_RESULTS_DIR = Path("../results/phase1/")
CSV_OUT_DIR      = BASE_RESULTS_DIR / "csv"
CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# ACUMULADORES
# =========================================================
rows_models = []
rows_metrics = []
rows_weights = []
rows_summary = []

# =========================================================
# LEITURA DOS FINAL.json
# =========================================================
final_files = sorted(BASE_RESULTS_DIR.glob("*-FINAL.json"))

if not final_files:
    raise FileNotFoundError("Nenhum arquivo *-FINAL.json encontrado.")

for file in final_files:
    dataset_name = file.stem.replace("-FINAL", "")
    print(f"📊 Processando dataset: {dataset_name}")

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # =====================================================
    # 1) RESUMO POR DATASET
    # =====================================================
    rows_summary.append({
        "Dataset": dataset_name,
        "SelectionStrategy": data.get("SelectionStrategy"),
        "BestModelIndex": data["BestMedianModel"]["Index"],
        "BestModelName": data["BestMedianModel"]["Name"],
        "QtdModelsSelectedMedian": data.get("QtdModelsSelectedMedian")
    })

    # =====================================================
    # 2) MODELOS (linha por modelo)
    # =====================================================
    for model in data["Models"]:
        rows_models.append({
            "Dataset": dataset_name,
            "ModelName": model["ModelName"],
            "ModelIndex": model["Index"],
            "MedianError": model["MedianError"],
            "MedianVariance": model["MedianVariance"],
            "MedianDiversity": model["MedianDiversity"],
            "MedianConsensusVar": model["MedianConsensusVar"],
            "MedianDF": model["MedianDF"],
            "CombinedScoreMedian": model["CombinedScoreMedian"],
            "SelectedMedian": model["SelectedMedian"]
        })

    # =====================================================
    # 3) MÉTRICAS GLOBAIS (uma linha por dataset)
    # =====================================================
    metrics = data["Metrics"]

    rows_metrics.append({
        "Dataset": dataset_name,
        "MedianError_Min": min(metrics["ErrorsMedian"]),
        "MedianError_Max": max(metrics["ErrorsMedian"]),
        "MedianError_Mean": sum(metrics["ErrorsMedian"]) / len(metrics["ErrorsMedian"]),
        "MedianVariance_Mean": sum(metrics["VarianceMedian"]) / len(metrics["VarianceMedian"]),
        "MedianDiversity_Mean": sum(metrics["DiversityMedian"]) / len(metrics["DiversityMedian"]),
        "MedianConsensusVar": sum(metrics["ConsensusVarMedian"]) / len(metrics["ConsensusVarMedian"]),
        "MedianDF_Mean": sum(metrics["DoubleFaultMedian"]) / len(metrics["DoubleFaultMedian"])
    })

    # =====================================================
    # 4) PESOS OTIMIZADOS (uma linha por dataset)
    # =====================================================
    weights = data.get("Weights")
    if weights:
        rows_weights.append({
            "Dataset": dataset_name,
            "alpha": weights.get("alpha"),
            "beta": weights.get("beta"),
            "gamma": weights.get("gamma"),
            "delta": weights.get("delta"),
            "epsilon": weights.get("epsilon"),
            "sum": weights.get("sum"),
            "OptimizationMethod": weights.get("optimization", {}).get("method"),
            "OptimizationObjective": weights.get("optimization", {}).get("objective"),
            "K_select": weights.get("optimization", {}).get("K_select"),
            "K_select": weights.get("optimization", {}).get("K_select"),
            "optimizer": weights.get("optimization", {}).get("optimizer"),
            "n_trials": weights.get("optimization", {}).get("n_trials"),
            "selection_mode": weights.get("optimization", {}).get("selection_mode")
        })

# =========================================================
# SALVAR CSVs
# =========================================================
df_models = pd.DataFrame(rows_models)
df_metrics = pd.DataFrame(rows_metrics)
df_weights = pd.DataFrame(rows_weights)
df_summary = pd.DataFrame(rows_summary)

df_models.to_csv(CSV_OUT_DIR / "phase1_models_median.csv", index=False)
df_metrics.to_csv(CSV_OUT_DIR / "phase1_metrics_summary.csv", index=False)
df_weights.to_csv(CSV_OUT_DIR / "phase1_weights_per_dataset.csv", index=False)
df_summary.to_csv(CSV_OUT_DIR / "phase1_dataset_summary.csv", index=False)

print("\n✅ CSVs gerados com sucesso:")
print(" - phase1_models_median.csv")
print(" - phase1_metrics_summary.csv")
print(" - phase1_weights_per_dataset.csv")
print(" - phase1_dataset_summary.csv")
