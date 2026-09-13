# -*- coding: utf-8 -*-
"""
Fase 3 (Auxiliar)
-----------------
Gera CSV com MSE individual de cada modelo base selecionado.

Saída:
Dataset | Exec | Fold | IndexModel | MSE
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from main.best_params import get_model_name
from main.configs import datasets_used, datasets_used_reduzido, datasets_synthetic, path_DREAM, num_execucoes, num_folds, models_used, models_fixed_used

# ------------------------------------------------------------------
# Configurações
# ------------------------------------------------------------------
datasets   = datasets_used_reduzido()
executions = num_execucoes()
folds      = num_folds()
models_used = models_used()
#models_used = models_fixed_used()

paths = path_DREAM()

SCRIPT_DIR = Path(__file__).resolve().parent

phase2_path = (SCRIPT_DIR / paths["results_path_phase2"]).resolve()
output_path = (SCRIPT_DIR / paths["results_path_phase3"]).resolve()

output_path.mkdir(parents=True, exist_ok=True)

rows = []

# ------------------------------------------------------------------
# Execução
# ------------------------------------------------------------------
print("\nGerando CSV de erros individuais dos modelos base...\n")

for dataset in datasets:
    for exec_id in range(executions):
        exec_folder = phase2_path / dataset / f"exec_{exec_id}"

        for fold in range(folds):
            npz_file = exec_folder / f"fold_{fold}.npz"

            if not npz_file.exists():
                continue

            data = np.load(npz_file, allow_pickle=True)

            y_test = data["y_test"]
            base_predictions_test = data["base_predictions_test"]
            regressor_indices = data["regressor_indices"]

            # Avaliar cada modelo individualmente
            for pos, model_idx in enumerate(regressor_indices):
                y_pred = base_predictions_test[:, pos]
                mse = mean_squared_error(y_test, y_pred)

                rows.append({
                    "Dataset": dataset,
                    "Exec": exec_id,
                    "Fold": fold,
                    "IndexModel": int(model_idx),
                    "ModelName": get_model_name(models_used[model_idx]),
                    "MSE": float(mse)
                })

# ------------------------------------------------------------------
# Salvar CSV
# ------------------------------------------------------------------
df = pd.DataFrame(rows)
csv_path = output_path / "dream_base_models_mse.csv"
df.to_csv(csv_path, index=False)

print(f"CSV gerado com sucesso em:\n{csv_path}")
