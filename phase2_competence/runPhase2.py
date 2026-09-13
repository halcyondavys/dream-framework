# arquivo: runPhase2.py

from main.configs import *
import time
import pandas as pd
import os
from runPhase2_Competence import runPhase2_Competence
from sklearn.exceptions import ConvergenceWarning
import warnings

print("Executando a Fase 2 - DREAM")

# =========================
# CONFIGURAÇÃO DO K
# =========================
# Opções:
# "fixed"      -> usa k_fixed
# "adaptive_heuristic"   -> usa sqrt(N) se N<=1000, ln(N) se N>1000
# "adaptive_theoretical" -> usa N^(2/(d+2))
K_STRATEGY = "adaptive_heuristic"

K_PARAMS = {
    "k_fixed": 10,
    "k_min": 3,
    "k_max": 20,
    "n_threshold": 1000
}

# Parâmetros - datasets_used_reduzido() ou datasets_used() ou datasets_synthetic()
datasets_used = datasets_used()
executions    = num_execucoes()

# Caminhos dos resultados da Fase 1 (sequencial)
paths = path_DREAM()

results_path_phase1 = paths["results_path_phase1"]
output_csv_path     = paths["output_csv_path_phase2"]

os.makedirs(output_csv_path, exist_ok=True)

execution_times = []

warnings.filterwarnings("ignore", category=ConvergenceWarning)

start_time = time.time()

for dataset in datasets_used:
    dataset_start_time = time.time()

    consolidated_file = f'{results_path_phase1}{dataset}-consolidated.json'
    final_file = f'{results_path_phase1}{dataset}-FINAL.json'

    if not os.path.exists(consolidated_file) or not os.path.exists(final_file):
        print(f"Arquivos não encontrados para {dataset}. Pulando dataset.")
        continue

    print(f'\n==============================')
    print(f'Dataset: {dataset}')
    print(f'==============================')

    for j in range(executions):
        print(f'>> Dataset={dataset} | Execution: {j}')
        runPhase2_Competence(
            dataset                = dataset,
            execution              = j,
            json_final_file        = final_file,
            json_consolidated_file = consolidated_file,
            k_strategy             = K_STRATEGY,
            k_params               = K_PARAMS)

    dataset_end_time = time.time()
    exec_time = dataset_end_time - dataset_start_time
    execution_times.append({"Dataset": dataset, "Tempo de Execução (segundos)": exec_time})

    print(f"\nTempo para processar {dataset}: {exec_time:.2f} segundos")

    # Salvar CSV parcial
    df = pd.DataFrame(execution_times)
    df.to_csv(os.path.join(output_csv_path, 'dream_execution_times_phase2.csv'), index=False)

end_time = time.time()
total_execution_time = end_time - start_time
execution_times.append({"Dataset": "TOTAL", "Tempo de Execução (segundos)": total_execution_time})

print(f"\nTempo total de execução: {total_execution_time:.2f} segundos")

df = pd.DataFrame(execution_times)
df.to_csv(os.path.join(output_csv_path, 'dream_execution_times_phase2.csv'), index=False)

print(f"Resultados salvos em {output_csv_path}")
print("Fim da Fase 2 - DREAM")
