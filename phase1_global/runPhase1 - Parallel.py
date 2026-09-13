from main.configs         import *
from loadData             import *
from normalizeMatrix import normalize_matrix
from process_execution    import process_execution
from combination_criteria import combined_score, dynamic_threshold
from joblib               import Parallel, delayed
import logging
import numpy as np
import time
import json
import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

np.random.seed(42)

# Parâmetros
GridSearch        = False
executions       = num_execucoes()
folds            = num_folds()
models_used      = models_used()
datasets_used    = datasets_used_reduzido()

if GridSearch:
    models_used = models_used()
else:
    models_used = models_fixed_used()

numberRegressors  = len(models_used)

# Lista para armazenar resultados
execution_times = []

# Medir o tempo total
start_time = time.time()

results_path_phase1, results_path_phase2, output_csv_path_phase1, output_csv_path_phase2, model_save_root = path_DREAM()

results_path_phase1 = '../main/DREAM/parallel/results/PHASE1/'
model_save_root     = '../main/DREAM/parallel/models_treinados/'

warnings.filterwarnings("ignore", category=ConvergenceWarning)

for dataset_name in datasets_used:

    logger = logging.getLogger(f"phase1_{dataset_name}")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Arquivo de log
    file_handler = logging.FileHandler(f"./logs/phase1_{dataset_name}.log", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

    # Console (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s | %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Iniciando processamento da Fase 1 - Dataset: {dataset_name}")

    dataset_start_time = time.time()

    data, labels = load_data(f'../data/{dataset_name}.data')

    # Normalizar dados manualmente usando Min-Max Scaling
    # Escala os valores das features e dos rótulos para o intervalo [0, 1]
    # A normalização é aplicada coluna por coluna, e evita divisão por zero quando max == min
    data_normalized, labels_normalized = normalize_matrix(data, labels)

    # Concatena os rótulos normalizados como primeira coluna com os dados normalizados
    # Resultado: matriz completa com (label | features)
    file_data = np.concatenate((labels_normalized.reshape(-1, 1), data_normalized), axis=1)

    # Inicializar arrays para acumular métricas
    errors_median_accum = np.zeros((executions, numberRegressors))
    errors_mean_accum   = np.zeros((executions, numberRegressors))
    diversity_accum     = np.zeros((executions, numberRegressors))
    variance_accum      = np.zeros((executions, numberRegressors))
    consolidated_results = []

    # Inicializar listas para acúmulo global
    all_errors_global      = []
    all_variances_global   = []
    all_diversities_global = []
    all_var_preds_global   = []
    all_df_global          = []

    results = Parallel(n_jobs=-1)(
        delayed(process_execution)(
            execution,
            file_data,
            dataset_name,
            labels,
            GridSearch,
            models_used,
            numberRegressors,
            folds,
            model_save_root
        )
        for execution in range(executions)
    )

    for res in results:
        all_errors_global.append(res['errors'])
        all_variances_global.append(res['variances'])
        all_diversities_global.append(res['diversities'])
        all_var_preds_global.append(res['var_preds'])
        all_df_global.append(res['df_metrics'])
        consolidated_results.append(res['consolidated'])

    # Salvar arquivo consolidado
    consolidated_file = f'{results_path_phase1}{dataset_name}-consolidated.json'
    with open(consolidated_file, 'w') as f:
        json.dump(consolidated_results, f, indent=4)

    # --------------------------------------------------------
    # 🔍 Análise Ablation: Média vs. Mediana para as métricas
    # --------------------------------------------------------

    # 🔸 Média
    errors_mean_final       = np.mean(np.vstack(all_errors_global), axis=0)
    variance_mean_final     = np.mean(np.vstack(all_variances_global), axis=0)
    diversity_mean_final    = np.mean(np.vstack(all_diversities_global), axis=0)
    varpreds_mean_final     = np.mean(np.vstack(all_var_preds_global), axis=0)
    double_fault_mean_final = np.mean(np.vstack(all_df_global), axis=0)

    # 🔸 Mediana
    errors_median_final       = np.median(np.vstack(all_errors_global), axis=0)
    variance_median_final     = np.median(np.vstack(all_variances_global), axis=0)
    diversity_median_final    = np.median(np.vstack(all_diversities_global), axis=0)
    varpreds_median_final     = np.median(np.vstack(all_var_preds_global), axis=0)
    double_fault_median_final = np.median(np.vstack(all_df_global), axis=0)

    # Melhor modelo
    index_better_model = int(np.argmin(errors_mean_final))
    better_model_name = models_used[index_better_model]

    # Pesos
    alpha, beta, gamma, delta, epsilon = 0.25, 0.20, 0.25, 0.15, 0.15

    # Scores
    combined_scores_mean = combined_score(
        div_corr=diversity_mean_final,
        var_error=variance_mean_final,
        mse_error=errors_mean_final,
        var_preds=varpreds_mean_final,
        df_score=double_fault_mean_final,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        epsilon=epsilon
    )

    combined_scores_median = combined_score(
        div_corr=diversity_median_final,
        var_error=variance_median_final,
        mse_error=errors_median_final,
        var_preds=varpreds_median_final,
        df_score=double_fault_median_final,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        epsilon=epsilon
    )

    # Determinar dinamicamente o número ideal de modelos
    num_models_to_select_mean   = dynamic_threshold(combined_scores_mean)
    num_models_to_select_median = dynamic_threshold(combined_scores_median)

    # Seleção com base nas melhores pontuações
    selected_mean   = np.argsort(combined_scores_mean)[-num_models_to_select_mean:]
    selected_median = np.argsort(combined_scores_median)[-num_models_to_select_median:]

    # 🔸 Garantir inclusão do melhor modelo (baseado no erro médio real)
    if index_better_model not in selected_mean:
        selected_mean = np.append(selected_mean, index_better_model)
    if index_better_model not in selected_median:
        selected_median = np.append(selected_median, index_better_model)

    all_models_info = [
        {
            "ModelName": models_used[i],
            "Index": int(i),
            "MeanError": float(errors_mean_final[i]),
            "MedianError": float(errors_median_final[i]),
            "MeanVariance":float(variance_mean_final[i]),
            "MedianVariance":float(variance_median_final[i]),
            "MeanDiversity": float(diversity_mean_final[i]),
            "MedianDiversity": float(diversity_median_final[i]),
            "MeanVarPreds": float(varpreds_mean_final[i]),
            "MedianVarPreds": float(varpreds_median_final[i]),
            "MeanDF": float(double_fault_mean_final[i]),
            "MedianDF": float(double_fault_median_final[i]),
            "CombinedScoreMean": float(combined_scores_mean[i]),
            "CombinedScoreMedian": float(combined_scores_median[i]),
            "SelectedMean": i in selected_mean,
            "SelectedMedian": i in selected_median
        }
        for i in range(len(models_used))
    ]

    consolidated_results_final = {
        "index_better_model": index_better_model,
        "better_model_name": better_model_name,
        "NumModelsSelectedMean": len(selected_mean),
        "NumModelsSelectedMedian": len(selected_median),
        "AllModels": all_models_info,
        "Metrics": {
            "ErrorsMean": errors_mean_final.tolist(),
            "ErrorsMedian": errors_median_final.tolist(),
            "VarianceMean": variance_mean_final.tolist(),
            "VarianceMedian": variance_median_final.tolist(),
            "DiversityMean": diversity_mean_final.tolist(),
            "DiversityMedian": diversity_median_final.tolist(),
            "VarPredsMean": varpreds_mean_final.tolist(),
            "VarPredsMedian": varpreds_median_final.tolist(),
            "DoubleFaultMean": double_fault_mean_final.tolist(),
            "DoubleFaultMedian": double_fault_median_final.tolist(),
            "CombinedScoresMean": combined_scores_mean.tolist(),
            "CombinedScoresMedian": combined_scores_median.tolist()
        }
    }

    # Salvar em arquivo JSON
    with open(f'{results_path_phase1}{dataset_name}-FINAL.json', 'w') as f:
        json.dump(consolidated_results_final, f, indent=4)

    logger.info(f"Resultados finais salvos para {dataset_name}")

    dataset_end_time = time.time()
    execution_time = dataset_end_time - dataset_start_time
    execution_times.append({"Dataset": dataset_name, "Tempo de Execução (segundos)": execution_time})

    logger.info(f"Tempo para {dataset_name}: {execution_time:.2f}s")

    # Salvar resultados em um arquivo CSV
    output_file = '../main/DREAM/parallel/results/PHASE1/csv/Dream_Execution_times_phase1.csv'
    output_file1 = results_path_phase1 + 'csv/Dream_Execution_times_phase1.csv'
    df = pd.DataFrame(execution_times)
    df.to_csv(output_file, index=False)

# Tempo total
total_execution_time = time.time() - start_time
execution_times.append({"Dataset": "TOTAL", "Tempo de Execução (segundos)": total_execution_time})

# Salvar tempos em CSV
output_file  = '../main/DREAM/parallel/results/PHASE1/csv/Dream_Execution_times_phase1.csv'
output_file1 = results_path_phase1 + 'csv/Dream_Execution_times_phase1.csv'
df = pd.DataFrame(execution_times)
df.to_csv(output_file, index=False)

logger.info(f"Tempo total da Fase 1: {total_execution_time:.2f}s")
logger.info("Fim da Fase 1")
