from main.configs            import *
from main.best_params        import *
from loadData                import *
from buildFolds              import build_folds
from normalizeMatrix         import normalize_matrix
from EnsembleGeneration      import ensemble_generation
from calculateDiversity      import calculate_diversity
from combination_criteria    import combined_score, dynamic_threshold, optimize_sdf_weights
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from sklearn.exceptions      import ConvergenceWarning
from pathlib                 import Path

from main.debug_tools import (
    build_weights_block,
    build_models_block,
    build_metrics_block
)

import time
import json
import joblib
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

print("Executando a Fase 1")

# Parâmetros - datasets_used_reduzido() ou datasets_used() ou datasets_synthetic()
GridSearch        = True
Optimized_Weights = True
synthetic_Base    = False
executions        = num_execucoes()
folds             = num_folds()
datasets_used     = datasets_used()

if GridSearch:
    models_used = models_used()
else:
    models_used = models_fixed_used()

numberRegressors  = len(models_used)
model_names       = [get_model_name(m) for m in models_used]

# Lista para armazenar resultados
paths = path_DREAM()

model_save_root     = paths["model_save_root"]
results_path_phase1 = paths["results_path_phase1"]
output_csv_path     = paths["output_csv_path_phase1"]

execution_times = []
start_time = time.time()

for dataset_name in datasets_used:
    dataset_start_time = time.time()

    if synthetic_Base:
        data, labels = load_data(f'../data/synthetic/{dataset_name}.data')
    else:
        data, labels = load_data(f'../data/real/{dataset_name}.data')

    # 1. Carregar perfil estatístico para checar outliers
    #row_stats = df_stats[df_stats['Dataset'] == dataset_name]
    #num_outliers = int(row_stats['num_outliers'].values[0]) if not row_stats.empty else 0
    #ratio_outliers = num_outliers / len(labels)

    # 2. Normalizar dados usando MinMaxScaler ou RobustScaler baseado no perfil estatístico
    #data_normalized, labels_normalized, scaler_x, scaler_y = normalize_matrix_outliers(data, labels, ratio_outliers)
    data_normalized, labels_normalized = normalize_matrix(data, labels)

    # 3. Salvar os scalers para uso futuro (Fase 4)
    #save_path = Path(f"{model_save_root}/{dataset_name}")
    #save_path.mkdir(parents=True, exist_ok=True)  # Garante que a pasta existe

    #joblib.dump(scaler_x, f"{model_save_root}/{dataset_name}/scaler_x.pkl")
    #joblib.dump(scaler_y, f"{model_save_root}/{dataset_name}/scaler_y.pkl")

    #Concatena os rótulos normalizados como primeira coluna com os dados normalizados
    file_data = np.concatenate((labels_normalized.reshape(-1, 1), data_normalized), axis=1)

    # Inicializar listas para acúmulo global
    consolidated_results     = []
    detailed_rows            = []
    all_valid_y_true_global  = []
    all_valid_preds_global   = []
    all_errors_global        = []
    all_variances_global     = []
    all_diversities_global   = []
    all_consensus_var_global = []
    all_df_global            = []

    print(f'Dataset: {dataset_name}')

    if GridSearch:
        # Etapa de ajuste de hiperparâmetros (apenas uma vez)
        print("Ajustando hiperparâmetros globais...")

        # 90% dos dados para RandomSearch
        X_train_tune, _, y_train_tune, _ = train_test_split(
            data_normalized, labels_normalized, test_size=0.1, random_state=42
        )

        best_params_dict = {}
        for model_name in models_used:
            print(f"Buscando hiperparâmetros para {model_name}...")
            params = get_or_create_best_params(dataset_name, model_name, X_train_tune, y_train_tune)
            best_params_dict[model_name] = params
            print(f"✓ {model_name} -> {params}")
    else:
        best_params_dict = {}

    for execution in range(executions):
        print(f'>> Dataset={dataset_name} | Exec={execution + 1}')

        # Construir Folds
        train_index, valid_index, test_index = build_folds(labels, folds, 7, 2, 1)

        errors_regressors_list = []
        div_corr_accum = np.zeros((folds, numberRegressors))
        consensus_var_accum  = np.zeros((folds, numberRegressors))
        div_df_accum   = np.zeros((folds, numberRegressors))

        for fold in range(folds):
            train = train_index[:, fold].astype(bool)
            valid = valid_index[:, fold].astype(bool)
            test  = test_index[:, fold].astype(bool)

            data_train = file_data[train, :]
            data_valid = file_data[valid, :]

            # Treinamento do modelo
            model_pool = ensemble_generation(data_train, models_used, GridSearch, best_params_dict)

            # Criar diretório para salvar os modelos desta execução/fold
            fold_model_dir = Path(f"{model_save_root}/{dataset_name}/exec_{execution}/fold_{fold}")
            fold_model_dir.mkdir(parents=True, exist_ok=True)

            # Gera matriz de previsões dos modelos sobre o conjunto de validação
            # Cada coluna representa as previsões de um modelo
            # Formato resultante: (n_amostras, n_modelos)
            # Previsões e salvamento dos modelos
            predictions = []
            for model_idx, model in enumerate(model_pool):
                pred = model.predict(data_valid[:, 1:])
                predictions.append(pred)

                # ✅ Salvar modelo treinado
                model_path = fold_model_dir / f"model_{model_idx}.pkl"
                joblib.dump(model, model_path)

            predictions = np.column_stack(predictions)

            # Acumula os rótulos verdadeiros do conjunto de validação
            all_valid_y_true_global.append(data_valid[:, 0])

            # Acumula as previsões brutas de todos os modelos no conjunto de validação
            all_valid_preds_global.append(predictions)

            # Calcula o erro quadrático local de cada modelo para cada instância de validação
            errors_local = (predictions - data_valid[:, 0].reshape(-1, 1)) ** 2
            errors_regressors_list.append(errors_local)

            # salvar erro médio por fold (para análise estatística futura)
            for model_idx in range(numberRegressors):
                mse_fold = np.mean(errors_local[:, model_idx])

                detailed_rows.append({
                    "Dataset": dataset_name,
                    "Execution": execution,
                    "Fold": fold,
                    "Model_ID": model_idx,
                    "ModelName": get_model_name(models_used[model_idx]),
                    "MSE": mse_fold
                })

            # Calcular diversidade e acumular por modelo
            div_corr, consensus_var, div_df = calculate_diversity(predictions, data_valid[:, 0])
            div_corr_accum[fold]       = div_corr
            consensus_var_accum[fold]  = consensus_var
            div_df_accum[fold]         = div_df

        # End Folders da Execução[i] do DataSet[j]

        # Métricas por execução (considerando todos os folds)
        # Agrupa todos os erros quadráticos dos modelos em todos os folds de validação
        errors_regressors_all = np.vstack(errors_regressors_list)
        variance_per_model    = np.var(errors_regressors_all, axis=0)

        # Armazenar para uso global na ablation
        all_errors_global.append(errors_regressors_all)
        all_variances_global.append(variance_per_model)
        all_diversities_global.append(div_corr_accum)
        all_consensus_var_global.append(consensus_var_accum)
        all_df_global.append(div_df_accum)

        consolidated_results.append({
            "Execution": execution,
            "ErrorsRegressors": errors_regressors_all.tolist(),
            "DiversityCorrelation": div_corr_accum.tolist(),
            "ConsensusVariance": consensus_var_accum.tolist(),
            "DoubleFault": div_df_accum.tolist(),
            "VariancePerModel": variance_per_model.tolist(),
            "TrainIndex": train_index.tolist(),
            "ValidIndex": valid_index.tolist(),
            "TestIndex": test_index.tolist(),
            "Data": file_data.tolist()
        })

    # End das 20 Execução do DataSet[i]

    # Salvar CSV detalhado por fold (análise estatística)
    df_detail = pd.DataFrame(detailed_rows)
    df_detail.to_csv(f"{results_path_phase1}{dataset_name}-phase1-detailed-fold-errors.csv", index=False)

    # Salvar arquivo consolidado
    consolidated_file = f'{results_path_phase1}{dataset_name}-consolidated.json'
    with open(consolidated_file, 'w') as f:
        json.dump(consolidated_results, f)

    # 🔸 Mediana
    errors_median_final        = np.median(np.vstack(all_errors_global), axis=0)
    variance_median_final      = np.median(np.vstack(all_variances_global), axis=0)
    diversity_median_final     = np.median(np.vstack(all_diversities_global), axis=0)
    consensus_var_median_final = np.median(np.vstack(all_consensus_var_global), axis=0)
    double_fault_median_final  = np.median(np.vstack(all_df_global), axis=0)

    # Melhor modelo
    index_better_median_model = int(np.argmin(errors_median_final))
    better_median_model_name  = get_model_name(models_used[index_better_median_model])

    # ----------------------------------------------------------------
    # 🔍 CONSOLIDAÇÃO FINAL PARA OTIMIZAÇÃO DE PESOS
    # ----------------------------------------------------------------

    # Consolida rótulos verdadeiros de todas as execuções/folds
    y_true_validation_all = np.hstack(all_valid_y_true_global)

    # Consolida previsões de todos os modelos em todas as execuções/folds
    # Formato resultante: (Total de Amostras de Validação, Número de Modelos)
    predictions_validation_all = np.vstack(all_valid_preds_global)

    norm = MinMaxScaler()
    div_norm = norm.fit_transform(diversity_median_final.reshape(-1, 1)).ravel()
    var_norm = norm.fit_transform(variance_median_final.reshape(-1, 1)).ravel()
    mse_norm = norm.fit_transform(errors_median_final.reshape(-1, 1)).ravel()
    consensus_var_norm = norm.fit_transform(consensus_var_median_final.reshape(-1, 1)).ravel()
    df_norm = norm.fit_transform(double_fault_median_final.reshape(-1, 1)).ravel()

    if Optimized_Weights:

        # --- Criando o Dicionário de Otimização ---
        metrics_data_optimization = {
            'y_true_validation': y_true_validation_all,  # Rótulos de validação de todas as amostras
            'preds_validation': predictions_validation_all,  # Previsões dos modelos para essas amostras
            'div_corr': div_norm,
            'var_error': var_norm,
            'mse_error': mse_norm,
            'consensus_var': consensus_var_norm,
            'df_score': df_norm
        }

        # Chamada da Otimização
        #method = "grid-search", "optuna", "ga"
        method = "optuna"
        #objective = "minimize ensemble MSE", "minimize ensemble MSE using dynamic threshold"
        objective = "minimize ensemble MSE using dynamic threshold"
        K_FIXO = 5  # Exemplo: buscar os melhores pesos para um ensemble de 5 modelos

        best_weights, best_mse = optimize_sdf_weights(
            metrics_data=metrics_data_optimization,
            K_select=K_FIXO,
            method=method,
            n_trials=200,
            random_state=42,
            selection_mode="dynamic_threshold"
        )

        # Usar os pesos otimizados na seleção do ensemble:
        alpha, beta, gamma, delta, epsilon = best_weights
    else:
        # Pesos Fixos
        method = "Fixo"
        objective = "None"
        K_FIXO = 0
        alpha, beta, gamma, delta, epsilon = 0.25, 0.20, 0.25, 0.15, 0.15
        best_weights = alpha, beta, gamma, delta, epsilon

    # Scores
    combined_scores_median = combined_score(
        div_corr=div_norm,
        var_error=var_norm,
        mse_error=mse_norm,
        consensus_var=consensus_var_norm,
        df_score=df_norm,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        epsilon=epsilon
    )

    # Determinar dinamicamente o número ideal de modelos
    num_models_to_select_median = dynamic_threshold(combined_scores_median)

    # Seleção com base nas melhores pontuações
    selected_median = np.argsort(combined_scores_median)[-num_models_to_select_median:]

    # 🔸 Garantir inclusão do melhor modelo (baseado no erro mediano real)
    if index_better_median_model not in selected_median:
        selected_median = np.append(selected_median, index_better_median_model)

    # -------------------------------------------------------------
    # 🔸 Construção FINAL.json (FONTE ÚNICA)
    # -------------------------------------------------------------
    consolidated_results_final = {
        "SelectionStrategy": "Median-based",
        "BestMedianModel": {
            "Index": index_better_median_model,
            "Name": better_median_model_name
        },
        "QtdModelsSelectedMedian": int(len(selected_median)),

        "Weights": build_weights_block(
            best_weights,
            optimization_meta={
                "method": method,
                "objective": objective,
                "K_select": K_FIXO,
                "optimizer": method,
                "n_trials": 200,
                "selection_mode": "dynamic_threshold",
                "random_state": 42
            }
        ),

        "Models": build_models_block(
            model_names      = model_names,
            errors_median    = errors_median_final,
            variance_median  = variance_median_final,
            diversity_median = diversity_median_final,
            consensus_var_median  = consensus_var_median_final,
            df_median        = double_fault_median_final,
            combined_scores  = combined_scores_median,
            selected_models  = selected_median
        ),

        "Metrics": build_metrics_block(
            errors_median    = errors_median_final,
            variance_median  = variance_median_final,
            diversity_median = diversity_median_final,
            consensus_var_median  = consensus_var_median_final,
            df_median        = double_fault_median_final,
            combined_scores  = combined_scores_median
        )
    }

    # Salvar em arquivo JSON
    with open(f'{results_path_phase1}{dataset_name}-FINAL.json', 'w') as f:
        json.dump(consolidated_results_final, f)

    print(f"Arquivo FINAL.json salvo para o dataset {dataset_name}!")

    dataset_end_time = time.time()
    execution_time = dataset_end_time - dataset_start_time
    execution_times.append({"Dataset": dataset_name, "Tempo de Execução (segundos)": execution_time})

    print(f"Tempo para processar {dataset_name}: {execution_time:.2f} segundos")

    # Salvar resultados em um arquivo CSV
    df = pd.DataFrame(execution_times)
    df.to_csv(os.path.join(output_csv_path, 'Dream_Execution_times_phase1.csv'), index=False)

# Tempo total
total_execution_time = time.time() - start_time
execution_times.append({"Dataset": "TOTAL", "Tempo de Execução (segundos)": total_execution_time})

# Salvar tempos em CSV
df = pd.DataFrame(execution_times)
df.to_csv(os.path.join(output_csv_path, 'Dream_Execution_times_phase1.csv'), index=False)

print(f"Tempo total de execução: {total_execution_time:.2f} segundos")
print("Fim da Fase 1")
