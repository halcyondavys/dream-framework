def process_execution(
    execution,
    file_data,
    dataset_name,
    labels,
    GridSearch,
    modelsUsed,
    numberRegressors,
    folds,
    model_save_root
):
    import numpy as np
    from pathlib            import Path
    from buildFolds         import build_folds
    from normalizeMatrix    import normalize_by_fold
    from EnsembleGeneration import ensemble_generation
    from calculateDiversity import calculate_diversity
    import joblib

    # Construir Folds
    train_index, valid_index, test_index = build_folds(labels, folds, 7, 2, 1)

    errors_regressors_list = []
    div_corr_accum = np.zeros((folds, numberRegressors))
    div_var_accum  = np.zeros((folds, numberRegressors))
    div_df_accum   = np.zeros((folds, numberRegressors))

    for fold in range(folds):
        # Máscaras booleanas
        train = train_index[:, fold].astype(bool)
        valid = valid_index[:, fold].astype(bool)
        test  = test_index[:, fold].astype(bool)

        data_train = file_data[train, :]
        data_valid = file_data[valid, :]
        data_test = file_data[test, :]

        # Treinamento do ensemble
        model_pool = ensemble_generation(numberRegressors, data_train, 0, modelsUsed, GridSearch)

        # Salvamento dos modelos treinados
        fold_model_dir = Path(f"{model_save_root}/{dataset_name}/exec_{execution}/fold_{fold}")
        fold_model_dir.mkdir(parents=True, exist_ok=True)

        # Previsões e salvamento dos modelos
        predictions = []
        for model_idx, model in enumerate(model_pool):
            pred = model.predict(data_valid[:, 1:])
            predictions.append(pred)

            # ✅ Salvar modelo treinado
            model_path = fold_model_dir / f"model_{model_idx}.pkl"
            joblib.dump(model, model_path)

        predictions = np.column_stack(predictions)

        # Cálculo do erro local por modelo
        errors_local = (predictions - data_valid[:, 0].reshape(-1, 1)) ** 2
        errors_regressors_list.append(errors_local)

        # Cálculo das métricas de diversidade
        div_corr, div_var, div_df = calculate_diversity(predictions, data_valid[:, 0])
        div_corr_accum[fold] = div_corr # Correlação (cálculo manual com estabilidade numérica)
        div_var_accum[fold]  = div_var # Variância da diferença com os demais
        div_df_accum[fold]   = div_df # Double Fault: erro alto em x e nos demais (> 0.01)

    # Empilhamento dos erros
    errors_regressors_all = np.vstack(errors_regressors_list)
    variance_per_model    = np.var(errors_regressors_all, axis=0)

    return {
        "execution": execution,
        "errors": errors_regressors_all,
        "variances": variance_per_model,
        "diversities": div_corr_accum,
        "var_preds": div_var_accum,
        "df_metrics": div_df_accum,
        "consolidated": {
            "Execution": execution,
            "ErrorsRegressors": errors_regressors_all.tolist(),
            "DiversityCorrelation": div_corr_accum.tolist(),
            "VariancePredictions": div_var_accum.tolist(),
            "DoubleFault": div_df_accum.tolist(),
            "VariancePerModel": variance_per_model.tolist(),
            "TrainIndex": train_index.tolist(),
            "ValidIndex": valid_index.tolist(),
            "TestIndex": test_index.tolist(),
            "Data": file_data.tolist()
        }
    }
