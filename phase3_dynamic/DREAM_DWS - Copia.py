import numpy as np

def DREAM_DWS(predictions, competence_scores, mse_values=None,
              min_relative_threshold=0.8, penalize_by_mse=True, return_debug=False):
    """
    DREAM_DWS inspirado no MINE (Range-Based Selection)
    ---------------------------------------------------
    Seleciona modelos dentro de um intervalo de competência relativo ao spread (Max - Min).

    Parâmetros:
    - min_relative_threshold: (0.0 a 1.0).
      Se 0.1, seleciona modelos nos top 10% da amplitude entre o pior e o melhor.
      Se 0.5, seleciona modelos na metade superior da amplitude (Top 50% do range).
      Valor recomendado para DREAM: 0.2 a 0.5 (Ajuste fino).
    """

    # Proteção
    competence_scores = np.maximum(competence_scores, 1e-12)

    # 1. Penalização Global (Mantemos pois o DREAM é heterogêneo)
    if penalize_by_mse and mse_values is not None:
        mse_norm = (mse_values - np.min(mse_values)) / (np.ptp(mse_values) + 1e-9)
        global_factor = 1.0 - mse_norm
        competence_scores *= (global_factor[np.newaxis, :] ** 0.2)

    # 2. Seleção Baseada em Intervalo (Lógica MINE adaptada para 'Maior é Melhor')
    # C_best = Max Competence
    # C_worst = Min Competence
    c_max = np.max(competence_scores, axis=1, keepdims=True)
    c_min = np.min(competence_scores, axis=1, keepdims=True)

    # Amplitude da competência na instância atual
    spread = c_max - c_min

    # Threshold de corte: Queremos estar no topo do spread.
    # Ex: Se threshold=0.2, aceitamos qualquer score >= Max - (Spread * 0.2)
    # Isso é dinâmico: Se o spread é zero (todos iguais), seleciona todos.
    cutoff_value = c_max - (spread * min_relative_threshold)

    # Máscara de Seleção
    mask = competence_scores >= cutoff_value

    # Fallback (Garante pelo menos o melhor)
    no_selection = np.sum(mask, axis=1) == 0
    if np.any(no_selection):
        best_idx = np.argmax(competence_scores[no_selection], axis=1)
        mask[no_selection, best_idx] = True

    # 3. Ponderação Direta (Igual simpleDW do MINE)
    scores_masked = np.where(mask, competence_scores, 0.0)

    # Normalização dos pesos
    sum_scores = np.sum(scores_masked, axis=1, keepdims=True) + 1e-12
    weights = scores_masked / sum_scores

    # 4. Predição
    y_pred = np.sum(predictions * weights, axis=1)

    if not return_debug:
        return y_pred

    debug = {
        "cutoff": cutoff_value.squeeze(),
        "n_selected": np.sum(mask, axis=1),
        "best_idx": np.argmax(competence_scores, axis=1),
        "mask": mask,
        "weights": weights,
    }
    return y_pred, debug
