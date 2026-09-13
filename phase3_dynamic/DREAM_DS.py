import numpy as np
from scipy.special import softmax

def DREAM_DS(predictions, competence_scores, mse_values=None, dynamic_threshold=0.3, return_debug=False):
    """
    DREAM_DS Range-Based
    Seleciona apenas a 'Nata' da competência.
    dynamic_threshold=0.3 -> Seleciona modelos no Top 30% do range de competência.
    """

    # Penalização Global
    if mse_values is not None:
        mse_norm = (mse_values - np.min(mse_values)) / (np.ptp(mse_values) + 1e-9)
        competence_scores *= (1.0 - mse_norm[np.newaxis, :])

    c_max = np.max(competence_scores, axis=1, keepdims=True)
    c_min = np.min(competence_scores, axis=1, keepdims=True)
    spread = c_max - c_min

    # Corte bem restrito (Top 10% ou 5% do spread)
    cutoff = c_max - (spread * dynamic_threshold)

    mask = competence_scores >= cutoff

    # Fallback
    no_selection = np.sum(mask, axis=1) == 0
    if np.any(no_selection):
        best_idx = np.argmax(competence_scores[no_selection], axis=1)
        mask[no_selection, best_idx] = True

    scores_filtered = np.where(mask, competence_scores, 0.0)
    sum_scores = np.sum(scores_filtered, axis=1, keepdims=True) + 1e-12
    weights = scores_filtered / sum_scores

    y_pred = np.sum(predictions * weights, axis=1)

    if not return_debug:
        return y_pred

    debug = {
        "cutoff": cutoff.squeeze(),
        "n_selected": np.sum(mask, axis=1),
        "best_idx": np.argmax(competence_scores, axis=1),
        "mask": mask,
        "weights": weights,
    }
    return y_pred, debug



