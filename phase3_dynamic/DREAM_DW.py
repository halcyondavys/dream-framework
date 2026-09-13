import numpy as np
from scipy.special import softmax

def DREAM_DW(predictions, competence_scores, return_debug=False):
    """
    DREAM_DW (Dynamic Weighting) - Versão "Inverse Error Weighting" (MINE-Aligned)
    -----------------------------------------------------------------------------
    A matemática mais robusta para regressão: O peso é proporcional ao inverso do erro.

    Como na Phase 2 já calculamos: competence_scores = 1 / (MSE_local + epsilon),
    aqui basta normalizar esses scores para que somem 1.

    Isso elimina hiperparâmetros como 'temperatura' do Softmax.
    """

    # 1. Proteção numérica (embora Phase 2 já deva tratar)
    # Garante que scores negativos ou zeros absolutos não quebrem a divisão
    scores = np.maximum(competence_scores, 1e-12)

    # 2. Normalização Simples (L1 Norm)
    # W_i = C_i / Sum(C)
    sum_scores = np.sum(scores, axis=1, keepdims=True) + 1e-12
    weights = scores / sum_scores

    # 3. Combinação Linear Ponderada
    # y_final = w1*y1 + w2*y2 + ...
    y_pred = np.sum(predictions * weights, axis=1)

    if not return_debug:
        return y_pred

    debug = {"weights": weights}
    return y_pred, debug