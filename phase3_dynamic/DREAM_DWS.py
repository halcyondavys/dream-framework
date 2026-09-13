import numpy as np

def DREAM_DWS(predictions, competence_scores, min_relative_threshold=0.5, return_debug=False):
    """
    DREAM_DWS — Seleção Dinâmica com Ponderação Uniforme
    -----------------------------------------------------
    Seleciona um subconjunto de modelos dentro de um intervalo relativo de
    competência (Range-Based Selection) e combina suas predições com pesos
    uniformes entre os selecionados.

    Diferenças arquiteturais em relação ao DREAM-DS e DREAM-DW:

        DREAM-DS  → seleciona (λ=0.3) + penalização global MSE + pesos proporcionais à competência ajustada
        DREAM-DW  → todos os modelos                            + pesos proporcionais à competência local
        DREAM-DWS → seleciona (λ=0.5) + sem penalização        + pesos UNIFORMES entre os selecionados

    A ponderação uniforme assume que, dentro da janela de competência
    selecionada, as diferenças entre os modelos são suficientemente pequenas
    para não justificar diferenciação — posição mais conservadora que o DS.

    Parâmetros:
    -----------
    predictions : np.ndarray, shape (n_instances, n_models)
        Predições dos modelos base para cada instância de teste.
    competence_scores : np.ndarray, shape (n_instances, n_models)
        Competências locais estimadas na Fase 2 (c = 1 / (erro + eps)).
        Maior valor = maior competência.
    min_relative_threshold : float in [0.0, 1.0], default=0.5
        Controla a largura do intervalo de seleção.
        cutoff = c_max - lambda * (c_max - c_min)
        λ=0.5 → seleciona modelos na metade superior do range de competência.
        λ=0.3 → mais restritivo (comportamento próximo ao DS sem penalização).
        λ=1.0 → seleciona todos (equivalente ao DW com pesos uniformes).
    return_debug : bool, default=False
        Se True, retorna dicionário de diagnóstico além da predição.

    Retorna:
    --------
    y_pred : np.ndarray, shape (n_instances,)
    debug  : dict (apenas se return_debug=True)
        - cutoff      : limiar de seleção por instância
        - n_selected  : quantidade de modelos selecionados por instância
        - best_idx    : índice do modelo de maior competência por instância
        - mask        : máscara booleana de seleção (n_instances, n_models)
        - weights     : pesos uniformes aplicados (n_instances, n_models)
    """

    # Proteção numérica
    competence_scores = np.maximum(competence_scores, 1e-12)

    # 1. Critério de seleção baseado em intervalo relativo de competência
    c_max = np.max(competence_scores, axis=1, keepdims=True)
    c_min = np.min(competence_scores, axis=1, keepdims=True)
    spread = c_max - c_min

    # cutoff = c_max - λ * spread
    # Modelos com competência >= cutoff são selecionados.
    cutoff_value = c_max - (spread * min_relative_threshold)

    mask = competence_scores >= cutoff_value

    # Fallback: garante ao menos o modelo de maior competência
    no_selection = np.sum(mask, axis=1) == 0
    if np.any(no_selection):
        best_idx = np.argmax(competence_scores[no_selection], axis=1)
        mask[no_selection, best_idx] = True

    # 2. Ponderação uniforme entre os modelos selecionados
    # Todos os modelos dentro da janela recebem o mesmo peso: 1 / n_selecionados
    n_selected = np.sum(mask, axis=1, keepdims=True).astype(float)
    weights = mask.astype(float) / n_selected   # shape (n_instances, n_models)

    # 3. Predição final
    y_pred = np.sum(predictions * weights, axis=1)

    if not return_debug:
        return y_pred

    debug = {
        "cutoff":     cutoff_value.squeeze(),
        "n_selected": np.sum(mask, axis=1),
        "best_idx":   np.argmax(competence_scores, axis=1),
        "mask":       mask,
        "weights":    weights,
    }
    return y_pred, debug