# debug_tools.py
"""
Utilitários para construção de blocos estruturados do FINAL.json
Framework DREAM — Phase 1
"""

from typing import List, Dict


def build_weights_block(weights, optimization_meta=None) -> Dict:
    """
    Constrói o bloco de pesos otimizados.
    """
    alpha, beta, gamma, delta, epsilon = weights

    block = {
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "delta": float(delta),
        "epsilon": float(epsilon),
        "sum": float(alpha + beta + gamma + delta + epsilon)
    }

    if optimization_meta is not None:
        block["optimization"] = optimization_meta

    return block


def build_models_block(
    model_names: List[str],
    errors_median,
    variance_median,
    diversity_median,
    consensus_var_median,
    df_median,
    combined_scores,
    selected_models
) -> List[Dict]:
    """
    Constrói a lista de modelos com métricas medianas e decisão final.
    """
    models_info = []

    for i, name in enumerate(model_names):
        models_info.append({
            "ModelName": name,
            "Index": int(i),
            "MedianError": float(errors_median[i]),
            "MedianVariance": float(variance_median[i]),
            "MedianDiversity": float(diversity_median[i]),
            "MedianConsensusVar": float(consensus_var_median[i]),
            "MedianDF": float(df_median[i]),
            "CombinedScoreMedian": float(combined_scores[i]),
            "SelectedMedian": bool(i in selected_models)
        })

    return models_info


def build_metrics_block(
    errors_median,
    variance_median,
    diversity_median,
    consensus_var_median,
    df_median,
    combined_scores
) -> Dict:
    """
    Constrói o bloco de métricas consolidadas (medianas).
    """
    return {
        "ErrorsMedian": errors_median.tolist(),
        "VarianceMedian": variance_median.tolist(),
        "DiversityMedian": diversity_median.tolist(),
        "ConsensusVarMedian": consensus_var_median.tolist(),
        "DoubleFaultMedian": df_median.tolist(),
        "CombinedScoresMedian": combined_scores.tolist()
    }
