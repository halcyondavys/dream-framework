# arquivo: competence_scores.py
import numpy as np
from knn import knn

def calculate_dream_rmse_matrix(
    X_ref,
    X_query,
    pred_ref_cv,
    y_ref,
    k=10,
    return_neighbors=False
):
    """
    Matriz de erro local utilizada pelo DREAM.
    RMSE ponderado por distância no espaço KNN
    """

    n_query  = X_query.shape[0]
    n_models = pred_ref_cv.shape[1]

    error_matrix = np.zeros((n_query, n_models))

    neighbors_idx  = None
    neighbors_dist = None

    if return_neighbors:
        neighbors_idx  = np.zeros((n_query, k), dtype=int)
        neighbors_dist = np.zeros((n_query, k), dtype=float)

    for i in range(n_query):
        # KNN exatamente como no MINE
        distances, indices = knn(X_ref, X_query[i], k)

        if return_neighbors:
            # se por algum motivo vier menos que k (dataset pequeno), preencha só o que existe
            kk = len(indices)
            neighbors_idx[i, :kk] = indices
            neighbors_dist[i, :kk] = distances

        # Pesos por distância (IDW implícito)
        # Evita divisão por zero com epsilon
        dist_safe = distances + 1e-12
        inv_dist  = 1.0 / dist_safe

        # Normaliza para que a soma dos pesos seja 1
        dist_w = inv_dist / np.sum(inv_dist)

        # Dados da vizinhança
        val_targets = y_ref[indices]           # (K,)
        pred_vals   = pred_ref_cv[indices, :]  # (K, n_models)

        # Erro quadrático ponderado por distância
        # (K, 1) - (K, n_models) -> (K, n_models)
        sq_error = (val_targets[:, None] - pred_vals) ** 2

        # 5. Aplica a ponderação
        # Multiplica cada erro pelo peso do vizinho correspondente
        weighted_sq_error = sq_error * dist_w[:, None]

        # RMSE local
        error_matrix[i, :] = np.sqrt(
            np.sum(weighted_sq_error, axis=0)
        )

    if return_neighbors:
        return error_matrix, neighbors_idx, neighbors_dist

    return error_matrix
