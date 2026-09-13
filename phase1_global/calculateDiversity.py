import numpy as np

def calculate_diversity(predictions: np.ndarray,
                        y_true: np.ndarray,
                        epsilon: float = 1e-9,
                        df_quantile: float = 0.80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula três métricas de diversidade por modelo (maior = melhor):

    1) div_corr: diversidade por correlação (1 - média(|corr|) par-a-par)
    2) consensus_var: variância do desvio em relação ao consenso (normalizada em [0,1])
    3) div_df: 1 - taxa de co-falhas (double fault) com limiar adaptativo por quantil do erro²
    """
    # --- Checagens básicas ---
    if predictions.ndim != 2:
        raise ValueError("predictions deve ser 2D (N, M).")
    if y_true.ndim != 1:
        raise ValueError("y_true deve ser 1D (N,).")

    N, M = predictions.shape
    if N != y_true.shape[0]:
        raise ValueError("Dimensão incompatível: predictions.shape[0] != y_true.shape[0].")

    # Caso degenerado: apenas um modelo no pool
    if M == 1:
        # Sem pares para correlação; var do desvio e co-fault contra "outros" não fazem sentido.
        # Definimos diversidade neutra (0.0) para correlação e var; DF = 1 - falha (com limiar global).
        err2           = (predictions[:, 0] - y_true)**2
        thr            = np.quantile(err2, df_quantile)
        cofault        = np.mean(err2 > thr)  # "cofault" vira falha do próprio
        div_corr       = np.array([0.0], dtype=float)
        consensus_var  = np.array([0.0], dtype=float)
        div_df         = np.array([1.0 - cofault], dtype=float)
        return div_corr, consensus_var, div_df

    # ----------------------------------------------------------------
    # 1) Diversidade por correlação: 1 - média(|corr(i,j)|), j != i
    # ----------------------------------------------------------------
    # Padronização por coluna (z-score) para correlação estável
    X        = predictions.astype(float)
    X_center = X - X.mean(axis=0, keepdims=True)
    X_std    = X.std(axis=0, keepdims=True) + epsilon
    Z        = X_center / X_std  # (N, M)

    # Matriz de correlação aproximada: (Z^T Z)/N, com diag=1
    C = (Z.T @ Z) / (N)
    # Remover self-corr na média par-a-par
    C_abs = np.abs(C)
    np.fill_diagonal(C_abs, 0.0)

    # média(|corr|) por modelo com os demais
    mean_abs_corr = C_abs.sum(axis=1) / (M - 1)
    div_corr      = 1.0 - mean_abs_corr
    # Garante faixa [0,1] por segurança numérica
    div_corr = np.clip(div_corr, 0.0, 1.0)

    # ----------------------------------------------------------------
    # 2) Variância do desvio ao consenso: var(yhat_i - mean_{j≠i} yhat_j)
    #    Normalizada por min-max entre modelos para ficar em [0,1]
    # ----------------------------------------------------------------
    # média dos "outros" para cada i: (sum - yhat_i) / (M-1)
    sum_all       = X.sum(axis=1, keepdims=True) # (N,1)
    mean_others   = (sum_all - X) / (M - 1) # (N,M)
    dev           = X - mean_others # (N,M)
    var_per_model = dev.var(axis=0)  # (M,)

    # normalização min-max segura
    vmin, vmax = var_per_model.min(), var_per_model.max()
    if vmax - vmin < epsilon:
        consensus_var = np.zeros_like(var_per_model)
    else:
        consensus_var = (var_per_model - vmin) / (vmax - vmin + epsilon)

    # ----------------------------------------------------------------
    # 3) Double Fault adaptativo:
    #    - Define "falha" por quantil do erro²
    #    - Cofault_i = proporção de instâncias em que (err²_i > thr_i) e (err²_others_mean > thr_global)
    #    - div_df = 1 - cofault_i  (maior = melhor)
    # ----------------------------------------------------------------
    err2 = (X - y_true.reshape(-1, 1))**2  # (N,M)

    # limiar por modelo (quantil) e limiar global (quantil da média dos erros)
    thr_i            = np.quantile(err2, df_quantile, axis=0)         # (M,)
    err2_mean_others = ((err2.sum(axis=1, keepdims=True) - err2) / (M - 1))  # (N,M)
    thr_global       = np.quantile(err2_mean_others, df_quantile)  # escalar

    fault_i = err2 > (thr_i.reshape(1, -1))
    fault_g = err2_mean_others > thr_global
    cofault = (fault_i & fault_g).mean(axis=0)  # (M,)
    div_df  = 1.0 - cofault
    div_df  = np.clip(div_df, 0.0, 1.0)

    return div_corr.astype(float), consensus_var.astype(float), div_df.astype(float)

