"""
Gerador fatorial paramétrico para o Estudo de Ablação com Datasets Sintéticos
Controlados (DREAM) — versão 3, com F2 redesenhado por famílias funcionais
ortogonais (correção da causa raiz identificada na divergência
Esperado x Observado).

Alterações em relação à versão anterior:
- F2 deixa de interpolar convexamente em direção a uma função comum;
  passa a controlar o grau de deslocamento de cada especialista em
  direção a uma família funcional distinta (trigonométrica de alta
  frequência, polinomial de ordem superior com inflexão, interação
  multiplicativa), preservando maior alinhamento entre o fator
  manipulado e a heterogeneidade de viés indutivo exigida dos modelos
  de regressão do pool.
- F1 (kappa, função de gating) e F3 (m_irrelevantes, n_samples)
  permanecem inalterados — já validados na rodada anterior.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product


# ---------------------------------------------------------------------------
# Função comum e termos especialistas (F2 redesenhado)
# ---------------------------------------------------------------------------

def _h_common(X):
    """Função comum h(x), base compartilhada por todos os especialistas."""
    return np.sin(np.pi * X[:, 0]) + X[:, 1]


def _delta_expert(idx, X):
    """
    Termos de desvio por família funcional distinta:
      idx=0 -> trigonométrica de alta frequência (favorece SVR/MLP)
      idx=1 -> polinomial de ordem superior com inflexão (favorece CART)
      idx=2 -> interação multiplicativa entre atributos (favorece KNN)
    """
    if idx == 0:
        return np.sin(4 * np.pi * X[:, 0]) - X[:, 1] ** 2
    if idx == 1:
        return 4 * (X[:, 0] - 0.5) ** 3
    if idx == 2:
        return 3 * X[:, 0] * X[:, 2] - 1.5 * X[:, 1] * X[:, 2]
    raise ValueError(idx)


def _blended_experts(X, delta):
    """
    f_i^eff = h(x) + delta * Delta_f_i(x)

    delta=0 -> todos os especialistas colapsam em h(x) (baixa dispersão
               de competência real esperada entre os modelos do pool).
    delta=1 -> cada especialista assume integralmente sua família
               funcional distinta (alta dispersão de competência real
               esperada, alinhada a vieses indutivos diferenciados).
    """
    h = _h_common(X)
    D = np.column_stack([_delta_expert(i, X) for i in range(3)])
    return h[:, np.newaxis] + delta * D


def _softmax(z, axis=1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _gating_weights(X, kappa):
    """F1 — inalterado."""
    z = np.column_stack([
        kappa * (X[:, 0] - 0.5),
        -kappa * np.abs(X[:, 1] - 0.5),
        kappa * (X[:, 2] - 0.5),
    ])
    return _softmax(z, axis=1)


def _add_irrelevant_features(X, n_irrelevant, rng):
    """F3a — inalterado."""
    if n_irrelevant <= 0:
        return X
    Z = rng.uniform(0, 1, size=(X.shape[0], n_irrelevant))
    return np.hstack([X, Z])


# ---------------------------------------------------------------------------
# Gerador de célula única
# ---------------------------------------------------------------------------

def make_cell(kappa, delta, n_samples, n_irrelevant=0, noise=0.05,
              random_state=42):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0, 1, size=(n_samples, 3))

    W = _gating_weights(X, kappa)
    F = _blended_experts(X, delta)
    y = np.sum(W * F, axis=1)
    y += rng.normal(0.0, noise, size=n_samples)

    X = _add_irrelevant_features(X, n_irrelevant, rng)
    return X, y


# ---------------------------------------------------------------------------
# Parâmetros por nível de fator — F1 e F3 inalterados
# ---------------------------------------------------------------------------

F1_LEVELS = {"continuo": 1.5, "discreto": 12.0}
F2_LEVELS = {"baixo": 0.3, "alto": 1.0}
F3_LEVELS = {
    "baixo": {"n_samples": 2000, "n_irrelevant": 0},
    "f3a":   {"n_samples": 2000, "n_irrelevant": 15},
    "f3b":   {"n_samples": 300,  "n_irrelevant": 0},
}

EXPECTED_WINNER = {
    ("continuo", "baixo", "baixo"): "DW",
    ("continuo", "baixo", "f3a"):   "DS",
    ("continuo", "baixo", "f3b"):   "DW",
    ("continuo", "alto",  "baixo"): "DW",
    ("continuo", "alto",  "f3a"):   "DWS",
    ("continuo", "alto",  "f3b"):   "DW",
    ("discreto", "baixo", "baixo"): "DWS",
    ("discreto", "baixo", "f3a"):   "DWS",
    ("discreto", "baixo", "f3b"):   "DW",
    ("discreto", "alto",  "baixo"): "DS",
    ("discreto", "alto",  "f3a"):   "DWS",
    ("discreto", "alto",  "f3b"):   "DWS",
}


def save_as_dream_format(X, y, output_path):
    data = np.hstack([y.reshape(-1, 1), X])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, data, delimiter=',', fmt='%.10f')


def generate_all_cells(output_dir, noise=0.05, random_state_base=100):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    idx = 0
    for f1_name, f2_name, f3_name in product(F1_LEVELS, F2_LEVELS, F3_LEVELS):
        idx += 1
        kappa = F1_LEVELS[f1_name]
        delta = F2_LEVELS[f2_name]
        f3_params = F3_LEVELS[f3_name]
        rs = random_state_base + idx

        X, y = make_cell(
            kappa=kappa, delta=delta,
            n_samples=f3_params["n_samples"],
            n_irrelevant=f3_params["n_irrelevant"],
            noise=noise, random_state=rs,
        )

        name = f"C{idx:02d}-{f1_name}-{f2_name}-{f3_name}"
        out_path = output_dir / f"{name}.data"
        save_as_dream_format(X, y, out_path)

        expected = EXPECTED_WINNER[(f1_name, f2_name, f3_name)]
        summary.append({
            "Cell": f"C{idx:02d}", "Dataset": name,
            "F1": f1_name, "kappa": kappa,
            "F2": f2_name, "delta": delta,
            "F3": f3_name,
            "n_samples": f3_params["n_samples"],
            "n_irrelevant": f3_params["n_irrelevant"],
            "random_state": rs,
            "expected_winner": expected,
            "output": str(out_path),
        })
        print(f"[OK] {name:35s} | n={f3_params['n_samples']:5d} "
              f"| m_irr={f3_params['n_irrelevant']:2d} | rs={rs} "
              f"| esperado={expected}")

    log_path = output_dir / "generation_log_fatorial.csv"
    pd.DataFrame(summary).to_csv(log_path, index=False)
    print(f"\nLog salvo em: {log_path}")
    return summary


if __name__ == "__main__":
    generate_all_cells(
        output_dir="../data/synthetic_fatorial",
        noise=0.05,
        random_state_base=100,
    )