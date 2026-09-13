# arquivo: extractPhase3LocalExample.py
# -*- coding: utf-8 -*-
"""Extrai uma decisão local da Fase 3 do DREAM sem alterar resultados oficiais.

O script carrega um único artefato PHASE2.npz, executa DREAM-DS, DREAM-DW e
DREAM-DWS para o fold informado e salva somente o detalhamento da instância
escolhida. Nenhum CSV consolidado da Fase 3 é sobrescrito.

Exemplo de uso:
    python extractPhase3LocalExample.py

Outro recorte:
    python extractPhase3LocalExample.py \
        --dataset abalone --exec 0 --fold 0 --test-idx 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

# Correspondência validada no arquivo phase1_models_median.csv.
MODEL_NAMES = {
    0: "cart",
    1: "linear",
    2: "mlp",
    3: "svr",
    4: "knn",
    5: "rbf",
    6: "elm",
    7: "ridge",
    8: "huber",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extrai, para auditoria, os dados de uma decisão local da Fase 3 "
            "sem sobrescrever os resultados consolidados."
        )
    )
    parser.add_argument("--dataset", default="energy_efficiency")
    parser.add_argument("--exec", dest="exec_id", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--test-idx", type=int, default=3)
    parser.add_argument(
        "--npz-file",
        type=Path,
        default=None,
        help="Caminho explícito do PHASE2.npz. Se omitido, usa main.configs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Arquivo CSV de saída. Se omitido, salva na pasta da Fase 3.",
    )
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="Exibe os campos disponíveis no PHASE2.npz.",
    )
    return parser.parse_args()


def resolve_paths(
    args: argparse.Namespace,
    paths: dict[str, str],
) -> tuple[Path, Path]:
    phase2_path = (SCRIPT_DIR / paths["results_path_phase2"]).resolve()
    phase3_path = (SCRIPT_DIR / paths["results_path_phase3"]).resolve()

    if args.npz_file is not None:
        npz_file = args.npz_file.expanduser().resolve()
    else:
        npz_file = (
            phase2_path
            / args.dataset
            / f"exec_{args.exec_id}"
            / f"fold_{args.fold}.npz"
        )

    if args.output is not None:
        output_file = args.output.expanduser().resolve()
    else:
        filename = (
            f"dream_phase3_audit_local_{args.dataset}"
            f"_exec{args.exec_id}_fold{args.fold}_idx{args.test_idx}.csv"
        )
        output_file = phase3_path / filename

    return npz_file, output_file


def require_fields(data: Any, required: Iterable[str]) -> None:
    missing = [field for field in required if field not in data.files]
    if missing:
        raise KeyError(
            "Campos obrigatórios ausentes no PHASE2.npz: "
            + ", ".join(missing)
        )


def optional_instance_value(
    data: Any,
    candidates: Iterable[str],
    test_idx: int,
) -> float | int | None:
    """Obtém um campo opcional escalar ou indexado por instância."""
    available = {name.lower(): name for name in data.files}

    for candidate in candidates:
        actual_name = available.get(candidate.lower())
        if actual_name is None:
            continue

        value = np.asarray(data[actual_name])

        if value.ndim == 0:
            return value.item()

        if value.size == 1:
            return value.reshape(-1)[0].item()

        if value.shape[0] > test_idx:
            selected = np.asarray(value[test_idx])
            if selected.size == 1:
                return selected.reshape(-1)[0].item()

    return None


def validate_shapes(
    y_test: np.ndarray,
    predictions: np.ndarray,
    local_errors: np.ndarray,
    regressor_indices: np.ndarray,
    test_idx: int,
) -> None:
    if predictions.ndim != 2:
        raise ValueError("base_predictions_test deve possuir duas dimensões.")

    if local_errors.shape != predictions.shape:
        raise ValueError(
            "dream_error_test e base_predictions_test possuem formas diferentes: "
            f"{local_errors.shape} versus {predictions.shape}."
        )

    if predictions.shape[0] != len(y_test):
        raise ValueError("Número de previsões diferente do tamanho de y_test.")

    if predictions.shape[1] != len(regressor_indices):
        raise ValueError(
            "Número de colunas de previsão diferente do número de regressores."
        )

    if not 0 <= test_idx < len(y_test):
        raise IndexError(
            f"test_idx={test_idx} inválido. O fold possui {len(y_test)} instâncias, "
            f"com índices de 0 a {len(y_test) - 1}."
        )


def top_two(values: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, dtype=float))[::-1]
    if len(ordered) == 1:
        return float(ordered[0]), float("nan")
    return float(ordered[0]), float(ordered[1])


def main() -> None:
    args = parse_args()

    # Imports locais para que ``--help`` funcione mesmo quando o arquivo ainda
    # não foi copiado para a pasta do projeto DREAM.
    try:
        from main.configs import path_DREAM
        from DREAM_DS import DREAM_DS
        from DREAM_DW import DREAM_DW
        from DREAM_DWS import DREAM_DWS
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Execute este script na mesma pasta do runPhase3.py, dentro do "
            "projeto DREAM, para que main.configs e os módulos DREAM_DS, "
            "DREAM_DW e DREAM_DWS possam ser importados."
        ) from exc

    npz_file, output_file = resolve_paths(args, path_DREAM())

    if not npz_file.exists():
        raise FileNotFoundError(f"Arquivo PHASE2.npz não encontrado: {npz_file}")

    with np.load(npz_file, allow_pickle=True) as data:
        require_fields(
            data,
            [
                "y_train",
                "y_test",
                "base_predictions_test",
                "dream_error_test",
                "mse_train_models",
                "regressor_indices",
            ],
        )

        if args.list_fields:
            print("Campos disponíveis no PHASE2.npz:")
            for field in data.files:
                print(f"  - {field}")

        y_train = np.asarray(data["y_train"]).reshape(-1)
        y_test = np.asarray(data["y_test"]).reshape(-1)
        predictions = np.asarray(data["base_predictions_test"], dtype=float)
        local_errors = np.asarray(data["dream_error_test"], dtype=float)
        mse_train_models = np.asarray(data["mse_train_models"], dtype=float)
        regressor_indices = np.asarray(data["regressor_indices"], dtype=int)

        validate_shapes(
            y_test,
            predictions,
            local_errors,
            regressor_indices,
            args.test_idx,
        )

        competence = 1.0 / (local_errors + 1e-12)

        y_ds, debug_ds = DREAM_DS(
            predictions,
            competence.copy(),
            mse_train_models,
            dynamic_threshold=0.3,
            return_debug=True,
        )
        y_dw, debug_dw = DREAM_DW(
            predictions,
            competence.copy(),
            return_debug=True,
        )
        y_dws, debug_dws = DREAM_DWS(
            predictions,
            competence.copy(),
            min_relative_threshold=0.5,
            return_debug=True,
        )

        weights_ds = np.asarray(debug_ds["weights"], dtype=float)
        weights_dw = np.asarray(debug_dw["weights"], dtype=float)
        weights_dws = np.asarray(debug_dws["weights"], dtype=float)

        i = args.test_idx
        y_true = float(y_test[i])
        competence_row = competence[i]
        c1, c2 = top_two(competence_row)

        k_value = optional_instance_value(
            data,
            ["K_Value", "K_Effective", "k_value", "k_effective", "K"],
            i,
        )
        mean_knn_dist = optional_instance_value(
            data,
            ["MeanKnnDist", "mean_knn_dist", "knn_mean_distance"],
            i,
        )
        median_knn_dist = optional_instance_value(
            data,
            ["MedianKnnDist", "median_knn_dist", "knn_median_distance"],
            i,
        )
        std_knn_dist = optional_instance_value(
            data,
            ["StdKnnDist", "std_knn_dist", "knn_std_distance"],
            i,
        )

        n_selected_ds = debug_ds.get("n_selected")
        n_selected_dws = debug_dws.get("n_selected")
        if n_selected_ds is None:
            n_selected_ds_value = int(np.sum(weights_ds[i] > 0))
        else:
            n_selected_ds_value = int(np.asarray(n_selected_ds)[i])

        if n_selected_dws is None:
            n_selected_dws_value = int(np.sum(weights_dws[i] > 0))
        else:
            n_selected_dws_value = int(np.asarray(n_selected_dws)[i])

        common = {
            "Dataset": args.dataset,
            "Exec": args.exec_id,
            "Fold": args.fold,
            "TestIdx": i,
            "Y_True": y_true,
            "Prediction_DS": float(y_ds[i]),
            "Prediction_DW": float(y_dw[i]),
            "Prediction_DWS": float(y_dws[i]),
            "AbsoluteError_DS": float(abs(y_true - y_ds[i])),
            "AbsoluteError_DW": float(abs(y_true - y_dw[i])),
            "AbsoluteError_DWS": float(abs(y_true - y_dws[i])),
            "N_Selected_DS": n_selected_ds_value,
            "N_Selected_DWS": n_selected_dws_value,
            "Competence_Max": float(np.max(competence_row)),
            "Competence_Min": float(np.min(competence_row)),
            "Competence_Spread": float(np.ptp(competence_row)),
            "MarginCompetence_Diff": float(c1 - c2),
            "MarginCompetence_Ratio": float(c1 / (c2 + 1e-12)),
            "K_Value": k_value,
            "MeanKnnDist": mean_knn_dist,
            "MedianKnnDist": median_knn_dist,
            "StdKnnDist": std_knn_dist,
            "N_Reference": int(len(y_train)),
        }

        rows = []
        for model_pos, regressor_index in enumerate(regressor_indices):
            prediction = float(predictions[i, model_pos])
            rows.append(
                {
                    **common,
                    "ModelPos": int(model_pos),
                    "RegressorIndex": int(regressor_index),
                    "ModelName": MODEL_NAMES.get(
                        int(regressor_index),
                        f"model_{int(regressor_index)}",
                    ),
                    "Prediction_Model": prediction,
                    "AbsoluteError_Model": float(abs(y_true - prediction)),
                    "EstimatedLocalRMSE": float(local_errors[i, model_pos]),
                    "Competence": float(competence_row[model_pos]),
                    "Weight_DS": float(weights_ds[i, model_pos]),
                    "Weight_DW": float(weights_dw[i, model_pos]),
                    "Weight_DWS": float(weights_dws[i, model_pos]),
                    "Selected_DS": int(weights_ds[i, model_pos] > 0),
                    "Selected_DWS": int(weights_dws[i, model_pos] > 0),
                }
            )

    result = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)

    print(f"Arquivo de origem: {npz_file}")
    print(f"Arquivo gerado: {output_file}")
    print(f"Modelos registrados: {len(result)}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
