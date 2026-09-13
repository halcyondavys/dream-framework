from pathlib import Path
import re
import math
import pandas as pd


# ============================================================
# Configurações
# ============================================================
input_path = Path(
    "../results/PHASE2/parquet/phase2_competence_audit.parquet"
)

output_dir = Path(
    "../results/PHASE2/parquet/phase2_competence_audit_parts"
)

MAX_MB = 40
SAFETY_FACTOR = 0.85  # margem para evitar arquivos próximos demais de 40 MB

output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Funções auxiliares
# ============================================================

def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def sanitize_filename(value) -> str:
    value = str(value)
    value = re.sub(r"[^\w\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def save_parquet(df: pd.DataFrame, path: Path):
    df.to_parquet(
        path,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )


# ============================================================
# Leitura do arquivo original
# ============================================================

df = pd.read_parquet(input_path, engine="pyarrow")

if "Dataset" not in df.columns:
    raise ValueError(
        "A coluna 'Dataset' não foi encontrada no arquivo Parquet. "
        "Verifique o nome correto da coluna antes de particionar."
    )


# ============================================================
# Particionamento por Dataset e, se necessário, por partes
# ============================================================

created_files = []

for dataset_name, df_dataset in df.groupby("Dataset", sort=True):
    dataset_safe = sanitize_filename(dataset_name)

    base_filename = f"phase2_competence_audit__dataset={dataset_safe}"

    dataset_path = output_dir / f"{base_filename}.parquet"

    # Primeiro tenta salvar o dataset inteiro
    save_parquet(df_dataset, dataset_path)

    dataset_size = file_size_mb(dataset_path)

    # Se ficou abaixo do limite, mantém o arquivo
    if dataset_size <= MAX_MB:
        created_files.append({
            "Dataset": dataset_name,
            "Arquivo": dataset_path.name,
            "Linhas": len(df_dataset),
            "Tamanho_MB": dataset_size
        })
        continue

    # Se ficou acima do limite, remove e divide em partes menores
    dataset_path.unlink()

    total_rows = len(df_dataset)

    estimated_rows_per_part = int(
        total_rows * ((MAX_MB * SAFETY_FACTOR) / dataset_size)
    )

    rows_per_part = max(1, estimated_rows_per_part)

    part_number = 1

    for start in range(0, total_rows, rows_per_part):
        end = start + rows_per_part

        df_part = df_dataset.iloc[start:end]

        part_path = output_dir / (
            f"{base_filename}__part={part_number:03d}.parquet"
        )

        save_parquet(df_part, part_path)

        created_files.append({
            "Dataset": dataset_name,
            "Arquivo": part_path.name,
            "Linhas": len(df_part),
            "Tamanho_MB": file_size_mb(part_path)
        })

        part_number += 1


# ============================================================
# Relatório dos arquivos gerados
# ============================================================

df_report = pd.DataFrame(created_files)

report_path = output_dir / "partition_report.csv"

df_report.to_csv(report_path, index=False, encoding="utf-8-sig")

print("Particionamento concluído.")
print(f"Pasta de saída: {output_dir}")
print(f"Relatório: {report_path}")
print()
print(df_report.sort_values(["Dataset", "Arquivo"]))
print()
print("Maior arquivo gerado:")
print(df_report.loc[df_report["Tamanho_MB"].idxmax()])