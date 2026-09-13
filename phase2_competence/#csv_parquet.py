from pathlib import Path
import re
import pandas as pd


# ============================================================
# Configurações
# ============================================================

input_path = Path(
    "../results/PHASE2/csv/phase2_competence_audit.csv"
)

output_dir = Path(
    "../results/PHASE2/parquet/phase2_competence_audit_parts"
)

MAX_MB = 40
SAFETY_FACTOR = 0.85  # margem para evitar arquivos muito próximos de 40 MB

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
# Leitura do arquivo CSV original
# ============================================================

df = pd.read_csv(
    input_path,
    low_memory=False
)

if "Dataset" not in df.columns:
    raise ValueError(
        "A coluna 'Dataset' não foi encontrada no arquivo CSV. "
        "Verifique o nome correto da coluna antes de particionar."
    )


# ============================================================
# Particionamento por Dataset e, se necessário, por partes
# ============================================================

created_files = []

for dataset_name, df_dataset in df.groupby("Dataset", sort=True):
    dataset_safe = sanitize_filename(dataset_name)

    base_filename = f"phase2_competence_audit_k10__dataset={dataset_safe}"

    dataset_path = output_dir / f"{base_filename}.parquet"

    # Primeiro tenta salvar o dataset inteiro
    save_parquet(df_dataset, dataset_path)

    dataset_size = file_size_mb(dataset_path)

    # Se o arquivo do dataset ficou abaixo do limite, mantém
    if dataset_size <= MAX_MB:
        created_files.append({
            "Dataset": dataset_name,
            "Arquivo": dataset_path.name,
            "Linhas": len(df_dataset),
            "Tamanho_MB": round(dataset_size, 2)
        })
        continue

    # Se o dataset ficou acima de 40 MB, remove e divide em partes menores
    dataset_path.unlink()

    total_rows = len(df_dataset)

    estimated_rows_per_part = int(
        total_rows * ((MAX_MB * SAFETY_FACTOR) / dataset_size)
    )

    rows_per_part = max(1, estimated_rows_per_part)

    start = 0
    part_number = 1

    while start < total_rows:
        current_rows = min(rows_per_part, total_rows - start)

        while True:
            end = start + current_rows

            df_part = df_dataset.iloc[start:end]

            part_path = output_dir / (
                f"{base_filename}__part={part_number:03d}.parquet"
            )

            save_parquet(df_part, part_path)

            part_size = file_size_mb(part_path)

            # Se a parte ficou dentro do limite, mantém
            if part_size <= MAX_MB:
                created_files.append({
                    "Dataset": dataset_name,
                    "Arquivo": part_path.name,
                    "Linhas": len(df_part),
                    "Tamanho_MB": round(part_size, 2)
                })

                start = end
                part_number += 1
                break

            # Se uma única linha já ultrapassar 40 MB, não há como dividir mais
            if current_rows == 1:
                created_files.append({
                    "Dataset": dataset_name,
                    "Arquivo": part_path.name,
                    "Linhas": len(df_part),
                    "Tamanho_MB": round(part_size, 2),
                    "Aviso": "Arquivo acima de 40 MB mesmo com apenas 1 linha"
                })

                start = end
                part_number += 1
                break

            # Se ficou acima do limite, remove e reduz o número de linhas da parte
            part_path.unlink()

            new_rows = int(
                current_rows * ((MAX_MB * SAFETY_FACTOR) / part_size)
            )

            if new_rows >= current_rows:
                new_rows = current_rows // 2

            current_rows = max(1, new_rows)


# ============================================================
# Relatório dos arquivos gerados
# ============================================================

df_report = pd.DataFrame(created_files)

report_path = output_dir / "partition_report_k10.csv"

df_report.to_csv(
    report_path,
    index=False,
    encoding="utf-8-sig"
)

print("Particionamento concluído.")
print(f"Pasta de saída: {output_dir}")
print(f"Relatório: {report_path}")
print()

print(df_report.sort_values(["Dataset", "Arquivo"]))

print()
print("Maior arquivo gerado:")
print(df_report.loc[df_report["Tamanho_MB"].idxmax()])