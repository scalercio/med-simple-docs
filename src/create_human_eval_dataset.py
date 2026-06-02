import pandas as pd
import numpy as np

RANDOM_SEED = 42

QWEN_PATH_FLESCH = "data/splits/challenge_hard_flesch.parquet"
GEMMA_PATH_FLESCH = "data/splits/challenge_hard_flesch_gemma4-31b.parquet"
QWEN_PATH_SIM = "data/splits/challenge_hard_sim.parquet"
GEMMA_PATH_SIM = "data/splits/challenge_hard_sim_gemma4-31b.parquet"
HUMAN_EVAL_PATH = "data/splits/challenge_hard_human_eval.parquet"

DROP_COLS = [
    "d_sari",
    "d_sari_add",
    "d_sari_keep",
    "d_sari_del",
    "sim",
    "flesch_original",
    "flesch_paraphrase",
    "flesch_diff",
]


def build_comparison_dataframe(df_qwen, df_gemma, n_medicines=50, random_seed=42):
    rng = np.random.default_rng(random_seed)

    # Garante que os índices estejam alinhados
    df_qwen = df_qwen.reset_index(drop=True)
    df_gemma = df_gemma.reset_index(drop=True)

    assert len(df_qwen) == len(df_gemma), "Os dataframes têm tamanhos diferentes."

    assert df_qwen["pdf_filename"].equals(df_gemma["pdf_filename"]), (
        "Os dataframes não parecem estar na mesma ordem: pdf_filename difere."
    )

    assert df_qwen["medicine_name"].equals(df_gemma["medicine_name"]), (
        "Os dataframes não parecem estar na mesma ordem: medicine_name difere."
    )

    # Escolhe 1 índice por medicine_name
    unique_indices = (
        df_qwen
        .groupby("medicine_name", sort=False)
        .sample(n=1, random_state=random_seed)
        .index
        .to_list()
    )

    n = min(n_medicines, len(unique_indices))

    selected_indices = rng.choice(
        unique_indices,
        size=n,
        replace=False
    )

    rows = []

    for idx in selected_indices:
        qwen_row = {
            "pdf_filename": df_qwen.loc[idx, "pdf_filename"],
            "docs": df_qwen.loc[idx, "informacoes_ao_paciente"],
            "simple_doc": df_qwen.loc[idx, "qwen3.6_27b_simplified"],
            "medicine_name": df_qwen.loc[idx, "medicine_name"],
            "llm": "qwen3.6_27b_simplified",
        }

        gemma_row = {
            "pdf_filename": df_gemma.loc[idx, "pdf_filename"],
            "docs": df_gemma.loc[idx, "informacoes_ao_paciente"],
            "simple_doc": df_gemma.loc[idx, "gemma4_31b_simplified"],
            "medicine_name": df_gemma.loc[idx, "medicine_name"],
            "llm": "gemma4_31b_simplified",
        }

        pair = [qwen_row, gemma_row]
        rng.shuffle(pair)

        rows.extend(pair)

    return pd.DataFrame(
        rows,
        columns=[
            "pdf_filename",
            "docs",
            "simple_doc",
            "medicine_name",
            "llm",
        ]
    )


# =========================
# Carregar arquivos
# =========================

df_qwen = pd.read_parquet(QWEN_PATH_SIM)
df_gemma = pd.read_parquet(GEMMA_PATH_SIM)

df_qwen = df_qwen.drop(columns=DROP_COLS, errors="ignore")
df_gemma = df_gemma.drop(columns=DROP_COLS, errors="ignore")


# =========================
# Criar dataframe comparativo
# =========================

comparison_df_sim = build_comparison_dataframe(
    df_qwen=df_qwen,
    df_gemma=df_gemma,
    n_medicines=50,
    random_seed=RANDOM_SEED,
)

comparison_df_sim["source_file"] = QWEN_PATH_SIM

print(comparison_df_sim.shape)
print(comparison_df_sim.head())

df_qwen = pd.read_parquet(QWEN_PATH_FLESCH)
df_gemma = pd.read_parquet(GEMMA_PATH_FLESCH)

df_qwen = df_qwen.drop(columns=DROP_COLS, errors="ignore")
df_gemma = df_gemma.drop(columns=DROP_COLS, errors="ignore")


# =========================
# Criar dataframe comparativo
# =========================

comparison_df_flesch = build_comparison_dataframe(
    df_qwen=df_qwen,
    df_gemma=df_gemma,
    n_medicines=50,
    random_seed=RANDOM_SEED,
)

comparison_df_flesch["source_file"] = QWEN_PATH_FLESCH

print(comparison_df_flesch.shape)
print(comparison_df_flesch.head())

df_final = pd.concat([comparison_df_sim, comparison_df_flesch], ignore_index=True)

df_final.to_parquet(
    HUMAN_EVAL_PATH,
    index=False
)