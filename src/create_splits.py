import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RANDOM_SEED = 42

INPUT_PARQUET = "simplifications/final_reprocess_tamarine_qwen3.6_27b_simplified_v2.parquet.final"
OUT_DIR = "data/splits"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_parquet(INPUT_PARQUET)

df = df.drop(
    columns=[
        "bula_simplificada_1",
        "bula_simplificada_2",
        "bula_simplificada_3",
    ],
    errors="ignore"  # evita erro se alguma coluna não existir
)
# Nome do medicamento: parte antes de "__"
import unicodedata

def normalize_medicine_name(text):
    text = text.strip().upper()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    return text

df["medicine_name"] = (
    df["pdf_filename"]
    .str.split("__")
    .str[0]
    .apply(normalize_medicine_name)
)

print("Medicamentos únicos:", df["medicine_name"].nunique())


# =========================
# 1. Criar challenge sets
# =========================

sim_p90 = df["sim"].quantile(0.9)
flesch_p90 = df["flesch_diff"].quantile(0.9)

sim_p10 = df["sim"].quantile(0.1)
flesch_p10 = df["flesch_diff"].quantile(0.1)

good_candidates_sim = df[
    (df["sim"] >= sim_p90) 
].copy()

good_candidates_flesch = df[
    (df["flesch_diff"] >= flesch_p90) 
].copy()

hard_candidates_sim = df[
    (df["sim"] <= sim_p10) 
].copy()

hard_candidates_flesch = df[
    (df["flesch_diff"] <= flesch_p10)
].copy()

good_meds_sim = (
    good_candidates_sim["medicine_name"]
    .drop_duplicates()
    .sample(
        n=min(200, good_candidates_sim["medicine_name"].nunique()),
        random_state=RANDOM_SEED
    )
)

good_meds_flesch = (
    good_candidates_flesch["medicine_name"]
    .drop_duplicates()
    .sample(
        n=min(200, good_candidates_flesch["medicine_name"].nunique()),
        random_state=RANDOM_SEED
    )
)

hard_candidates_sim = hard_candidates_sim[
    ~hard_candidates_sim["medicine_name"].isin(good_meds_sim)
]

hard_candidates_flesch = hard_candidates_flesch[
    ~hard_candidates_flesch["medicine_name"].isin(good_meds_flesch)
]

hard_meds_sim = (
    hard_candidates_sim["medicine_name"]
    .drop_duplicates()
    .sample(
        n=min(200, hard_candidates_sim["medicine_name"].nunique()),
        random_state=RANDOM_SEED
    )
)

hard_meds_flesch = (
    hard_candidates_flesch["medicine_name"]
    .drop_duplicates()
    .sample(
        n=min(200, hard_candidates_flesch["medicine_name"].nunique()),
        random_state=RANDOM_SEED
    )
)

challenge_good_sim = good_candidates_sim[good_candidates_sim["medicine_name"].isin(good_meds_sim)].copy()
challenge_good_flesch = good_candidates_flesch[good_candidates_flesch["medicine_name"].isin(good_meds_flesch)].copy()
challenge_hard_sim = hard_candidates_sim[hard_candidates_sim["medicine_name"].isin(hard_meds_sim)].copy()
challenge_hard_flesch = hard_candidates_flesch[hard_candidates_flesch["medicine_name"].isin(hard_meds_flesch)].copy()


# =========================
# 2. Criar train/val/test
# =========================
# Aqui usamos o df completo, incluindo os medicamentos dos challenge sets.

HUMAN_REVISED_MEDS = [
    "ATENOLOL",
    "BESILATO DE ANLODIPINO",
    "CLORIDRATO DE PROPRANOLOL",
    "CAPTOPRIL",
    "ESPIRONOLACTONA",
    "FUROSEMIDA",
    "HIDROCLOROTIAZIDA",
    "LOSARTANA POTASSICA",
    "MALEATO DE ENALAPRIL",
    "SUCCINATO DE METOPROLOL",
]

HUMAN_REVISED_MEDS = set(HUMAN_REVISED_MEDS)

# Separa obrigatoriamente esses medicamentos para o teste
human_test_df = df[df["medicine_name"].isin(HUMAN_REVISED_MEDS)].copy()

# O restante será dividido normalmente
df_remaining = df[~df["medicine_name"].isin(HUMAN_REVISED_MEDS)].copy()

print("Medicamentos revisados por humanos encontrados:")
print(human_test_df["medicine_name"].drop_duplicates().sort_values().to_list())

print("Total de linhas human_test_df:", len(human_test_df))
print("Medicamentos únicos human_test_df:", human_test_df["medicine_name"].nunique())

gss1 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.8,
    random_state=RANDOM_SEED
)

train_idx, temp_idx = next(gss1.split(df_remaining, groups=df_remaining['medicine_name']))

train_df = df_remaining.iloc[train_idx].copy()
temp_df = df_remaining.iloc[temp_idx].copy()

gss2 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.5,
    random_state=RANDOM_SEED
)

val_idx, test_idx = next(
    gss2.split(temp_df, groups=temp_df["medicine_name"])
)

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

# Adiciona os medicamentos revisados por humanos ao teste
test_df = pd.concat([test_df, human_test_df], ignore_index=True)

# =========================
# 3. Salvar arquivos
# =========================

train_df.to_parquet(f"{OUT_DIR}/train.parquet", index=False)
val_df.to_parquet(f"{OUT_DIR}/val.parquet", index=False)
test_df.to_parquet(f"{OUT_DIR}/test.parquet", index=False)

challenge_good_sim.to_parquet(f"{OUT_DIR}/challenge_good_sim.parquet", index=False)
challenge_good_flesch.to_parquet(f"{OUT_DIR}/challenge_good_flesch.parquet", index=False)
challenge_hard_sim.to_parquet(f"{OUT_DIR}/challenge_hard_sim.parquet", index=False)
challenge_hard_flesch.to_parquet(f"{OUT_DIR}/challenge_hard_flesch.parquet", index=False)


# =========================
# 4. Checagens
# =========================

def summarize(name, data):
    print(f"{name}:")
    print(f"  linhas: {len(data)}")
    print(f"  medicamentos únicos: {data['medicine_name'].nunique()}")
    print()

summarize("train", train_df)
summarize("val", val_df)
summarize("test", test_df)
summarize("challenge_good_sim", challenge_good_sim)
summarize("challenge_good_flesch", challenge_good_flesch)
summarize("challenge_hard_sim", challenge_hard_sim)
summarize("challenge_hard_flesch", challenge_hard_flesch)

# Verifica que train/val/test não compartilham medicamentos entre si
split_sets = {
    "train": set(train_df["medicine_name"]),
    "val": set(val_df["medicine_name"]),
    "test": set(test_df["medicine_name"]),
}

for a in split_sets:
    for b in split_sets:
        if a < b:
            overlap = split_sets[a] & split_sets[b]
            assert len(overlap) == 0, (
                f"Vazamento entre {a} e {b}: {len(overlap)} medicamentos"
            )

# Verifica que good e hard não compartilham medicamentos
overlap_challenges_sim = (
    set(challenge_good_sim["medicine_name"]) &
    set(challenge_hard_sim["medicine_name"])
)

assert len(overlap_challenges_sim) == 0, (
    f"Vazamento entre challenge_good_sim e challenge_hard_sim: "
    f"{len(overlap_challenges_sim)} medicamentos"
)


overlap_challenges_flesch = (
    set(challenge_good_flesch["medicine_name"]) &
    set(challenge_hard_flesch["medicine_name"])
)

assert len(overlap_challenges_flesch) == 0, (
    f"Vazamento entre challenge_good_flesch e challenge_hard_flesch: "
    f"{len(overlap_challenges_flesch)} medicamentos"
)

print("Tudo certo:")
print("- train/val/test não compartilham medicamentos entre si;")
print("- challenge_good_sim e challenge_hard_sim não compartilham medicamentos entre si;")
print("- challenge_good_flesch e challenge_hard_flesch não compartilham medicamentos entre si;")
print("- challenge sets podem estar distribuídos em train/val/test.")

missing_human_meds = HUMAN_REVISED_MEDS - set(df["medicine_name"])

if missing_human_meds:
    print("Atenção: estes medicamentos revisados não foram encontrados no dataset:")
    print(sorted(missing_human_meds))