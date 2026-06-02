import pandas as pd
from utils import filtrar_por_flesch_diff, filtrar_por_similaridade, filtrar_por_diversidade, load_parquets, filter_errors

#parquet_path = "/home/arthur/nlp/repo/simplification/legal-doc-simplification-data/datastf_paraphrases_v2.parquet"
parquet_path = '/home/arthur/nlp/repo/simplification/med-simple-docs/simplifications/final_reprocess_tamarine_qwen3.6_27b_simplified_v2.parquet'
#parquet_path = '/home/arthurscalercio/repo/legal-doc-simplification-data/mlp_pt_BRCAD-5.parquet.finalflesch'

if isinstance(parquet_path, list):
    df = load_parquets(parquet_path)
    save_file = "acordaos_tcu"+".parquet.final"
else:
    df=pd.read_parquet(parquet_path)
    save_file = parquet_path + ".final"
print(len(df))
df = df[
    df["qwen3.6_27b_simplified"].notna() &
    (df["qwen3.6_27b_simplified"].str.strip() != "")
]
print(len(df))
df = filtrar_por_flesch_diff(df)
print(len(df))
#df.to_parquet(save_file+'.flesch_filtered')
df = filtrar_por_similaridade(df)
print(len(df))
#df = filtrar_por_diversidade(df)
#print(len(df))
df.to_parquet(save_file)
