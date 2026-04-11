#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd


PDF_DIR = Path("/home/arthur/nlp/repo/simplification/med-simple-docs/oncologia")
PARQUET_IN = Path(
    "/home/arthur/nlp/repo/simplification/med-simple-docs/simplifications/"
    "informacoes_ao_paciente_v4__simplificado3__Qwen3-8B-Instruct.parquet"
)
PARQUET_OUT = Path(
    "/home/arthur/nlp/repo/simplification/med-simple-docs/simplifications/"
    "oncologia__subset_reset_simplified__docs_simple_rev1_rev2.parquet"
)


def main() -> int:
    # 1) Carrega nomes dos arquivos da pasta em uma lista
    if not PDF_DIR.exists():
        print(f"[ERRO] Pasta não existe: {PDF_DIR}", file=sys.stderr)
        return 2

    pdf_names = sorted([p.name for p in PDF_DIR.iterdir() if p.is_file()])
    pdf_set = set(pdf_names)

    if not pdf_names:
        print(f"[AVISO] Nenhum arquivo encontrado em: {PDF_DIR}", file=sys.stderr)

    # 2) Lê parquet e filtra registros cujo pdf_filename está na lista
    if not PARQUET_IN.exists():
        print(f"[ERRO] Parquet não existe: {PARQUET_IN}", file=sys.stderr)
        return 2

    df = pd.read_parquet(PARQUET_IN)

    required_cols = {
        "pdf_filename",
        "informacoes_ao_paciente",
        "bula_simplificada_1",
        "bula_simplificada_2",
        "bula_simplificada_3",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERRO] Colunas faltando no parquet: {sorted(missing)}", file=sys.stderr)
        return 2

    df_f = df[df["pdf_filename"].isin(pdf_set)].copy()

    # 3) Torna NaN as colunas de simplificação
    for col in ["bula_simplificada_1", "bula_simplificada_2", "bula_simplificada_3"]:
        df_f[col] = pd.NA

    # 4) Renomeia colunas para o formato do seu app
    df_f = df_f.rename(
        columns={
            "informacoes_ao_paciente": "docs",
            "bula_simplificada_1": "simple_doc",
            "bula_simplificada_2": "rev1",
            "bula_simplificada_3": "rev2",
        }
    )

    # (Opcional) Mantém só as colunas que interessam ao app, preservando pdf_filename
    out_cols = ["pdf_filename", "docs", "simple_doc", "rev1", "rev2"]
    df_out = df_f[out_cols].copy()

    # 5) Salva parquet
    PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(PARQUET_OUT, index=False)

    print(f"[OK] Arquivos na pasta: {len(pdf_names)}")
    print(f"[OK] Registros filtrados: {len(df_out)}")
    print(f"[OK] Salvo em: {PARQUET_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
