#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import unicodedata
from collections import Counter
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


# --------------------------
# Normalização / utilitários
# --------------------------

def strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch) != "Mn"
    )

def normalize_for_match(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = strip_accents(s)
    s = s.upper()
    # normaliza NBSP e afins para espaço comum
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s

def clean_join_lines(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\u00A0", " ")
    # junta hifenização no fim da linha
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # remove espaços antes de newline
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----------------------------------------
# Extração de texto por página (robusta)
# ----------------------------------------

def extract_pages_pdfplumber(pdf_path: str, top_crop: float = 0.08, bottom_crop: float = 0.08) -> List[str]:
    import pdfplumber
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            h = page.height
            crop_box = (0, h * top_crop, page.width, h * (1 - bottom_crop))
            cropped = page.crop(crop_box)
            txt = cropped.extract_text(layout=True, x_tolerance=2, y_tolerance=2) or ""
            pages_text.append(txt)
    return pages_text

def extract_pages_pymupdf(pdf_path: str) -> List[str]:
    import fitz
    pages_text = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            txt = page.get_text("text") or ""
            pages_text.append(txt)
    finally:
        doc.close()
    return pages_text

def remove_repeated_headers_footers(pages: List[str], head_lines: int = 2, foot_lines: int = 2, min_repeats: int = 3) -> List[str]:
    head_counter = Counter()
    foot_counter = Counter()

    split_pages = []
    for p in pages:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        split_pages.append(lines)
        for ln in lines[:head_lines]:
            head_counter[normalize_for_match(ln)] += 1
        for ln in lines[-foot_lines:]:
            foot_counter[normalize_for_match(ln)] += 1

    common_heads = {ln for ln, c in head_counter.items() if c >= min_repeats}
    common_foots = {ln for ln, c in foot_counter.items() if c >= min_repeats}

    cleaned_pages = []
    for lines in split_pages:
        cleaned = []
        for i, ln in enumerate(lines):
            nln = normalize_for_match(ln)
            if i < head_lines and nln in common_heads:
                continue
            if i >= max(0, len(lines) - foot_lines) and nln in common_foots:
                continue
            cleaned.append(ln)
        cleaned_pages.append("\n".join(cleaned))
    return cleaned_pages

def extract_full_text(pdf_path: str) -> str:
    try:
        pages = extract_pages_pdfplumber(pdf_path)
        if sum(len(p.strip()) for p in pages) < 200:
            raise ValueError("Extração muito pequena via pdfplumber.")
    except Exception:
        pages = extract_pages_pymupdf(pdf_path)

    pages = remove_repeated_headers_footers(pages)
    text = "\n\n".join(pages)
    return clean_join_lines(text)


# ----------------------------------------
# Localização do trecho de interesse
# ----------------------------------------

# Início: 1. PARA QUE (ESTE|ESSE) (MEDICAMENTO|PRODUTO) É INDICADO?
# Obs: usamos matching na versão normalizada SEM ACENTOS, por isso "E" em vez de "É".
START_RE = re.compile(
    r"(?is)"
    r"(^|[\n\r])\s*"
    r"(?:1\s*[\.\-–—)]?\s*)?"
    r"PARA\s+QUE\s+"
    r"(ESTE|ESSE)\s+"
    r"(MEDICAMENTO|PRODUTO)\s+"
    r"(E|FOI)\s+INDICADO\s*\??"
)

# Fim: apenas "DIZERES LEGAIS" (sem "III")
END_RE = re.compile(
    r"(?is)\bDIZERES\s+LEGAIS\b"
)

def slice_informacoes_ao_paciente(full_text: str, min_chars: int = 500) -> Optional[str]:
    if not full_text or len(full_text.strip()) < 50:
        return None

    norm = normalize_for_match(full_text)

    m_start = START_RE.search(norm)
    if not m_start:
        return None

    # tenta achar o fim depois do início
    m_end = END_RE.search(norm, pos=m_start.end())
    if not m_end:
        return None

    # Para cortar no texto original de forma estável:
    # 1) achamos start no original com regex tolerante a acento (É/E) e hífens
    start_orig = re.search(
        r"(?is)(^|[\n\r])\s*(?:1\s*[\.\-–—)]?\s*)?PARA\s+QUE\s+(ESTE|ESSE)\s+(MEDICAMENTO|PRODUTO)\s+(É|E|FOI)\s+INDICADO\s*\??",
        full_text
    )
    cut_start = start_orig.start() if start_orig else 0

    # 2) achamos end no original por "DIZERES LEGAIS"
    end_orig = re.search(
        r"(?is)\bDIZERES\s+LEGAIS\b",
        full_text[cut_start:]
    )
    if not end_orig:
        return None

    cut_end = cut_start + end_orig.start()

    section = clean_join_lines(full_text[cut_start:cut_end])

    if len(section) < min_chars:
        return None

    return section


# ----------------------------------------
# Pipeline incremental: só falhas do CSV
# ----------------------------------------

def find_pdf_by_basename(pdf_dir: str, basename: str) -> Optional[str]:
    """
    Procura um PDF com esse basename em pdf_dir (recursivo).
    Retorna o caminho completo se achar.
    """
    # tenta caminho direto
    direct = os.path.join(pdf_dir, basename)
    if os.path.isfile(direct):
        return direct

    # busca recursiva
    for root, _, files in os.walk(pdf_dir):
        for fn in files:
            if fn == basename:
                return os.path.join(root, fn)
    return None

def process_one(pdf_path: str, min_chars: int) -> Tuple[str, Optional[str], Optional[str]]:
    try:
        full_text = extract_full_text(pdf_path)
        section = slice_informacoes_ao_paciente(full_text, min_chars=min_chars)
        if not section:
            return (os.path.basename(pdf_path), None, "nao_encontrou_inicio_ou_fim_v2")
        return (os.path.basename(pdf_path), section, None)
    except Exception as e:
        return (os.path.basename(pdf_path), None, f"erro: {type(e).__name__}: {e}")

def main():
    ap = argparse.ArgumentParser(
        description="Reprocessa somente PDFs listados no CSV de erro e tenta extrair INFORMAÇÕES AO PACIENTE com variações (ESSE/PRODUTO) e fim por DIZERES LEGAIS."
    )
    ap.add_argument("--pdf_dir", required=True, help="Pasta onde estão os PDFs (busca recursiva).")
    ap.add_argument("--input_parquet", required=True, help="Parquet existente (já processados).")
    ap.add_argument("--errors_csv", required=True, help="CSV de erros com coluna pdf_filename (basenames).")
    ap.add_argument("--output_parquet", required=True, help="Novo parquet de saída (não sobrescreve o input).")
    ap.add_argument("--min_chars", type=int, default=500, help="Tamanho mínimo do texto extraído para aceitar (default=500).")
    ap.add_argument("--output_new_errors_csv", default=None, help="(Opcional) CSV com falhas após esta segunda passada.")
    args = ap.parse_args()

    # carrega parquet existente
    df_old = pd.read_parquet(args.input_parquet)
    # garante colunas esperadas
    if "pdf_filename" not in df_old.columns or "informacoes_ao_paciente" not in df_old.columns:
        raise SystemExit("Parquet de entrada precisa ter colunas: pdf_filename, informacoes_ao_paciente")

    # carrega erros
    df_err = pd.read_csv(args.errors_csv)
    if "pdf_filename" not in df_err.columns:
        raise SystemExit("CSV de erros precisa ter coluna: pdf_filename")

    # só os basenames únicos
    failed_names = sorted(set(df_err["pdf_filename"].astype(str).tolist()))

    new_rows = []
    new_errors = []

    for basename in tqdm(failed_names, desc="Reprocessando falhas"):
        pdf_path = find_pdf_by_basename(args.pdf_dir, basename)
        if not pdf_path:
            new_errors.append({"pdf_filename": basename, "error": "arquivo_nao_encontrado_no_pdf_dir"})
            continue

        fn, section, err = process_one(pdf_path, min_chars=args.min_chars)
        if section:
            new_rows.append({"pdf_filename": fn, "informacoes_ao_paciente": section})
        else:
            new_errors.append({"pdf_filename": fn, "error": err or "desconhecido"})

    df_new = pd.DataFrame(new_rows)

    # concatena com o parquet antigo
    # (se um pdf_filename já existir no parquet, mantemos o que já existe; opcionalmente você pode escolher substituir)
    if not df_new.empty:
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        # remove duplicatas por pdf_filename mantendo o primeiro (o antigo)
        df_combined = df_combined.drop_duplicates(subset=["pdf_filename"], keep="first")
    else:
        df_combined = df_old.copy()

    df_combined.to_parquet(args.output_parquet, index=False)

    if args.output_new_errors_csv:
        pd.DataFrame(new_errors).to_csv(args.output_new_errors_csv, index=False)

    print("\nResumo:")
    print(f"- Já existiam no parquet: {len(df_old)}")
    print(f"- Novos extraídos agora:   {len(df_new)}")
    print(f"- Total final (sem dup):   {len(df_combined)}")
    print(f"- Ainda com erro:          {len(new_errors)}")
    print(f"Output parquet: {args.output_parquet}")
    if args.output_new_errors_csv:
        print(f"Erros desta rodada: {args.output_new_errors_csv}")

if __name__ == "__main__":
    main()
