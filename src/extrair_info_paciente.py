#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import unicodedata
from collections import Counter
from typing import List, Optional, Tuple, Dict

import pandas as pd
from tqdm import tqdm


# --------------------------
# Normalização / utilitários
# --------------------------

def strip_accents(s: str) -> str:
    # remove diacríticos (acentos) para facilitar matching
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch) != "Mn"
    )

def normalize_for_match(s: str) -> str:
    """
    Normaliza para matching:
    - remove acentos
    - upper
    - normaliza whitespace
    """
    s = s.replace("\u00ad", "")  # soft hyphen
    s = strip_accents(s)
    s = s.upper()
    s = re.sub(r"[ \t]+", " ", s)
    return s

def clean_join_lines(text: str) -> str:
    """
    Limpa texto após extração:
    - remove soft hyphen
    - corrige hifenização no fim da linha (ex: "medi-\ncamento" -> "medicamento")
    - normaliza múltiplas quebras
    """
    text = text.replace("\u00ad", "")
    # junta palavras quebradas por hifenização no final da linha
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # normaliza quebras de linha: mantém parágrafos, mas remove quebras "soltas"
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----------------------------------------
# Extração de texto por página (robusta)
# ----------------------------------------

def extract_pages_pdfplumber(pdf_path: str, top_crop: float = 0.08, bottom_crop: float = 0.08) -> List[str]:
    """
    Extrai texto por página usando pdfplumber.
    Faz um recorte em porcentagem do topo e do rodapé para reduzir cabeçalho/rodapé.
    """
    import pdfplumber

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            h = page.height
            # recorta topo e rodapé (heurística)
            crop_box = (0, h * top_crop, page.width, h * (1 - bottom_crop))
            cropped = page.crop(crop_box)
            # layout=True ajuda em alguns PDFs; x_tolerance/y_tolerance dá estabilidade
            txt = cropped.extract_text(layout=True, x_tolerance=2, y_tolerance=2) or ""
            pages_text.append(txt)
    return pages_text

def extract_pages_pymupdf(pdf_path: str) -> List[str]:
    """
    Fallback: extrai texto por página usando PyMuPDF (fitz).
    """
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
    """
    Remove linhas que aparecem repetidamente no topo/rodapé das páginas.
    Heurística:
    - coleta as N primeiras e N últimas linhas de cada página
    - remove linhas que se repetem >= min_repeats
    """
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
            # remove se for cabeçalho comum e estiver nas primeiras linhas
            if i < head_lines and nln in common_heads:
                continue
            # remove se for rodapé comum e estiver nas últimas linhas
            if i >= max(0, len(lines) - foot_lines) and nln in common_foots:
                continue
            cleaned.append(ln)
        cleaned_pages.append("\n".join(cleaned))
    return cleaned_pages

def extract_full_text(pdf_path: str) -> str:
    """
    Extrai o texto completo (todas as páginas) de modo robusto.
    """
    # tentativa 1: pdfplumber com crop
    try:
        pages = extract_pages_pdfplumber(pdf_path)
        # se vier vazio demais, cai pro fallback
        if sum(len(p.strip()) for p in pages) < 200:
            raise ValueError("Extração muito pequena via pdfplumber.")
    except Exception:
        pages = extract_pages_pymupdf(pdf_path)

    # remove headers/footers repetidos (melhora bastante em bulas)
    pages = remove_repeated_headers_footers(pages)

    text = "\n\n".join(pages)
    text = clean_join_lines(text)
    return text


# ----------------------------------------
# Localização do trecho de interesse
# ----------------------------------------

# Começo: subseção 1 (bem tolerante a variações)
START_RE = re.compile(
    r"(?is)"                                 # ignorecase + dotall (via flags no find)
    r"(^|[\n\r])\s*"                         # início de linha
    r"1\s*[\.\-–—)]?\s*"                     # "1." ou "1 -" ou "1)"
    r"PARA\s+QUE\s+ESTE\s+MEDICAMENTO\s+E\s+INDICADO\s*\??"  # sem acento (normalizado)
)

# Fim: imediatamente antes de "Em caso de uso de grande quantidade..."
# (normalizamos sem acento, então "QUANTIDADE" etc.)
END_RE = re.compile(
    r"(?is)"
    r"\bEM\s+CASO\s+DE\s+USO\s+DE\s+GRANDE\s+QUANTIDADE\b"
)

END_PRIMARY_RE = re.compile(
    r"(?is)\bIII\s*[-–—]\s*DIZERES\s+LEGAIS\b"
)

def slice_informacoes_ao_paciente(full_text: str) -> Optional[str]:
    """
    Retorna o trecho entre:
    - início da subseção 1 ("1. PARA QUE ESTE MEDICAMENTO É INDICADO?")
    - até imediatamente antes de "Em caso de uso de grande quantidade"
    """
    if not full_text or len(full_text.strip()) < 50:
        return None

    # Para localizar com robustez, fazemos matching em versão normalizada,
    # mas cortamos usando índices do texto original.
    norm = normalize_for_match(full_text)

    m_start = START_RE.search(norm)
    if not m_start:
        return None

    start_idx = m_start.start()

    # tenta primeiro o final padrão das bulas
    m_end = END_PRIMARY_RE.search(norm, pos=m_start.end())

    # fallback: frase de superdosagem (caso não haja dizeres legais)
    if not m_end:
        m_end = END_RE.search(norm, pos=m_start.end())
        
    if not m_end:
        # se não achou o "Em caso...", melhor não inventar.
        return None

    end_idx = m_end.start()

    # Mapear índices de norm -> original é 1-para-1? Não, porque removemos acentos.
    # Então fazemos uma estratégia segura: achar os mesmos padrões no ORIGINAL via regex tolerante.
    # (Isso evita problemas de índices.)
    # 1) achar o start no original (regex com acentos opcional)
    start_orig = re.search(
        r"(?is)(^|[\n\r])\s*1\s*[\.\-–—)]?\s*PARA\s+QUE\s+ESTE\s+MEDICAMENTO\s+(É|E)\s+INDICADO\s*\??",
        full_text
    )
    if not start_orig:
        # fallback: corta aproximando pelo índice do norm (pode errar pouco)
        cut_start = max(0, start_idx)
    else:
        cut_start = start_orig.start()

    # 2) achar o end no original
    # tenta cortar no original pelo "III - DIZERES LEGAIS"
    end_orig = re.search(
        r"(?is)\bIII\s*[-–—]\s*DIZERES\s+LEGAIS\b",
        full_text[cut_start:]
    )
    
    if not end_orig:
        # fallback: frase de superdosagem
        end_orig = re.search(
            r"(?is)\bEm\s+caso\s+de\s+uso\s+de\s+grande\s+quantidade\b",
            full_text[cut_start:]
        )
    if not end_orig:
        # fallback: aproxima pelo end_idx do norm
        cut_end = len(full_text)
    else:
        cut_end = cut_start + end_orig.start()

    section = full_text[cut_start:cut_end]
    section = clean_join_lines(section)

    # sanity check: deve conter pelo menos "2." e "9." na maioria das bulas
    # (não eliminamos se não tiver, mas ajuda a evitar lixo)
    if len(section) < 500:
        return None

    return section


# ----------------------------------------
# Pipeline principal
# ----------------------------------------

def iter_pdf_files(input_dir: str) -> List[str]:
    pdfs = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fn))
    pdfs.sort()
    return pdfs

def process_pdf(pdf_path: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Retorna (filename, section_text, error).
    """
    try:
        full_text = extract_full_text(pdf_path)
        section = slice_informacoes_ao_paciente(full_text)
        if not section:
            return (os.path.basename(pdf_path), None, "nao_encontrou_inicio_ou_fim")
        return (os.path.basename(pdf_path), section, None)
    except Exception as e:
        return (os.path.basename(pdf_path), None, f"erro: {type(e).__name__}: {e}")

def main():
    ap = argparse.ArgumentParser(description="Extrai a seção INFORMAÇÕES AO PACIENTE (subseções 1-9) de PDFs de bulas e salva em Parquet.")
    ap.add_argument("--input_dir", required=True, help="Pasta com PDFs")
    ap.add_argument("--output_parquet", required=True, help="Caminho do parquet de saída")
    ap.add_argument("--output_errors_csv", default=None, help="(Opcional) CSV com arquivos que falharam e motivo")
    args = ap.parse_args()

    pdfs = iter_pdf_files(args.input_dir)
    if not pdfs:
        raise SystemExit(f"Nenhum PDF encontrado em: {args.input_dir}")

    rows = []
    errors = []

    for pdf_path in tqdm(pdfs, desc="Processando PDFs"):
        fn, section, err = process_pdf(pdf_path)
        if section:
            rows.append({"pdf_filename": fn, "informacoes_ao_paciente": section})
        else:
            errors.append({"pdf_filename": fn, "error": err or "desconhecido"})

    df = pd.DataFrame(rows)
    # garante colunas mesmo se vazio
    if df.empty:
        df = pd.DataFrame(columns=["pdf_filename", "informacoes_ao_paciente"])

    df.to_parquet(args.output_parquet, index=False)

    if args.output_errors_csv:
        pd.DataFrame(errors).to_csv(args.output_errors_csv, index=False)

    print(f"\nOK: {len(rows)} extraídos | Falhas: {len(errors)}")
    print(f"Parquet: {args.output_parquet}")
    if args.output_errors_csv:
        print(f"Erros: {args.output_errors_csv}")

if __name__ == "__main__":
    main()
