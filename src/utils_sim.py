import re
import unicodedata
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


SECTION_SPECS = [
    ("sec_1", "1. PARA QUE ESTE MEDICAMENTO É INDICADO?"),
    ("sec_2", "2. COMO ESTE MEDICAMENTO FUNCIONA?"),
    ("sec_3", "3. QUANDO NÃO DEVO USAR ESTE MEDICAMENTO?"),
    ("sec_4", "4. O QUE DEVO SABER ANTES DE USAR ESTE MEDICAMENTO?"),
    ("sec_5", "5. ONDE, COMO E POR QUANTO TEMPO POSSO GUARDAR ESTE MEDICAMENTO?"),
    ("sec_6", "6. COMO DEVO USAR ESTE MEDICAMENTO?"),
    ("sec_7", "7. O QUE DEVO FAZER QUANDO EU ME ESQUECER DE USAR ESTE MEDICAMENTO?"),
    ("sec_8", "8. QUAIS OS MALES QUE ESTE MEDICAMENTO PODE ME CAUSAR?"),
    ("sec_9", "9. O QUE FAZER SE ALGUÉM USAR UMA QUANTIDADE MAIOR DO QUE A INDICADA DESTE MEDICAMENTO?"),
]


SECTION_WEIGHTS = {
    "sec_1": 0.08,
    "sec_2": 0.07,
    "sec_3": 0.16,
    "sec_4": 0.16,
    "sec_5": 0.06,
    "sec_6": 0.18,
    "sec_7": 0.09,
    "sec_8": 0.15,
    "sec_9": 0.05,
}


def normalize_text_for_match(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper()

    text = text.replace("ESSE MEDICAMENTO", "ESTE MEDICAMENTO")
    text = text.replace("ESSE PRODUTO", "ESTE MEDICAMENTO")
    text = text.replace("ESTE PRODUTO", "ESTE MEDICAMENTO")

    text = re.sub(r"[ \t]+", " ", text)
    return text


def normalize_with_mapping(text: str):
    normalized_chars = []
    mapping = []

    for idx, ch in enumerate(text):
        decomposed = unicodedata.normalize("NFKD", ch)

        for dch in decomposed:
            if not unicodedata.combining(dch):
                normalized_chars.append(dch.upper())
                mapping.append(idx)

    normalized_text = "".join(normalized_chars)

    normalized_text = normalized_text.replace("ESSE MEDICAMENTO", "ESTE MEDICAMENTO")
    normalized_text = normalized_text.replace("ESSE PRODUTO", "ESTE MEDICAMENTO")
    normalized_text = normalized_text.replace("ESTE PRODUTO", "ESTE MEDICAMENTO")

    return normalized_text, mapping


def build_section_header_patterns() -> List[Tuple[str, re.Pattern]]:
    patterns = []

    flexible_headers = {
        "sec_1": r"(?:1\s*[.\-–—)]?\s*)?PARA\s+QUE\s+(?:(?:ESTE\s+(?:MEDICAMENTO|PRODUTO)\s+(?:E|FOI)\s+INDICADO)|(?:SERVE\s+ESTE\s+(?:MEDICAMENTO|PRODUTO)))\??",
        "sec_2": r"(?:2\s*[.\-–—)]?\s*)?COMO\s+ESTE\s+(?:MEDICAMENTO|PRODUTO)\s+FUNCIONA\??",
        "sec_3": r"(?:3\s*[.\-–—)]?\s*)?QUANDO\s+NAO\s+DEVO\s+(?:USAR|UTILIZAR)\s+ESTE\s+(?:MEDICAMENTO|PRODUTO)\??",
        "sec_4": r"(?:4\s*[.\-–—)]?\s*)?O\s+QUE\s+DEVO\s+SABER\s+ANTES\s+DE\s+USAR\s+ESTE\s+(?:MEDICAMENTO|PRODUTO)\??",
        "sec_5": r"(?:5\s*[.\-–—)]?\s*)?(?:(?:ONDE,\s*COMO\s+E\s+POR\s+QUANTO\s+TEMPO\s+POSSO\s+GUARDAR)|(?:COMO\s+GUARDAR))\s+ESTE\s+(?:MEDICAMENTO|PRODUTO)\??",
        "sec_6": r"(?:6\s*[.\-–—)]?\s*)?COMO\s+DEVO\s+USAR\s+ESTE\s+(?:MEDICAMENTO|PRODUTO)\??",
        "sec_7": r"(?:7\s*[.\-–—)]?\s*)?O\s+QUE\s+(?:DEVO\s+)?FAZER\s+(QUANDO|SE)\s+(?:EU\s+)?ME\s+ESQUECER\s+DE\s+USAR\s+ESTE\s+(?:MEDICAMENTO|PRODUTO)\??",
        "sec_8": r"(?:8\s*[.\-–—)]?\s*)?QUAIS\s+OS\s+MALES\s+QUE\s+ESTE\s+(?:MEDICAMENTO|PRODUTO)\s+PODE\s+(?:ME\s+)?CAUSAR\??",
        "sec_9": r"(?:9\s*[.\-–—)]?\s*)?O\s+QUE\s+FAZER\s+SE\s+ALGUEM\s+USAR\s+UMA\s+QUANTIDADE\s+MAIOR\s+DO\s+QUE\s+A\s+INDICADA(?:\s+DESTE\s+(?:MEDICAMENTO|PRODUTO))?\??",
    }

    for sec_id, pattern in flexible_headers.items():
        patterns.append((sec_id, re.compile(pattern, re.IGNORECASE)))

    return patterns


SECTION_PATTERNS = build_section_header_patterns()


def find_section_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Retorna (section_id, start_original, end_original)
    """

    normalized_text, mapping = normalize_with_mapping(text)

    matches = []

    for sec_id, pattern in SECTION_PATTERNS:
        m = pattern.search(normalized_text)
        if m:
            start_norm = m.start()
            end_norm = m.end()

            # converte para índices do texto original
            start_orig = mapping[start_norm]
            end_orig = mapping[end_norm - 1] + 1  # fim exclusivo

            matches.append((sec_id, start_orig, end_orig))

    matches.sort(key=lambda x: x[1])
    return matches


def extract_leaflet_sections(text: str) -> Dict[str, str]:
    """
    Extrai as 9 seções principais da bula.
    Retorna dict com sec_1 ... sec_9.
    Se uma seção não for encontrada, retorna string vazia.
    """
    if not isinstance(text, str) or not text.strip():
        return {sec_id: "" for sec_id, _ in SECTION_SPECS}

    spans = find_section_spans(text)
    result = {sec_id: "" for sec_id, _ in SECTION_SPECS}

    if not spans:
        return result

    for i, (sec_id, start, header_end) in enumerate(spans):
        content_start = header_end
        content_end = spans[i + 1][1] if i + 1 < len(spans) else len(text)
        section_text = text[content_start:content_end].strip()
        result[sec_id] = section_text

    return result


def chunk_text(text: str, max_words: int = 120, overlap_words: int = 30) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, max_words - overlap_words)

    for start in range(0, len(words), step):
        chunk_words = words[start:start + max_words]
        if not chunk_words:
            continue
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        if start + max_words >= len(words):
            break

    return chunks


def max_alignment_bidirectional_weighted(
    text_a: str,
    text_b: str,
    model: SentenceTransformer,
    max_words: int = 120,
    overlap_words: int = 30,
) -> Dict[str, float]:
    """
    Calcula:
      - orig_to_simp
      - simp_to_orig
      - semantic_f1
    usando max-alignment bidirecional entre chunks.
    """
    chunks_a = chunk_text(text_a, max_words=max_words, overlap_words=overlap_words)
    chunks_b = chunk_text(text_b, max_words=max_words, overlap_words=overlap_words)

    if len(chunks_a) == 0 and len(chunks_b) == 0:
        return {
            "orig_to_simp": np.nan,
            "simp_to_orig": np.nan,
            "semantic_f1": np.nan,
        }

    if len(chunks_a) == 0 or len(chunks_b) == 0:
        return {
            "orig_to_simp": 0.0,
            "simp_to_orig": 0.0,
            "semantic_f1": 0.0,
        }

    emb_a = model.encode(
        chunks_a,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    emb_b = model.encode(
        chunks_b,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    sim_matrix = cosine_similarity(emb_a, emb_b)

    weights_a = np.array([len(c.split()) for c in chunks_a], dtype=float)
    weights_b = np.array([len(c.split()) for c in chunks_b], dtype=float)

    max_a = sim_matrix.max(axis=1)  # cada chunk da original busca melhor match na simplificada
    max_b = sim_matrix.max(axis=0)  # cada chunk da simplificada busca melhor match na original

    orig_to_simp = float(np.average(max_a, weights=weights_a))
    simp_to_orig = float(np.average(max_b, weights=weights_b))

    if (orig_to_simp + simp_to_orig) == 0:
        semantic_f1 = 0.0
    else:
        semantic_f1 = float(
            2 * orig_to_simp * simp_to_orig / (orig_to_simp + simp_to_orig)
        )

    return {
        "orig_to_simp": orig_to_simp,
        "simp_to_orig": simp_to_orig,
        "semantic_f1": semantic_f1,
    }


def compute_section_similarity(
    original_text: str,
    simplified_text: str,
    model: SentenceTransformer,
    max_words: int = 120,
    overlap_words: int = 30,
) -> Dict[str, Dict[str, float]]:
    """
    Calcula similaridade por seção.
    Retorna algo como:
    {
      "sec_1": {"orig_to_simp": ..., "simp_to_orig": ..., "semantic_f1": ...},
      ...
    }
    """
    original_sections = extract_leaflet_sections(original_text)
    simplified_sections = extract_leaflet_sections(simplified_text)

    section_scores = {}

    for sec_id, _ in SECTION_SPECS:
        orig_sec = original_sections.get(sec_id, "").strip()
        if not orig_sec:
            print('não achou source '+sec_id)
        simp_sec = simplified_sections.get(sec_id, "").strip()
        if not simp_sec:
            print('não achou simple '+sec_id)
            print("\n"+ simplified_text)

        if not orig_sec and not simp_sec:
            section_scores[sec_id] = {
                "orig_to_simp": np.nan,
                "simp_to_orig": np.nan,
                "semantic_f1": np.nan,
            }
            continue

        if orig_sec and not simp_sec:
            section_scores[sec_id] = {
                "orig_to_simp": 0.0,
                "simp_to_orig": 0.0,
                "semantic_f1": 0.0,
            }
            continue

        if simp_sec and not orig_sec:
            section_scores[sec_id] = {
                "orig_to_simp": 0.0,
                "simp_to_orig": 0.0,
                "semantic_f1": 0.0,
            }
            continue

        section_scores[sec_id] = max_alignment_bidirectional_weighted(
            orig_sec,
            simp_sec,
            model=model,
            max_words=max_words,
            overlap_words=overlap_words,
        )

    return section_scores


def compute_document_similarity_from_sections(
    section_scores: Dict[str, Dict[str, float]],
    section_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Agrega os semantic_f1 das seções em um score final do documento.
    Ignora seções com NaN (ausentes nos dois textos).
    """
    if section_weights is None:
        section_weights = SECTION_WEIGHTS

    weighted_sum = 0.0
    weight_total = 0.0

    for sec_id, score_dict in section_scores.items():
        sec_score = score_dict.get("semantic_f1", np.nan)
        sec_weight = section_weights.get(sec_id, 0.0)

        if np.isnan(sec_score):
            continue

        weighted_sum += sec_weight * sec_score
        weight_total += sec_weight

    if weight_total == 0:
        return np.nan

    return float(weighted_sum / weight_total)


def calculate_section_based_similarity(
    df: pd.DataFrame,
    original_col: str = "docs",
    simplified_col: str = "qwen3_8b_simplified",
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    max_words: int = 120,
    overlap_words: int = 30,
) -> pd.DataFrame:
    """
    Calcula similaridade por seção e score final do documento.
    Retorna um novo dataframe com colunas adicionais.
    """
    model = SentenceTransformer(model_name)
    df = df.copy()

    doc_scores = []

    for sec_id, _ in SECTION_SPECS:
        df[f"{sec_id}_orig_to_simp"] = pd.NA
        df[f"{sec_id}_simp_to_orig"] = pd.NA
        df[f"{sec_id}_semantic_f1"] = pd.NA

    for idx, row in df.iterrows():
        original_text = "" if pd.isna(row[original_col]) else str(row[original_col]).strip()
        simplified_text = "" if pd.isna(row[simplified_col]) else str(row[simplified_col]).strip()

        section_scores = compute_section_similarity(
            original_text=original_text,
            simplified_text=simplified_text,
            model=model,
            max_words=max_words,
            overlap_words=overlap_words,
        )

        for sec_id in section_scores:
            df.at[idx, f"{sec_id}_orig_to_simp"] = section_scores[sec_id]["orig_to_simp"]
            df.at[idx, f"{sec_id}_simp_to_orig"] = section_scores[sec_id]["simp_to_orig"]
            df.at[idx, f"{sec_id}_semantic_f1"] = section_scores[sec_id]["semantic_f1"]

        doc_sim = compute_document_similarity_from_sections(section_scores)
        doc_scores.append(doc_sim)

    df["semantic_similarity_sections"] = doc_scores
    return df