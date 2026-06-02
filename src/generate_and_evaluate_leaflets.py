#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from utils import calculate_d_sari, calcular_similaridade_sliding
from utils_sim import calculate_section_based_similarity


DEFAULT_SYSTEM_PROMPT = (
    "Você é um assistente especialista em simplificação textual de bulas."
)

DEFAULT_USER_TEMPLATE = (
    "Simplifique a bula abaixo, mas mantenha o sentido original. "
    "Retorne somente o texto simplificado.\n\n"
    "Siga estas orientações:\n"
    "1. Substitua termos técnicos por sinônimos simples; se inevitável, explique entre parênteses.\n"
    "2. Prefira voz ativa à passiva.\n"
    "3. Divida sentenças longas em sentenças menores e mais simples.\n"
    "4. Ignore sentenças irrelevantes.\n"
    "5. Resolva anáforas e pronomes quando necessário.\n"
    "6. Mantenha números como doses, frequência, quantidades e vias de administração.\n"
    "7. Mantenha as nove seções da bula.\n\n"
    "Texto original:\n"
    "{text}\n\n"
    "Texto simplificado: "
)

THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)

HEADER_RE = re.compile(
    r"""(?isx)
    ^\s*
    (?:\*{1,3}\s*)?
    \b(?:BULA|TEXTO)\b
    (?:
        \s+\b(?:SIMPLIFICADA|SIMPLIFICADO|SIMPLES)\b
      |
        \s*\(
            [^)]{0,80}?
            \b(?:SIMPLIFICADA|SIMPLIFICADO|SIMPLES)\b
            [^)]{0,80}?
        \)
    )
    [^\n]*
    (?:\s*\*{1,3})?
    \s*
    (?:\n[ \t]*\n)?
    (?:\n[ \t]*[-–—]{3,}[ \t]*\n)?
    \s*
    """
)


def clean_llm_output(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = THINK_BLOCK_RE.sub("", text)
    text = THINK_TAG_RE.sub("", text)
    text = HEADER_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_chat_contents(data: Dict[str, Any]) -> List[str]:
    choices = data.get("choices", [])
    outs = []
    for ch in choices:
        msg = ch.get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            outs.append(content.strip())
    return outs


def lmstudio_chat_completion_single(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = 240,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    api_key: str = "lm-studio",
    seed: Optional[int] = None,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    if seed is not None:
        payload["seed"] = seed

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    outputs = _extract_chat_contents(data)

    if not outputs:
        raise RuntimeError("O LM Studio não retornou conteúdo em choices.message.content.")

    return outputs[0]


def generate_simplification(
    text: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_template: str,
    timeout: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    api_key: str,
    seed: Optional[int] = None,
) -> str:
    user_prompt = user_template.format(text=text)

    output = lmstudio_chat_completion_single(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_key=api_key,
        seed=seed,
    )
    return clean_llm_output(output)


def safe_mean(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) == 0:
        return float("nan")
    return float(valid.mean())


def is_filled(value) -> bool:
    if pd.isna(value):
        return False
    if not isinstance(value, str):
        value = str(value)
    return bool(value.strip())


def ensure_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "d_sari",
        "d_sari_add",
        "d_sari_keep",
        "d_sari_del",
        "sim",
    ]
    for col in metric_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera simplificações de bulas com LM Studio e avalia com D_SARI e SIM."
    )

    parser.add_argument(
        "--input-parquet",
        default="data/bulas_all_v2.parquet",
        help="Caminho do parquet de entrada.",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/bulas_all_v2_qwen3_8b_simplified.parquet",
        help="Caminho do parquet de saída.",
    )
    parser.add_argument(
        "--model",
        default="Qwen3-8B-Instruct",
        help="Nome do modelo carregado no LM Studio.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:1234",
        help="URL base do servidor OpenAI-compatible do LM Studio.",
    )
    parser.add_argument(
        "--api-key",
        default="lm-studio",
        help="Bearer token. O LM Studio geralmente aceita qualquer valor.",
    )
    parser.add_argument(
        "--text-col",
        default="docs",
        help="Coluna com o texto original.",
    )
    parser.add_argument(
        "--reference-col",
        default="rev1",
        help="Coluna com o texto simplificado de referência.",
    )
    parser.add_argument(
        "--output-col",
        default="qwen3_8b_simplified",
        help="Nome da coluna que armazenará a simplificação gerada pelo LLM.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Número máximo de linhas a processar. 0 = processa tudo.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Salva checkpoint a cada N linhas geradas.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-template", default=DEFAULT_USER_TEMPLATE)

    args = parser.parse_args()

    input_path = Path(args.input_parquet)
    output_path = Path(args.output_parquet)

    if not input_path.exists() and not output_path.exists():
        print(
            f"[ERRO] Nem o input nem o output existem. "
            f"input={input_path} | output={output_path}",
            file=sys.stderr,
        )
        return 1

    # Carregamento inteligente:
    # - se output existe, prioriza ele
    # - se não existe, usa input
    if output_path.exists():
        print(f"[INFO] Arquivo de saída existente encontrado. Carregando: {output_path}")
        df = pd.read_parquet(output_path)
        loaded_from_output = True
    else:
        print(f"[INFO] Arquivo de saída não existe. Carregando input: {input_path}")
        df = pd.read_parquet(input_path)
        loaded_from_output = False

    required_cols = [args.text_col, args.reference_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[ERRO] Colunas ausentes no parquet: {missing_cols}", file=sys.stderr)
        print(f"[INFO] Colunas disponíveis: {list(df.columns)}", file=sys.stderr)
        return 1

    if args.output_col not in df.columns:
        df[args.output_col] = pd.NA

    df = ensure_metric_columns(df)

    n_rows = len(df)
    if args.limit > 0:
        n_rows = min(n_rows, args.limit)

    # Decide quais linhas ainda precisam de geração
    rows_to_generate = []
    for idx in range(n_rows):
        if not is_filled(df.at[idx, args.output_col]):
            rows_to_generate.append(idx)

    if loaded_from_output:
        if len(rows_to_generate) == 0:
            print("[INFO] Todas as simplificações já existem no arquivo de saída.")
            print("[INFO] Nenhuma nova geração será feita. Apenas métricas serão recalculadas.")
        else:
            print(
                f"[INFO] Arquivo de saída já existe, mas ainda há "
                f"{len(rows_to_generate)} linhas sem simplificação."
            )
            print("[INFO] Só as linhas faltantes serão geradas; depois as métricas serão recalculadas.")
    else:
        print("[INFO] Nenhum arquivo de saída existente. As simplificações serão geradas do zero.")

    print(f"[INFO] Total de linhas consideradas: {n_rows}")
    print(f"[INFO] Modelo: {args.model}")
    print(f"[INFO] Coluna de entrada: {args.text_col}")
    print(f"[INFO] Coluna de referência: {args.reference_col}")
    print(f"[INFO] Coluna de saída: {args.output_col}")

    generated_now = 0
    failures = 0

    # Geração apenas se houver linhas faltantes
    for count, idx in enumerate(rows_to_generate, start=1):
        original_text = df.at[idx, args.text_col]
        original_text = "" if pd.isna(original_text) else str(original_text).strip()

        if not original_text:
            print(f"[WARN] idx={idx}: texto original vazio. Pulando geração.")
            failures += 1
            continue

        try:
            generated_text = generate_simplification(
                text=original_text,
                base_url=args.base_url,
                model=args.model,
                system_prompt=args.system_prompt,
                user_template=args.user_template,
                timeout=args.timeout,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                api_key=args.api_key,
                seed=args.seed + idx,
            )

            df.at[idx, args.output_col] = generated_text
            generated_now += 1

            print(
                f"[OK] idx={idx} | "
                f"chars_in={len(original_text)} | chars_out={len(generated_text)}"
            )

        except Exception as e:
            failures += 1
            print(f"[ERRO] idx={idx} | {e}", file=sys.stderr)

        if args.checkpoint_every > 0 and generated_now > 0 and generated_now % args.checkpoint_every == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            print(f"[CHECKPOINT] Salvo em: {output_path}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Recalcula D_SARI para todas as linhas que tiverem original + referência + simplificação
    print("[INFO] Recalculando D_SARI...")
    for idx in range(n_rows):
        original_text = df.at[idx, args.text_col]
        reference_text = df.at[idx, args.reference_col]
        generated_text = df.at[idx, args.output_col]

        original_text = "" if pd.isna(original_text) else str(original_text).strip()
        reference_text = "" if pd.isna(reference_text) else str(reference_text).strip()
        generated_text = "" if pd.isna(generated_text) else str(generated_text).strip()

        if original_text and reference_text and generated_text:
            try:
                score, add_score, keep_score, del_score = calculate_d_sari(
                    original_text,
                    generated_text,
                    reference_text,
                )
                df.at[idx, "d_sari"] = score
                df.at[idx, "d_sari_add"] = add_score
                df.at[idx, "d_sari_keep"] = keep_score
                df.at[idx, "d_sari_del"] = del_score
            except Exception as e:
                df.at[idx, "d_sari"] = pd.NA
                df.at[idx, "d_sari_add"] = pd.NA
                df.at[idx, "d_sari_keep"] = pd.NA
                df.at[idx, "d_sari_del"] = pd.NA
                print(f"[ERRO] D_SARI idx={idx} | {e}", file=sys.stderr)
        else:
            df.at[idx, "d_sari"] = pd.NA
            df.at[idx, "d_sari_add"] = pd.NA
            df.at[idx, "d_sari_keep"] = pd.NA
            df.at[idx, "d_sari_del"] = pd.NA

    # Recalcula SIM
    print("[INFO] Recalculando SIM...")
    try:
        sim_scores = calcular_similaridade_sliding(df, args.output_col)
        df["sim"] = sim_scores
#        df = calculate_section_based_similarity(
#            df,
#            original_col=args.text_col,
#            simplified_col=args.output_col,
#            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#            max_words=120,
#            overlap_words=30,
#        )
#
#        df["sim"] = df["semantic_similarity_sections"]

    except Exception as e:
        print(
            "[ERRO] Falha ao calcular SIM com calcular_similaridade_sliding(df). "
            "Verifique se a função espera nomes específicos de colunas.",
            file=sys.stderr,
        )
        print(f"Detalhe do erro: {e}", file=sys.stderr)
        df["sim"] = pd.NA

    # Salva tudo novamente
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print("\n========== RESULTADOS FINAIS ==========")
    print(f"Novas simplificações geradas nesta execução: {generated_now}")
    print(f"Falhas na geração: {failures}")
    print(f"D_SARI médio:      {safe_mean(df['d_sari']):.6f}")
    print(f"D_SARI ADD médio:  {safe_mean(df['d_sari_add']):.6f}")
    print(f"D_SARI KEEP médio: {safe_mean(df['d_sari_keep']):.6f}")
    print(f"D_SARI DEL médio:  {safe_mean(df['d_sari_del']):.6f}")
    print(f"SIM médio:         {safe_mean(df['sim']):.6f}")
    print(f"\nArquivo salvo em: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())