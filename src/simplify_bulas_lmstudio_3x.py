#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import time
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import requests


DEFAULT_SYSTEM_PROMPT = (
    "Você é um especialista em simplificação textual."
)

DEFAULT_USER_TEMPLATE = (
    "Simplifique a bula abaixo para um público leigo, mas mantenha o sentido original. Retorne só o texto simplificado.\n"
    "Regras:\n"
    "1) Não invente nada.\n"
    "2) Preserve todos os números/unidades/doses/frequências/vias.\n"
    "3) Mantenha a estrutura por seções quando possível.\n"
    "4) Evite jargão; se inevitável, explique entre parênteses.\n\n"
    "BULA (TEXTO ORIGINAL):\n"
    "{text}\n\n"
    "BULA (VERSÃO SIMPLIFICADA): /no_think"
)

import re

#THINK_RE = re.compile(r"<think>\s*.*?\s*</think>\s*", re.DOTALL | re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE   = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)  # pega <think> e </think> soltos

#def strip_think_block(text: str) -> str:
#    if not isinstance(text, str):
#        return text
#    return THINK_RE.sub("", text).lstrip()

# Cabeçalho “BULA SIMPLIFICADA …” com variações, e remove separadores tipo --- logo depois
HEADER_RE = re.compile(
    r"""(?isx)
    ^\s*
    (?:\*{1,3}\s*)?                     # abre ** ou *** (opcional)

    \bBULA\b                             # "BULA"
    (?:                                  # variações aceitas após BULA:
        \s+\b(?:SIMPLIFICADA|SIMPLES)\b  #   "BULA SIMPLIFICADA" / "BULA SIMPLES"
      |                                  # ou
        \s*\(                            #   "BULA ( ... )"
            [^)]{0,60}?                  #   texto curto dentro do parênteses
            \b(?:SIMPLIFICADO|SIMPLIFICADA|SIMPLES)\b
            [^)]{0,60}?                  #   pode ter "TEXTO" etc
        \)
    )

    [^\n]*                               # resto da linha (ex.: – AAS® Protect (...))
    (?:\s*\*{1,3})?                      # fecha **/*** (opcional)
    \s*
    (?:\n[ \t]*\n)?                      # linha em branco opcional
    (?:\n[ \t]*[-–—]{3,}[ \t]*\n)?       # separador --- opcional
    \s*
    """,
)


#def strip_header(text: str) -> str:
#    return HEADER_RE.sub("", text).lstrip()

def clean_llm_output(text: str) -> str:
    if not isinstance(text, str):
        return text

    # 1) remove bloco completo <think>...</think>
    text = THINK_BLOCK_RE.sub("", text)

    # 2) remove quaisquer tags <think> ou </think> soltas (em qualquer lugar)
    text = THINK_TAG_RE.sub("", text)

    # 3) remove cabeçalho “BULA SIMPLIFICADA ...” no início
    text = HEADER_RE.sub("", text)

    # 4) normaliza quebras de linha
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def sanitize_for_filename(s: str, max_len: int = 80) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-\.]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s)
    return s[:max_len].strip("_") or "model"


def _extract_chat_contents(data: Dict[str, Any]) -> List[str]:
    # OpenAI-like: {"choices":[{"message":{"content":"..."}}...]}
    choices = data.get("choices", [])
    outs: List[str] = []
    for ch in choices:
        msg = ch.get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            outs.append(content.strip())
    return outs

import json

def _shorten(s: str, max_chars: int = 800) -> str:
    s = s if s is not None else ""
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"... [truncado, len={len(s)}]"


def _debug_dump_http_error(prefix: str, err: requests.HTTPError, payload: dict):
    resp = getattr(err, "response", None)
    status = getattr(resp, "status_code", None)
    text = getattr(resp, "text", "")
    headers = dict(getattr(resp, "headers", {}) or {})

    # Evita logar a bula inteira
    payload_safe = dict(payload)
    if "messages" in payload_safe:
        msgs = []
        for m in payload_safe["messages"]:
            m2 = dict(m)
            if m2.get("role") == "user":
                m2["content"] = _shorten(m2.get("content", ""), 600)
            msgs.append(m2)
        payload_safe["messages"] = msgs

    print(f"[HTTP ERRO] {prefix} | status={status}", file=sys.stderr)
    print(f"--- response.headers ---\n{json.dumps(headers, ensure_ascii=False, indent=2)}", file=sys.stderr)
    print(f"--- response.text ---\n{_shorten(text, 2000)}", file=sys.stderr)
    print(f"--- request.payload (safe) ---\n{json.dumps(payload_safe, ensure_ascii=False, indent=2)}", file=sys.stderr)

def lmstudio_chat_completions_n(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    n: int = 3,
    timeout: int = 180,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    api_key: str = "lm-studio",
    seed: Optional[int] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    debug: bool = True,
) -> List[str]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": n,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if seed is not None:
        payload["seed"] = seed

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        #print("[DEBUG] n_requested=", payload.get("n"), "n_choices=", len(data.get("choices", [])))
        outs = _extract_chat_contents(data)
        return outs
    except requests.HTTPError as e:
        # Mostra detalhes do 400
        if debug and getattr(e, "response", None) is not None:
            resp = e.response
            if resp.status_code in (400, 401, 403, 404, 409, 413, 422, 429, 500, 503):
                _debug_dump_http_error(prefix="chat_completions", err=e, payload=payload)
        raise



def lmstudio_chat_completion_single(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = 180,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    api_key: str = "lm-studio",
    seed: Optional[int] = None,
) -> str:
    outs = lmstudio_chat_completions_n(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        n=1,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_key=api_key,
        seed=seed,
    )
    if not outs:
        raise RuntimeError("Servidor retornou 0 saídas.")
    return outs[0]


def get_three_simplifications(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    api_key: str,
    prefer_single_call: bool,
    base_seed: int,
) -> Tuple[str, str, str]:
    
    temps = [temperature, temperature, temperature]
    seeds = [base_seed, base_seed + 1, base_seed + 2]

    out1 = lmstudio_chat_completion_single(
        base_url, model, system_prompt, user_prompt,
        timeout=timeout, temperature=temps[0], top_p=top_p, max_tokens=max_tokens, api_key=api_key, seed=seeds[0]
    )
    out2 = lmstudio_chat_completion_single(
        base_url, model, system_prompt, user_prompt,
        timeout=timeout, temperature=temps[1], top_p=top_p, max_tokens=max_tokens, api_key=api_key, seed=seeds[1]
    )
    out3 = lmstudio_chat_completion_single(
        base_url, model, system_prompt, user_prompt,
        timeout=timeout, temperature=temps[2], top_p=top_p, max_tokens=max_tokens, api_key=api_key, seed=seeds[2]
    )
    return out1, out2, out3


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gera 3 simplificações por bula usando modelo servido pelo LM Studio (OpenAI-compatible)."
    )
    ap.add_argument("--model", required=True, help="Nome do modelo no LM Studio (ex.: Qwen3-8B-Instruct).")
    ap.add_argument("--input-parquet", required=True, help="Parquet de entrada.")
    ap.add_argument("--text-col", required=True, help="Coluna com o texto original da bula.")
    ap.add_argument("--output-dir", default=".", help="Diretório de saída (default: .).")
    ap.add_argument("--out-col-prefix", default="bula_simplificada", help="Prefixo das colunas (default: bula_simplificada).")

    ap.add_argument("--base-url", default="http://127.0.0.1:1234", help="Base URL do servidor do LM Studio.")
    ap.add_argument("--api-key", default="lm-studio", help="Bearer token (LM Studio geralmente ignora, mas exige header).")

    ap.add_argument("--temperature", type=float, default=0.7, help="Temperatura base (diversidade).")
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--sleep", type=float, default=0.0)

    ap.add_argument("--limit", type=int, default=0, help="Processa só N linhas (0 = tudo).")
    ap.add_argument("--resume", action="store_true", help="Se output existir, retoma e processa só faltantes.")
    ap.add_argument("--checkpoint-every", type=int, default=20, help="Checkpoint a cada N linhas.")

    ap.add_argument("--prefer-single-call", action="store_true",
                    help="Tenta usar n=3 numa única chamada; se não suportar, faz fallback automático.")
    ap.add_argument("--base-seed", type=int, default=12345, help="Seed base (usada para diversidade).")

    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--user-template", default=DEFAULT_USER_TEMPLATE)

    ap.add_argument("--id-col", default="", help="Opcional: coluna de id para logs (ex.: 'id').")
    args = ap.parse_args()

    if not os.path.exists(args.input_parquet):
        print(f"[ERRO] input-parquet não existe: {args.input_parquet}", file=sys.stderr)
        return 2

    df = pd.read_parquet(args.input_parquet)
    if args.text_col not in df.columns:
        print(f"[ERRO] text-col '{args.text_col}' não existe. Colunas: {list(df.columns)}", file=sys.stderr)
        return 2

    # Colunas de saída
    out_cols = [f"{args.out_col_prefix}_{i}" for i in (1, 2, 3)]

    model_tag = sanitize_for_filename(args.model)
    in_base = os.path.splitext(os.path.basename(args.input_parquet))[0]
    out_name = f"{in_base}__simplificado3__{model_tag}.parquet"
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, out_name)

    # Resume
    if args.resume and os.path.exists(out_path):
        df_out = pd.read_parquet(out_path)
        for c in df.columns:
            if c not in df_out.columns:
                df_out[c] = df[c].values
        df = df_out
        print(f"[INFO] Resume: carreguei {out_path}")
    else:
        for c in out_cols:
            if c not in df.columns:
                df[c] = pd.NA

    n = len(df)
    if args.limit and args.limit > 0:
        n = min(n, args.limit)

    # Define quais linhas precisam ser processadas (qualquer uma das 3 colunas faltando)
    to_process: List[int] = []
    for i in range(n):
        missing_any = False
        for c in out_cols:
            v = df.at[i, c]
            if pd.isna(v) or (isinstance(v, str) and not v.strip()):
                missing_any = True
                break
        if missing_any:
            to_process.append(i)

    print(f"[INFO] Linhas consideradas: {n}")
    print(f"[INFO] Linhas a processar: {len(to_process)}")
    print(f"[INFO] Saída: {out_path}")

    processed = 0
    failures = 0
    last_checkpoint = time.time()

    for idx in to_process:
        raw_text = df.at[idx, args.text_col]
        raw_text = "" if raw_text is None else str(raw_text).strip()

        row_id = None
        if args.id_col and args.id_col in df.columns:
            row_id = df.at[idx, args.id_col]
        prefix = f"idx={idx}" + (f" id={row_id}" if row_id is not None else "")

        if not raw_text:
            for c in out_cols:
                df.at[idx, c] = ""
            processed += 1
            continue

        user_prompt = args.user_template.format(text=raw_text)

        try:
            s1, s2, s3 = get_three_simplifications(
                base_url=args.base_url,
                model=args.model,
                system_prompt=args.system_prompt,
                user_prompt=user_prompt,
                timeout=args.timeout,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                api_key=args.api_key,
                prefer_single_call=args.prefer_single_call,
                base_seed=args.base_seed + idx * 10,  # muda seed por linha
            )
            s1 = clean_llm_output(s1)
            #print(s1+'\nDIVISAO\n')
            s2 = clean_llm_output(s2)
            #print(s2+'\nDIVISAO\n')
            s3 = clean_llm_output(s3)
            #print(s3+'\nDIVISAO\n')
            df.at[idx, out_cols[0]] = s1
            df.at[idx, out_cols[1]] = s2
            df.at[idx, out_cols[2]] = s3
            processed += 1
            print(f"[OK] {prefix} | chars_in={len(raw_text)} | out=({len(s1)},{len(s2)},{len(s3)})")
        except Exception as e:
            failures += 1
            for c in out_cols:
                df.at[idx, c] = pd.NA
            print(f"[ERRO] {prefix} | {e}", file=sys.stderr)

        if args.checkpoint_every > 0 and processed % args.checkpoint_every == 0:
            df.to_parquet(out_path, index=False)
            dt = time.time() - last_checkpoint
            last_checkpoint = time.time()
            print(f"[CHECKPOINT] salvei {out_path} | +{args.checkpoint_every} linhas | dt={dt:.1f}s")

        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

    df.to_parquet(out_path, index=False)
    print(f"[DONE] Salvo: {out_path} | processadas={processed} | falhas={failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
