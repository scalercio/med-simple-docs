#!/usr/bin/env python3
"""
Compara o Qwen3.5 original com um checkpoint fine-tuned em bulas.

Fluxo:
1. Lê data/bulas_all_v2.parquet.
2. Executa inferência com o modelo base.
3. Libera a GPU.
4. Executa inferência com o modelo fine-tuned.
5. Calcula D-SARI e SIM para as duas saídas.
6. Salva incrementalmente em Parquet, permitindo retomada.

IMPORTANTE:
Ajuste o import das funções calculate_d_sari e
calcular_similaridade_sliding para o módulo usado no seu projeto.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm
from unsloth import FastVisionModel


DEFAULT_INSTRUCTION = (
    "Simplifique a bula abaixo, mas mantenha o sentido original. "
    "Retorne somente o texto simplificado.\n\n"
    "Siga estas orientações:\n"
    "1. Use linguagem simples, clara e direta.\n"
    "2. Preserve todas as informações importantes do texto original.\n"
    "3. Não acrescente informações que não estejam no texto original.\n"
    "4. Explique ou substitua termos técnicos quando possível.\n"
    "5. Mantenha avisos, contraindicações, doses e orientações de segurança.\n"
    "6. Não inclua comentários sobre a tarefa nem introduções como "
    "'Texto simplificado'."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara inferência do Qwen3.5 base e fine-tuned."
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/bulas_all_v2.parquet"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/bulas_all_v2_qwen35_comparison.parquet"),
    )

    parser.add_argument(
        "--base-model",
        default="unsloth/Qwen3.5-9B",
        help="Modelo Qwen original usado antes do fine-tuning.",
    )
    parser.add_argument(
        "--finetuned-model",
        required=True,
        help=(
            "Diretório do checkpoint/adapter salvo pelo Unsloth ou nome de um "
            "repositório no Hugging Face."
        ),
    )

    parser.add_argument(
        "--text-col",
        default="docs",
        help="Coluna com a bula original.",
    )
    parser.add_argument(
        "--reference-col",
        default="rev1",
        help="Coluna com a simplificação de referência para D-SARI.",
    )
    parser.add_argument(
        "--base-output-col",
        default="qwen35_base_simplified",
    )
    parser.add_argument(
        "--finetuned-output-col",
        default="qwen35_finetuned_simplified",
    )

    parser.add_argument(
        "--metrics-module",
        default="src.metrics",
        help=(
            "Módulo que contém calculate_d_sari e "
            "calcular_similaridade_sliding. Ex.: src.metrics"
        ),
    )

    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Salva o parquet a cada N novas gerações.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Primeiro índice posicional que poderá ser processado.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Número máximo de linhas; útil para testes.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("base", "finetuned"),
        default=("base", "finetuned"),
        help="Permite executar apenas um dos modelos.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Gera novamente mesmo quando a coluna já possui saída.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def atomic_save_parquet(df: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    df.to_parquet(tmp_file, index=False)
    os.replace(tmp_file, output_file)


def load_or_initialize_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.output_file.exists() and not args.overwrite:
        print(f"[INFO] Retomando arquivo existente: {args.output_file}")
        df = pd.read_parquet(args.output_file)
    else:
        print(f"[INFO] Lendo entrada: {args.input_file}")
        df = pd.read_parquet(args.input_file)

    required = [args.text_col]
    if not args.skip_metrics:
        required.append(args.reference_col)

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"Colunas ausentes no dataframe: {missing}. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )

    for output_col in (args.base_output_col, args.finetuned_output_col):
        if output_col not in df.columns:
            df[output_col] = pd.Series(pd.NA, index=df.index, dtype="string")

    return df


def build_prompt(original_text: str) -> str:
    return (
        DEFAULT_INSTRUCTION
        + "\n\nBula original:\n"
        + original_text
        + "\n\nBula simplificada: "
    )


def load_model_and_processor(
    model_name: str,
    args: argparse.Namespace,
):
    print(f"[INFO] Carregando modelo: {model_name}")

    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    FastVisionModel.for_inference(model)
    model.eval()

    return model, processor


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)

    if isinstance(batch, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    raise TypeError(f"Tipo de batch não suportado: {type(batch)!r}")


@torch.inference_mode()
def generate_one(
    model: Any,
    processor: Any,
    original_text: str,
    args: argparse.Namespace,
) -> str:
    prompt = build_prompt(original_text)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    input_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Reserva espaço para a resposta. O truncamento afeta apenas a entrada.
    max_input_tokens = args.max_seq_length - args.max_new_tokens
    if max_input_tokens <= 0:
        raise ValueError(
            "--max-new-tokens precisa ser menor que --max-seq-length."
        )

    inputs = processor(
        text=[input_text],
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        padding=False,
    )

    device = next(model.parameters()).device
    inputs = move_batch_to_device(inputs, device)
    input_length = inputs["input_ids"].shape[1]

    do_sample = args.temperature > 0

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "do_sample": do_sample,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }

    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
        if args.min_p > 0:
            generation_kwargs["min_p"] = args.min_p

    generated_ids = model.generate(**inputs, **generation_kwargs)
    response_ids = generated_ids[:, input_length:]

    generated_text = processor.batch_decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return generated_text


def infer_dataframe(
    df: pd.DataFrame,
    model_name: str,
    output_col: str,
    label: str,
    args: argparse.Namespace,
) -> None:
    model, processor = load_model_and_processor(model_name, args)

    stop = len(df)
    if args.limit is not None:
        stop = min(stop, args.start_index + args.limit)

    positions = range(args.start_index, stop)
    generated_since_save = 0

    try:
        for position in tqdm(positions, desc=f"Inferência {label}"):
            row_index = df.index[position]

            existing = clean_text(df.at[row_index, output_col])
            if existing and not args.overwrite:
                continue

            original_text = clean_text(df.at[row_index, args.text_col])
            if not original_text:
                df.at[row_index, output_col] = pd.NA
                continue

            try:
                generated_text = generate_one(
                    model=model,
                    processor=processor,
                    original_text=original_text,
                    args=args,
                )
                df.at[row_index, output_col] = generated_text or pd.NA

            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                print(
                    f"\n[ERRO] CUDA OOM em position={position}, "
                    f"index={row_index}: {exc}",
                    file=sys.stderr,
                )
                df.at[row_index, output_col] = pd.NA

            except Exception as exc:
                print(
                    f"\n[ERRO] Inferência {label} em position={position}, "
                    f"index={row_index}: {exc}",
                    file=sys.stderr,
                )
                df.at[row_index, output_col] = pd.NA

            generated_since_save += 1
            if generated_since_save >= args.save_every:
                atomic_save_parquet(df, args.output_file)
                generated_since_save = 0

    finally:
        atomic_save_parquet(df, args.output_file)

        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print(f"[INFO] Modelo {label} removido da memória.")


def import_metric_functions(module_name: str):
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Não foi possível importar o módulo de métricas '{module_name}'. "
            "Use --metrics-module com o caminho correto."
        ) from exc

    missing = [
        name
        for name in ("calculate_d_sari", "calcular_similaridade_sliding")
        if not hasattr(module, name)
    ]
    if missing:
        raise AttributeError(
            f"O módulo '{module_name}' não contém: {missing}"
        )

    return module.calculate_d_sari, module.calcular_similaridade_sliding


def calculate_d_sari_columns(
    df: pd.DataFrame,
    output_col: str,
    prefix: str,
    calculate_d_sari: Any,
    args: argparse.Namespace,
) -> None:
    metric_columns = {
        "d_sari": f"{prefix}_d_sari",
        "d_sari_add": f"{prefix}_d_sari_add",
        "d_sari_keep": f"{prefix}_d_sari_keep",
        "d_sari_del": f"{prefix}_d_sari_del",
    }

    for column in metric_columns.values():
        if column not in df.columns:
            df[column] = pd.NA

    print(f"[INFO] Calculando D-SARI para {output_col}...")

    for position in tqdm(range(len(df)), desc=f"D-SARI {prefix}"):
        row_index = df.index[position]

        original_text = clean_text(df.at[row_index, args.text_col])
        reference_text = clean_text(df.at[row_index, args.reference_col])
        generated_text = clean_text(df.at[row_index, output_col])

        if not (original_text and reference_text and generated_text):
            for column in metric_columns.values():
                df.at[row_index, column] = pd.NA
            continue

        try:
            score, add_score, keep_score, del_score = calculate_d_sari(
                original_text,
                generated_text,
                reference_text,
            )
            df.at[row_index, metric_columns["d_sari"]] = score
            df.at[row_index, metric_columns["d_sari_add"]] = add_score
            df.at[row_index, metric_columns["d_sari_keep"]] = keep_score
            df.at[row_index, metric_columns["d_sari_del"]] = del_score

        except Exception as exc:
            for column in metric_columns.values():
                df.at[row_index, column] = pd.NA
            print(
                f"[ERRO] D-SARI {prefix}, position={position}, "
                f"index={row_index}: {exc}",
                file=sys.stderr,
            )


def calculate_sim_column(
    df: pd.DataFrame,
    output_col: str,
    prefix: str,
    calcular_similaridade_sliding: Any,
) -> None:
    sim_col = f"{prefix}_sim"
    print(f"[INFO] Calculando SIM para {output_col}...")

    try:
        sim_scores = calcular_similaridade_sliding(df, output_col)
        if len(sim_scores) != len(df):
            raise ValueError(
                f"A função retornou {len(sim_scores)} scores para "
                f"{len(df)} linhas."
            )
        df[sim_col] = sim_scores

    except Exception as exc:
        print(
            f"[ERRO] Falha ao calcular SIM para '{output_col}'. "
            "Verifique se calcular_similaridade_sliding aceita "
            "(df, output_col) e se reconhece a coluna original.",
            file=sys.stderr,
        )
        print(f"Detalhe: {exc}", file=sys.stderr)
        df[sim_col] = pd.NA


def calculate_metrics(df: pd.DataFrame, args: argparse.Namespace) -> None:
    calculate_d_sari, calcular_similaridade_sliding = import_metric_functions(
        args.metrics_module
    )

    metric_jobs = []
    if "base" in args.models:
        metric_jobs.append(
            (args.base_output_col, "qwen35_base")
        )
    if "finetuned" in args.models:
        metric_jobs.append(
            (args.finetuned_output_col, "qwen35_finetuned")
        )

    for output_col, prefix in metric_jobs:
        calculate_d_sari_columns(
            df=df,
            output_col=output_col,
            prefix=prefix,
            calculate_d_sari=calculate_d_sari,
            args=args,
        )
        atomic_save_parquet(df, args.output_file)

        calculate_sim_column(
            df=df,
            output_col=output_col,
            prefix=prefix,
            calcular_similaridade_sliding=calcular_similaridade_sliding,
        )
        atomic_save_parquet(df, args.output_file)


def print_summary(df: pd.DataFrame, args: argparse.Namespace) -> None:
    columns = [
        "qwen35_base_d_sari",
        "qwen35_base_sim",
        "qwen35_finetuned_d_sari",
        "qwen35_finetuned_sim",
    ]
    available = [column for column in columns if column in df.columns]
    if not available:
        return

    print("\n[RESUMO] Médias:")
    print(df[available].apply(pd.to_numeric, errors="coerce").mean())


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível.")

    torch.backends.cuda.matmul.allow_tf32 = True

    df = load_or_initialize_dataframe(args)
    atomic_save_parquet(df, args.output_file)

    start_time = time.time()

    if "base" in args.models:
        infer_dataframe(
            df=df,
            model_name=args.base_model,
            output_col=args.base_output_col,
            label="base",
            args=args,
        )

    if "finetuned" in args.models:
        infer_dataframe(
            df=df,
            model_name=args.finetuned_model,
            output_col=args.finetuned_output_col,
            label="fine-tuned",
            args=args,
        )

    if not args.skip_metrics:
        calculate_metrics(df, args)

    atomic_save_parquet(df, args.output_file)
    print_summary(df, args)

    elapsed = time.time() - start_time
    print(f"\n[INFO] Concluído em {elapsed / 60:.1f} minutos.")
    print(f"[INFO] Resultado salvo em: {args.output_file}")


if __name__ == "__main__":
    main()
