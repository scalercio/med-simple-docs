# finetune_qwen35_9b_bulas.py

import argparse
import pandas as pd
import torch

from datasets import Dataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


instruction = """Simplifique a bula de medicamento abaixo para pacientes leigos.

Regras:
- mantenha as informações essenciais;
- não invente informações;
- não remova contraindicações, alertas ou efeitos adversos importantes;
- use linguagem clara e acessível;
- preserve o sentido médico e farmacêutico."""


def convert_to_conversation(sample, original_col, simplified_col):
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": instruction + "\n\nBula original:\n" + str(sample[original_col]).strip(),
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": str(sample[simplified_col]).strip(),
                }
            ],
        },
    ]

    return {"messages": conversation}


def load_dataset_from_parquet(path, original_col, simplified_col):
    df = pd.read_parquet(path)

    df = df[[original_col, simplified_col]].dropna()
    df = df[
        (df[original_col].astype(str).str.strip() != "") &
        (df[simplified_col].astype(str).str.strip() != "")
    ]

    dataset = Dataset.from_pandas(df, preserve_index=False)

    dataset = dataset.map(
        lambda sample: convert_to_conversation(
            sample,
            original_col=original_col,
            simplified_col=simplified_col,
        ),
        remove_columns=dataset.column_names,
        num_proc=1,
    )

    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default="train_80.parquet")
    parser.add_argument("--val_file", default="val_20.parquet")

    parser.add_argument("--original_col", default="informacoes_ao_paciente")
    parser.add_argument("--simplified_col", default="simple_doc")

    parser.add_argument("--model_name", default="unsloth/Qwen3.5-9B")
    parser.add_argument("--output_dir", default="outputs_qwen35_9b_bulas")

    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)

    parser.add_argument("--seed", type=int, default=3407)

    args = parser.parse_args()

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        use_gradient_checkpointing = "unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,

        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    train_dataset = load_dataset_from_parquet(
        args.train_file,
        args.original_col,
        args.simplified_col,
    )

    eval_dataset = load_dataset_from_parquet(
        args.val_file,
        args.original_col,
        args.simplified_col,
    )

    print(f"Treino: {len(train_dataset)} exemplos")
    print(f"Validação: {len(eval_dataset)} exemplos")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            max_seq_length=args.max_seq_length,

            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,

            num_train_epochs=args.epochs,

            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",

            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=1.0,

            eval_strategy="steps",
            eval_steps=args.eval_steps,

            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=3,

            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            logging_steps=args.logging_steps,

            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),

            output_dir=args.output_dir,
            seed=args.seed,
            report_to="none",

            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": False},
        ),
    )

    trainer.train()

    trainer.save_model(f"{args.output_dir}/best_lora_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/best_lora_adapter")

    print("Treinamento finalizado.")
    print(f"Melhor adapter salvo em: {args.output_dir}/best_lora_adapter")


if __name__ == "__main__":
    main()