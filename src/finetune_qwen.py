# finetune_qwen35_9b_bulas.py

import argparse
import pandas as pd
import torch

from datasets import Dataset
from unsloth import FastVisionModel, FastLanguageModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

class AssistantOnlyCollator:
    def __init__(self, base_collator, tokenizer):
        self.base_collator = base_collator
        self.tokenizer = tokenizer

        raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

        self.assistant_ids = raw_tokenizer(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )["input_ids"]

        print("assistant_ids:", self.assistant_ids)

    def __call__(self, features):
        batch = self.base_collator(features)

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        for i in range(input_ids.shape[0]):
            ids = input_ids[i].tolist()

            start = None
            for j in range(len(ids) - len(self.assistant_ids) + 1):
                if ids[j:j + len(self.assistant_ids)] == self.assistant_ids:
                    start = j + len(self.assistant_ids)
                    break

            if start is None:
                labels[i, :] = -100
            else:
                labels[i, :start] = -100

            if "attention_mask" in batch:
                labels[i][batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        return batch


instruction = """Simplifique a bula de medicamento abaixo para pacientes leigos.

Regras:
- mantenha as informações essenciais;
- não invente informações;
- não remova contraindicações, alertas ou efeitos adversos importantes;
- use linguagem clara e acessível;
- preserve o sentido médico e farmacêutico."""


def is_valid_after_collator(example, data_collator, min_valid_labels=20, min_label_ratio=0.05):
    try:
        batch = data_collator([example])
    except Exception as e:
        print(f"Erro ao processar exemplo: {e}")
        return False

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    if input_ids.numel() == 0:
        return False

    if input_ids.shape[0] != labels.shape[0]:
        return False

    valid_labels = (labels != -100).sum().item()

    if valid_labels < min_valid_labels:
        return False

    label_ratio = valid_labels / labels.shape[0]

    if label_ratio < min_label_ratio:
        return False

    return True

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

    parser.add_argument("--train_file", default="data/splits/train_80.parquet")
    parser.add_argument("--val_file", default="data/splits/val_20.parquet")

    parser.add_argument("--original_col", default="informacoes_ao_paciente")
    parser.add_argument("--simplified_col", default="qwen3.6_27b_simplified")

    parser.add_argument("--model_name", default="unsloth/Qwen3.5-9B")
    parser.add_argument("--output_dir", default="outputs_qwen35_9b_bulas")

    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=2.0)

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
    
    print("Modelo carregado.", flush=True)

    print("Aplicando LoRA...", flush=True)

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
    
    print("LoRA aplicada.", flush=True)

    print("Carregando datasets...", flush=True)

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

    base_collator = UnslothVisionDataCollator(model, tokenizer)

    data_collator = AssistantOnlyCollator(
        base_collator=base_collator,
        tokenizer=tokenizer,
    )
    
    print(f"Treino antes do filtro: {len(train_dataset)}")

    train_dataset = train_dataset.filter(
        lambda example: is_valid_after_collator(
            example,
            data_collator=data_collator,
            min_valid_labels=20,
            min_label_ratio=0.05,
        ),
        num_proc=1,
    )

    print(f"Treino após filtro: {len(train_dataset)}")


    print(f"Validação antes do filtro: {len(eval_dataset)}")

    eval_dataset = eval_dataset.filter(
        lambda example: is_valid_after_collator(
            example,
            data_collator=data_collator,
            min_valid_labels=20,
            min_label_ratio=0.05,
        ),
        num_proc=1,
    )

    print(f"Validação após filtro: {len(eval_dataset)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            max_seq_length=args.max_seq_length,

            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,

            #num_train_epochs=args.epochs,
            max_steps = 60,

            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",

            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=1.0,

            eval_strategy="steps",
            eval_steps=args.eval_steps,
            eval_accumulation_steps=1,
            prediction_loss_only=True,
            #assistant_only_loss=True,

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
            dataset_kwargs={"skip_prepare_dataset": True},
            torch_empty_cache_steps=1,
        ),
    )
#    batch = next(iter(trainer.get_train_dataloader()))
#
#    input_ids = batch["input_ids"]
#    labels = batch["labels"]
#    attention_mask = batch["attention_mask"]
#
#    print("input_ids:", input_ids.shape)
#    print("labels:", labels.shape)
#    print("attention_mask:", attention_mask.shape)
#
#    pad_positions = attention_mask == 0
#
#    print("Pads no batch:", pad_positions.sum().item())
#    print("Pads entrando na loss:", ((labels != -100) & pad_positions).sum().item())
#    
#    
#    
#    
#    i = 0
#
#    full_text = tokenizer.decode(
#        input_ids[i],
#        skip_special_tokens=False,
#    )
#
#    loss_ids = labels[i][labels[i] != -100]
#
#    loss_text = tokenizer.decode(
#        loss_ids,
#        skip_special_tokens=False,
#    )
#
#    print("=" * 80)
#    print("INPUT COMPLETO")
#    print("=" * 80)
#    print(full_text[:6000])
#
#    print("=" * 80)
#    print("TEXTO QUE ENTRA NA LOSS")
#    print("=" * 80)
#    print(loss_text[:6000])
#    
#    
#    
#    
#    i = 0
#
#    valid_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
#
#    print("Primeira posição com loss:", valid_positions[0].item())
#    print("Última posição com loss:", valid_positions[-1].item())
#    print("Total de tokens com loss:", len(valid_positions))
#
#    start = max(valid_positions[0].item() - 30, 0)
#    end = min(valid_positions[0].item() + 80, input_ids.shape[1])
#
#    print("=" * 80)
#    print("TRECHO EM TORNO DO INÍCIO DA LOSS")
#    print("=" * 80)
#
#    for pos in range(start, end):
#        token = tokenizer.decode([input_ids[i, pos].item()], skip_special_tokens=False)
#        marker = "<LOSS>" if labels[i, pos] != -100 else "      "
#        print(f"{pos:04d} {marker} {repr(token)}")
    

    trainer.train()

    #trainer.save_model(f"{args.output_dir}/best_lora_adapter")
    #tokenizer.save_pretrained(f"{args.output_dir}/best_lora_adapter")

    print("Treinamento finalizado.")
    print(f"Melhor adapter salvo em: {args.output_dir}/best_lora_adapter")


if __name__ == "__main__":
    main()