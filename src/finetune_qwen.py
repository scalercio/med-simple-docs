import argparse
import math
import random
from typing import Iterator, List, Optional

import types
import pandas as pd
import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from datasets import Dataset
from torch.utils.data import BatchSampler, DataLoader
from transformers.trainer_utils import seed_worker
from trl import SFTConfig, SFTTrainer

import inspect

print("SFTTrainer module:", SFTTrainer.__module__)
print("SFTTrainer file:", inspect.getfile(SFTTrainer))
print("SFTTrainer signature:")
print(inspect.signature(SFTTrainer.__init__))

INSTRUCTION = """Simplifique a bula de medicamento abaixo para pacientes leigos.

Regras:
- mantenha as informações essenciais;
- não invente informações;
- não remova contraindicações, alertas ou efeitos adversos importantes;
- use linguagem clara e acessível;
- preserve o sentido médico e farmacêutico."""


def find_subsequence(sequence: List[int], pattern: List[int], start: int = 0) -> Optional[int]:
    """Retorna a primeira posição de pattern em sequence, ou None."""
    if not pattern:
        return None

    last_start = len(sequence) - len(pattern)
    for pos in range(start, last_start + 1):
        if sequence[pos : pos + len(pattern)] == pattern:
            return pos
    return None


class AssistantOnlyCollator:
    """
    Usa o collator multimodal da Unsloth para tokenização/padding e substitui
    os labels para que somente a resposta do assistant participe da loss.
    """

    def __init__(self, base_collator, tokenizer):
        self.base_collator = base_collator
        self.tokenizer = tokenizer

        # Em Qwen3.5, tokenizer pode ser um processor multimodal.
        raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

        self.assistant_ids = raw_tokenizer(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )["input_ids"]

        self.end_ids = raw_tokenizer(
            "<|im_end|>",
            add_special_tokens=False,
        )["input_ids"]

        print("assistant_ids:", self.assistant_ids)
        print("assistant_end_ids:", self.end_ids)

    def __call__(self, features):
        # Campos auxiliares, como total_length, não devem chegar ao collator base.
        clean_features = [{"messages": feature["messages"]} for feature in features]
        batch = self.base_collator(clean_features)

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        for row in range(input_ids.shape[0]):
            ids = input_ids[row].tolist()
            assistant_marker = find_subsequence(ids, self.assistant_ids)

            if assistant_marker is None:
                labels[row, :] = -100
            else:
                response_start = assistant_marker + len(self.assistant_ids)
                labels[row, :response_start] = -100

            if "attention_mask" in batch:
                labels[row][batch["attention_mask"][row] == 0] = -100

        batch["labels"] = labels
        return batch

    def response_is_complete(self, input_ids: torch.Tensor) -> bool:
        """Confirma que o marcador final do assistant não foi truncado."""
        ids = input_ids.tolist()
        assistant_marker = find_subsequence(ids, self.assistant_ids)
        if assistant_marker is None:
            return False

        response_start = assistant_marker + len(self.assistant_ids)
        return find_subsequence(ids, self.end_ids, start=response_start) is not None


class LengthBucketBatchSampler(BatchSampler):
    """
    Forma batches com exemplos de comprimentos próximos.

    A cada época:
    1. ordena os índices pelo comprimento total;
    2. divide a lista em buckets maiores que um batch;
    3. embaralha exemplos dentro de cada bucket;
    4. forma os batches;
    5. embaralha a ordem dos batches.

    Isso reduz padding sem ordenar rigidamente toda a época do menor para o maior.
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        bucket_multiplier: int = 50,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 3407,
    ):
        if batch_size < 1:
            raise ValueError("batch_size deve ser >= 1")
        if bucket_multiplier < 1:
            raise ValueError("bucket_multiplier deve ser >= 1")

        self.lengths = list(map(int, lengths))
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_multiplier
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        sorted_indices = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)

        buckets = [
            sorted_indices[start : start + self.bucket_size]
            for start in range(0, len(sorted_indices), self.bucket_size)
        ]

        batches: List[List[int]] = []
        for bucket in buckets:
            if self.shuffle:
                rng.shuffle(bucket)

            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        self.epoch += 1
        
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)

def make_bucketed_train_dataloader(bucket_multiplier):
    def get_bucketed_train_dataloader(self):
        batch_sampler = LengthBucketBatchSampler(
            lengths=self.train_dataset["total_length"],
            batch_size=self.args.per_device_train_batch_size,
            bucket_multiplier=bucket_multiplier,
            shuffle=True,
            seed=self.args.seed,
            drop_last=self.args.dataloader_drop_last,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    return get_bucketed_train_dataloader


class BucketedSFTTrainer(SFTTrainer):
    """SFTTrainer com bucketing por total_length apenas no treino."""

    def __init__(self, *args, bucket_multiplier: int = 50, **kwargs):
        self.bucket_multiplier = bucket_multiplier
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requer train_dataset.")

        if "total_length" not in self.train_dataset.column_names:
            raise ValueError("train_dataset precisa conter a coluna total_length.")

        batch_sampler = LengthBucketBatchSampler(
            lengths=self.train_dataset["total_length"],
            batch_size=self.args.per_device_train_batch_size,
            bucket_multiplier=self.bucket_multiplier,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
            seed=self.args.seed,
        )

        dataloader_kwargs = {
            "batch_sampler": batch_sampler,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "worker_init_fn": seed_worker,
        }

        if self.args.dataloader_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.args.dataloader_persistent_workers
            if self.args.dataloader_prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = DataLoader(self.train_dataset, **dataloader_kwargs)
        return self.accelerator.prepare(dataloader)


def convert_to_conversation(sample, original_col, simplified_col):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            INSTRUCTION
                            + "\n\nBula original:\n"
                            + str(sample[original_col]).strip()
                        ),
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
    }


def load_dataset_from_parquet(path, original_col, simplified_col):
    df = pd.read_parquet(path)
    df = df[[original_col, simplified_col]].dropna()
    df = df[
        (df[original_col].astype(str).str.strip() != "")
        & (df[simplified_col].astype(str).str.strip() != "")
    ]

    dataset = Dataset.from_pandas(df, preserve_index=False)
    return dataset.map(
        lambda sample: convert_to_conversation(
            sample,
            original_col=original_col,
            simplified_col=simplified_col,
        ),
        remove_columns=dataset.column_names,
        num_proc=1,
    )


def add_length_and_validity(
    example,
    data_collator,
    min_valid_labels=20,
    min_label_ratio=0.05,
):
    """
    Executa exatamente o mesmo collator usado no treino para medir:
    - comprimento real após tokenização/truncamento;
    - número e proporção de labels úteis;
    - presença do fim da resposta, garantindo que ela não foi truncada.
    """
    try:
        batch = data_collator([example])
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        attention_mask = batch.get("attention_mask")

        if input_ids.numel() == 0 or input_ids.shape[0] != labels.shape[0]:
            return {"total_length": 0, "valid_labels": 0, "label_ratio": 0.0, "is_valid": False}

        if attention_mask is not None:
            total_length = int(attention_mask[0].sum().item())
        else:
            total_length = int(input_ids.numel())

        valid_labels = int((labels != -100).sum().item())
        label_ratio = valid_labels / max(total_length, 1)
        complete_response = data_collator.response_is_complete(input_ids)

        is_valid = (
            complete_response
            and valid_labels >= min_valid_labels
            and label_ratio >= min_label_ratio
        )

        return {
            "total_length": total_length,
            "valid_labels": valid_labels,
            "label_ratio": float(label_ratio),
            "is_valid": bool(is_valid),
        }

    except Exception as exc:
        print(f"Erro ao analisar exemplo: {exc}")
        return {"total_length": 0, "valid_labels": 0, "label_ratio": 0.0, "is_valid": False}


def prepare_dataset(dataset, data_collator, keep_length: bool, description: str):
    print(f"{description} antes do filtro: {len(dataset)}")

    dataset = dataset.map(
        lambda example: add_length_and_validity(
            example,
            data_collator=data_collator,
            min_valid_labels=20,
            min_label_ratio=0.05,
        ),
        num_proc=1,
        desc=f"Tokenizando e medindo {description.lower()}",
    )

    dataset = dataset.filter(
        lambda example: example["is_valid"],
        num_proc=1,
        desc=f"Filtrando {description.lower()}",
    )

    columns_to_remove = ["valid_labels", "label_ratio", "is_valid"]
    if not keep_length:
        columns_to_remove.append("total_length")
    dataset = dataset.remove_columns(columns_to_remove)

    print(f"{description} após o filtro: {len(dataset)}")
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
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument(
        "--bucket_multiplier",
        type=int,
        default=50,
        help="Tamanho do bucket = batch_size * bucket_multiplier.",
    )

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
        use_gradient_checkpointing="unsloth",
    )
    print("Modelo carregado.", flush=True)

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

    base_collator = UnslothVisionDataCollator(model, tokenizer)
    data_collator = AssistantOnlyCollator(base_collator, tokenizer)

    # Mantemos total_length apenas no treino, pois o sampler precisa da coluna.
    train_dataset = prepare_dataset(
        train_dataset,
        data_collator,
        keep_length=True,
        description="Treino",
    )
    eval_dataset = prepare_dataset(
        eval_dataset,
        data_collator,
        keep_length=False,
        description="Validação",
    )

    lengths = train_dataset["total_length"]
    print(
        "Comprimentos do treino: "
        f"min={min(lengths)}, média={sum(lengths)/len(lengths):.1f}, max={max(lengths)}"
    )
    print(
        f"Bucketing: batch={args.batch_size}, "
        f"bucket_size={args.batch_size * args.bucket_multiplier}"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
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
    
    trainer.get_train_dataloader = types.MethodType(
        make_bucketed_train_dataloader(
            bucket_multiplier=args.bucket_multiplier,
        ),
        trainer,
    )

    trainer.train()

    best_dir = f"{args.output_dir}/best_lora_adapter"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    print("Treinamento finalizado.")
    print(f"Melhor adapter salvo em: {best_dir}")


if __name__ == "__main__":
    main()