from __future__ import annotations

import csv
import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ============================================================
# KONFIGURASI
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SEED = 42
MAX_LENGTH = 128
NUM_EPOCHS = 1

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "data" / "lora_dataset.json"
OUTPUT_DIR = SCRIPT_DIR / "outputs_lora"
ADAPTER_DIR = OUTPUT_DIR / "lora_adapter"

COMPARISON_FILE = OUTPUT_DIR / "before_after_comparison.csv"
HISTORY_FILE = OUTPUT_DIR / "training_history.csv"
LOSS_FILE = OUTPUT_DIR / "training_loss.png"
SUMMARY_FILE = OUTPUT_DIR / "training_summary.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32

SYSTEM_PROMPT = (
    "Kamu adalah asisten pembelajaran Generative AI. "
    "Jawab dalam Bahasa Indonesia dengan jelas, akurat, ringkas, "
    "dan mudah dipahami mahasiswa."
)


# ============================================================
# UTILITAS
# ============================================================

def print_environment() -> None:
    print("=" * 72)
    print("LAB 4.1 PART B — LoRA FINE-TUNING")
    print("=" * 72)
    print(f"Model           : {MODEL_ID}")
    print(f"Dataset         : {DATA_FILE}")
    print(f"Device          : {DEVICE}")
    print(f"CUDA tersedia   : {CUDA_AVAILABLE}")
    print(f"LoRA rank       : {LORA_RANK}")
    print(f"Epoch           : {NUM_EPOCHS}")
    print(f"Max length      : {MAX_LENGTH}")

    if CUDA_AVAILABLE:
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        print(
            "VRAM            :",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

    print("=" * 72)


def load_data() -> list[dict[str, str]]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_FILE}"
        )

    with DATA_FILE.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    if not 50 <= len(rows) <= 200:
        raise ValueError(
            f"Dataset harus berisi 50–200 contoh. "
            f"Jumlah saat ini: {len(rows)}"
        )

    print(f"Dataset berhasil dibaca: {len(rows)} contoh")
    return rows


def format_training_text(
    example: dict[str, str],
    tokenizer: Any,
) -> dict[str, str]:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": example["instruction"],
        },
        {
            "role": "assistant",
            "content": example["answer"],
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def generate_answer(
    model: Any,
    tokenizer: Any,
    question: str,
) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_device = model.get_input_embeddings().weight.device

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    model.eval()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return response or "(Tidak ada jawaban.)"


def save_training_results(
    trainer: Trainer,
    comparisons: list[dict[str, str]],
) -> None:
    history = trainer.state.log_history

    if history:
        columns = sorted(
            {
                key
                for row in history
                for key in row.keys()
            }
        )

        with HISTORY_FILE.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(history)

    loss_rows = [
        row
        for row in history
        if "loss" in row
    ]

    if loss_rows:
        steps = [
            row.get("step", index + 1)
            for index, row in enumerate(loss_rows)
        ]

        losses = [row["loss"] for row in loss_rows]

        plt.figure(figsize=(8, 5))
        plt.plot(steps, losses, marker="o")
        plt.title("Training Loss LoRA")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(LOSS_FILE, dpi=160)
        plt.close()

    with COMPARISON_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question",
                "before_lora",
                "after_lora",
            ],
        )

        writer.writeheader()
        writer.writerows(comparisons)


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:
    set_seed(SEED)
    print_environment()

    rows = load_data()

    print("\n[1/7] Memuat tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("[2/7] Memuat base model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )

    model = model.to(DEVICE)
    model.config.use_cache = True

    test_questions = [
        "Apa yang dimaksud dengan LoRA?",
        "Jelaskan konsep diffusion model secara singkat.",
        "Apa yang dimaksud dengan RAG?",
    ]

    print("\n[3/7] Prediksi sebelum fine-tuning...")

    before_answers: dict[str, str] = {}

    for question in test_questions:
        answer = generate_answer(
            model,
            tokenizer,
            question,
        )

        before_answers[question] = answer

        print("\nPertanyaan:", question)
        print("Sebelum LoRA:", answer)

    print("\n[4/7] Memasang adapter LoRA...")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n[5/7] Menyiapkan dataset...")

    dataset = Dataset.from_list(rows)

    dataset = dataset.map(
        lambda item: format_training_text(
            item,
            tokenizer,
        )
    )

    def tokenize(example: dict[str, str]) -> dict[str, Any]:
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        fp16=CUDA_AVAILABLE,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\n[6/7] Memulai training LoRA...")

    train_result = trainer.train()

    print(
        f"Training selesai. Loss akhir: "
        f"{train_result.training_loss:.6f}"
    )

    model.save_pretrained(
        ADAPTER_DIR,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(ADAPTER_DIR)

    print("\n[7/7] Prediksi setelah fine-tuning...")

    model.config.use_cache = True
    comparisons: list[dict[str, str]] = []

    for question in test_questions:
        after_answer = generate_answer(
            model,
            tokenizer,
            question,
        )

        comparisons.append(
            {
                "question": question,
                "before_lora": before_answers[question],
                "after_lora": after_answer,
            }
        )

        print("\n" + "=" * 72)
        print("Pertanyaan:", question)
        print("\nSebelum LoRA:")
        print(before_answers[question])
        print("\nSesudah LoRA:")
        print(after_answer)

    save_training_results(
        trainer,
        comparisons,
    )

    summary = {
        "model": MODEL_ID,
        "dataset_size": len(rows),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "epochs": NUM_EPOCHS,
        "max_length": MAX_LENGTH,
        "final_loss": float(
            train_result.training_loss
        ),
        "device": DEVICE,
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("LAB 4.1 PART B BERHASIL")
    print("=" * 72)
    print(f"Adapter       : {ADAPTER_DIR}")
    print(f"Perbandingan  : {COMPARISON_FILE}")
    print(f"Loss curve    : {LOSS_FILE}")
    print(f"Ringkasan     : {SUMMARY_FILE}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()

    except torch.cuda.OutOfMemoryError:
        print("\nGPU kehabisan VRAM.")
        print("Tutup browser, game, dan aplikasi yang memakai GPU.")
        print(
            "Jika masih gagal, ubah MODEL_ID menjadi "
            "'Qwen/Qwen2.5-0.5B-Instruct'."
        )

        torch.cuda.empty_cache()

    except Exception as error:
        print("\nPROGRAM ERROR")
        print(type(error).__name__, "-", error)
        traceback.print_exc()