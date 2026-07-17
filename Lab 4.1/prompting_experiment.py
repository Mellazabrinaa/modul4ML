from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)


# ============================================================
# LAB 4.1 PART A — PROMPT ENGINEERING
# ============================================================
# Eksperimen:
# 1. Classification
# 2. Summarization
# 3. Style transfer
# 4. Variasi temperature
# 5. Variasi top-p
# 6. Variasi system prompt
# ============================================================


# ============================================================
# 1. KONFIGURASI
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SEED = 42
MAX_NEW_TOKENS = 150

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_prompting"

CSV_FILE = OUTPUT_DIR / "prompting_results.csv"
JSON_FILE = OUTPUT_DIR / "prompting_results.json"
REPORT_FILE = OUTPUT_DIR / "prompting_report.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SYSTEM PROMPT UTAMA
# ============================================================

DEFAULT_SYSTEM_PROMPT = (
    "Kamu adalah asisten pembelajaran machine learning. "
    "Jawablah menggunakan Bahasa Indonesia yang jelas, ringkas, "
    "akurat, dan mudah dipahami mahasiswa."
)


# ============================================================
# 3. INFORMASI ENVIRONMENT
# ============================================================

def print_environment_info() -> None:
    """Menampilkan informasi Python, PyTorch, dan GPU."""

    print("=" * 72)
    print("LAB 4.1 PART A — PROMPT ENGINEERING")
    print("=" * 72)

    print(f"Model          : {MODEL_ID}")
    print(f"PyTorch        : {torch.__version__}")
    print(f"CUDA tersedia  : {CUDA_AVAILABLE}")
    print(f"Device         : {DEVICE}")
    print(f"Data type      : {DTYPE}")
    print(f"Seed           : {SEED}")
    print(f"Folder output  : {OUTPUT_DIR}")

    if CUDA_AVAILABLE:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
        )

        print(f"GPU            : {gpu_name}")
        print(f"VRAM           : {gpu_memory:.2f} GB")

    print("=" * 72)


# ============================================================
# 4. MEMUAT MODEL
# ============================================================

def load_model_and_tokenizer() -> tuple[Any, Any]:
    """Memuat tokenizer dan model Qwen."""

    print("\n[1/7] Memuat tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer berhasil dimuat.")

    print("\n[2/7] Memuat model...")
    print("Model akan diunduh saat pertama kali dijalankan.")

    model_kwargs: dict[str, Any] = {
        "dtype": DTYPE,
        "low_cpu_mem_usage": True,
    }

    if CUDA_AVAILABLE:
        # Jika VRAM tidak cukup, sebagian lapisan dapat dipindahkan
        # secara otomatis ke RAM.
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        **model_kwargs,
    )

    if not CUDA_AVAILABLE:
        model = model.to("cpu")

    model.eval()

    print("Model berhasil dimuat.")

    input_device = model.get_input_embeddings().weight.device
    print(f"Input model berada di: {input_device}")

    return model, tokenizer


# ============================================================
# 5. FUNGSI GENERASI TEKS
# ============================================================

def generate_response(
    model: Any,
    tokenizer: Any,
    user_prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[str, float]:
    """
    Menghasilkan jawaban menggunakan chat template Qwen.

    Parameter sampling hanya digunakan jika do_sample=True.
    """

    set_seed(SEED)

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    input_device = model.get_input_embeddings().weight.device

    model_inputs = {
        key: value.to(input_device)
        for key, value in model_inputs.items()
    }

    generation_arguments: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": 1.05,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generation_arguments["temperature"] = temperature
        generation_arguments["top_p"] = top_p

    start_time = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            **generation_arguments,
        )

    elapsed_time = time.perf_counter() - start_time

    input_length = model_inputs["input_ids"].shape[1]
    generated_tokens = generated[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    if not response:
        response = "(Model tidak menghasilkan jawaban.)"

    return response, elapsed_time


# ============================================================
# 6. MENYIMPAN SATU HASIL
# ============================================================

def add_result(
    results: list[dict[str, Any]],
    *,
    experiment: str,
    variant: str,
    prompt: str,
    system_prompt: str,
    response: str,
    temperature: float | None,
    top_p: float | None,
    do_sample: bool,
    elapsed_time: float,
) -> None:
    """Menambahkan hasil eksperimen ke daftar."""

    result = {
        "experiment": experiment,
        "variant": variant,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "response": response,
        "generation_time_seconds": round(elapsed_time, 3),
        "seed": SEED,
    }

    results.append(result)

    print("\n" + "-" * 72)
    print(f"Eksperimen : {experiment}")
    print(f"Varian     : {variant}")
    print(f"Temperature: {temperature}")
    print(f"Top-p      : {top_p}")
    print(f"Waktu      : {elapsed_time:.2f} detik")

    print("\nPrompt:")
    print(prompt)

    print("\nJawaban:")
    print(response)

    print("-" * 72)


# ============================================================
# 7. CORE TASKS
# ============================================================

def run_core_tasks(
    model: Any,
    tokenizer: Any,
    results: list[dict[str, Any]],
) -> None:
    """Classification, summarization, dan style transfer."""

    print("\n[3/7] Menjalankan tugas utama prompt engineering...")

    # --------------------------------------------------------
    # A. CLASSIFICATION
    # --------------------------------------------------------

    classification_prompt = """
Klasifikasikan sentimen ulasan berikut menjadi salah satu label:
POSITIF, NETRAL, atau NEGATIF.

Ulasan:
"Aplikasi pembelajaran ini sangat membantu saya memahami materi,
tetapi proses membuka halaman awal terkadang cukup lambat."

Jawab dengan format:
Label: ...
Alasan: ...
""".strip()

    response, elapsed = generate_response(
        model,
        tokenizer,
        classification_prompt,
        do_sample=False,
        max_new_tokens=80,
    )

    add_result(
        results,
        experiment="Classification",
        variant="Sentiment analysis",
        prompt=classification_prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response=response,
        temperature=None,
        top_p=None,
        do_sample=False,
        elapsed_time=elapsed,
    )

    # --------------------------------------------------------
    # B. SUMMARIZATION
    # --------------------------------------------------------

    summarization_prompt = """
Ringkas paragraf berikut menjadi tepat satu kalimat:

Generative AI merupakan cabang kecerdasan buatan yang mampu
menghasilkan konten baru seperti teks, gambar, audio, video, dan
model tiga dimensi. Teknologi ini mempelajari pola dalam data
pelatihan, kemudian menggunakan pola tersebut untuk membentuk
keluaran baru. Meskipun memiliki banyak manfaat, penggunaannya
juga menimbulkan tantangan seperti bias, pelanggaran hak cipta,
misinformasi, dan penyalahgunaan deepfake.
""".strip()

    response, elapsed = generate_response(
        model,
        tokenizer,
        summarization_prompt,
        do_sample=False,
        max_new_tokens=100,
    )

    add_result(
        results,
        experiment="Summarization",
        variant="One-sentence summary",
        prompt=summarization_prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response=response,
        temperature=None,
        top_p=None,
        do_sample=False,
        elapsed_time=elapsed,
    )

    # --------------------------------------------------------
    # C. STYLE TRANSFER
    # --------------------------------------------------------

    style_prompt = """
Ubah pesan informal berikut menjadi bahasa formal untuk dikirim
kepada dosen:

"Pak, saya belum bisa ngumpulin tugas hari ini karena laptop saya
tiba-tiba error. Boleh nggak saya kirim besok pagi?"
""".strip()

    response, elapsed = generate_response(
        model,
        tokenizer,
        style_prompt,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        max_new_tokens=100,
    )

    add_result(
        results,
        experiment="Style Transfer",
        variant="Informal to formal",
        prompt=style_prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response=response,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        elapsed_time=elapsed,
    )


# ============================================================
# 8. EKSPERIMEN TEMPERATURE
# ============================================================

def run_temperature_experiment(
    model: Any,
    tokenizer: Any,
    results: list[dict[str, Any]],
) -> None:
    """Membandingkan temperature rendah, sedang, dan tinggi."""

    print("\n[4/7] Menjalankan eksperimen temperature...")

    prompt = (
        "Berikan tiga ide kreatif penerapan Generative AI "
        "untuk meningkatkan pembelajaran mahasiswa di kampus."
    )

    temperatures = [0.2, 0.7, 1.3]

    for temperature in temperatures:
        response, elapsed = generate_response(
            model,
            tokenizer,
            prompt,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )

        add_result(
            results,
            experiment="Temperature",
            variant=f"temperature={temperature}",
            prompt=prompt,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            response=response,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            elapsed_time=elapsed,
        )


# ============================================================
# 9. EKSPERIMEN TOP-P
# ============================================================

def run_top_p_experiment(
    model: Any,
    tokenizer: Any,
    results: list[dict[str, Any]],
) -> None:
    """Membandingkan nucleus sampling top-p."""

    print("\n[5/7] Menjalankan eksperimen top-p...")

    prompt = (
        "Tuliskan paragraf pendek tentang masa depan "
        "kecerdasan buatan dalam pendidikan."
    )

    top_p_values = [0.5, 0.8, 0.95]

    for top_p in top_p_values:
        response, elapsed = generate_response(
            model,
            tokenizer,
            prompt,
            temperature=0.8,
            top_p=top_p,
            do_sample=True,
        )

        add_result(
            results,
            experiment="Top-p",
            variant=f"top_p={top_p}",
            prompt=prompt,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            response=response,
            temperature=0.8,
            top_p=top_p,
            do_sample=True,
            elapsed_time=elapsed,
        )


# ============================================================
# 10. EKSPERIMEN SYSTEM PROMPT
# ============================================================

def run_system_prompt_experiment(
    model: Any,
    tokenizer: Any,
    results: list[dict[str, Any]],
) -> None:
    """Membandingkan gaya jawaban berdasarkan peran sistem."""

    print("\n[6/7] Menjalankan eksperimen system prompt...")

    user_prompt = (
        "Jelaskan konsep diffusion model dalam maksimal "
        "tiga kalimat."
    )

    system_prompts = {
        "Dosen akademik": (
            "Kamu adalah dosen machine learning. "
            "Gunakan bahasa akademik yang tepat dan formal."
        ),
        "Tutor pemula": (
            "Kamu adalah tutor yang menjelaskan konsep rumit "
            "kepada mahasiswa pemula dengan analogi sederhana."
        ),
        "Asisten ringkas": (
            "Kamu adalah asisten teknis. Berikan jawaban sangat "
            "ringkas, langsung, dan tanpa penjelasan tambahan."
        ),
    }

    for variant, system_prompt in system_prompts.items():
        response, elapsed = generate_response(
            model,
            tokenizer,
            user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            max_new_tokens=120,
        )

        add_result(
            results,
            experiment="System Prompt",
            variant=variant,
            prompt=user_prompt,
            system_prompt=system_prompt,
            response=response,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            elapsed_time=elapsed,
        )


# ============================================================
# 11. SIMPAN HASIL
# ============================================================

def save_results(results: list[dict[str, Any]]) -> None:
    """Menyimpan hasil ke CSV, JSON, dan TXT."""

    print("\n[7/7] Menyimpan seluruh hasil...")

    columns = [
        "experiment",
        "variant",
        "prompt",
        "system_prompt",
        "temperature",
        "top_p",
        "do_sample",
        "response",
        "generation_time_seconds",
        "seed",
    ]

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(results)

    with JSON_FILE.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            results,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write("LAB 4.1 PART A — PROMPT ENGINEERING\n")
        report.write("=" * 72 + "\n\n")
        report.write(f"Model: {MODEL_ID}\n")
        report.write(f"Device: {DEVICE}\n")
        report.write(f"Seed: {SEED}\n")
        report.write(f"Jumlah eksperimen: {len(results)}\n\n")

        for index, result in enumerate(results, start=1):
            report.write("=" * 72 + "\n")
            report.write(f"EKSPERIMEN {index}\n")
            report.write("=" * 72 + "\n")

            report.write(
                f"Kategori    : {result['experiment']}\n"
            )
            report.write(
                f"Varian      : {result['variant']}\n"
            )
            report.write(
                f"Temperature : {result['temperature']}\n"
            )
            report.write(
                f"Top-p       : {result['top_p']}\n\n"
            )

            report.write("PROMPT:\n")
            report.write(result["prompt"] + "\n\n")

            report.write("JAWABAN:\n")
            report.write(result["response"] + "\n\n")

    print(f"CSV    : {CSV_FILE}")
    print(f"JSON   : {JSON_FILE}")
    print(f"Report : {REPORT_FILE}")


# ============================================================
# 12. PROGRAM UTAMA
# ============================================================

def main() -> None:
    print_environment_info()

    model, tokenizer = load_model_and_tokenizer()

    results: list[dict[str, Any]] = []

    run_core_tasks(model, tokenizer, results)
    run_temperature_experiment(model, tokenizer, results)
    run_top_p_experiment(model, tokenizer, results)
    run_system_prompt_experiment(model, tokenizer, results)

    save_results(results)

    print("\n" + "=" * 72)
    print("LAB 4.1 PART A BERHASIL DISELESAIKAN")
    print("=" * 72)
    print(f"Jumlah eksperimen : {len(results)}")
    print(f"Semua hasil ada di:\n{OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna.")

    except torch.cuda.OutOfMemoryError:
        print("\nGPU kehabisan VRAM.")
        print(
            "Tutup aplikasi yang menggunakan GPU, lalu jalankan ulang."
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as error:
        print("\n" + "=" * 72)
        print("PROGRAM MENGALAMI ERROR")
        print("=" * 72)
        print(f"Jenis error : {type(error).__name__}")
        print(f"Pesan       : {error}")
        print("\nTraceback lengkap:")

        traceback.print_exc()