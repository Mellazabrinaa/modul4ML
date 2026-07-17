from __future__ import annotations

import csv
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from PIL import Image, ImageDraw


# ============================================================
# LAB 4.2 — STABLE DIFFUSION
# ============================================================
#
# Eksperimen:
# 1. Gambar wajib berukuran 256 x 256
# 2. Guidance scale 2, 7, dan 15
# 3. Inference steps 10, 25, dan 50
# 4. Tanpa dan dengan negative prompt
# 5. Seed tetap untuk perbandingan yang adil
#
# ============================================================


# ============================================================
# 1. KONFIGURASI
# ============================================================

MODEL_ID = (
    "stable-diffusion-v1-5/"
    "stable-diffusion-v1-5"
)

PROMPT = (
    "wide angle interior photograph of a modern university "
    "smart classroom, organized rows of student desks and chairs, "
    "large interactive digital whiteboard, modern educational "
    "technology, bright natural daylight, realistic architecture, "
    "professional interior photography, sharp focus, highly detailed"
)

NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, abstract, distorted, "
    "deformed furniture, duplicate desks, floating objects, "
    "cropped, bad perspective, watermark, text, logo, noisy, "
    "oversaturated, dark image"
)

SEED = 42

# Ketentuan modul.
REQUIRED_SIZE = 256

# Resolusi eksperimen agar lebih jelas saat demo.
EXPERIMENT_SIZE = 512

DEFAULT_GUIDANCE = 7.0
DEFAULT_STEPS = 25

GUIDANCE_VALUES = [2.0, 7.0, 15.0]
STEP_VALUES = [10, 25, 50]

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

REQUIRED_DIR = OUTPUT_DIR / "01_required_256"
GUIDANCE_DIR = OUTPUT_DIR / "02_guidance_scale"
STEPS_DIR = OUTPUT_DIR / "03_inference_steps"
NEGATIVE_DIR = OUTPUT_DIR / "04_negative_prompt"
MAIN_DIR = OUTPUT_DIR / "05_main_result"

SUMMARY_FILE = OUTPUT_DIR / "experiment_summary.csv"
ANALYSIS_FILE = OUTPUT_DIR / "analysis_lab_4_2.txt"
COMPARISON_FILE = OUTPUT_DIR / "lab_4_2_comparison.png"


# ============================================================
# 2. MEMBUAT FOLDER OUTPUT
# ============================================================

def prepare_directories() -> None:
    """Membuat seluruh folder output."""

    directories = [
        REQUIRED_DIR,
        GUIDANCE_DIR,
        STEPS_DIR,
        NEGATIVE_DIR,
        MAIN_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# 3. INFORMASI ENVIRONMENT
# ============================================================

def print_environment() -> None:
    """Menampilkan informasi perangkat."""

    print("=" * 72)
    print("LAB 4.2 — STABLE DIFFUSION")
    print("=" * 72)

    print(f"Model             : {MODEL_ID}")
    print(f"PyTorch           : {torch.__version__}")
    print(f"CUDA tersedia     : {CUDA_AVAILABLE}")
    print(f"Device            : {DEVICE}")
    print(f"Data type         : {DTYPE}")
    print(f"Seed              : {SEED}")
    print(f"Ukuran wajib      : {REQUIRED_SIZE}x{REQUIRED_SIZE}")
    print(
        f"Ukuran eksperimen : "
        f"{EXPERIMENT_SIZE}x{EXPERIMENT_SIZE}"
    )
    print(f"Folder output     : {OUTPUT_DIR}")

    if CUDA_AVAILABLE:
        print(
            "GPU               :",
            torch.cuda.get_device_name(0),
        )

        vram = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
        )

        print(f"VRAM              : {vram:.2f} GB")

    print("=" * 72)


# ============================================================
# 4. MEMUAT PIPELINE
# ============================================================

def load_pipeline() -> StableDiffusionPipeline:
    """Memuat Stable Diffusion dan scheduler DPM."""

    print("\n[1/7] Memuat Stable Diffusion pipeline...")
    print(
        "Model akan diunduh saat pertama kali dijalankan."
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
    )

    # Scheduler DPM biasanya memberi hasil baik dengan
    # jumlah langkah yang relatif sedikit.
    pipe.scheduler = (
        DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
    )

    if CUDA_AVAILABLE:
        # Lebih aman untuk RTX dengan VRAM terbatas.
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    else:
        pipe = pipe.to("cpu")

    print("Pipeline berhasil dimuat.")
    return pipe


# ============================================================
# 5. MEMBUAT GENERATOR
# ============================================================

def create_generator() -> torch.Generator:
    """
    Membuat generator dengan seed yang sama.

    Seed diulang untuk setiap gambar agar parameter yang
    dibandingkan menjadi penyebab utama perbedaan hasil.
    """

    return torch.Generator(
        device="cpu"
    ).manual_seed(SEED)


# ============================================================
# 6. GENERASI SATU GAMBAR
# ============================================================

def generate_image(
    pipe: StableDiffusionPipeline,
    *,
    experiment: str,
    filename: str,
    output_directory: Path,
    prompt: str,
    negative_prompt: str | None,
    guidance_scale: float,
    inference_steps: int,
    image_size: int,
) -> dict[str, Any]:
    """Menghasilkan satu gambar dan mencatat waktu."""

    print("\n" + "-" * 72)
    print(f"Eksperimen      : {experiment}")
    print(f"File            : {filename}")
    print(f"Guidance scale  : {guidance_scale}")
    print(f"Inference steps : {inference_steps}")
    print(f"Ukuran          : {image_size}x{image_size}")
    print(
        "Negative prompt :",
        "Ya" if negative_prompt else "Tidak",
    )

    start_time = time.perf_counter()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=inference_steps,
        height=image_size,
        width=image_size,
        generator=create_generator(),
    )

    elapsed_time = time.perf_counter() - start_time

    image = result.images[0]

    output_path = output_directory / filename
    image.save(output_path)

    print(f"Tersimpan       : {output_path}")
    print(f"Waktu generasi  : {elapsed_time:.2f} detik")
    print("-" * 72)

    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()

    return {
        "experiment": experiment,
        "filename": filename,
        "output_path": str(output_path),
        "guidance_scale": guidance_scale,
        "inference_steps": inference_steps,
        "negative_prompt": (
            "Ya" if negative_prompt else "Tidak"
        ),
        "image_size": f"{image_size}x{image_size}",
        "seed": SEED,
        "generation_time_seconds": round(
            elapsed_time,
            3,
        ),
    }


# ============================================================
# 7. GAMBAR WAJIB 256 x 256
# ============================================================

def run_required_image(
    pipe: StableDiffusionPipeline,
) -> list[dict[str, Any]]:
    """Membuat gambar 256x256 sesuai ketentuan modul."""

    print("\n[2/7] Membuat gambar wajib 256x256...")

    record = generate_image(
        pipe,
        experiment="Required 256x256",
        filename="smart_classroom_256.png",
        output_directory=REQUIRED_DIR,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=DEFAULT_GUIDANCE,
        inference_steps=DEFAULT_STEPS,
        image_size=REQUIRED_SIZE,
    )

    return [record]


# ============================================================
# 8. EKSPERIMEN GUIDANCE SCALE
# ============================================================

def run_guidance_experiment(
    pipe: StableDiffusionPipeline,
) -> list[dict[str, Any]]:
    """Membandingkan guidance scale 2, 7, dan 15."""

    print("\n[3/7] Eksperimen guidance scale...")

    records = []

    for guidance in GUIDANCE_VALUES:
        record = generate_image(
            pipe,
            experiment="Guidance Scale",
            filename=f"guidance_{guidance:.0f}.png",
            output_directory=GUIDANCE_DIR,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=guidance,
            inference_steps=DEFAULT_STEPS,
            image_size=EXPERIMENT_SIZE,
        )

        records.append(record)

    return records


# ============================================================
# 9. EKSPERIMEN INFERENCE STEPS
# ============================================================

def run_steps_experiment(
    pipe: StableDiffusionPipeline,
) -> list[dict[str, Any]]:
    """Membandingkan 10, 25, dan 50 steps."""

    print("\n[4/7] Eksperimen inference steps...")

    records = []

    for steps in STEP_VALUES:
        record = generate_image(
            pipe,
            experiment="Inference Steps",
            filename=f"steps_{steps}.png",
            output_directory=STEPS_DIR,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=DEFAULT_GUIDANCE,
            inference_steps=steps,
            image_size=EXPERIMENT_SIZE,
        )

        records.append(record)

    return records


# ============================================================
# 10. EKSPERIMEN NEGATIVE PROMPT
# ============================================================

def run_negative_prompt_experiment(
    pipe: StableDiffusionPipeline,
) -> list[dict[str, Any]]:
    """Membandingkan hasil dengan dan tanpa negative prompt."""

    print("\n[5/7] Eksperimen negative prompt...")

    records = []

    records.append(
        generate_image(
            pipe,
            experiment="Negative Prompt",
            filename="without_negative_prompt.png",
            output_directory=NEGATIVE_DIR,
            prompt=PROMPT,
            negative_prompt=None,
            guidance_scale=DEFAULT_GUIDANCE,
            inference_steps=DEFAULT_STEPS,
            image_size=EXPERIMENT_SIZE,
        )
    )

    records.append(
        generate_image(
            pipe,
            experiment="Negative Prompt",
            filename="with_negative_prompt.png",
            output_directory=NEGATIVE_DIR,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=DEFAULT_GUIDANCE,
            inference_steps=DEFAULT_STEPS,
            image_size=EXPERIMENT_SIZE,
        )
    )

    return records


# ============================================================
# 11. HASIL UTAMA UNTUK DEMO
# ============================================================

def run_main_result(
    pipe: StableDiffusionPipeline,
) -> list[dict[str, Any]]:
    """Membuat hasil utama dengan konfigurasi seimbang."""

    print("\n[6/7] Membuat gambar utama untuk demo...")

    record = generate_image(
        pipe,
        experiment="Main Result",
        filename="smart_classroom_final.png",
        output_directory=MAIN_DIR,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=DEFAULT_GUIDANCE,
        inference_steps=DEFAULT_STEPS,
        image_size=EXPERIMENT_SIZE,
    )

    return [record]


# ============================================================
# 12. SIMPAN CSV
# ============================================================

def save_summary(
    records: list[dict[str, Any]],
) -> None:
    """Menyimpan konfigurasi dan waktu setiap eksperimen."""

    columns = [
        "experiment",
        "filename",
        "output_path",
        "guidance_scale",
        "inference_steps",
        "negative_prompt",
        "image_size",
        "seed",
        "generation_time_seconds",
    ]

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"\nCSV tersimpan: {SUMMARY_FILE}")


# ============================================================
# 13. MEMBUAT LAPORAN ANALISIS
# ============================================================

def save_analysis_template() -> None:
    """Membuat jawaban analisis awal sesuai aktivitas modul."""

    analysis = f"""
LAB 4.2 — ANALISIS STABLE DIFFUSION
===================================

Model:
{MODEL_ID}

Prompt:
{PROMPT}

Negative prompt:
{NEGATIVE_PROMPT}

Seed:
{SEED}


1. PENGARUH GUIDANCE SCALE
---------------------------

Guidance scale 2:
Model memiliki kebebasan generasi lebih besar. Hasil biasanya lebih
kreatif, tetapi dapat kurang sesuai dengan detail prompt.

Guidance scale 7:
Memberikan keseimbangan antara kreativitas dan kesesuaian terhadap
prompt. Nilai ini digunakan sebagai konfigurasi utama eksperimen.

Guidance scale 15:
Model mengikuti prompt dengan lebih kuat, tetapi hasil dapat terlihat
terlalu kaku, terlalu tajam, atau menampilkan artefak tertentu.


2. PENGARUH INFERENCE STEPS
----------------------------

Steps 10:
Proses paling cepat, tetapi detail dan struktur gambar dapat belum
stabil.

Steps 25:
Memberikan keseimbangan antara waktu generasi dan kualitas hasil.

Steps 50:
Membutuhkan waktu lebih lama. Kualitas dapat meningkat, tetapi
peningkatannya belum tentu sebanding dengan tambahan waktu.


3. PERAN NEGATIVE PROMPT
-------------------------

Negative prompt memberi informasi tentang elemen yang tidak diinginkan.
Pada eksperimen ini, negative prompt digunakan untuk mengurangi blur,
distorsi, duplikasi furnitur, teks, watermark, dan kualitas rendah.

Tiga contoh negative prompt:
1. blurry, low quality, low resolution
2. distorted, deformed furniture, duplicate objects
3. watermark, text, logo


4. CLASSIFIER-FREE GUIDANCE
----------------------------

Classifier-free guidance menggabungkan prediksi noise tanpa kondisi
dan prediksi noise dengan kondisi teks.

Secara sederhana:

epsilon_guided =
epsilon_unconditional +
s * (epsilon_conditional - epsilon_unconditional)

Keterangan:
- epsilon_unconditional adalah prediksi tanpa prompt.
- epsilon_conditional adalah prediksi yang menggunakan prompt.
- s adalah guidance scale.

Semakin besar nilai s, semakin kuat hasil diarahkan mengikuti prompt.
Nilai yang terlalu tinggi dapat menurunkan naturalitas atau menambah
artefak.


5. KESIMPULAN SEMENTARA
------------------------

Konfigurasi guidance scale 7 dan 25 inference steps digunakan sebagai
konfigurasi seimbang. Hasil akhir harus dinilai dengan membandingkan
gambar pada folder output dan waktu pada experiment_summary.csv.
""".strip()

    ANALYSIS_FILE.write_text(
        analysis,
        encoding="utf-8",
    )

    print(f"Analisis tersimpan: {ANALYSIS_FILE}")


# ============================================================
# 14. CONTACT SHEET
# ============================================================

def create_contact_sheet() -> None:
    """Menggabungkan delapan gambar perbandingan."""

    image_items = [
        (
            "Guidance 2",
            GUIDANCE_DIR / "guidance_2.png",
        ),
        (
            "Guidance 7",
            GUIDANCE_DIR / "guidance_7.png",
        ),
        (
            "Guidance 15",
            GUIDANCE_DIR / "guidance_15.png",
        ),
        (
            "Steps 10",
            STEPS_DIR / "steps_10.png",
        ),
        (
            "Steps 25",
            STEPS_DIR / "steps_25.png",
        ),
        (
            "Steps 50",
            STEPS_DIR / "steps_50.png",
        ),
        (
            "Tanpa Negative",
            NEGATIVE_DIR / "without_negative_prompt.png",
        ),
        (
            "Dengan Negative",
            NEGATIVE_DIR / "with_negative_prompt.png",
        ),
    ]

    loaded_images = []

    for title, image_path in image_items:
        if image_path.exists():
            image = Image.open(
                image_path
            ).convert("RGB")

            loaded_images.append(
                (title, image)
            )

    if not loaded_images:
        print("Tidak ada gambar untuk contact sheet.")
        return

    thumbnail_size = 256
    label_height = 40
    columns = 4
    rows = 2

    sheet_width = columns * thumbnail_size
    sheet_height = rows * (
        thumbnail_size + label_height
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    for index, (title, image) in enumerate(
        loaded_images
    ):
        row = index // columns
        column = index % columns

        x = column * thumbnail_size
        y = row * (
            thumbnail_size + label_height
        )

        thumbnail = image.resize(
            (thumbnail_size, thumbnail_size)
        )

        sheet.paste(
            thumbnail,
            (x, y),
        )

        draw.rectangle(
            [
                x,
                y + thumbnail_size,
                x + thumbnail_size,
                y + thumbnail_size + label_height,
            ],
            fill="white",
        )

        draw.text(
            (
                x + 10,
                y + thumbnail_size + 12,
            ),
            title,
            fill="black",
        )

    sheet.save(COMPARISON_FILE)

    print(
        f"Contact sheet tersimpan: "
        f"{COMPARISON_FILE}"
    )


# ============================================================
# 15. PROGRAM UTAMA
# ============================================================

def main() -> None:
    prepare_directories()
    print_environment()

    pipe = load_pipeline()

    all_records: list[dict[str, Any]] = []

    all_records.extend(
        run_required_image(pipe)
    )

    all_records.extend(
        run_guidance_experiment(pipe)
    )

    all_records.extend(
        run_steps_experiment(pipe)
    )

    all_records.extend(
        run_negative_prompt_experiment(pipe)
    )

    all_records.extend(
        run_main_result(pipe)
    )

    save_summary(all_records)
    save_analysis_template()
    create_contact_sheet()

    print("\n" + "=" * 72)
    print("LAB 4.2 BERHASIL DISELESAIKAN")
    print("=" * 72)

    print(f"Jumlah gambar : {len(all_records)}")
    print(f"Semua hasil   : {OUTPUT_DIR}")

    print("\nStruktur output:")
    print("01_required_256/")
    print("02_guidance_scale/")
    print("03_inference_steps/")
    print("04_negative_prompt/")
    print("05_main_result/")
    print("experiment_summary.csv")
    print("analysis_lab_4_2.txt")
    print("lab_4_2_comparison.png")

    print("=" * 72)


if __name__ == "__main__":
    try:
        main()

    except torch.cuda.OutOfMemoryError:
        print("\nGPU kehabisan VRAM.")
        print(
            "Tutup browser, game, dan aplikasi lain "
            "yang menggunakan GPU."
        )

        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna.")

    except Exception as error:
        print("\nPROGRAM ERROR")
        print(type(error).__name__, "-", error)
        traceback.print_exc()