from __future__ import annotations

import csv
import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# LAB 4.4 — TINY DDPM FROM SCRATCH ON 2D TOY DATA
# ============================================================
#
# Eksperimen:
# 1. Data sintetis empat Gaussian
# 2. Forward diffusion pada t = 0, T/4, T/2, 3T/4, T
# 3. Training MLP untuk memprediksi noise
# 4. Reverse diffusion dan sampel baru
# 5. Perbandingan T = 100 dengan T = 1000
# 6. Perbandingan MLP 2 hidden layer dan 4 hidden layer
# 7. Dokumentasi DDIM
#
# ============================================================


# ============================================================
# 1. KONFIGURASI
# ============================================================

SEED = 42

N_DATA = 10_000
N_GENERATED = 2_000
BATCH_SIZE = 512

HIDDEN_DIM = 128
TIME_EMBED_DIM = 16

BETA_START = 1e-4
BETA_END = 0.02

T_BASE = 1000
T_SHORT = 100

TRAIN_STEPS_BASE = 2_000
TRAIN_STEPS_SHORT = 1_500
TRAIN_STEPS_DEEP = 2_000

LEARNING_RATE = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

FORWARD_FILE = OUTPUT_DIR / "01_forward_diffusion_timesteps.png"
MAIN_FILE = OUTPUT_DIR / "02_original_noisy_generated.png"
TIMESTEP_FILE = OUTPUT_DIR / "03_compare_t100_vs_t1000.png"
DEPTH_FILE = OUTPUT_DIR / "04_loss_2layer_vs_4layer.png"
DENOISING_FILE = OUTPUT_DIR / "05_reverse_denoising_process.png"

SUMMARY_FILE = OUTPUT_DIR / "experiment_summary.csv"
METRICS_FILE = OUTPUT_DIR / "training_metrics.json"
ANALYSIS_FILE = OUTPUT_DIR / "analysis_lab_4_4.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. DATA SINTETIS: EMPAT GAUSSIAN
# ============================================================

def generate_four_gaussians(
    n_samples: int,
    standard_deviation: float = 0.28,
) -> torch.Tensor:
    """
    Membuat empat kelompok Gaussian dalam susunan grid.

        (-2, 2)       (2, 2)

        (-2, -2)      (2, -2)
    """

    centers = torch.tensor(
        [
            [-2.0, -2.0],
            [-2.0, 2.0],
            [2.0, -2.0],
            [2.0, 2.0],
        ],
        dtype=torch.float32,
    )

    cluster_ids = torch.randint(
        low=0,
        high=len(centers),
        size=(n_samples,),
    )

    noise = standard_deviation * torch.randn(
        n_samples,
        2,
    )

    points = centers[cluster_ids] + noise
    return points


# ============================================================
# 4. DIFFUSION SCHEDULE
# ============================================================

class DiffusionSchedule:
    def __init__(
        self,
        total_timesteps: int,
        device: str,
    ) -> None:
        self.total_timesteps = total_timesteps

        self.betas = torch.linspace(
            BETA_START,
            BETA_END,
            total_timesteps,
            device=device,
        )

        self.alphas = 1.0 - self.betas

        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0,
        )

        self.alpha_bars_previous = torch.cat(
            [
                torch.ones(
                    1,
                    device=device,
                ),
                self.alpha_bars[:-1],
            ]
        )

        self.posterior_variance = (
            self.betas
            * (1.0 - self.alpha_bars_previous)
            / (1.0 - self.alpha_bars)
        )


# ============================================================
# 5. FORWARD DIFFUSION
# ============================================================

def q_sample(
    x_0: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Menghasilkan x_t secara langsung:

    x_t =
    sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
    """

    if noise is None:
        noise = torch.randn_like(x_0)

    alpha_bar_t = schedule.alpha_bars[
        timesteps
    ].unsqueeze(1)

    x_t = (
        torch.sqrt(alpha_bar_t) * x_0
        + torch.sqrt(1.0 - alpha_bar_t) * noise
    )

    return x_t, noise


# ============================================================
# 6. TIME EMBEDDING
# ============================================================

def sinusoidal_time_embedding(
    timesteps: torch.Tensor,
    total_timesteps: int,
    embedding_dim: int,
) -> torch.Tensor:
    """
    Membuat sinusoidal embedding untuk timestep.
    """

    half_dim = embedding_dim // 2

    frequencies = torch.exp(
        -math.log(10_000)
        * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        )
        / max(half_dim - 1, 1)
    )

    normalized_time = (
        timesteps.float()
        / max(total_timesteps - 1, 1)
    ).unsqueeze(1)

    angles = normalized_time * frequencies.unsqueeze(0) * 1000

    embedding = torch.cat(
        [
            torch.sin(angles),
            torch.cos(angles),
        ],
        dim=1,
    )

    return embedding


# ============================================================
# 7. MLP NOISE PREDICTOR
# ============================================================

class NoisePredictor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        hidden_layers: int,
        time_embedding_dim: int,
    ) -> None:
        super().__init__()

        input_dim = 2 + time_embedding_dim

        layers: list[nn.Module] = [
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.SiLU(),
        ]

        for _ in range(hidden_layers - 1):
            layers.extend(
                [
                    nn.Linear(
                        hidden_dim,
                        hidden_dim,
                    ),
                    nn.SiLU(),
                ]
            )

        layers.append(
            nn.Linear(
                hidden_dim,
                2,
            )
        )

        self.network = nn.Sequential(*layers)
        self.time_embedding_dim = time_embedding_dim

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        total_timesteps: int,
    ) -> torch.Tensor:

        time_embedding = sinusoidal_time_embedding(
            timesteps,
            total_timesteps,
            self.time_embedding_dim,
        )

        model_input = torch.cat(
            [
                x_t,
                time_embedding,
            ],
            dim=1,
        )

        return self.network(model_input)


# ============================================================
# 8. TRAINING
# ============================================================

def train_model(
    data: torch.Tensor,
    *,
    total_timesteps: int,
    hidden_layers: int,
    train_steps: int,
    experiment_name: str,
) -> tuple[
    NoisePredictor,
    DiffusionSchedule,
    list[float],
    float,
]:
    """
    Melatih model untuk memprediksi noise epsilon.
    """

    print("\n" + "=" * 72)
    print(f"TRAINING: {experiment_name}")
    print("=" * 72)
    print(f"Timestep       : {total_timesteps}")
    print(f"Hidden layers  : {hidden_layers}")
    print(f"Training steps : {train_steps}")
    print(f"Device         : {DEVICE}")
    print("=" * 72)

    schedule = DiffusionSchedule(
        total_timesteps,
        DEVICE,
    )

    model = NoisePredictor(
        hidden_dim=HIDDEN_DIM,
        hidden_layers=hidden_layers,
        time_embedding_dim=TIME_EMBED_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    losses: list[float] = []
    start_time = time.perf_counter()

    for step in range(1, train_steps + 1):
        indices = torch.randint(
            low=0,
            high=len(data),
            size=(BATCH_SIZE,),
            device=DEVICE,
        )

        x_0 = data[indices]

        timesteps = torch.randint(
            low=0,
            high=total_timesteps,
            size=(BATCH_SIZE,),
            device=DEVICE,
        )

        noise = torch.randn_like(x_0)

        x_t, target_noise = q_sample(
            x_0,
            timesteps,
            schedule,
            noise,
        )

        predicted_noise = model(
            x_t,
            timesteps,
            total_timesteps,
        )

        loss = F.mse_loss(
            predicted_noise,
            target_noise,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        losses.append(float(loss.item()))

        if (
            step == 1
            or step % 250 == 0
            or step == train_steps
        ):
            recent_loss = float(
                np.mean(losses[-100:])
            )

            print(
                f"Step {step:4d}/{train_steps} "
                f"| loss terakhir: {loss.item():.5f} "
                f"| rata-rata 100 step: {recent_loss:.5f}"
            )

    elapsed_time = time.perf_counter() - start_time

    print(
        f"Training selesai dalam "
        f"{elapsed_time:.2f} detik."
    )

    return (
        model,
        schedule,
        losses,
        elapsed_time,
    )


# ============================================================
# 9. REVERSE DIFFUSION
# ============================================================

@torch.no_grad()
def sample_ddpm(
    model: NoisePredictor,
    schedule: DiffusionSchedule,
    *,
    n_samples: int,
    save_snapshots: bool = False,
) -> tuple[
    torch.Tensor,
    dict[int, torch.Tensor],
]:
    """
    Menghasilkan titik baru dari noise menggunakan reverse diffusion.
    """

    model.eval()

    total_timesteps = schedule.total_timesteps

    x = torch.randn(
        n_samples,
        2,
        device=DEVICE,
    )

    snapshots: dict[int, torch.Tensor] = {}

    if save_snapshots:
        snapshots[total_timesteps] = (
            x.detach().cpu().clone()
        )

    snapshot_steps = {
        int(3 * total_timesteps / 4),
        int(total_timesteps / 2),
        int(total_timesteps / 4),
        0,
    }

    for timestep_value in reversed(
        range(total_timesteps)
    ):
        timesteps = torch.full(
            size=(n_samples,),
            fill_value=timestep_value,
            device=DEVICE,
            dtype=torch.long,
        )

        predicted_noise = model(
            x,
            timesteps,
            total_timesteps,
        )

        beta_t = schedule.betas[
            timestep_value
        ]

        alpha_t = schedule.alphas[
            timestep_value
        ]

        alpha_bar_t = schedule.alpha_bars[
            timestep_value
        ]

        model_mean = (
            x
            - (
                beta_t
                / torch.sqrt(
                    1.0 - alpha_bar_t
                )
            )
            * predicted_noise
        ) / torch.sqrt(alpha_t)

        if timestep_value > 0:
            posterior_variance_t = (
                schedule.posterior_variance[
                    timestep_value
                ]
            )

            random_noise = torch.randn_like(x)

            x = (
                model_mean
                + torch.sqrt(
                    posterior_variance_t
                )
                * random_noise
            )
        else:
            x = model_mean

        if (
            save_snapshots
            and timestep_value in snapshot_steps
        ):
            snapshots[timestep_value] = (
                x.detach().cpu().clone()
            )

    return x.detach().cpu(), snapshots


# ============================================================
# 10. VISUALISASI HELPER
# ============================================================

def scatter_plot(
    axis: Any,
    points: torch.Tensor | np.ndarray,
    title: str,
) -> None:
    if isinstance(points, torch.Tensor):
        values = points.detach().cpu().numpy()
    else:
        values = points

    axis.scatter(
        values[:, 0],
        values[:, 1],
        s=5,
        alpha=0.45,
    )

    axis.set_title(title)
    axis.set_aspect("equal")
    axis.set_xlim(-5, 5)
    axis.set_ylim(-5, 5)
    axis.grid(alpha=0.2)


def moving_average(
    values: list[float],
    window: int = 50,
) -> np.ndarray:
    array = np.asarray(values)

    if len(array) < window:
        return array

    weights = np.ones(window) / window

    return np.convolve(
        array,
        weights,
        mode="valid",
    )


# ============================================================
# 11. FORWARD DIFFUSION TIMESTEPS
# ============================================================

def create_forward_visualization(
    data: torch.Tensor,
) -> None:
    schedule = DiffusionSchedule(
        T_BASE,
        DEVICE,
    )

    sample_data = data[:2_000]

    fixed_noise = torch.randn_like(sample_data)

    displayed_timesteps = [
        0,
        T_BASE // 4,
        T_BASE // 2,
        3 * T_BASE // 4,
        T_BASE - 1,
    ]

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )

    for axis, timestep_value in zip(
        axes,
        displayed_timesteps,
    ):
        timesteps = torch.full(
            (len(sample_data),),
            timestep_value,
            device=DEVICE,
            dtype=torch.long,
        )

        x_t, _ = q_sample(
            sample_data,
            timesteps,
            schedule,
            noise=fixed_noise,
        )

        displayed_label = (
            "T"
            if timestep_value == T_BASE - 1
            else str(timestep_value)
        )

        scatter_plot(
            axis,
            x_t,
            f"t = {displayed_label}",
        )

    figure.suptitle(
        "Forward Diffusion: Struktur Data Secara Bertahap Menjadi Noise"
    )

    plt.tight_layout()
    plt.savefig(FORWARD_FILE, dpi=160)
    plt.close()


# ============================================================
# 12. VISUALISASI HASIL UTAMA
# ============================================================

def create_main_visualization(
    data: torch.Tensor,
    generated_samples: torch.Tensor,
) -> None:
    schedule = DiffusionSchedule(
        T_BASE,
        DEVICE,
    )

    original_subset = data[:2_000]

    middle_timesteps = torch.full(
        (len(original_subset),),
        T_BASE // 2,
        device=DEVICE,
        dtype=torch.long,
    )

    noisy_middle, _ = q_sample(
        original_subset,
        middle_timesteps,
        schedule,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
    )

    scatter_plot(
        axes[0],
        original_subset,
        "Data Asli: Empat Gaussian",
    )

    scatter_plot(
        axes[1],
        noisy_middle,
        f"Data Berisik: t = {T_BASE // 2}",
    )

    scatter_plot(
        axes[2],
        generated_samples,
        "Sampel Baru Hasil Reverse Diffusion",
    )

    plt.tight_layout()
    plt.savefig(MAIN_FILE, dpi=160)
    plt.close()


# ============================================================
# 13. PERBANDINGAN T=100 DAN T=1000
# ============================================================

def create_timestep_comparison(
    original_data: torch.Tensor,
    samples_t100: torch.Tensor,
    samples_t1000: torch.Tensor,
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
    )

    scatter_plot(
        axes[0],
        original_data[:2_000],
        "Data Asli",
    )

    scatter_plot(
        axes[1],
        samples_t100,
        "Generated Samples: T = 100",
    )

    scatter_plot(
        axes[2],
        samples_t1000,
        "Generated Samples: T = 1000",
    )

    plt.tight_layout()
    plt.savefig(TIMESTEP_FILE, dpi=160)
    plt.close()


# ============================================================
# 14. LOSS 2 LAYER VS 4 LAYER
# ============================================================

def create_depth_comparison(
    losses_2_layer: list[float],
    losses_4_layer: list[float],
) -> None:
    smoothed_2 = moving_average(
        losses_2_layer,
        window=50,
    )

    smoothed_4 = moving_average(
        losses_4_layer,
        window=50,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        smoothed_2,
        label="MLP 2 hidden layer",
    )

    plt.plot(
        smoothed_4,
        label="MLP 4 hidden layer",
    )

    plt.title(
        "Perbandingan Training Loss: MLP 2 vs 4 Hidden Layer"
    )

    plt.xlabel("Training Step")
    plt.ylabel("Moving Average Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(DEPTH_FILE, dpi=160)
    plt.close()


# ============================================================
# 15. VISUALISASI REVERSE DENOISING
# ============================================================

def create_denoising_visualization(
    snapshots: dict[int, torch.Tensor],
) -> None:
    ordered_steps = [
        T_BASE,
        3 * T_BASE // 4,
        T_BASE // 2,
        T_BASE // 4,
        0,
    ]

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )

    for axis, step in zip(
        axes,
        ordered_steps,
    ):
        if step not in snapshots:
            axis.axis("off")
            continue

        title = (
            "Noise Awal"
            if step == T_BASE
            else f"Reverse t = {step}"
        )

        scatter_plot(
            axis,
            snapshots[step],
            title,
        )

    figure.suptitle(
        "Reverse Diffusion: Noise Secara Bertahap Menjadi Data"
    )

    plt.tight_layout()
    plt.savefig(DENOISING_FILE, dpi=160)
    plt.close()


# ============================================================
# 16. SIMPAN METRIK DAN ANALISIS
# ============================================================

def save_summary(
    experiments: list[dict[str, Any]],
) -> None:
    columns = [
        "experiment",
        "timesteps",
        "hidden_layers",
        "training_steps",
        "final_loss",
        "mean_last_100_loss",
        "training_time_seconds",
        "device",
    ]

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(experiments)


def save_analysis(
    metrics: dict[str, Any],
) -> None:
    analysis = f"""
LAB 4.4 — ANALISIS TINY DDPM
=============================

1. VISUALISASI FORWARD DIFFUSION
--------------------------------

Pada t = 0, empat kelompok Gaussian masih terlihat jelas. Saat timestep
bertambah menjadi T/4 dan T/2, posisi titik mulai menyebar dan struktur
kelompok semakin sulit dikenali. Pada 3T/4 dan T, distribusi mendekati
noise Gaussian sehingga informasi data asli hampir seluruhnya hilang.

2. PERBANDINGAN T = 100 DAN T = 1000
-------------------------------------

Final loss T=100:
{metrics["t100_final_loss"]:.6f}

Final loss T=1000:
{metrics["t1000_final_loss"]:.6f}

T=100 menggunakan proses noising dan denoising yang lebih pendek sehingga
sampling lebih cepat. Untuk data sederhana, model masih dapat menghasilkan
pola yang cukup baik. Namun, perubahan pada setiap langkah menjadi lebih
besar sehingga proses reverse diffusion cenderung kurang halus.

T=1000 menyediakan perubahan noise yang lebih bertahap. Proses sampling
lebih lambat, tetapi model mendapat lintasan denoising yang lebih halus dan
biasanya lebih stabil untuk distribusi data yang kompleks.

3. MLP 2 LAYER VS 4 LAYER
---------------------------

Final loss MLP 2 hidden layer:
{metrics["two_layer_final_loss"]:.6f}

Final loss MLP 4 hidden layer:
{metrics["four_layer_final_loss"]:.6f}

Model yang lebih dalam memiliki kapasitas representasi lebih besar, tetapi
tidak otomatis memberikan hasil lebih baik. Pada data 2D sederhana, MLP
dua hidden layer sering sudah cukup. MLP empat hidden layer dapat membantu
jika optimasi stabil, tetapi juga menambah waktu dan kompleksitas training.

4. DDIM
--------

DDIM atau Denoising Diffusion Implicit Model menggunakan proses sampling
non-Markovian yang dapat dibuat deterministik. DDIM tidak harus mengunjungi
seluruh timestep training ketika melakukan sampling.

Dengan memilih subset timestep, misalnya 50 langkah dari 1000 langkah,
DDIM dapat menghasilkan sampel jauh lebih cepat. Model prediksi noise yang
sama tetap dapat digunakan, tetapi aturan transisinya berbeda dari sampling
DDPM stokastik.

5. LoRA STABLE DIFFUSION
-------------------------

Modul juga meminta training LoRA untuk Stable Diffusion melalui Google
Colab serta menunjukkan notebook dan output cell pada saat kelas. Bagian
tersebut merupakan pekerjaan terpisah dari implementasi Tiny DDPM lokal ini.

KESIMPULAN
-----------

Tiny DDPM berhasil mempelajari proses penambahan dan penghilangan noise
pada data 2D. Model menghasilkan sampel baru dari noise, dan seluruh proses
dapat divisualisasikan dengan jelas melalui forward dan reverse diffusion.
""".strip()

    ANALYSIS_FILE.write_text(
        analysis,
        encoding="utf-8",
    )


# ============================================================
# 17. PROGRAM UTAMA
# ============================================================

def main() -> None:
    set_seed(SEED)

    print("=" * 72)
    print("LAB 4.4 — TINY DDPM FROM SCRATCH")
    print("=" * 72)
    print(f"Device      : {DEVICE}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA        : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(
            "GPU         :",
            torch.cuda.get_device_name(0),
        )

    print(f"Output      : {OUTPUT_DIR}")
    print("=" * 72)

    data = generate_four_gaussians(
        N_DATA
    ).to(DEVICE)

    print("\n[1/8] Membuat visualisasi forward diffusion...")
    create_forward_visualization(data)

    print("\n[2/8] Training T=1000, MLP 2 hidden layer...")

    (
        model_t1000_2,
        schedule_t1000,
        losses_t1000_2,
        time_t1000_2,
    ) = train_model(
        data,
        total_timesteps=T_BASE,
        hidden_layers=2,
        train_steps=TRAIN_STEPS_BASE,
        experiment_name="T1000_MLP_2_LAYER",
    )

    torch.save(
        model_t1000_2.state_dict(),
        MODEL_DIR / "ddpm_t1000_mlp2.pth",
    )

    print("\n[3/8] Sampling T=1000...")

    samples_t1000, denoising_snapshots = sample_ddpm(
        model_t1000_2,
        schedule_t1000,
        n_samples=N_GENERATED,
        save_snapshots=True,
    )

    print("\n[4/8] Training T=100, MLP 2 hidden layer...")

    (
        model_t100_2,
        schedule_t100,
        losses_t100_2,
        time_t100_2,
    ) = train_model(
        data,
        total_timesteps=T_SHORT,
        hidden_layers=2,
        train_steps=TRAIN_STEPS_SHORT,
        experiment_name="T100_MLP_2_LAYER",
    )

    torch.save(
        model_t100_2.state_dict(),
        MODEL_DIR / "ddpm_t100_mlp2.pth",
    )

    print("\n[5/8] Sampling T=100...")

    samples_t100, _ = sample_ddpm(
        model_t100_2,
        schedule_t100,
        n_samples=N_GENERATED,
        save_snapshots=False,
    )

    print("\n[6/8] Training T=1000, MLP 4 hidden layer...")

    (
        model_t1000_4,
        _,
        losses_t1000_4,
        time_t1000_4,
    ) = train_model(
        data,
        total_timesteps=T_BASE,
        hidden_layers=4,
        train_steps=TRAIN_STEPS_DEEP,
        experiment_name="T1000_MLP_4_LAYER",
    )

    torch.save(
        model_t1000_4.state_dict(),
        MODEL_DIR / "ddpm_t1000_mlp4.pth",
    )

    print("\n[7/8] Membuat seluruh visualisasi...")

    create_main_visualization(
        data,
        samples_t1000,
    )

    create_timestep_comparison(
        data,
        samples_t100,
        samples_t1000,
    )

    create_depth_comparison(
        losses_t1000_2,
        losses_t1000_4,
    )

    create_denoising_visualization(
        denoising_snapshots,
    )

    experiments = [
        {
            "experiment": "T1000_MLP_2_LAYER",
            "timesteps": T_BASE,
            "hidden_layers": 2,
            "training_steps": TRAIN_STEPS_BASE,
            "final_loss": losses_t1000_2[-1],
            "mean_last_100_loss": float(
                np.mean(losses_t1000_2[-100:])
            ),
            "training_time_seconds": round(
                time_t1000_2,
                3,
            ),
            "device": DEVICE,
        },
        {
            "experiment": "T100_MLP_2_LAYER",
            "timesteps": T_SHORT,
            "hidden_layers": 2,
            "training_steps": TRAIN_STEPS_SHORT,
            "final_loss": losses_t100_2[-1],
            "mean_last_100_loss": float(
                np.mean(losses_t100_2[-100:])
            ),
            "training_time_seconds": round(
                time_t100_2,
                3,
            ),
            "device": DEVICE,
        },
        {
            "experiment": "T1000_MLP_4_LAYER",
            "timesteps": T_BASE,
            "hidden_layers": 4,
            "training_steps": TRAIN_STEPS_DEEP,
            "final_loss": losses_t1000_4[-1],
            "mean_last_100_loss": float(
                np.mean(losses_t1000_4[-100:])
            ),
            "training_time_seconds": round(
                time_t1000_4,
                3,
            ),
            "device": DEVICE,
        },
    ]

    save_summary(experiments)

    metrics = {
        "t100_final_loss": losses_t100_2[-1],
        "t1000_final_loss": losses_t1000_2[-1],
        "two_layer_final_loss": losses_t1000_2[-1],
        "four_layer_final_loss": losses_t1000_4[-1],
    }

    with METRICS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    save_analysis(metrics)

    print("\n[8/8] Semua output berhasil disimpan.")

    print("\n" + "=" * 72)
    print("LAB 4.4 BERHASIL DISELESAIKAN")
    print("=" * 72)
    print(f"Output folder : {OUTPUT_DIR}")
    print("\nFile utama:")
    print("01_forward_diffusion_timesteps.png")
    print("02_original_noisy_generated.png")
    print("03_compare_t100_vs_t1000.png")
    print("04_loss_2layer_vs_4layer.png")
    print("05_reverse_denoising_process.png")
    print("experiment_summary.csv")
    print("analysis_lab_4_4.txt")
    print("models/")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna.")

    except torch.cuda.OutOfMemoryError:
        print("\nGPU kehabisan VRAM.")
        print(
            "Turunkan BATCH_SIZE menjadi 256 "
            "dan jalankan ulang."
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as error:
        print("\nPROGRAM ERROR")
        print(type(error).__name__, "-", error)
        traceback.print_exc()