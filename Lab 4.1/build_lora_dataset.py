import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_FILE = DATA_DIR / "lora_dataset.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

topics = [
    (
        "generative AI",
        "Generative AI adalah cabang kecerdasan buatan yang mampu menghasilkan konten baru seperti teks, gambar, audio, atau video berdasarkan pola yang dipelajari dari data."
    ),
    (
        "large language model",
        "Large language model atau LLM adalah model bahasa berukuran besar yang dilatih pada data teks dalam jumlah sangat banyak untuk memahami dan menghasilkan bahasa manusia."
    ),
    (
        "prompt engineering",
        "Prompt engineering adalah proses merancang instruksi yang jelas dan efektif agar model AI menghasilkan keluaran yang sesuai dengan tujuan pengguna."
    ),
    (
        "diffusion model",
        "Diffusion model adalah model generatif yang belajar menghasilkan data dengan membalik proses penambahan noise secara bertahap sampai kembali menjadi data bermakna."
    ),
    (
        "transformer",
        "Transformer adalah arsitektur jaringan saraf yang menggunakan mekanisme attention untuk memahami hubungan antar token dalam urutan data secara efisien."
    ),
    (
        "tokenizer",
        "Tokenizer adalah komponen yang mengubah teks menjadi potongan kecil yang disebut token agar bisa diproses oleh model bahasa."
    ),
    (
        "fine-tuning",
        "Fine-tuning adalah proses melatih ulang model yang sudah ada menggunakan dataset yang lebih spesifik agar performanya lebih baik pada tugas tertentu."
    ),
    (
        "LoRA",
        "LoRA atau Low-Rank Adaptation adalah teknik fine-tuning efisien yang menambahkan parameter kecil pada model besar tanpa melatih semua bobot model."
    ),
    (
        "QLoRA",
        "QLoRA adalah pengembangan LoRA yang menggabungkan quantization sehingga fine-tuning model besar bisa dilakukan dengan penggunaan memori yang lebih kecil."
    ),
    (
        "RLHF",
        "RLHF atau Reinforcement Learning from Human Feedback adalah metode alignment model dengan memanfaatkan umpan balik manusia untuk meningkatkan kualitas respons."
    ),
    (
        "DPO",
        "DPO atau Direct Preference Optimization adalah metode pelatihan berbasis preferensi yang lebih sederhana dibanding RLHF karena tidak memerlukan reward model terpisah."
    ),
    (
        "RAG",
        "RAG atau Retrieval-Augmented Generation adalah pendekatan yang menggabungkan pencarian informasi eksternal dengan generasi teks agar jawaban model lebih akurat dan kontekstual."
    ),
    (
        "overfitting",
        "Overfitting adalah kondisi saat model terlalu cocok dengan data latih sehingga performanya menurun ketika diuji pada data baru."
    ),
    (
        "underfitting",
        "Underfitting adalah kondisi saat model terlalu sederhana sehingga gagal menangkap pola penting pada data latih maupun data uji."
    ),
    (
        "temperature pada LLM",
        "Temperature pada LLM mengatur tingkat kerandoman keluaran model; nilai rendah membuat jawaban lebih deterministik, sedangkan nilai tinggi membuat jawaban lebih beragam."
    ),
    (
        "top-p sampling",
        "Top-p sampling adalah teknik pemilihan token yang membatasi pilihan pada kumpulan token dengan probabilitas kumulatif tertentu agar keluaran lebih terkontrol."
    ),
    (
        "system prompt",
        "System prompt adalah instruksi tingkat awal yang mendefinisikan peran, gaya, dan perilaku model selama percakapan."
    ),
    (
        "classifier-free guidance",
        "Classifier-free guidance adalah teknik pada diffusion model untuk mengontrol seberapa kuat hasil generasi mengikuti prompt yang diberikan."
    ),
    (
        "negative prompt",
        "Negative prompt adalah instruksi tambahan yang memberi tahu model tentang elemen yang tidak diinginkan dalam hasil generasi."
    ),
    (
        "stable diffusion",
        "Stable Diffusion adalah model generatif gambar berbasis diffusion yang dapat menghasilkan gambar dari deskripsi teks secara efisien."
    ),
]

instruction_templates = [
    "Jelaskan secara singkat apa itu {topic}.",
    "Apa yang dimaksud dengan {topic}? Berikan penjelasan yang jelas untuk mahasiswa pemula.",
    "Terangkan konsep {topic} dalam 2 sampai 3 kalimat menggunakan Bahasa Indonesia yang mudah dipahami.",
]

dataset = []

for topic, answer in topics:
    for template in instruction_templates:
        dataset.append(
            {
                "instruction": template.format(topic=topic),
                "answer": answer,
            }
        )

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Dataset berhasil disimpan di: {OUTPUT_FILE}")
print(f"Jumlah contoh: {len(dataset)}")