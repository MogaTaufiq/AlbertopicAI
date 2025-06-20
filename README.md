# AlbertopicAI: Sistem Pemodelan Topik Artikel Ilmiah dengan Antarmuka Web

AlbertopicAI adalah proyek _end-to-end_ yang dirancang untuk melakukan pemodelan topik pada artikel ilmiah dari arXiv. Proyek ini mencakup tahapan mulai dari pengambilan data, pra-pemrosesan teks, pelatihan model _machine learning_ menggunakan BERTopic, hingga penyajian interaktif melalui antarmuka web. Proyek ini juga menerapkan beberapa prinsip MLOps dengan penggunaan Docker untuk kontainerisasi dan Prometheus serta Grafana untuk monitoring (meskipun implementasi monitoring lebih lanjut mungkin diperlukan).

## ✨ Fitur Utama

-   **Pengambilan Data (Scraping):**
    -   Mengambil metadata artikel (judul, abstrak, penulis, tahun, DOI) dari arXiv API.
    -   Implementasi _scraping_ secara bertahap berdasarkan rentang tanggal untuk mengumpulkan data dalam jumlah besar.
    -   Mekanisme _checkpointing_ untuk melanjutkan proses _scraping_ yang terhenti.
    -   Menyimpan data mentah dalam format JSON Lines (`.jsonl`).
-   **Pra-pemrosesan Teks:**
    -   Membersihkan teks artikel (judul dan abstrak) dengan menghilangkan karakter non-alfanumerik, spasi berlebih, dan _stopwords_.
    -   Menyimpan data yang telah diproses dan data asli ke dalam file CSV.
-   **Pemodelan Topik:**
    -   Menggunakan **BERTopic** untuk mengidentifikasi topik-topik tersembunyi dalam kumpulan artikel.
    -   Menggunakan _custom embeddings_ yang dihasilkan oleh model `SentenceTransformer('paraphrase-MiniLM-L6-v2')`.
    -   Menyimpan model BERTopic yang telah dilatih (`.pkl`) dan hasil pemetaan topik per artikel (`.csv`).
    -   Dasar integrasi dengan Prometheus untuk metrik model (misalnya, _coherence score_).
-   **API Backend (FastAPI):**
    -   Menyediakan endpoint RESTful untuk berinteraksi dengan sistem.
    -   Menyajikan antarmuka web statis (`index.html`).
    -   Endpoint untuk menganalisis topik dari file PDF yang diunggah.
    -   Endpoint untuk menganalisis topik dari input teks (judul dan abstrak).
    -   Endpoint untuk mencari artikel dalam dataset berdasarkan kueri topik.
-   **Antarmuka Web (Frontend):**
    -   Halaman web interaktif yang dibangun dengan HTML, Tailwind CSS, dan JavaScript.
    -   Memungkinkan pengguna mengunggah PDF, memasukkan teks, atau mencari berdasarkan topik.
    -   Menampilkan prediksi topik untuk input pengguna dan daftar artikel terkait dari dataset.
-   **Kontainerisasi & Orkestrasi:**
    -   `Dockerfile` untuk membangun _image_ aplikasi.
    -   `docker-compose.yml` untuk menjalankan aplikasi FastAPI bersama dengan layanan Prometheus dan Grafana.
-   **Monitoring (Dasar):**
    -   `prometheus.yml` untuk konfigurasi Prometheus.
    -   Metrik dari model BERTopic dan API diekspos untuk dikumpulkan oleh Prometheus dan divisualisasikan di Grafana.

## 📂 Struktur Proyek

```bash
AlbertAI/
├── data/                     # Direktori untuk menyimpan data
│   ├── final/                # Model akhir dan hasil topik
│   │   ├── bertopic_model.pkl
│   │   └── topic_results.csv
│   ├── processed_data/       # Data setelah pra-pemrosesan
│   │   └── processed_articles.csv
│   └── rawdata/              # Data mentah hasil scraping
│       ├── arxiv_cs_articles_by_date.jsonl
│       └── scraper_checkpoint_by_date.json
├── src/                      # Kode sumber aplikasi
│   ├── main_api.py           # Logika API FastAPI
│   ├── modelling.py          # Skrip untuk training model BERTopic
│   ├── preprocess.py         # Skrip untuk pra-pemrosesan teks
│   └── scraper.py            # Skrip untuk scraping data arXiv
├── static/                   # File statis untuk frontend
│   └── index.html            # Halaman web utama
├── .dockerignore             # File yang diabaikan oleh Docker
├── .gitignore                # File yang diabaikan oleh Git
├── docker-compose.yml        # Konfigurasi Docker Compose
├── Dockerfile                # Instruksi untuk membangun image Docker
├── download_nltk_resources.py # Skrip bantu untuk mengunduh resource NLTK
├── prometheus.yml            # Konfigurasi Prometheus
├── README.md                 # File ini
└── requirements.txt          # Daftar dependensi Python
```

## ⚙️ Setup dan Instalasi

### Prasyarat

-   Python 3.9+
-   pip (Python package installer)
-   Docker Engine
-   Docker Compose

### Langkah-langkah Instalasi

1.  **Clone Repository:**

    ```bash
    git clone <URL_REPOSITORY_ANDA>
    cd AlbertopicAI
    ```

2.  **Buat dan Aktifkan Lingkungan Virtual (Direkomendasikan):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Untuk macOS/Linux
    # .venv\Scripts\activate   # Untuk Windows
    ```

3.  **Instal Dependensi Python:**
    Pastikan file `requirements.txt` Anda sudah mencakup semua paket yang diperlukan, termasuk `PyMuPDF` dan `python-multipart`.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Unduh Resource NLTK:**
    Jalankan skrip berikut untuk mengunduh paket NLTK yang diperlukan (`punkt`, `stopwords`, `wordnet`):

    ```bash
    python3 download_nltk_resources.py
    ```

    Jika Anda menjalankan di dalam Docker, pastikan langkah ini sudah ada di `Dockerfile`.

5.  **Unduh Model spaCy (Jika Digunakan Aktif):**
    Jika pra-pemrosesan Anda secara aktif menggunakan model spaCy, unduh model yang diperlukan:
    ```bash
    python3 -m spacy download en_core_web_sm
    ```
    Pastikan ini juga ada di `Dockerfile` jika belum.

## 🚀 Menjalankan Aplikasi dengan Docker Compose

Cara termudah untuk menjalankan seluruh tumpukan aplikasi (FastAPI, Prometheus, Grafana) adalah dengan Docker Compose.

1.  **Pastikan Docker Engine dan Docker Compose berjalan.**
2.  **Bangun dan Jalankan Kontainer:**
    Dari direktori root proyek (`AlbertAI/`), jalankan:

    ```bash
    docker-compose up --build
    ```

    Opsi `--build` akan memaksa Docker untuk membangun ulang _image_ jika ada perubahan pada `Dockerfile` atau kode sumber. Untuk menjalankan di latar belakang, tambahkan flag `-d`: `docker-compose up --build -d`.

3.  **Akses Layanan:**
    -   **Antarmuka Web AlbertopicAI:** Buka browser dan navigasi ke `http://localhost:8000/`
    -   **Prometheus:** Akses di `http://localhost:9090/`
    -   **Grafana:** Akses di `http://localhost:3000/` (Login default biasanya `admin`/`admin`, bisa diubah di `docker-compose.yml`). Anda perlu mengkonfigurasi Grafana untuk mengambil data dari Prometheus.

## 🛠️ Alur Kerja Pipeline Data & Model (Lokal/Manual)

Sebelum API dapat berfungsi penuh dengan model yang relevan, Anda perlu menjalankan pipeline data dan pelatihan model:

1.  **Scraping Data (`scraper.py`):**
    Jalankan skrip ini untuk mengumpulkan artikel dari arXiv. Skrip akan menggunakan checkpoint untuk melanjutkan jika terhenti.

    ```bash
    python3 src/scraper.py
    ```

    Ini akan menghasilkan file `.jsonl` di `data/rawdata/`.

2.  **Pra-pemrosesan Data (`preprocess.py`):**
    Setelah data mentah terkumpul, jalankan skrip pra-pemrosesan:

    ```bash
    python3 src/preprocess.py
    ```

    Ini akan membaca file `.jsonl` dan menghasilkan file `.csv` di `data/processed_data/` yang berisi teks yang sudah dibersihkan dan kolom-kolom asli.

3.  **Pelatihan Model (`modelling.py`):**
    Gunakan data yang telah diproses untuk melatih model BERTopic:
    ```bash
    python3 src/modelling.py
    ```
    Ini akan menghasilkan:
    -   `data/final/bertopic_model.pkl`: Model BERTopic yang dilatih.
    -   `data/final/topic_results.csv`: Artikel beserta ID topik yang ditetapkan.
        Pastikan model `paraphrase-MiniLM-L6-v2` dari SentenceTransformers berhasil diunduh saat pertama kali skrip ini dijalankan.

Setelah langkah-langkah ini selesai dan file model (`.pkl`) serta hasil topik (`.csv`) ada di direktori `data/final/`, aplikasi FastAPI akan memuatnya saat startup dan siap melayani permintaan analisis.

## 🔌 Endpoint API Utama

Aplikasi FastAPI menyediakan beberapa endpoint berikut (diasumsikan berjalan di `http://localhost:8000`):

-   **`GET /`**: Menyajikan halaman web utama (`index.html`).
-   **`POST /api/analyze/pdf`**: Menerima unggahan file PDF, menganalisis topiknya, dan mengembalikan artikel terkait.
-   **`POST /api/analyze/text`**: Menerima input judul dan abstrak, menganalisis topiknya, dan mengembalikan artikel terkait.
-   **`GET /api/search/topic`**: Menerima kueri teks (kata kunci topik) dan mengembalikan artikel yang relevan dari dataset.

(Anda mungkin memiliki endpoint "legacy" seperti `/scrape_legacy`, `/preprocess_legacy`, `/model_legacy` yang bisa digunakan untuk memicu langkah-langkah pipeline secara manual melalui API, namun fungsionalitas utamanya adalah melalui endpoint `/api/...` di atas).

## 💻 Teknologi yang Digunakan

-   **Python 3.9+**
-   **Backend & API:**
    -   FastAPI
    -   Uvicorn (ASGI Server)
-   **Pemodelan Topik & NLP:**
    -   BERTopic
    -   Sentence-Transformers (untuk embeddings)
    -   spaCy (untuk pra-pemrosesan, jika diaktifkan)
    -   NLTK (untuk pra-pemrosesan)
    -   Scikit-learn
    -   Gensim
-   **Manipulasi Data:**
    -   Pandas
    -   NumPy
-   **Web Scraping:**
    -   Requests
    -   Beautiful Soup 4
-   **Frontend:**
    -   HTML5
    -   Tailwind CSS
    -   JavaScript (vanilla)
-   **PDF Processing:**
    -   PyMuPDF (fitz)
-   **Kontainerisasi:**
    -   Docker
    -   Docker Compose
-   **Monitoring:**
    -   Prometheus
    -   Grafana (untuk visualisasi metrik dari Prometheus)
-   **Lainnya:**
    -   `python-multipart` (untuk unggahan file FastAPI)

## 🚀 Potensi Pengembangan Selanjutnya

-   Integrasi pipeline data (scraping, preprocessing, modelling) agar bisa dipicu secara otomatis atau terjadwal (misalnya, menggunakan Airflow, Prefect, atau cron jobs).
-   Implementasi _database_ untuk menyimpan artikel dan hasil topik agar lebih _scalable_ dan mudah dikelola daripada file CSV/JSONL.
-   Peningkatan granularitas dan kualitas topik dengan _tuning_ parameter BERTopic lebih lanjut, eksplorasi model embedding yang berbeda, atau teknik pra-pemrosesan yang lebih canggih.
-   Pengembangan fitur _feedback_ dari pengguna terhadap relevansi topik atau artikel terkait.
-   Optimasi performa API, terutama untuk ekstraksi teks PDF dan inferensi model.
-   Pengembangan _dashboard_ Grafana yang lebih detail untuk memantau metrik model dan performa sistem.
-   Implementasi autentikasi dan otorisasi jika diperlukan.
-   Peningkatan UI/UX antarmuka web.
    <<<<<<< HEAD

## 📄 Lisensi

(Jika ada, sebutkan lisensi proyek Anda di sini, misalnya: MIT License, Apache 2.0, dll.)

---

# _README ini dibuat untuk memberikan gambaran umum proyek AlbertopicAI. Sesuaikan detailnya lebih lanjut sesuai dengan implementasi spesifik Anda._

> > > > > > > 4f12336c3450f757af06022e4b9e644c328bcb02
