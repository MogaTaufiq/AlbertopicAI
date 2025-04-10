# AlbertopicAI

**AlbertopicAI** adalah proyek riset scraping dan klasifikasi topik artikel secara otomatis menggunakan teknologi Python, FastAPI, dan preprocessing berbasis NLP.

## 🚀 Fitur

-   Scraping judul artikel dari URL berita.
-   Preprocessing teks menggunakan NLTK dan spaCy.
-   API untuk mengakses hasil scraping dan preprocessing.
-   Siap dijalankan di GitHub Codespaces dengan Docker.

## 📁 Struktur Proyek

.
├── data/
│ ├── rawdata/
│ └── processed_data/
├── src/
│ ├── scraping/
│ │ └── scraper.py
│ ├── preprocessing/
│ │ ├── preprocessing.py
│ │ └── text_cleaning.py
│ ├── scraping_api.py
│ └── preprocessed_api.py
├── .devcontainer/
│ ├── Dockerfile
│ └── devcontainer.json
├── requirements.txt
└── README.md

## 🧪 Menjalankan Secara Lokal

### 1. Clone repository

```bash
git clone https://github.com/MogaTaufiq/AlbertopicAI.git
cd AlbertopicAI

2. Install dependencies

Pastikan kamu sudah install Python 3.10+

pip install -r requirements.txt

3. Jalankan API

uvicorn src.scraping_api:app --reload --port 8000

Lalu akses:
	•	http://localhost:8000/scrape?url=https://example.com
	•	http://localhost:8000/preprocessed

🐳 Menjalankan di GitHub Codespaces

1. Buka dengan Codespaces

Klik <> Code → Codespaces → Create codespace on main.

2. Tunggu environment selesai dibuild

3. Jalankan server FastAPI

uvicorn src.scraping_api:app --host 0.0.0.0 --port 8000

4. Akses melalui forwarded port

GitHub akan membuka port 8000 sebagai public preview.

📦 API Endpoint

Endpoint	Method	Deskripsi
/scrape	GET	Melakukan scraping judul artikel dari URL
/preprocessed	GET	Menampilkan hasil preprocessing CSV

🤖 Teknologi yang Digunakan
	•	Python 3
	•	FastAPI
	•	BeautifulSoup
	•	NLTK
	•	spaCy
	•	Docker + GitHub Codespaces

📜 Lisensi

Proyek ini menggunakan lisensi MIT.

⸻

Feel free to fork, eksperimen, dan kontribusi ya!

---

Silakan copas isi di atas ke dalam `README.md` kamu. Kalau mau sekalian aku bantu generate `requirements.txt` juga atau lanjut bantu setup Docker-nya di Codespace, tinggal bilang!
```

# README.md
