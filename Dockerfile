# Gunakan image Python versi 3.10 yang lebih ringan
FROM python:3.10-slim

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Install dependencies sistem yang diperlukan (jika ada library yang perlu dikompilasi)
RUN apt-get update && \
    apt-get install -y gcc g++ make libx11-dev && \
    rm -rf /var/lib/apt/lists/*

# Salin file requirements.txt terlebih dahulu untuk memanfaatkan cache Docker
COPY requirements.txt .

# Install dependensi Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download model NLP untuk NLTK dan SpaCy sebelum menyalin kode
RUN python -m nltk.downloader punkt stopwords wordnet && \
    python -m spacy download en_core_web_sm

# Salin semua direktori yang diperlukan oleh aplikasi
COPY src ./src
COPY static ./static
# --- PERUBAHAN KRITIS ---
# Salin direktori data yang berisi model yang telah dilatih dan hasil topik
COPY data ./data

# --- SARAN PERBAIKAN ---
# Membuat pengguna non-root dan mengubah kepemilikan direktori kerja
# Ini memastikan aplikasi memiliki izin yang benar jika perlu menulis file.
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app

# Beralih ke pengguna non-root
USER appuser

# Mengekspos port 8000 untuk FastAPI
EXPOSE 8000

# Menentukan perintah untuk menjalankan aplikasi menggunakan Uvicorn
CMD ["uvicorn", "src.main_api:app", "--host", "0.0.0.0", "--port", "8000"]
