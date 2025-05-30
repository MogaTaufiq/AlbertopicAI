# Gunakan image Python versi 3.10 yang lebih ringan
FROM python:3.10-slim

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Install dependencies sistem yang diperlukan (gcc, g++, make, dll.)
RUN apt-get update && \
    apt-get install -y gcc g++ make libx11-dev && \
    rm -rf /var/lib/apt/lists/* # Membersihkan cache apt untuk mengurangi ukuran image

# Salin file requirements.txt terlebih dahulu untuk memanfaatkan cache Docker layer
COPY requirements.txt .

# Install dependensi Python dari requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin direktori source code aplikasi
COPY src ./src

# Salin direktori static yang berisi index.html dan aset frontend lainnya
COPY static ./static

# Download model NLP untuk NLTK dan SpaCy
RUN python -m nltk.downloader punkt stopwords wordnet && \
    python -m spacy download en_core_web_sm

# Membuat pengguna non-root untuk keamanan
RUN useradd -m appuser
USER appuser

# Mengekspos port 8000 untuk FastAPI
EXPOSE 8000

# Menentukan perintah untuk menjalankan aplikasi menggunakan Uvicorn
CMD ["uvicorn", "src.main_api:app", "--host", "0.0.0.0", "--port", "8000"]
