# Gunakan Python image resmi
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy semua file ke container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Download resource NLTK dan spaCy
RUN python -m nltk.downloader punkt stopwords wordnet && \
    python -m spacy download en_core_web_sm

# Jalankan scraping API
CMD ["uvicorn", "src.main_api:app", "--host", "0.0.0.0", "--port", "8000"]