FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the src folder into the container
COPY src ./src

# Download NLTK and spaCy resources
RUN python -m nltk.downloader punkt stopwords wordnet && \
    python -m spacy download en_core_web_sm

EXPOSE 8000

CMD ["uvicorn", "src.main_api:app", "--host", "0.0.0.0", "--port", "8000"]
