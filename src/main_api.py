import os
import re
import pickle
import time
import pandas as pd
import numpy as np
from sentence_transformers import util, SentenceTransformer
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import fitz  # PyMuPDF

# Impor fungsi dari modul lain dalam paket 'src'
from .preprocess import preprocess_text_pipeline

# --- Konfigurasi Path ---
BASE_DATA_PATH = "data"
STATIC_FILES_DIR = "static" 

# --- Variabel Global untuk Model dan Data ---
TOPIC_MODEL = None
TOPIC_RESULTS_DF = None
SENTENCE_MODEL = None

# --- Stopwords ---
try:
    from nltk.corpus import stopwords
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except ImportError:
    print("NLTK stopwords tidak ditemukan. Pastikan NLTK terinstal dan resource diunduh.")
    ENGLISH_STOPWORDS = set()

# --- Fungsi Baru untuk Menemukan Model dan Hasil Terbaru ---
def find_latest_model_paths():
    """
    Menemukan path untuk model .pkl dan hasil .csv terbaru berdasarkan ID numerik tertinggi.
    Mengembalikan tuple (model_path, results_path).
    """
    model_dir = os.path.join(BASE_DATA_PATH, "final", "bertopic_model")
    results_dir = os.path.join(BASE_DATA_PATH, "final", "topic_results")
    
    latest_id = -1
    
    # Pastikan direktori ada sebelum memindai
    if not os.path.isdir(results_dir):
        print(f"Peringatan: Direktori hasil topik '{results_dir}' tidak ditemukan.")
        return None, None

    # Cari ID tertinggi dari file hasil (bisa juga dari file model)
    for filename in os.listdir(results_dir):
        match = re.search(r'_(\d+)\.csv$', filename)
        if match:
            current_id = int(match.group(1))
            if current_id > latest_id:
                latest_id = current_id

    if latest_id == -1:
        print("Tidak ditemukan file model atau hasil topik yang valid dengan ID berurutan.")
        return None, None
        
    # Buat path lengkap berdasarkan ID tertinggi yang ditemukan
    latest_model_path = os.path.join(model_dir, f"bertopic_model_{latest_id}.pkl")
    latest_results_path = os.path.join(results_dir, f"topic_results_{latest_id}.csv")
    
    print(f"Menemukan file terbaru dengan ID: {latest_id}")
    return latest_model_path, latest_results_path

# --- Fungsi Pemuatan Model yang Diperbarui ---
def load_model_and_data():
    """Memuat model BERTopic dan data hasil topik versi terbaru."""
    global TOPIC_MODEL, TOPIC_RESULTS_DF, SENTENCE_MODEL

    model_path, results_path = find_latest_model_paths()

    if model_path and os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                TOPIC_MODEL = pickle.load(f)
            print(f"Model BERTopic terbaru berhasil dimuat dari: {model_path}")
        except Exception as e:
            print(f"Error memuat model BERTopic dari {model_path}: {e}")
            TOPIC_MODEL = None
    else:
        print(f"Peringatan: File model terbaru tidak ditemukan di path yang diharapkan ({model_path}).")
        TOPIC_MODEL = None

    if results_path and os.path.exists(results_path):
        try:
            TOPIC_RESULTS_DF = pd.read_csv(results_path)
            if 'Topic' in TOPIC_RESULTS_DF.columns:
                TOPIC_RESULTS_DF['Topic'] = pd.to_numeric(TOPIC_RESULTS_DF['Topic'], errors='coerce')
            print(f"Hasil topik terbaru berhasil dimuat dari: {results_path}")
        except Exception as e:
            print(f"Error memuat hasil topik dari {results_path}: {e}")
            TOPIC_RESULTS_DF = None
    else:
        print(f"Peringatan: File hasil topik terbaru tidak ditemukan di path yang diharapkan ({results_path}).")
        TOPIC_RESULTS_DF = None

    try:
        SENTENCE_MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("Model SentenceTransformer berhasil dimuat.")
    except Exception as e:
        print(f"Error memuat SentenceTransformer: {e}")
        SENTENCE_MODEL = None

# --- Pengaturan Aplikasi FastAPI ---
app = FastAPI(title="AlbertopicAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("Memulai aplikasi FastAPI...")
    load_model_and_data()

# --- Pydantic Models (tidak berubah) ---
class TextAnalysisRequest(BaseModel):
    title: str
    abstract: str

class ArticleResponse(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[str] = None
    topic_keywords: Optional[str] = None
    abstract_snippet: Optional[str] = None

class AnalysisResponse(BaseModel):
    predicted_topic: Optional[str] = None
    related_articles: List[ArticleResponse] = []

class TopicSearchResponse(BaseModel):
    articles: List[ArticleResponse] = []

# --- Fungsi Helper (tidak berubah) ---
def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        print(f"Error mengekstrak teks PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal mengekstrak teks dari PDF: {e}")

def get_topic_representation(topic_id: int) -> str:
    if TOPIC_MODEL is None: return "Model tidak tersedia"
    try:
        topic_words_scores = TOPIC_MODEL.get_topic(topic_id)
        if topic_words_scores:
            return ", ".join([word for word, score in topic_words_scores[:5]])
        return f"Topik {topic_id} (tidak ada kata kunci)"
    except Exception as e:
        return f"Topik {topic_id} (error)"

def find_related_articles_by_topic_id(topic_id: int, current_article_title: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    related = []
    if TOPIC_RESULTS_DF is None or TOPIC_RESULTS_DF.empty: return related
    
    matched_articles_df = TOPIC_RESULTS_DF[TOPIC_RESULTS_DF['Topic'] == topic_id]
    
    for _, row in matched_articles_df.head(limit).iterrows():
        if current_article_title and row.get('title', '').strip().lower() == current_article_title.strip().lower(): continue
        
        authors_list = []
        authors_data = row.get('authors')
        if isinstance(authors_data, str):
            try: authors_list = eval(authors_data)
            except: authors_list = [authors_data]
        elif isinstance(authors_data, list): authors_list = authors_data

        related.append({
            "title": row.get('title', 'Tanpa Judul'),
            "authors": authors_list,
            "year": str(row.get('year', 'N/A')),
            "topic_keywords": get_topic_representation(int(row.get('Topic'))) if pd.notna(row.get('Topic')) else "N/A",
            "abstract_snippet": (str(row.get('abstract', ''))[:150] + '...') if pd.notna(row.get('abstract')) and str(row.get('abstract', '')) else "Abstrak tidak tersedia."
        })
    return related

# --- API Endpoints ---
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_index_html():
    index_html_path = os.path.join(STATIC_FILES_DIR, "index.html")
    if not os.path.exists(index_html_path):
        raise HTTPException(status_code=404, detail="Halaman utama (index.html) tidak ditemukan.")
    return FileResponse(index_html_path)
    
@app.post("/api/analyze/pdf", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_pdf_endpoint(pdf_file: UploadFile = File(...)):
    if TOPIC_MODEL is None: raise HTTPException(status_code=503, detail="Model BERTopic tidak tersedia.")
    
    contents = await pdf_file.read()
    if not contents: raise HTTPException(status_code=400, detail="File PDF kosong.")
    
    extracted_text = extract_text_from_pdf(contents)
    if not extracted_text.strip(): raise HTTPException(status_code=400, detail="Tidak ada teks yang dapat diekstrak dari PDF.")
    
    processed_text = preprocess_text_pipeline(extracted_text, ENGLISH_STOPWORDS)
    
    try:
        topic_ids, probs = TOPIC_MODEL.transform([processed_text])
        predicted_topic_id = topic_ids[0] if topic_ids else -1
        predicted_topic_str = get_topic_representation(predicted_topic_id)
        
        related = find_related_articles_by_topic_id(predicted_topic_id)
        return AnalysisResponse(predicted_topic=predicted_topic_str, related_articles=related)
    
    except Exception as e:
        import traceback
        print(f"Error saat analisis PDF (traceback): {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat analisis PDF: {e}")

@app.post("/api/analyze/text", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_text_endpoint(request: TextAnalysisRequest):
    if TOPIC_MODEL is None: raise HTTPException(status_code=503, detail="Model BERTopic tidak tersedia.")
    
    text_to_analyze = request.title + " " + request.abstract
    processed_text = preprocess_text_pipeline(text_to_analyze, ENGLISH_STOPWORDS)

    try:
        topic_ids, _ = TOPIC_MODEL.transform([processed_text])
        predicted_topic_id = topic_ids[0] if topic_ids else -1
        predicted_topic_str = get_topic_representation(predicted_topic_id)
        
        related = find_related_articles_by_topic_id(predicted_topic_id, current_article_title=request.title)
        return AnalysisResponse(predicted_topic=predicted_topic_str, related_articles=related)
    
    except Exception as e:
        import traceback
        print(f"Error saat analisis teks (traceback): {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat analisis teks: {e}")

@app.get("/api/search/topic", response_model=TopicSearchResponse, tags=["Search"])
async def search_topic_endpoint(query: str = Query(..., min_length=3)):
    if TOPIC_MODEL is None or SENTENCE_MODEL is None: raise HTTPException(status_code=503, detail="Model tidak tersedia.")
    if TOPIC_RESULTS_DF is None or TOPIC_RESULTS_DF.empty: raise HTTPException(status_code=503, detail="Data artikel tidak tersedia.")

    try:
        query_embedding = SENTENCE_MODEL.encode(query)
        
        topic_ids = sorted([tid for tid in TOPIC_MODEL.get_topics() if tid != -1])
        if not topic_ids: return TopicSearchResponse(articles=[])

        topic_representations = [" ".join(word for word, _ in TOPIC_MODEL.get_topic(tid)) for tid in topic_ids]
        topic_embeddings = SENTENCE_MODEL.encode(topic_representations)
        
        similarities = util.cos_sim(query_embedding, topic_embeddings)
        top_topic_index = np.argmax(similarities[0])
        highest_similarity_score = similarities[0][top_topic_index]

        SIMILARITY_THRESHOLD = 0.2
        if highest_similarity_score > SIMILARITY_THRESHOLD:
            target_topic_id = topic_ids[top_topic_index]
            print(f"Query '{query}' paling mirip dengan Topik ID: {target_topic_id} (Skor: {highest_similarity_score:.4f})")
            articles = find_related_articles_by_topic_id(target_topic_id)
            return TopicSearchResponse(articles=articles)
        else:
            print(f"Tidak ada topik yang cukup mirip ditemukan untuk query: '{query}' (Skor tertinggi: {highest_similarity_score:.4f})")
            return TopicSearchResponse(articles=[])

    except Exception as e:
        import traceback
        print(f"Error saat mencari topik (traceback): {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat mencari topik: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
