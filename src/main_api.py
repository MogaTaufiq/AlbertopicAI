import os
import pickle
import time
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from collections import Counter as PyCounter
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

# Import functions and models
from .scraper import scrape_article_titles, save_titles_to_csv
from .preprocess import preprocess_text_pipeline
from .modelling import perform_bertopic_modeling, save_model as save_bertopic_model, save_topic_results as save_bertopic_topic_results, evaluate_coherence

# Configuration Paths
BASE_DATA_PATH = "data"
RAW_DATA_PATH = os.path.join(BASE_DATA_PATH, "rawdata", "arxiv_cs_articles.jsonl")
PROCESSED_DATA_PATH = os.path.join(BASE_DATA_PATH, "processed_data", "processed_articles.csv")
FINAL_MODEL_PATH = os.path.join(BASE_DATA_PATH, "final", "bertopic_model.pkl")
TOPIC_RESULTS_PATH = os.path.join(BASE_DATA_PATH, "final", "topic_results.csv")
STATIC_FILES_DIR = "static" 

# Global Variables for Models and Data
TOPIC_MODEL = None
TOPIC_RESULTS_DF = None
SENTENCE_MODEL = None

# Stopwords
try:
    from nltk.corpus import stopwords
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except ImportError:
    print("NLTK stopwords not found. Ensure NLTK is installed and resources are downloaded.")
    ENGLISH_STOPWORDS = set()

# Load model and data at startup
def load_model_and_data():
    global TOPIC_MODEL, TOPIC_RESULTS_DF, SENTENCE_MODEL

    if not os.path.exists(FINAL_MODEL_PATH):
        print(f"Warning: Model file {FINAL_MODEL_PATH} not found.")
    else:
        try:
            with open(FINAL_MODEL_PATH, 'rb') as f:
                TOPIC_MODEL = pickle.load(f)
            print("BERTopic model loaded.")
        except Exception as e:
            print(f"Error loading BERTopic model: {e}")
            TOPIC_MODEL = None

    if not os.path.exists(TOPIC_RESULTS_PATH):
        print(f"Warning: Topic results file {TOPIC_RESULTS_PATH} not found.")
    else:
        try:
            TOPIC_RESULTS_DF = pd.read_csv(TOPIC_RESULTS_PATH)
            if 'Topic' in TOPIC_RESULTS_DF.columns:
                TOPIC_RESULTS_DF['Topic'] = pd.to_numeric(TOPIC_RESULTS_DF['Topic'], errors='coerce')
            print("Topic results loaded.")
        except Exception as e:
            print(f"Error loading topic results: {e}")
            TOPIC_RESULTS_DF = None

    try:
        SENTENCE_MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("SentenceTransformer model loaded.")
    except Exception as e:
        print(f"Error loading SentenceTransformer: {e}")
        SENTENCE_MODEL = None

# FastAPI app setup
app = FastAPI(title="AlbertopicAI API", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to a specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and data when FastAPI starts
@app.on_event("startup")
async def startup_event():
    load_model_and_data()

# Pydantic models for requests
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

# Helper Functions
def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting PDF text: {e}")

def get_topic_representation(topic_id: int) -> str:
    if TOPIC_MODEL is None:
        return "Model not available"
    try:
        topic_words_scores = TOPIC_MODEL.get_topic(topic_id)
        if topic_words_scores:
            return ", ".join([word for word, score in topic_words_scores[:5]])
        return f"Topic {topic_id} (no keywords)"
    except Exception as e:
        print(f"Error getting topic representation for ID {topic_id}: {e}")
        return f"Topic {topic_id} (error)"

def find_related_articles_by_topic_id(topic_id: int, current_article_title: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    related = []
    if TOPIC_RESULTS_DF is None or TOPIC_RESULTS_DF.empty or 'Topic' not in TOPIC_RESULTS_DF.columns:
        return related

    matched_articles_df = TOPIC_RESULTS_DF[TOPIC_RESULTS_DF['Topic'] == topic_id]
    
    for _, row in matched_articles_df.iterrows():
        if current_article_title and row.get('title', '').strip().lower() == current_article_title.strip().lower():
            continue
        
        authors_list = []
        authors_data = row.get('authors')
        if isinstance(authors_data, str):
            try:
                parsed_authors = eval(authors_data)
                if isinstance(parsed_authors, list):
                    authors_list = parsed_authors
            except:
                authors_list = [authors_data]
        elif isinstance(authors_data, list):
            authors_list = authors_data

        related.append({
            "title": row.get('title', 'No Title'),
            "authors": authors_list,
            "year": str(row.get('year', 'N/A')),
            "topic_keywords": get_topic_representation(int(row.get('Topic'))) if pd.notna(row.get('Topic')) else "N/A",
            "abstract_snippet": (str(row.get('abstract', ''))[:150] + '...') if pd.notna(row.get('abstract')) and str(row.get('abstract', '')) else "Abstract not available."
        })
        if len(related) >= limit:
            break
    return related

# API Endpoints

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_index_html():
    index_html_path = os.path.join(STATIC_FILES_DIR, "index.html")
    if not os.path.exists(index_html_path):
        from fastapi import HTTPException # Impor di dalam fungsi jika belum di atas
        raise HTTPException(status_code=404, detail="Halaman utama (index.html) tidak ditemukan.")
    return FileResponse(index_html_path)
    
@app.post("/api/analyze/pdf", response_model=AnalysisResponse)
async def analyze_pdf_endpoint(pdf_file: UploadFile = File(...)):
    if TOPIC_MODEL is None:
        raise HTTPException(status_code=503, detail="BERTopic model not available.")
    
    contents = await pdf_file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty PDF file.")
    
    extracted_text = extract_text_from_pdf(contents)
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from PDF.")
    
    processed_text = preprocess_text_pipeline(extracted_text, ENGLISH_STOPWORDS)
    
    try:
        predicted_topic_ids, _ = TOPIC_MODEL.transform([processed_text])
        predicted_topic_id = predicted_topic_ids[0] if predicted_topic_ids else -1
        predicted_topic_str = get_topic_representation(predicted_topic_id)
        
        related = find_related_articles_by_topic_id(predicted_topic_id)
        return AnalysisResponse(predicted_topic=predicted_topic_str, related_articles=related)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during PDF analysis: {str(e)}")

@app.post("/api/analyze/text", response_model=AnalysisResponse)
async def analyze_text_endpoint(request: TextAnalysisRequest):
    if TOPIC_MODEL is None:
        raise HTTPException(status_code=503, detail="BERTopic model not available.")
    
    if not request.title.strip() or not request.abstract.strip():
        raise HTTPException(status_code=400, detail="Title and Abstract cannot be empty.")
    
    text_to_analyze = request.title + " " + request.abstract
    processed_text = preprocess_text_pipeline(text_to_analyze, ENGLISH_STOPWORDS)

    try:
        predicted_topic_ids, _ = TOPIC_MODEL.transform([processed_text])
        predicted_topic_id = predicted_topic_ids[0] if predicted_topic_ids else -1
        predicted_topic_str = get_topic_representation(predicted_topic_id)
        
        related = find_related_articles_by_topic_id(predicted_topic_id, current_article_title=request.title)
        return AnalysisResponse(predicted_topic=predicted_topic_str, related_articles=related)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during text analysis: {str(e)}")

@app.get("/api/search/topic", response_model=TopicSearchResponse)
async def search_topic_endpoint(query: str = Query(..., min_length=3)):
    if TOPIC_MODEL is None:
        raise HTTPException(status_code=503, detail="BERTopic model not available.")
    if TOPIC_RESULTS_DF is None or TOPIC_RESULTS_DF.empty:
        raise HTTPException(status_code=503, detail="Topic data not available.")
    
    try:
        similar_topics_info = TOPIC_MODEL.find_topics(query, top_n=1)
        found_articles_response = []

        if similar_topics_info and similar_topics_info[0][1] > 0.1:
            target_topic_id = similar_topics_info[0][0]
            matched_df = TOPIC_RESULTS_DF[TOPIC_RESULTS_DF['Topic'] == target_topic_id]

            for _, row in matched_df.head(10).iterrows():
                authors_list = []
                authors_data = row.get('authors')
                if isinstance(authors_data, str):
                    try:
                        parsed_authors = eval(authors_data)
                        if isinstance(parsed_authors, list):
                            authors_list = parsed_authors
                    except:
                        authors_list = [authors_data]
                elif isinstance(authors_data, list):
                    authors_list = authors_data

                found_articles_response.append(ArticleResponse(
                    title=row.get('title', 'No Title'),
                    authors=authors_list,
                    year=str(row.get('year', 'N/A')),
                    topic_keywords=get_topic_representation(int(row.get('Topic'))) if pd.notna(row.get('Topic')) else "N/A",
                    abstract_snippet=(str(row.get('abstract', ''))[:150] + '...') if pd.notna(row.get('abstract')) and str(row.get('abstract', '')) else "Abstract not available."
                ))

        return TopicSearchResponse(articles=found_articles_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching for topics: {str(e)}")

# Running the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
