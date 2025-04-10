from fastapi import FastAPI
from src.scraping.scraper import scrape_article_titles, save_titles_to_csv
from src.preprocessing.pipeline import run_preprocessing_pipeline
import os
import pandas as pd

app = FastAPI()

RAW_DATA_PATH = "data/rawdata/article_titles.csv"
PROCESSED_DATA_PATH = "data/processed_data/processed_titles.csv"

@app.get("/scrape")
async def scrape_titles(url: str):
    try:
        article_titles = scrape_article_titles(url)
        save_titles_to_csv(article_titles)
        return {"message": f"Scraped {len(article_titles)} titles successfully."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/preprocessed")
async def get_preprocessed_titles():
    if not os.path.exists(RAW_DATA_PATH):
        return {"error": "Scraped data not found. Please run the scrape service first."}
    
    try:
        run_preprocessing_pipeline()  # Jalankan pipeline preproses
        df = pd.read_csv(PROCESSED_DATA_PATH)
        return {"preprocessed_titles": df['Processed_Text'].dropna().tolist()}
    except Exception as e:
        return {"error": str(e)}