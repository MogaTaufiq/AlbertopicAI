from fastapi import FastAPI
from src.scraping.scraper import scrape_article_titles, save_titles_to_csv
from src.preprocessing.preprocess import preprocess_data
from src.modelling.modelling import (
    perform_bertopic_modeling,
    save_model,
    save_topic_results,
    evaluate_coherence,
    load_data,
)
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter
import os
import pandas as pd
import time
from collections import Counter
from nltk.corpus import stopwords

app = FastAPI()

# Initialize Prometheus metrics
coherence_score_gauge = Gauge('bertopic_coherence_score', 'Coherence Score from BERTopic')
num_topics_gauge = Gauge('bertopic_num_topics', 'Number of topics found by BERTopic')
training_time_gauge = Gauge('bertopic_training_time', 'Training time of BERTopic model in seconds')
topic_freq_gauge = Gauge('bertopic_topic_freq', 'Frequency of each topic', ['topic'])
docs_per_topic_gauge = Gauge('bertopic_docs_per_topic', 'Number of documents assigned to each topic', ['topic'])

# Expose metrics to Prometheus
Instrumentator().instrument(app).expose(app)

RAW_DATA_PATH = "data/rawdata/article_titles.csv"
PROCESSED_DATA_PATH = "data/processed_data/processed_titles.csv"
FINAL_MODEL_PATH = "data/final/bertopic_model.pkl"
TOPIC_RESULTS_PATH = "data/final/topic_results.csv"


@app.get("/scrape")
async def scrape_titles(url: str):
    try:
        article_titles = scrape_article_titles(url)
        save_titles_to_csv(article_titles)
        return {"message": f"Scraped {len(article_titles)} titles successfully."}
    except Exception as e:
        return {"error": str(e)}


@app.get("/preprocess")
async def run_preprocessing():
    if not os.path.exists(RAW_DATA_PATH):
        return {"error": "Scraped data not found. Please run the scrape service first."}
    try:
        stopwords_set = set(stopwords.words('english'))
        preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH, stopwords_set)
        return {"message": "Preprocessing completed successfully."}
    except Exception as e:
        return {"error": str(e)}


@app.get("/preprocessed")
async def get_preprocessed_titles():
    if not os.path.exists(PROCESSED_DATA_PATH):
        return {"error": "Processed data not found. Please run the preprocess service first."}
    try:
        df = load_data(PROCESSED_DATA_PATH)
        if df is None:
            return {"error": "Failed to load processed data."}
        return {"preprocessed_titles": df['Processed_Text'].dropna().tolist()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/model")
async def perform_modeling():
    if not os.path.exists(PROCESSED_DATA_PATH):
        return {"error": "Processed data not found. Please run the preprocess service first."}
    try:
        df = load_data(PROCESSED_DATA_PATH)
        if df is None:
            return {"error": "Failed to load processed data properly."}

        start_time = time.time()

        topic_model, topics = perform_bertopic_modeling(df)

        training_time = time.time() - start_time
        training_time_gauge.set(training_time)

        save_model(topic_model)
        save_topic_results(df, topics)

        coherence_score = evaluate_coherence(df, topic_model)
        coherence_score_gauge.set(coherence_score)
        num_topics_gauge.set(len(set(topics)))

        topic_freq = Counter(topics)
        for topic, freq in topic_freq.items():
            topic_freq_gauge.labels(topic=str(topic)).set(freq)

        docs_per_topic = Counter(topics)
        for topic, count in docs_per_topic.items():
            docs_per_topic_gauge.labels(topic=str(topic)).set(count)

        return {
            "message": "Modeling completed successfully",
            "coherence_score": coherence_score,
            "training_time": training_time
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/model/status")
async def get_model_status():
    if not os.path.exists(FINAL_MODEL_PATH):
        return {"error": "Model not found. Please run the model service first."}
    try:
        return {"message": f"Model and topic results are saved at {FINAL_MODEL_PATH} and {TOPIC_RESULTS_PATH}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
