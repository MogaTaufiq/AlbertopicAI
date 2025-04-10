from fastapi import FastAPI
import os
import pandas as pd
from src.preprocessing.preprocess import preprocess_data  # sesuaikan nama modul/folder
from nltk.corpus import stopwords

app = FastAPI()

# Path input dan output
rawdata_path = os.path.join("data", "rawdata", "article_titles.csv")
processed_path = os.path.join("data", "processed_data", "processed_titles.csv")

@app.get("/preprocess")
async def run_preprocessing():
    try:
        if not os.path.exists(rawdata_path):
            return {"error": "Scraped data not found. Please run the scrape service first."}
        
        stopwords_set = set(stopwords.words('english'))
        preprocess_data(rawdata_path, processed_path, stopwords_set)

        return {"message": "Preprocessing completed successfully."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/preprocessed")
async def get_preprocessed_titles():
    try:
        if not os.path.exists(processed_path):
            return {"error": "Processed data not found. Please run the preprocess service first."}
        
        df = pd.read_csv(processed_path)
        if 'Processed_Text' not in df.columns:
            return {"error": "No 'Processed_Text' column found in the processed file."}
        
        return {"preprocessed_titles": df['Processed_Text'].dropna().tolist()}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Jalankan di port berbeda dari scraping_api