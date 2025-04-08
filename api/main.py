from fastapi import FastAPI
from .scraper_service import scrape_articles
from .data_service import get_processed_data

app = FastAPI()

@app.get("/scrape")
async def scrape():
    """
    Menjalankan proses scraping untuk mengambil artikel.
    """
    result = scrape_articles()
    return {"status": "scraping started", "details": result}

@app.get("/data")
async def get_data():
    """
    Menampilkan data yang sudah diproses (hasil scraping).
    """
    data = get_processed_data()
    return {"status": "success", "data": data}
