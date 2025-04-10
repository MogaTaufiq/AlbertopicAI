from fastapi import FastAPI
from src.scraping.scraper import scrape_article_titles, save_titles_to_csv
import os

app = FastAPI()

@app.get("/scrape")
async def scrape_titles(url: str):
    try:
        # Melakukan scraping judul artikel dari URL yang diberikan
        article_titles = scrape_article_titles(url)
        
        # Menyimpan hasil scraping ke file CSV
        save_titles_to_csv(article_titles)
        
        return {"message": f"Scraped {len(article_titles)} titles successfully."}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
