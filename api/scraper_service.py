from src.scraping.scraper import scrape_article_titles, save_titles_to_csv

def scrape_articles():
    """
    Memulai proses scraping artikel dan menyimpannya dalam file.
    """
    url = 'https://ijaseit.insightsociety.org/index.php/ijaseit'
    titles = scrape_article_titles(url)
    file_path = save_titles_to_csv(titles)
    return {"file_path": file_path, "titles_count": len(titles)}
