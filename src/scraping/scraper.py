import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

def scrape_article_titles(url: str, max_results=1000):
    """
    Scrape article titles from a given URL (e.g., arXiv).
    """
    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query=cat:cs.*&start=0&max_results={max_results}"
    url = base_url + query

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "lxml")

    entries = soup.find_all("entry")
    data = []

    for entry in entries:
        title = entry.title.text.strip().replace('\n', ' ')
        data.append({
            "title": title,
        })
    
    return data

def save_titles_to_csv(titles, filename="arxiv_cs_articles.csv"):
    """
    Save the scraped article titles to a CSV file in data/rawdata.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
    save_path = os.path.join(project_root, "AlbertopicAI", "data", "rawdata", filename)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(titles)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} titles to {save_path}")
    return save_path

if __name__ == "__main__":
    url = "http://export.arxiv.org/api/query?search_query=cat:cs.*&start=0&max_results=1000"
    titles = scrape_article_titles(url)
    save_titles_to_csv(titles)
