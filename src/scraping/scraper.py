import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

def scrape_arxiv_cs_articles(max_results=200):
    """
    Scrape recent Computer Science articles from arXiv using the arXiv API.
    """
    base_url = "http://export.arxiv.org/api/query?"
    query = "search_query=cat:cs.*&start=0&max_results={}".format(max_results)
    url = base_url + query

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "lxml")

    entries = soup.find_all("entry")

    data = []
    for entry in entries:
        title = entry.title.text.strip().replace('\n', ' ')
        abstract = entry.summary.text.strip().replace('\n', ' ')
        data.append({
            "title": title,
            "abstract": abstract
        })

    return data

def save_articles_to_csv(articles, filename="arxiv_cs_articles.csv"):
    """
    Save the scraped articles to a CSV file in data/rawdata.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
    save_path = os.path.join(project_root, "AlbertopicAI", "data", "rawdata", filename)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(articles)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} articles to {save_path}")
    return save_path

if __name__ == "__main__":
    articles = scrape_arxiv_cs_articles()
    save_articles_to_csv(articles)