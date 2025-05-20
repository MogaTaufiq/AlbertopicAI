# scraper_dataset.py
import requests
from bs4 import BeautifulSoup
import json
import os

def scrape_arxiv(query="machine learning", max_results=10):
    """
    Scrape arXiv articles menggunakan BeautifulSoup
    """
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"all:{query}"
    url = f"{base_url}search_query={search_query}&start=0&max_results={max_results}"
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")  # Gunakan parser XML
    
    entries = soup.find_all("entry")
    
    dataset = []
    for entry in entries:
        # Extract authors
        authors = [author.find("name").text for author in entry.find_all("author")]
        
        # Extract published year
        published = entry.published.text.strip()[:4] if entry.published else ""
        
        # Extract DOI (jika ada)
        doi_tag = entry.find("arxiv:doi")
        doi = doi_tag.text.strip() if doi_tag else ""
        
        paper = {
            "title": entry.title.text.strip(),
            "abstract": entry.summary.text.strip(),
            "authors": authors,
            "journal_conference_name": entry.find("arxiv:journal_ref").text.strip() if entry.find("arxiv:journal_ref") else "arXiv",
            "publisher": "arXiv",
            "year": published,
            "doi": doi,
            "group_name": "default_group"
        }
        dataset.append(paper)
    
    return dataset

def save_to_json(data, filename="data/arxiv_dataset.json"):
    """Simpan ke format JSON"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} entries to {filename}")

if __name__ == "__main__":
    # Ganti parameter sesuai kebutuhan
    articles = scrape_arxiv(query="machine learning", max_results=10)
    save_to_json(articles)