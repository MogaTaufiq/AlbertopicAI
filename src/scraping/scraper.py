import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def scrape_article_titles(url):
    """Scrape article titles from the specified URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Misalnya, kita ingin mendapatkan semua judul artikel dari elemen <h3>
    titles = soup.find_all('h3')  # Sesuaikan dengan elemen HTML yang sesuai di situs target
    article_titles = [title.get_text(strip=True) for title in titles]

    return article_titles

def save_titles_to_csv(titles, filename='article_titles.csv'):
    """Save scraped titles to a CSV file in the 'rawdata' folder."""
    # Tentukan jalur folder rawdata di dalam data menggunakan jalur absolut
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Mendapatkan direktori dari skrip saat ini
    file_path = os.path.join(base_dir, '..', '..', 'data', 'rawdata', filename)  # Naik satu level ke root folder

    # Membuat folder rawdata jika belum ada
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    df = pd.DataFrame(titles, columns=['Title'])
    df.to_csv(file_path, index=False)
    print(f"File saved to {file_path}")

if __name__ == "__main__":
    url = 'https://ijaseit.insightsociety.org/index.php/ijaseit'  # URL dari IJASEIT atau sumber lain yang relevan
    article_titles = scrape_article_titles(url)
    save_titles_to_csv(article_titles)
    print(f"Saved {len(article_titles)} titles to article_titles.csv in 'rawdata' folder.")
