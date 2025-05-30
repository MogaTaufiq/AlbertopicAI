import os
import requests
from bs4 import BeautifulSoup # Pastikan BeautifulSoup diinstal untuk XML
import json
import time

# --- Konfigurasi ---
ARXIV_API_URL = "http://export.arxiv.org/api/query?"
DEFAULT_QUERY = "cat:cs.*"  # Kategori Computer Science, bisa diubah
TARGET_ARTICLES_GOAL = 10000  # Target total artikel
BATCH_SIZE_PER_ITERATION = 500  # Jumlah artikel per panggilan API / per iterasi
OUTPUT_FILENAME = "data/rawdata/arxiv_cs_articles.jsonl"
CHECKPOINT_FILENAME = "data/rawdata/scraper_checkpoint.json"

# Pastikan direktori data/rawdata ada
os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

# --- Fungsi Checkpoint ---
def load_checkpoint():
    """Memuat checkpoint terakhir (next_start_index dan total_articles_scraped)."""
    if os.path.exists(CHECKPOINT_FILENAME):
        with open(CHECKPOINT_FILENAME, 'r') as f:
            try:
                checkpoint = json.load(f)
                return checkpoint.get('next_start_index', 0), checkpoint.get('total_articles_scraped', 0)
            except json.JSONDecodeError:
                print(f"Peringatan: File checkpoint '{CHECKPOINT_FILENAME}' rusak. Memulai dari awal.")
                return 0, 0
    return 0, 0

def save_checkpoint(next_start_index, total_articles_scraped):
    """Menyimpan checkpoint saat ini."""
    with open(CHECKPOINT_FILENAME, 'w') as f:
        json.dump({'next_start_index': next_start_index, 'total_articles_scraped': total_articles_scraped}, f, indent=2)
    print(f"Checkpoint disimpan: next_start_index={next_start_index}, total_articles_scraped={total_articles_scraped}")

# --- Fungsi Parsing Data ---
def parse_arxiv_entry(entry_soup):
    """Mengekstrak data dari satu <entry> XML menjadi format JSON yang diinginkan."""
    
    # Fungsi pembantu untuk mendapatkan teks dengan aman
    def get_text_safely(element, tag_name, namespace_uri=None):
        if namespace_uri: # Jika ada namespace spesifik (meskipun BS4 kadang bisa tanpa ini)
            found = element.find(tag_name, namespaces=namespace_uri)
        else: # Mencoba tanpa namespace eksplisit dulu
            found = element.find(tag_name)
            # Jika tidak ditemukan, coba dengan prefix 'atom:' karena itu umum di Atom feed
            if not found:
                 found = element.find(f"atom:{tag_name}")

        return found.text.strip().replace('\n', ' ') if found else ""

    title = get_text_safely(entry_soup, 'title')
    abstract = get_text_safely(entry_soup, 'summary') # Di arXiv, 'summary' adalah abstrak

    authors_elements = entry_soup.find_all('author') # atau 'atom:author'
    if not authors_elements and entry_soup.find('atom:author'): # Coba dengan prefix jika find_all gagal
        authors_elements = entry_soup.find_all('atom:author')
        
    authors = [get_text_safely(author_element, 'name') for author_element in authors_elements if get_text_safely(author_element, 'name')]
    
    published_date = get_text_safely(entry_soup, 'published')
    year = published_date.split('-')[0] if published_date else ""

    # Mencari DOI
    doi = ""
    doi_link_tag = entry_soup.find('link', {'title': 'doi'}) # Preferensi format link DOI
    if doi_link_tag and doi_link_tag.get('href'):
        doi = doi_link_tag['href'].replace('http://dx.doi.org/', '').replace('https://doi.org/', '')
    else: # Jika tidak ada di link, coba tag arxiv:doi
        doi_tag = entry_soup.find('arxiv:doi') # Membutuhkan namespace 'arxiv'
        if doi_tag:
            doi = doi_tag.text.strip()

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal_conference_name": "arXiv", # Sesuai permintaan
        "publisher": "arXiv", # Sesuai permintaan
        "year": year,
        "doi": doi,
        "group_name": "default_group" # Sesuai permintaan
    }

# --- Fungsi Scraping ---
def scrape_arxiv_batch(api_query, start_index, batch_size_limit):
    """Mengambil satu batch artikel dari arXiv API."""
    params = {
        "search_query": api_query,
        "start": start_index,
        "max_results": batch_size_limit,
        "sortBy": "submittedDate", # Urutkan berdasarkan tanggal submit
        "sortOrder": "descending" # Dari yang terbaru (untuk mendapatkan artikel cs/recent)
                                   # atau 'ascending' jika ingin dari yang terlama
    }
    
    print(f"Mengambil data dari arXiv: start={start_index}, max_results={batch_size_limit}")
    try:
        response = requests.get(ARXIV_API_URL, params=params)
        response.raise_for_status()  # Akan error jika status code 4xx atau 5xx
    except requests.exceptions.RequestException as e:
        print(f"Error saat mengambil data dari arXiv: {e}")
        return []

    # Menggunakan 'xml' parser dari lxml jika tersedia dan lebih baik, atau parser XML bawaan BS4
    soup = BeautifulSoup(response.content, "xml") 
    
    entries = soup.find_all("entry") # atau "atom:entry"
    if not entries and soup.find("atom:entry"): # Coba dengan prefix 'atom'
        entries = soup.find_all("atom:entry")

    scraped_articles = []
    if not entries:
        print("Tidak ada entri artikel ditemukan dalam respons API untuk batch ini.")
        return []

    for entry_element in entries:
        article_details = parse_arxiv_entry(entry_element)
        scraped_articles.append(article_details)
    
    return scraped_articles

# --- Fungsi Penyimpanan Data ---
def append_articles_to_jsonl(list_of_articles, output_filepath):
    """Menambahkan daftar artikel ke file JSON Lines."""
    with open(output_filepath, 'a', encoding='utf-8') as f:
        for article in list_of_articles:
            f.write(json.dumps(article) + '\n')

# --- Fungsi Utama Scraper ---
def run_incremental_scraper(query_string=DEFAULT_QUERY,
                            target_total_articles=TARGET_ARTICLES_GOAL,
                            batch_fetch_size=BATCH_SIZE_PER_ITERATION):
    """Menjalankan scraper secara bertahap dengan checkpoint."""
    
    current_start_index, total_scraped_count = load_checkpoint()
    
    print(f"--- Memulai Scraper arXiv ---")
    print(f"Target: {target_total_articles} artikel.")
    print(f"Status Saat Ini: {total_scraped_count} artikel telah di-scrape.")
    print(f"Melanjutkan dari start_index: {current_start_index}.")

    if total_scraped_count >= target_total_articles:
        print("Target artikel telah tercapai atau terlampaui. Tidak ada yang perlu di-scrape.")
        return

    while total_scraped_count < target_total_articles:
        print(f"\nIterasi Baru: (Terkumpul: {total_scraped_count}/{target_total_articles})")
        
        # Tentukan berapa banyak yang akan di-scrape di iterasi ini
        remaining_needed = target_total_articles - total_scraped_count
        current_batch_to_fetch = min(batch_fetch_size, remaining_needed)
        
        if current_batch_to_fetch <= 0: # Seharusnya tidak terjadi jika logika di atas benar
            break

        newly_fetched_articles = scrape_arxiv_batch(
            api_query=query_string,
            start_index=current_start_index,
            batch_size_limit=current_batch_to_fetch
        )
        
        if not newly_fetched_articles:
            print("Tidak ada artikel baru yang ditemukan. Kemungkinan sudah mencapai akhir hasil pencarian atau ada masalah API.")
            break # Hentikan jika tidak ada artikel baru
        
        append_articles_to_jsonl(newly_fetched_articles, OUTPUT_FILENAME)
        
        num_actually_scraped_this_batch = len(newly_fetched_articles)
        total_scraped_count += num_actually_scraped_this_batch
        current_start_index += num_actually_scraped_this_batch # Penting: update berdasarkan jumlah *aktual*
        
        save_checkpoint(current_start_index, total_scraped_count)
        print(f"Berhasil mengambil dan menyimpan {num_actually_scraped_this_batch} artikel baru.")
        print(f"Total artikel terkumpul: {total_scraped_count}.")
        
        # Jeda untuk menghindari request berlebihan ke API
        print("Memberi jeda 3 detik sebelum batch berikutnya...")
        time.sleep(3)

    print(f"\n--- Scraper Selesai ---")
    print(f"Total artikel yang berhasil di-scrape: {total_scraped_count}.")
    if total_scraped_count < target_total_articles:
        print(f"Peringatan: Target {target_total_articles} artikel tidak tercapai.")

# --- Eksekusi Skrip ---
if __name__ == "__main__":
    # Contoh menjalankan scraper:
    # Anda bisa mengganti parameter jika diperlukan, misalnya:
    # run_incremental_scraper(query_string="cat:cs.AI", target_total_articles=5000, batch_fetch_size=200)
    run_incremental_scraper()

    # Untuk menguji checkpointing:
    # 1. Jalankan `run_incremental_scraper(target_total_articles=10, batch_fetch_size=5)`
    #    Ini akan mengambil 5, lalu 5 lagi. File checkpoint akan dibuat/diupdate.
    # 2. Jalankan lagi `run_incremental_scraper(target_total_articles=15, batch_fetch_size=5)`
    #    Ini akan melanjutkan dari index 10 dan mengambil 5 artikel berikutnya.
    # 3. Untuk reset, hapus file `arxiv_cs_articles.jsonl` dan `scraper_checkpoint.json`.