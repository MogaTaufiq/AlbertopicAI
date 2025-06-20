import os
import requests
from bs4 import BeautifulSoup # Pastikan BeautifulSoup diinstal
import json
import time
import calendar # Untuk mendapatkan jumlah hari dalam sebulan
from datetime import datetime, timedelta # Untuk manipulasi tanggal

# --- Konfigurasi ---
ARXIV_API_URL = "http://export.arxiv.org/api/query?"
DEFAULT_BASE_QUERY = "cat:cs.*"  # Kategori Computer Science, bisa diubah
TARGET_ARTICLES_GOAL = 10000     # Target total artikel yang ingin dikumpulkan
BATCH_SIZE_PER_ITERATION = 200   # Jumlah artikel per panggilan API (lebih kecil lebih aman)
OUTPUT_FILENAME = "data/rawdata/arxiv_cs_articles_by_date.jsonl" # Nama file output baru
CHECKPOINT_FILENAME = "data/rawdata/scraper_checkpoint_by_date.json" # File checkpoint baru

# Konfigurasi Rentang Tanggal Default untuk Scraping
# Skrip akan mengambil data mundur dari END_YEAR/END_MONTH hingga START_YEAR/START_MONTH
CURRENT_DATETIME = datetime.now()
DEFAULT_END_YEAR = CURRENT_DATETIME.year
DEFAULT_END_MONTH = CURRENT_DATETIME.month
# Defaultnya, ambil data hingga 3 tahun ke belakang dari bulan saat ini
three_years_ago = CURRENT_DATETIME - timedelta(days=3*365)
DEFAULT_START_YEAR = three_years_ago.year
DEFAULT_START_MONTH = three_years_ago.month

# Pastikan direktori data/rawdata ada
os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

# --- Fungsi Checkpoint ---
def load_checkpoint():
    """
    Memuat checkpoint terakhir.
    Mengembalikan: current_year, current_month, current_month_start_index, total_articles_scraped_overall.
    Jika checkpoint tidak ada atau rusak, mengembalikan nilai default untuk memulai dari awal.
    """
    if os.path.exists(CHECKPOINT_FILENAME):
        with open(CHECKPOINT_FILENAME, 'r') as f:
            try:
                cp = json.load(f)
                # Pastikan semua kunci ada, jika tidak, anggap checkpoint tidak valid
                if all(k in cp for k in ['current_processing_year', 'current_processing_month', 
                                         'current_month_start_index', 'total_articles_scraped_overall']):
                    return (cp['current_processing_year'],
                            cp['current_processing_month'],
                            cp['current_month_start_index'],
                            cp['total_articles_scraped_overall'])
                else:
                    print(f"Peringatan: File checkpoint '{CHECKPOINT_FILENAME}' tidak lengkap. Memulai dari awal.")
            except json.JSONDecodeError:
                print(f"Peringatan: File checkpoint '{CHECKPOINT_FILENAME}' rusak. Memulai dari awal.")
    return None, None, 0, 0 # Default jika tidak ada checkpoint atau rusak

def save_checkpoint(year_to_process, month_to_process, month_start_idx, total_scraped_overall):
    """Menyimpan checkpoint saat ini ke file JSON."""
    checkpoint_data = {
        'current_processing_year': year_to_process,
        'current_processing_month': month_to_process,
        'current_month_start_index': month_start_idx,
        'total_articles_scraped_overall': total_scraped_overall
    }
    with open(CHECKPOINT_FILENAME, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    # Komentar di bawah bisa diaktifkan untuk debugging lebih detail
    # print(f"Checkpoint disimpan: Y={year_to_process}, M={month_to_process}, StartIdxBulan={month_start_idx}, TotalKeseluruhan={total_scraped_overall}")

# --- Fungsi Helper Tanggal ---
def get_arxiv_date_query_for_month(year, month, base_query=DEFAULT_BASE_QUERY):
    """Membuat string filter tanggal untuk kueri arXiv untuk satu bulan penuh."""
    # Mendapatkan hari terakhir dalam bulan dan tahun yang diberikan
    _, last_day_of_month = calendar.monthrange(year, month)
    
    # Format tanggal sesuai yang diharapkan arXiv API: YYYYMMDD
    start_date_str = f"{year:04d}{month:02d}01" # Tanggal 1
    end_date_str = f"{year:04d}{month:02d}{last_day_of_month:02d}" # Hari terakhir bulan itu
    
    date_filter = f"submittedDate:[{start_date_str} TO {end_date_str}]"
    
    # Gabungkan dengan base_query jika ada
    if base_query and base_query.strip():
        return f"({base_query}) AND {date_filter}"
    else:
        return date_filter

def get_previous_month_year(year, month):
    """Mendapatkan tahun dan bulan sebelumnya dari tahun dan bulan yang diberikan."""
    if month == 1: # Jika bulan Januari, bulan sebelumnya adalah Desember tahun lalu
        return year - 1, 12
    else: # Jika bukan Januari, cukup kurangi bulannya
        return year, month - 1

# --- Fungsi Parsing Data dari Entri XML arXiv ---
def parse_arxiv_entry(entry_soup):
    """Mengekstrak data dari satu elemen <entry> XML menjadi format JSON yang diinginkan."""
    
    # Fungsi pembantu untuk mendapatkan teks dengan aman dari tag, mencoba dengan prefix 'atom:' jika perlu
    def get_text_safely(element, tag_name):
        found_tag = element.find(tag_name)
        if not found_tag: # Jika tidak ditemukan, coba dengan prefix 'atom:'
             found_tag = element.find(f"atom:{tag_name}")
        return found_tag.text.strip().replace('\n', ' ') if found_tag else ""

    title = get_text_safely(entry_soup, 'title')
    abstract = get_text_safely(entry_soup, 'summary') # Di arXiv, 'summary' adalah abstrak

    authors_elements = entry_soup.find_all('author')
    if not authors_elements and entry_soup.find('atom:author'): # Coba dengan prefix jika find_all gagal
        authors_elements = entry_soup.find_all('atom:author')
        
    authors = [get_text_safely(author_element, 'name') for author_element in authors_elements if get_text_safely(author_element, 'name')]
    
    published_date = get_text_safely(entry_soup, 'published')
    year_str = published_date.split('-')[0] if published_date else "" # Ambil tahun dari tanggal publikasi

    # Mencari DOI
    doi = ""
    # Preferensi format link DOI
    doi_link_tag = entry_soup.find('link', {'title': 'doi'}) 
    if doi_link_tag and doi_link_tag.get('href'):
        doi = doi_link_tag['href'].replace('http://dx.doi.org/', '').replace('https://doi.org/', '')
    else: # Jika tidak ada di link, coba tag arxiv:doi (membutuhkan namespace 'arxiv')
        doi_tag = entry_soup.find('arxiv:doi') 
        if doi_tag:
            doi = doi_tag.text.strip()

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors, # Disimpan sebagai list of strings
        "journal_conference_name": "arXiv", # Sesuai permintaan
        "publisher": "arXiv", # Sesuai permintaan
        "year": year_str,
        "doi": doi,
        "group_name": "default_group" # Sesuai permintaan
    }

# --- Fungsi Scraping per Batch ---
def scrape_arxiv_batch(api_query_with_date_filter, start_index_in_month_batch, batch_size_limit):
    """Mengambil satu batch artikel dari arXiv API untuk kueri dan rentang tanggal tertentu."""
    params = {
        "search_query": api_query_with_date_filter,
        "start": start_index_in_month_batch,
        "max_results": batch_size_limit,
        "sortBy": "submittedDate", # Urutkan berdasarkan tanggal submit (dalam rentang tanggal yang sudah difilter)
        "sortOrder": "descending"  # Dari yang terbaru dalam rentang tersebut
    }
    
    # print(f"  Mengambil data: start_idx_bulan={start_index_in_month_batch}, query='{api_query_with_date_filter}'") # Untuk debugging
    try:
        response = requests.get(ARXIV_API_URL, params=params)
        response.raise_for_status()  # Akan error jika status code 4xx atau 5xx
    except requests.exceptions.RequestException as e:
        print(f"  Error saat request API: {e}")
        return [] # Kembalikan list kosong jika ada error

    # Menggunakan 'xml' parser dari lxml jika tersedia (lebih baik), atau parser XML bawaan BS4
    soup = BeautifulSoup(response.content, "xml") 
    
    entries = soup.find_all("entry") # atau "atom:entry"
    if not entries and soup.find("atom:entry"): # Coba dengan prefix 'atom' jika tidak ditemukan
        entries = soup.find_all("atom:entry")

    scraped_articles_in_batch = [parse_arxiv_entry(entry_element) for entry_element in entries]
    return scraped_articles_in_batch

# --- Fungsi Penyimpanan Data ---
def append_articles_to_jsonl(list_of_articles_to_append, output_filepath):
    """Menambahkan daftar artikel (list of dicts) ke file JSON Lines."""
    with open(output_filepath, 'a', encoding='utf-8') as f:
        for article_dict in list_of_articles_to_append:
            f.write(json.dumps(article_dict) + '\n')

# --- Fungsi Utama Scraper dengan Iterasi Tanggal ---
def run_scraper_by_date_range(
    base_query_str=DEFAULT_BASE_QUERY,
    target_total_articles_to_scrape=TARGET_ARTICLES_GOAL,
    batch_size_to_fetch=BATCH_SIZE_PER_ITERATION,
    # Batas tanggal paling lama untuk diambil datanya
    limit_stop_year=DEFAULT_START_YEAR, 
    limit_stop_month=DEFAULT_START_MONTH,
    # Tanggal mulai iterasi (mundur dari sini)
    iteration_start_year=DEFAULT_END_YEAR,
    iteration_start_month=DEFAULT_END_MONTH
):
    # Memuat status dari checkpoint
    cp_year, cp_month, cp_month_start_idx, total_scraped_count_overall = load_checkpoint()

    # Menentukan titik awal iterasi berdasarkan checkpoint atau default
    if cp_year is not None and cp_month is not None: # Jika ada checkpoint valid
        current_iter_year = cp_year
        current_iter_month = cp_month
        current_month_start_index_val = cp_month_start_idx
        print(f"--- Melanjutkan Scraper dari Checkpoint ---")
        print(f"Tahun: {current_iter_year}, Bulan: {current_iter_month}, Start Index Bulan Ini: {current_month_start_index_val}")
    else: # Jika tidak ada checkpoint, mulai dari awal (sesuai parameter fungsi)
        current_iter_year = iteration_start_year
        current_iter_month = iteration_start_month
        current_month_start_index_val = 0
        total_scraped_count_overall = 0 # Mulai dari nol jika tidak ada checkpoint
        print(f"--- Memulai Scraper Baru ---")
        print(f"Mulai dari Tahun: {current_iter_year}, Bulan: {current_iter_month}")
    
    print(f"Target: {target_total_articles_to_scrape} artikel. Saat ini terkumpul: {total_scraped_count_overall}.")
    print(f"Akan berhenti jika mencapai atau melewati tanggal {limit_stop_month:02d}-{limit_stop_year} atau target artikel terpenuhi.")

    # Loop utama: terus berjalan selama target artikel belum tercapai
    while total_scraped_count_overall < target_total_articles_to_scrape:
        # Cek apakah sudah melewati batas tanggal stop yang ditentukan
        if current_iter_year < limit_stop_year or \
           (current_iter_year == limit_stop_year and current_iter_month < limit_stop_month):
            print(f"Telah mencapai atau melewati batas tanggal stop ({limit_stop_month:02d}-{limit_stop_year}). Menghentikan scraper.")
            break

        print(f"\nMemproses bulan {current_iter_month:02d}-{current_iter_year} (Total terkumpul: {total_scraped_count_overall}/{target_total_articles_to_scrape})")
        
        # Buat kueri API untuk bulan dan tahun saat ini
        api_query_for_current_month = get_arxiv_date_query_for_month(current_iter_year, current_iter_month, base_query_str)
        
        articles_found_in_current_month_this_session = False # Flag untuk melacak apakah ada artikel di bulan ini

        # Loop paginasi di dalam bulan ini
        while total_scraped_count_overall < target_total_articles_to_scrape:
            remaining_needed_to_reach_goal = target_total_articles_to_scrape - total_scraped_count_overall
            current_batch_size_for_api = min(batch_size_to_fetch, remaining_needed_to_reach_goal)
            
            if current_batch_size_for_api <= 0: # Jika target sudah tercapai
                break

            print(f"  Batch untuk {current_iter_month:02d}-{current_iter_year}: start_idx={current_month_start_index_val}, minta_size={current_batch_size_for_api}")
            
            newly_fetched_articles_list = scrape_arxiv_batch(
                api_query_for_current_month,
                current_month_start_index_val,
                current_batch_size_for_api
            )

            if not newly_fetched_articles_list: # Jika tidak ada artikel baru di batch ini
                print(f"  Tidak ada artikel baru ditemukan untuk {current_iter_month:02d}-{current_iter_year} pada start_idx={current_month_start_index_val}.")
                # Jika ini adalah batch pertama bulan ini dan kosong, berarti bulan ini memang tidak ada artikel (atau sudah habis)
                if not articles_found_in_current_month_this_session and current_month_start_index_val == 0:
                    print(f"  Tidak ada artikel sama sekali di bulan {current_iter_month:02d}-{current_iter_year} (atau sudah semua diambil).")
                break # Keluar dari loop paginasi bulan ini, pindah ke bulan sebelumnya
            
            articles_found_in_current_month_this_session = True
            append_articles_to_jsonl(newly_fetched_articles_list, OUTPUT_FILENAME)
            
            num_actually_scraped_this_batch = len(newly_fetched_articles_list)
            total_scraped_count_overall += num_actually_scraped_this_batch
            current_month_start_index_val += num_actually_scraped_this_batch # Maju untuk batch berikutnya di bulan ini
            
            # Simpan checkpoint setelah setiap batch berhasil
            save_checkpoint(current_iter_year, current_iter_month, current_month_start_index_val, total_scraped_count_overall)
            print(f"  Berhasil menyimpan {num_actually_scraped_this_batch} artikel. Total keseluruhan: {total_scraped_count_overall}.")
            
            # Jika API mengembalikan lebih sedikit dari yang diminta, anggap akhir dari hasil untuk bulan ini
            if num_actually_scraped_this_batch < current_batch_size_for_api:
                print(f"  Batch terakhir untuk bulan {current_iter_month:02d}-{current_iter_year} (diterima {num_actually_scraped_this_batch} dari {current_batch_size_for_api} diminta). Pindah ke bulan sebelumnya.")
                break # Keluar dari loop paginasi bulan ini
            
            print("  Jeda 3 detik sebelum batch berikutnya dalam bulan yang sama...")
            time.sleep(3) # Jeda untuk menghormati API server
        
        # Setelah selesai dengan satu bulan (baik karena habis artikelnya atau target tercapai),
        # pindah ke bulan sebelumnya untuk iterasi berikutnya.
        current_iter_year, current_iter_month = get_previous_month_year(current_iter_year, current_iter_month)
        current_month_start_index_val = 0 # Reset start index untuk bulan baru
        
        # Simpan checkpoint sebelum memulai bulan baru (atau jika target sudah tercapai dan loop luar akan berhenti)
        save_checkpoint(current_iter_year, current_iter_month, current_month_start_index_val, total_scraped_count_overall)

    # Selesai loop utama
    print(f"\n--- Scraper Selesai ---")
    print(f"Total artikel yang berhasil di-scrape: {total_scraped_count_overall}.")
    if total_scraped_count_overall < target_total_articles_to_scrape:
        print(f"Peringatan: Target {target_total_articles_to_scrape} artikel tidak tercapai.")
    else:
        print(f"Target {target_total_articles_to_scrape} artikel telah tercapai atau terlampaui.")

# --- Bagian Eksekusi Skrip ---
if __name__ == "__main__":
    print("Memulai skrip scraper arXiv dengan iterasi tanggal...")
    run_scraper_by_date_range(
         base_query_str="cat:cs.AI OR cat:cs.LG",
         target_total_articles_to_scrape=5000,
         batch_size_to_fetch=100,
         limit_stop_year=2023,
         limit_stop_month=1,
         iteration_start_year=2024,
         iteration_start_month=5
    )
    print("Eksekusi scraper selesai.")

