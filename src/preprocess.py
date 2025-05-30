import os
import re
import string
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

# NLP_SPACY = None
# try:
#     NLP_SPACY = spacy.load('en_core_web_sm')
# except OSError:
#     print("Model spaCy 'en_core_web_sm' belum terinstal. Silakan jalankan:\npython -m spacy download en_core_web_sm")

def clean_text_basic(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def remove_stopwords(text, custom_stopwords):
    words = text.split()
    return ' '.join([word for word in words if word not in custom_stopwords])

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def clean_text_advanced(text, custom_stopwords=None):
    text = clean_text_basic(text)
    text = remove_extra_whitespace(text)
    if custom_stopwords:
        text = remove_stopwords(text, custom_stopwords)
    return text

def preprocess_text_pipeline(text_input, stopword_list):
    cleaned_text = clean_text_advanced(str(text_input), custom_stopwords=stopword_list)
    return cleaned_text

def load_data_from_jsonl(jsonl_filepath):
    data_list = []
    if not os.path.exists(jsonl_filepath):
        print(f"Error: File input {jsonl_filepath} tidak ditemukan!")
        return None
    with open(jsonl_filepath, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON di baris {line_number + 1} file {jsonl_filepath}: {e}")
                return None
    if not data_list:
        print(f"Tidak ada data yang berhasil dimuat dari {jsonl_filepath}")
        return None
    return pd.DataFrame(data_list)

def run_preprocess_pipeline(input_jsonl_path, output_csv_path, language_stopwords):
    print(f"Memulai pra-pemrosesan untuk file: {input_jsonl_path}")
    
    df = load_data_from_jsonl(input_jsonl_path)
    if df is None or df.empty:
        print("Gagal memuat data. Proses pra-pemrosesan dihentikan.")
        return

    if 'title' not in df.columns:
        print("Error: Kolom 'title' tidak ditemukan dalam data JSONL.")
        return

    print("Melakukan pra-pemrosesan pada kolom 'title'...")
    df['Processed_Title'] = df['title'].apply(lambda x: preprocess_text_pipeline(x, language_stopwords))

    if 'abstract' in df.columns:
        print("Melakukan pra-pemrosesan pada kolom 'abstract'...")
        df['Processed_Abstract'] = df['abstract'].apply(lambda x: preprocess_text_pipeline(x, language_stopwords))
    else:
        print("Peringatan: Kolom 'abstract' tidak ditemukan. Hanya 'title' yang akan diproses.")
        df['Processed_Abstract'] = "" 

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    try:
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"Pra-pemrosesan selesai. Data yang diproses disimpan di: {output_csv_path}")
        print(f"Jumlah baris yang diproses: {len(df)}")
    except Exception as e:
        print(f"Error saat menyimpan file CSV: {e}")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_data_jsonl_path = os.path.join(project_root, 'data', 'rawdata', 'arxiv_cs_articles.jsonl')
    processed_data_csv_path = os.path.join(project_root, 'data', 'processed_data', 'processed_articles.csv')
    english_stopwords = set(stopwords.words('english'))
    
    print("Menjalankan preprocess.py sebagai skrip mandiri...")
    run_preprocess_pipeline(raw_data_jsonl_path, processed_data_csv_path, english_stopwords)