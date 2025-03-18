import pandas as pd
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from .text_cleaning import clean_text_advanced
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Pastikan bahwa kita sudah mengunduh beberapa resources NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Muat model spaCy untuk lemmatization (spaCy lebih cepat dan akurat untuk lemmatization)
nlp = spacy.load('en_core_web_sm')

# Fungsi untuk preprocessing teks
def preprocess_text(text, stopwords):
    """Preprocess a single piece of text."""
    # Gunakan fungsi clean_text_advanced untuk pembersihan teks lanjutan
    cleaned_text = clean_text_advanced(text, stopwords)
    return cleaned_text

# Fungsi untuk memproses seluruh dataset
def preprocess_data(input_file_path, output_file_path, stopwords):
    """Load, preprocess, and save the dataset."""
    # Memastikan file ada dan dapat dibaca
    if not os.path.exists(input_file_path):
        print(f"Error: File {input_file_path} not found!")
        return
    
    # Membaca data dari CSV
    df = pd.read_csv(input_file_path)

    # Memastikan kolom 'Title' ada dalam data
    if 'Title' not in df.columns:
        print("Error: 'Title' column not found in the dataset.")
        return

    # Terapkan preprocessing pada setiap judul artikel
    df['Processed_Text'] = df['Title'].apply(lambda x: preprocess_text(x, stopwords))

    # Membuat folder untuk menyimpan hasil yang telah diproses jika belum ada
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Simpan data yang telah diproses ke dalam file CSV
    df.to_csv(output_file_path, index=False)
    print(f"Preprocessing completed and saved to {output_file_path}")

if __name__ == "__main__":
    # Path menuju file yang disimpan oleh scraper
    rawdata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'rawdata', 'article_titles.csv')
    
    # Mendapatkan direktori skrip saat ini untuk membuat path output relatif
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_path = os.path.join(base_dir, '..', '..', 'data', 'processed_data', 'processed_titles.csv')  # Path output
    
    # Daftar stopwords untuk preprocessing
    stopwords_set = set(stopwords.words('english'))

    # Panggil fungsi untuk memproses data
    preprocess_data(rawdata_path, processed_data_path, stopwords_set)
