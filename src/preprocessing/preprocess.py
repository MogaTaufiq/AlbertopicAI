import os
import re
import string
import sys
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Setup environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')

def clean_text_basic(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def remove_stopwords(text, stopwords):
    words = text.split()
    return ' '.join([word for word in words if word not in stopwords])

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def clean_text_advanced(text, stopwords=None):
    text = clean_text_basic(text)
    text = remove_extra_whitespace(text)
    if stopwords:
        text = remove_stopwords(text, stopwords)
    text = remove_punctuation(text)
    return text

def preprocess_text(text, stopwords):
    cleaned_text = clean_text_advanced(text, stopwords)
    return cleaned_text

def preprocess_data(input_file_path, output_file_path, stopwords):
    if not os.path.exists(input_file_path):
        print(f"Error: File {input_file_path} not found!")
        return
    
    df = pd.read_csv(input_file_path)

    # Pastikan kolom 'Title' ada
    if 'Title' not in df.columns:
        if 'title' in df.columns:
            df.rename(columns={'title': 'Title'}, inplace=True)
        else:
            print("Error: Neither 'Title' nor 'title' column found.")
            return

    # Periksa keberadaan 'abstract'
    if 'abstract' not in df.columns and 'Abstract' not in df.columns:
        print("Warning: Abstract not found. Only title will be processed.")
    elif 'Abstract' in df.columns and 'abstract' not in df.columns:
        df.rename(columns={'Abstract': 'abstract'}, inplace=True)

    # Preprocessing
    df['Processed_Title'] = df['Title'].apply(lambda x: preprocess_text(str(x), stopwords))

    if 'abstract' in df.columns:
        df['Processed_Abstract'] = df['abstract'].apply(lambda x: preprocess_text(str(x), stopwords))

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_csv(output_file_path, index=False)
    print(f"Preprocessing completed and saved to {output_file_path}")

if __name__ == "__main__":
    rawdata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'rawdata', 'arxiv_cs_articles.csv')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_path = os.path.join(base_dir, '..', '..', 'data', 'processed_data', 'processed_titles.csv')
    stopwords_set = set(stopwords.words('english'))
    preprocess_data(rawdata_path, processed_data_path, stopwords_set)