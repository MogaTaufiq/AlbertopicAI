import pandas as pd
import os
import logging

# Inisialisasi logging untuk mencatat proses eksekusi
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if os.path.exists(file_path):
        logging.info(f"Loading data from {file_path}")
        return pd.read_csv(file_path)
    else:
        logging.error(f"File {file_path} not found!")
        return None

def save_data(df, file_path):
    """
    Save the DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to be saved.
        file_path (str): Path to save the CSV file.
    """
    df.to_csv(file_path, index=False)
    logging.info(f"Data saved to {file_path}")

def clean_text(text):
    """
    Clean text by removing special characters, unnecessary spaces, etc.
    
    Args:
        text (str): The text to clean.
    
    Returns:
        str: Cleaned text.
    """
    # Hapus karakter non-alfabet dan ubah ke huruf kecil
    cleaned_text = ''.join(e for e in text if e.isalnum() or e.isspace()).lower()
    return cleaned_text

def create_dir(directory):
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory (str): Directory path to be created.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Directory {directory} created.")
    else:
        logging.info(f"Directory {directory} already exists.")

def print_top_words(model, vectorizer, n_top_words=10):
    """
    Print the top N words from a topic model.
    
    Args:
        model: The trained topic model (e.g., BERTopic or LDA).
        vectorizer: The vectorizer used for transforming the text data (e.g., TfidfVectorizer).
        n_top_words (int): Number of top words to display from each topic.
    """
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:")
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(" ".join(top_words))

def get_category_from_filename(filename):
    """
    Extract category from filename, assuming filenames are structured as 'category_filename.csv'.
    
    Args:
        filename (str): Name of the file (e.g., 'technology_article.csv').
    
    Returns:
        str: Extracted category.
    """
    return filename.split('_')[0]

def log_model_results(model_name, accuracy, additional_info=""):
    """
    Log the results of the model (e.g., accuracy, performance metrics).
    
    Args:
        model_name (str): Name of the model (e.g., 'Naive Bayes').
        accuracy (float): Accuracy score of the model.
        additional_info (str): Optional additional information to log.
    """
    logging.info(f"{model_name} Model Results:")
    logging.info(f"Accuracy: {accuracy * 100:.2f}%")
    if additional_info:
        logging.info(f"Additional Info: {additional_info}")

