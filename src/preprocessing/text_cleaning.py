import re
import string

def clean_text_basic(text):
    """
    Clean text by removing special characters, numbers, and punctuation.
    
    Args:
        text (str): The raw text to clean.
    
    Returns:
        str: Cleaned text without special characters or numbers.
    """
    # Menghapus angka dan karakter khusus selain huruf dan spasi
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hanya mempertahankan huruf dan spasi
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    return text

def remove_extra_whitespace(text):
    """
    Remove extra whitespaces from the text.
    
    Args:
        text (str): The text to remove extra whitespaces.
    
    Returns:
        str: Cleaned text with single spaces between words.
    """
    # Menghapus whitespace yang lebih dari satu dan menghapus whitespace di awal/akhir
    return re.sub(r'\s+', ' ', text).strip()

def remove_stopwords(text, stopwords):
    """
    Remove stopwords from the text.
    
    Args:
        text (str): The text to remove stopwords from.
        stopwords (set): A set of stopwords to be removed.
    
    Returns:
        str: Text with stopwords removed.
    """
    # Membagi teks menjadi kata-kata dan menghapus stopwords
    words = text.split()
    return ' '.join([word for word in words if word not in stopwords])

def remove_punctuation(text):
    """
    Remove punctuation from the text.
    
    Args:
        text (str): The text to remove punctuation from.
    
    Returns:
        str: Text without punctuation.
    """
    # Menghapus tanda baca dari teks
    return text.translate(str.maketrans('', '', string.punctuation))

def clean_text_advanced(text, stopwords=None):
    """
    Advanced text cleaning including removal of special characters, extra whitespaces, stopwords, and punctuation.
    
    Args:
        text (str): The raw text to clean.
        stopwords (set, optional): A set of stopwords to be removed. Defaults to None.
    
    Returns:
        str: Fully cleaned text.
    """
    # Bersihkan teks dengan beberapa tahapan
    text = clean_text_basic(text)  # Bersihkan karakter khusus dan angka
    text = remove_extra_whitespace(text)  # Hapus whitespace yang berlebihan
    if stopwords:
        text = remove_stopwords(text, stopwords)  # Hapus stopwords jika ada
    text = remove_punctuation(text)  # Hapus tanda baca
    return text

if __name__ == "__main__":
    # Contoh penggunaan
    sample_text = "This is a sample text, with some unwanted characters like # and numbers 123!"
    stopwords = {"is", "a", "with", "some", "and"}
    
    print("Original Text:")
    print(sample_text)
    
    cleaned_text = clean_text_advanced(sample_text, stopwords=stopwords)
    
    print("\nCleaned Text:")
    print(cleaned_text)
