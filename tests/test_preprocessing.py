import unittest
import pandas as pd
from nltk.corpus import stopwords
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import setelah penambahan path
from src.preprocessing.text_cleaning import (
    clean_text_basic, 
    remove_extra_whitespace, 
    remove_stopwords, 
    remove_punctuation, 
    clean_text_advanced
)
from src.preprocessing.preprocess import preprocess_text, preprocess_data

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Siapkan data uji
        self.sample_text = "This is a sample text, with some unwanted characters like # and numbers 123!"
        self.stopwords_set = set(stopwords.words('english'))

    def test_clean_text_basic(self):
        # Uji pembersihan teks dasar
        cleaned_text = clean_text_basic(self.sample_text)
        self.assertFalse(any(char.isdigit() for char in cleaned_text))
        self.assertTrue(cleaned_text.islower())

    def test_remove_extra_whitespace(self):
        # Uji penghapusan whitespace berlebih
        text_with_extra_space = "  text   with    extra   spaces  "
        cleaned_text = remove_extra_whitespace(text_with_extra_space)
        self.assertEqual(cleaned_text, "text with extra spaces")

    def test_remove_stopwords(self):
        # Uji penghapusan stopwords
        stopwords_test = {"is", "a", "with", "some", "and"}
        cleaned_text = remove_stopwords(self.sample_text, stopwords_test)
        for word in stopwords_test:
            self.assertNotIn(word, cleaned_text)

    def test_remove_punctuation(self):
        # Uji penghapusan tanda baca
        cleaned_text = remove_punctuation(self.sample_text)
        self.assertFalse(any(char in '!@#$%^&*()' for char in cleaned_text))

    def test_clean_text_advanced(self):
        # Uji pembersihan teks lanjutan
        cleaned_text = clean_text_advanced(self.sample_text, self.stopwords_set)
        self.assertTrue(len(cleaned_text) < len(self.sample_text))
        self.assertFalse(any(char.isdigit() for char in cleaned_text))

    def test_preprocess_text(self):
        # Uji fungsi preprocessing teks
        processed_text = preprocess_text(self.sample_text, self.stopwords_set)
        self.assertIsInstance(processed_text, str)

    def test_preprocess_data(self):
        # Buat data uji
        test_df = pd.DataFrame({'Title': [self.sample_text, "Another test text"]})
        test_input_path = 'test_input.csv'
        test_output_path = 'test_output.csv'
        
        # Simpan data uji
        test_df.to_csv(test_input_path, index=False)

        # Proses data
        preprocess_data(test_input_path, test_output_path, self.stopwords_set)

        # Periksa hasil
        processed_df = pd.read_csv(test_output_path)
        self.assertTrue('Processed_Text' in processed_df.columns)

        # Bersihkan file uji
        os.remove(test_input_path)
        os.remove(test_output_path)

if __name__ == '__main__':
    unittest.main()