import unittest
from preprocess import preprocess_text

class TestPreprocessing(unittest.TestCase):

    def test_preprocess_text(self):
        raw_text = "This is a Sample article! With stopwords."
        processed_text = preprocess_text(raw_text)
        
        # Memastikan teks telah diproses dengan benar
        self.assertIsInstance(processed_text, str, "Processed text should be a string")
        self.assertNotIn("sample", processed_text, "Stopword 'sample' should be removed")
        self.assertNotIn("stopwords", processed_text, "Stopword 'stopwords' should be removed")
        self.assertTrue(processed_text.islower(), "Processed text should be in lowercase")

if __name__ == '__main__':
    unittest.main()
