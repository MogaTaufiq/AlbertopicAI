import unittest
from classification import classify_articles
import pandas as pd
from sklearn.model_selection import train_test_split

class TestClassification(unittest.TestCase):

    def test_classify_articles(self):
        # Data contoh untuk klasifikasi
        data = {
            'Processed_Text': [
                'Machine learning and AI research', 
                'New trends in technology',
                'Medical advances and innovations'
            ],
            'Category': ['Technology', 'Technology', 'Health']
        }
        
        df = pd.DataFrame(data)
        
        # Memastikan bahwa klasifikasi dapat dijalankan tanpa error
        try:
            classify_articles(df)
            success = True
        except Exception as e:
            print(f"Error during classification: {e}")
            success = False
        
        self.assertTrue(success, "Classification should run without errors")

if __name__ == '__main__':
    unittest.main()
