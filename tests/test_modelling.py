import unittest
import os
import pandas as pd
import pickle
from src.modelling.modelling import (
    load_data, 
    perform_bertopic_modeling, 
    save_model, 
    save_topic_results
)

class TestModelling(unittest.TestCase):
    def setUp(self):
        # Buat data uji
        self.test_data = pd.DataFrame({
            'Processed_Text': [
                'machine learning', 
                'data science', 
                'artificial intelligence', 
                'deep learning', 
                'natural language processing'
            ]
        })
        self.test_input_path = 'test_input.csv'
        self.test_data.to_csv(self.test_input_path, index=False)

    def test_load_data(self):
        # Uji fungsi load_data
        df = load_data(self.test_input_path)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 5)

    def test_perform_bertopic_modeling(self):
        # Uji fungsi topic modeling
        topic_model, topics = perform_bertopic_modeling(self.test_data)
        
        # Periksa keluaran
        self.assertIsNotNone(topic_model)
        self.assertIsNotNone(topics)
        self.assertEqual(len(topics), len(self.test_data))

    def test_save_and_load_model(self):
        # Uji penyimpanan model
        topic_model, _ = perform_bertopic_modeling(self.test_data)
        model_path = os.path.join('data', 'final', 'test_model.pkl')
        
        # Simpan model
        with open(model_path, 'wb') as f:
            pickle.dump(topic_model, f)
        
        # Periksa apakah file model berhasil disimpan
        self.assertTrue(os.path.exists(model_path))

        # Muat kembali model
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Periksa keberhasilan pembacaan model
        self.assertIsNotNone(loaded_model)

        # Hapus file model uji
        os.remove(model_path)

    def test_save_topic_results(self):
        # Uji penyimpanan hasil topic modeling
        _, topics = perform_bertopic_modeling(self.test_data)
        output_path = os.path.join('data', 'final', 'test_topic_results.csv')
        
        # Simpan hasil topic
        save_topic_results(self.test_data, topics, 'test_topic_results.csv')
        
        # Periksa keberadaan file
        self.assertTrue(os.path.exists(output_path))
        
        # Periksa isi file
        df = pd.read_csv(output_path)
        self.assertTrue('Topic' in df.columns)
        
        # Hapus file uji
        os.remove(output_path)

    def tearDown(self):
        # Hapus file input uji
        if os.path.exists(self.test_input_path):
            os.remove(self.test_input_path)

if __name__ == '__main__':
    unittest.main()