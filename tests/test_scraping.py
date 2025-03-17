import unittest
import os
import sys
import pandas as pd
import shutil

# Tambahkan path proyek ke sistem
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraping.scraper import save_titles_to_csv

class TestScraping(unittest.TestCase):
    def setUp(self):
        # Siapkan direktori root proyek
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Definisikan path yang akurat untuk file output
        self.rawdata_dir = os.path.join(self.project_root, 'data', 'rawdata')
        self.output_file = os.path.join(self.rawdata_dir, 'test_titles.csv')
        
        # Pastikan direktori rawdata ada
        os.makedirs(self.rawdata_dir, exist_ok=True)
        
        # Data uji
        self.test_titles = ['Test Title 1', 'Test Title 2']

    def test_save_titles_to_csv(self):
        # Panggil fungsi save_titles_to_csv dengan judul uji
        saved_file_path = save_titles_to_csv(self.test_titles, 'test_titles.csv')
        
        # Periksa apakah file berhasil dibuat
        self.assertTrue(os.path.exists(saved_file_path))
        
        # Baca file CSV dan validasi
        df = pd.read_csv(saved_file_path)
        
        # Debug: Cetak isi DataFrame untuk investigasi
        print("DataFrame contents:")
        print(df)
        print(f"DataFrame length: {len(df)}")
        print(f"Test titles length: {len(self.test_titles)}")
        
        # Validasi jumlah baris
        self.assertEqual(len(df), len(self.test_titles), 
                         "Jumlah baris dalam CSV tidak sesuai dengan jumlah judul")
        
        # Validasi kolom
        self.assertTrue('Title' in df.columns, "Kolom 'Title' tidak ditemukan")
        
        # Validasi isi
        pd.testing.assert_frame_equal(
            df, 
            pd.DataFrame(self.test_titles, columns=['Title']),
            check_dtype=False
        )

    def tearDown(self):
        # Hapus file test jika ada
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

if __name__ == '__main__':
    unittest.main()