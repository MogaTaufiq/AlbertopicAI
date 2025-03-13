import unittest
import os
from scraper import scrape_article_titles, save_titles_to_csv

class TestScraper(unittest.TestCase):

    def test_scrape_article_titles(self):
        url = 'https://www.ijaseit.com'  # URL contoh, pastikan disesuaikan dengan website yang relevan
        titles = scrape_article_titles(url)
        
        # Memastikan scraping menghasilkan judul-judul artikel
        self.assertGreater(len(titles), 0, "Scraped titles should not be empty")
        self.assertIsInstance(titles, list, "Scraped titles should be a list")

    def test_save_titles_to_csv(self):
        titles = ["Article 1", "Article 2", "Article 3"]
        save_titles_to_csv(titles, 'test_titles.csv')
        
        # Memastikan file CSV telah disimpan
        self.assertTrue(os.path.exists('test_titles.csv'), "CSV file should be saved")
        
        # Hapus file setelah pengujian
        os.remove('test_titles.csv')

if __name__ == '__main__':
    unittest.main()
