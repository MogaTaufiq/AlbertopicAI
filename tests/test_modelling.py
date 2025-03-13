import unittest
from topic_modeling import topic_modeling
from bertopic import BERTopic

class TestTopicModeling(unittest.TestCase):

    def test_topic_modeling(self):
        # Data contoh untuk pemodelan topik
        texts = ["This is a science article.", "A study in technology.", "Research in machine learning."]
        
        model = BERTopic(language="english")
        topics, _ = topic_modeling(texts)
        
        # Memastikan bahwa hasil topik bukan kosong
        self.assertGreater(len(topics), 0, "Topic modeling should produce at least one topic")
        self.assertIsInstance(topics, list, "Topics should be a list")

if __name__ == '__main__':
    unittest.main()
