from bertopic import BERTopic
import pandas as pd

def topic_modeling(text_data):
    """Perform topic modeling using BERTopic."""
    model = BERTopic(language="english")
    topics, probs = model.fit_transform(text_data)
    return topics, probs

def save_topics_to_csv(topics, filename='topics.csv'):
    """Save the topic results to a CSV file."""
    df = pd.DataFrame(topics, columns=['Topic'])
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    # Muat data yang telah diproses
    df = pd.read_csv('processed_data.csv')
    
    # Melakukan pemodelan topik
    topics, _ = topic_modeling(df['Processed_Text'].tolist())
    
    # Simpan hasil topik ke dalam CSV
    save_topics_to_csv(topics)
    print(f"Topics saved to 'topics.csv'.")
