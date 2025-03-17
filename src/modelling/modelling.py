import pandas as pd
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pickle

def load_data(input_file_path):
    """Load preprocessed data from the specified path."""
    if not os.path.exists(input_file_path):
        print(f"Error: File {input_file_path} not found!")
        return None
    df = pd.read_csv(input_file_path)
    return df

def perform_bertopic_modeling(df):
    """Perform BERTopic for topic modeling."""
    # Menggunakan SentenceTransformer untuk menghasilkan embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Model transformer BERT yang ringan
    embeddings = model.encode(df['Processed_Text'].tolist(), show_progress_bar=True)

    # Menerapkan BERTopic
    topic_model = BERTopic(nr_topics=5)
    topics, _ = topic_model.fit_transform(df['Processed_Text'].tolist(), embeddings)
    
    return topic_model, topics

def save_model(topic_model, filename='bertopic_model.pkl'):
    """Save the trained BERTopic model to disk."""
    model_dir = os.path.join('..', '..', 'data', 'final')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    with open(model_path, 'wb') as f:
        pickle.dump(topic_model, f)
    print(f"Model saved to {model_path}")

def save_topic_results(df, topics, filename='topic_results.csv'):
    """Assign topics to the dataset and save the results."""
    df['Topic'] = topics
    result_file_path = os.path.join('..', '..', 'data', 'final', filename)
    df.to_csv(result_file_path, index=False)
    print(f"Topic results saved to {result_file_path}")

def print_top_words(topic_model):
    """Print the top words for each topic."""
    for i in range(topic_model.get_topic_freq().shape[0]):
        print(f"Topic #{i}:")
        print(topic_model.get_topic(i))
        print()

if __name__ == "__main__":
    # Path ke file data yang sudah diproses
    processed_data_path = os.path.join('..', '..', 'data', 'processed_data', 'processed_titles.csv')
    
    # Load data
    df = load_data(processed_data_path)
    if df is None:
        exit()

    # Perform BERTopic modeling
    topic_model, topics = perform_bertopic_modeling(df)

    # Print top words for each topic
    print_top_words(topic_model)

    # Save model and topic results
    save_model(topic_model)
    save_topic_results(df, topics)
