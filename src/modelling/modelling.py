import pandas as pd
import os
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pickle

def load_data(input_file_path):
    """Load preprocessed data and combine title and abstract into one text column."""
    if not os.path.exists(input_file_path):
        print(f"Error: File {input_file_path} not found!")
        return None
    df = pd.read_csv(input_file_path)

    if 'Processed_Title' not in df.columns:
        print("Error: Processed_Title column not found.")
        return None

    if 'Processed_Abstract' in df.columns:
        df['Processed_Text'] = df['Processed_Title'].fillna('') + ' ' + df['Processed_Abstract'].fillna('')
    else:
        print("Warning: Processed_Abstract column not found. Using only Processed_Title.")
        df['Processed_Text'] = df['Processed_Title']

    return df

def perform_bertopic_modeling(df):
    """Perform BERTopic for topic modeling."""
    # Menggunakan SentenceTransformer untuk menghasilkan embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Model transformer BERT yang ringan
    embeddings = model.encode(df['Processed_Text'].tolist(), show_progress_bar=True)

    # Menerapkan BERTopic
    topic_model = BERTopic(nr_topics="auto")
    topics, _ = topic_model.fit_transform(df['Processed_Text'].tolist(), embeddings)
    
    return topic_model, topics

def save_model(topic_model, filename='bertopic_model.pkl'):
    """Save the trained BERTopic model to disk."""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'final')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    with open(model_path, 'wb') as f:
        pickle.dump(topic_model, f)
    print(f"Model saved to {model_path}")

def save_topic_results(df, topics, filename='topic_results.csv'):
    """Assign topics to the dataset and save the results."""
    df['Topic'] = topics
    result_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'final', filename)
    df.to_csv(result_file_path, index=False)
    print(f"Topic results saved to {result_file_path}")

def print_top_words(topic_model):
    """Print the top words for each topic."""
    for i in range(topic_model.get_topic_freq().shape[0]):
        print(f"Topic #{i}:")
        print(topic_model.get_topic(i))
        print()

def evaluate_coherence(df, topic_model):
    """Evaluate the topic model using Coherence Score."""
    # Ambil topik dan kata kunci
    topics = topic_model.get_topics()
    topic_words = []
    for topic_id in topics:
        if topic_id == -1:
            continue
        words = [word for word, _ in topics[topic_id]]
        topic_words.append(words)

    # Tokenisasi ulang dokumen (untuk Gensim)
    tokenized_docs = [doc.split() for doc in df['Processed_Text'].tolist()]
    
    # Buat dictionary dan corpus
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    # Hitung coherence score
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    print(f"Coherence Score (c_v): {coherence_score:.4f}")
    return coherence_score

if __name__ == "__main__":
    # Absolute fallback jika __file__ error:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    processed_data_path = os.path.join(project_root, 'data', 'processed_data', 'processed_titles.csv')
    df = load_data(processed_data_path)
    if df is None:
        exit()
    
    topic_model, topics = perform_bertopic_modeling(df)
    
    # Coherence evaluation
    evaluate_coherence(df, topic_model)

    print_top_words(topic_model)
    save_model(topic_model)
    save_topic_results(df, topics)