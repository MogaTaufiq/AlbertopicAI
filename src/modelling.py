import os
import re
import pickle
import time
import pandas as pd

# Impor untuk MLflow
import mlflow
import mlflow.pyfunc

# Impor untuk BERTopic dan pemodelan
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

# Fungsi untuk memuat data (tidak berubah)
def load_data(input_file_path):
    """Memuat data yang telah diproses dan menggabungkan judul serta abstrak."""
    if not os.path.exists(input_file_path):
        print(f"Error: File {input_file_path} tidak ditemukan!")
        return None
    df = pd.read_csv(input_file_path)

    if 'Processed_Title' not in df.columns:
        print("Error: Kolom Processed_Title tidak ditemukan.")
        return None

    df['Processed_Title'] = df['Processed_Title'].fillna('')
    if 'Processed_Abstract' in df.columns:
        df['Processed_Abstract'] = df['Processed_Abstract'].fillna('')
        df['Processed_Text'] = df['Processed_Title'] + ' ' + df['Processed_Abstract']
    else:
        print("Peringatan: Kolom Processed_Abstract tidak ditemukan. Menggunakan hanya Processed_Title.")
        df['Processed_Text'] = df['Processed_Title']

    return df

# Fungsi untuk pelatihan model (ditambahkan calculate_probabilities)
def perform_bertopic_modeling(df, nr_topics="auto", min_topic_size=10, sentence_model_name='paraphrase-MiniLM-L6-v2'):
    """Melakukan pemodelan topik BERTopic."""
    print(f"Memulai pelatihan dengan parameter: nr_topics='{nr_topics}', min_topic_size={min_topic_size}")
    embedding_model = SentenceTransformer(sentence_model_name)
    embeddings = embedding_model.encode(df['Processed_Text'].tolist(), show_progress_bar=True)

    bertopic_nr_topics = None if nr_topics == "auto" else nr_topics

    # --- PERUBAHAN DI SINI: Aktifkan penghitungan probabilitas ---
    topic_model = BERTopic(
        nr_topics=bertopic_nr_topics,
        min_topic_size=min_topic_size,
        embedding_model=embedding_model,
        calculate_probabilities=True  # <-- PENTING untuk mengatasi TypeError
    )
    
    topics, _ = topic_model.fit_transform(df['Processed_Text'].tolist(), embeddings)
    return topic_model, topics

# --- PERUBAHAN DI SINI: Fungsi untuk mendapatkan ID berurutan ---
def get_next_run_id(base_dir="data/final/topic_results"):
    """Mencari ID run tertinggi yang ada dan mengembalikan ID berikutnya (mulai dari 0)."""
    os.makedirs(base_dir, exist_ok=True)
    max_id = -1
    for filename in os.listdir(base_dir):
        # Ekstrak angka dari nama file seperti 'topic_results_5.csv'
        match = re.search(r'_(\d+)\.csv$', filename)
        if match:
            current_id = int(match.group(1))
            if current_id > max_id:
                max_id = current_id
    return max_id + 1

# --- PERUBAHAN DI SINI: Fungsi untuk menyimpan ke subfolder yang berbeda ---
def save_local_artifacts(topic_model, df_with_topics, sequential_run_id):
    """Menyimpan model .pkl dan hasil topik .csv ke subfolder terpisah dengan ID berurutan."""
    base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'final')
    
    # Direktori untuk model .pkl
    model_output_dir = os.path.join(base_output_dir, "bertopic_model")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Direktori untuk hasil .csv
    results_output_dir = os.path.join(base_output_dir, "topic_results")
    os.makedirs(results_output_dir, exist_ok=True)

    # Simpan model .pkl
    model_path = os.path.join(model_output_dir, f"bertopic_model_{sequential_run_id}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(topic_model, f)
    print(f"Model disimpan secara lokal di: {model_path}")

    # Simpan hasil topik .csv
    results_path = os.path.join(results_output_dir, f"topic_results_{sequential_run_id}.csv")
    df_with_topics.to_csv(results_path, index=False)
    print(f"Hasil topik disimpan secara lokal di: {results_path}")
    
    return model_path, results_path

# Fungsi untuk evaluasi koherensi (tidak berubah)
def evaluate_coherence(df, topic_model):
    """Mengevaluasi model topik menggunakan Coherence Score (c_v)."""
    try:
        documents = df['Processed_Text'].astype(str).tolist()
        topics = topic_model.get_topics()
        valid_topic_words = [words for t_id, words in topics.items() if t_id != -1]
        if not valid_topic_words: return 0.0
        tokenized_docs = [doc.split() for doc in documents]
        dictionary = Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        if not corpus: return 0.0
        coherence_model = CoherenceModel(topics=valid_topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        print(f"Coherence Score (c_v): {coherence_score:.4f}")
        return coherence_score
    except Exception as e:
        print(f"Error saat menghitung koherensi: {e}")
        return 0.0

# ### MLFLOW ###: Wrapper untuk model
class BERTopicWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, topic_model):
        self.topic_model = topic_model
        
    # --- PERUBAHAN DI SINI: Dibuat lebih robust ---
    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            texts_to_transform = model_input.iloc[:, 0].tolist()
        else:
            texts_to_transform = model_input
            
        topic_ids, probabilities = self.topic_model.transform(texts_to_transform)
        
        # Penanganan kasus jika probabilitas adalah None
        if probabilities is None:
            probabilities_list = [None] * len(topic_ids)
        else:
            probabilities_list = [p.tolist() if p is not None else None for p in probabilities]

        return pd.DataFrame({
            'predicted_topic_id': topic_ids,
            'probabilities': probabilities_list
        })

# --- Bagian utama skrip ---
if __name__ == "__main__":
    # --- Konfigurasi Eksperimen ---
    NR_TOPICS_CONFIG = 50
    MIN_TOPIC_SIZE_CONFIG = 15
    SENTENCE_MODEL_NAME_CONFIG = 'paraphrase-MiniLM-L6-v2'

    mlflow.set_experiment("AlbertopicAI - Pelatihan Model Topik")

    with mlflow.start_run() as run:
        # --- PERUBAHAN DI SINI: Gunakan ID berurutan untuk file lokal ---
        sequential_run_id = get_next_run_id()
        print(f"Memulai Run. ID Lokal: {sequential_run_id}, ID MLflow: {run.info.run_id}")

        mlflow.log_param("local_run_id", sequential_run_id)
        mlflow.log_param("nr_topics", NR_TOPICS_CONFIG)
        mlflow.log_param("min_topic_size", MIN_TOPIC_SIZE_CONFIG)
        mlflow.log_param("sentence_model", SENTENCE_MODEL_NAME_CONFIG)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        processed_data_path = os.path.join(project_root, 'data', 'processed_data', 'processed_articles.csv')
        df = load_data(processed_data_path)

        if df is None:
            print("Gagal memuat data. Menghentikan proses.")
            exit()
        mlflow.log_param("jumlah_dokumen", len(df))

        print("Memulai pelatihan model...")
        start_time = time.time()
        topic_model, topics = perform_bertopic_modeling(
            df, 
            nr_topics=NR_TOPICS_CONFIG, 
            min_topic_size=MIN_TOPIC_SIZE_CONFIG,
            sentence_model_name=SENTENCE_MODEL_NAME_CONFIG
        )
        training_time = time.time() - start_time
        print(f"Pelatihan selesai dalam {training_time:.2f} detik.")

        coherence_score = evaluate_coherence(df, topic_model)
        num_topics_found = len(topic_model.get_topic_info())
        if -1 in topic_model.get_topic_info()['Topic'].values: num_topics_found -= 1
        
        print("Mencatat metrik ke MLflow...")
        mlflow.log_metric("coherence_score_cv", coherence_score)
        mlflow.log_metric("training_time_seconds", round(training_time, 2))
        mlflow.log_metric("jumlah_topik_ditemukan", num_topics_found)

        df['Topic'] = topics
        # --- PERUBAHAN DI SINI: Kirim ID berurutan ke fungsi penyimpanan ---
        _, local_results_path = save_local_artifacts(topic_model, df, sequential_run_id)

        print("Mencatat artefak ke MLflow...")
        mlflow.log_artifact(local_results_path, "hasil_topik")

        topic_info_df = topic_model.get_topic_info()
        topic_info_path = f"topic_info_{sequential_run_id}.csv"
        topic_info_df.to_csv(topic_info_path, index=False)
        mlflow.log_artifact(topic_info_path, "ringkasan_topik")
        os.remove(topic_info_path)

        print("Mencatat model ke MLflow...")
        input_example = pd.DataFrame(["Contoh dokumen tentang machine learning.", "Dokumen lain tentang komputasi kuantum."])
        mlflow.pyfunc.log_model(
            artifact_path="bertopic_model",
            python_model=BERTopicWrapper(topic_model),
            input_example=input_example
        )
        
        print(f"\nEksperimen dengan ID Lokal {sequential_run_id} berhasil dilacak di MLflow.")
