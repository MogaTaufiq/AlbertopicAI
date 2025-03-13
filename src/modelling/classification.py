from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

def classify_articles(df):
    """Classify articles into categories based on the title."""
    # Menggunakan TF-IDF untuk representasi teks
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Processed_Text'])
    y = df['Category']  # Pastikan ada kolom 'Category' di dataset Anda

    # Split data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Menggunakan Naive Bayes untuk klasifikasi
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Muat data yang telah diproses
    df = pd.read_csv('processed_data.csv')
    
    # Pastikan dataset sudah memiliki kategori untuk klasifikasi
    classify_articles(df)
