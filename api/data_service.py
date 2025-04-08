import pandas as pd
import os

def get_processed_data():
    """
    Mengambil data yang telah diproses (dari file CSV yang sudah dibersihkan).
    """
    processed_data_path = os.path.join('..', '..', 'data', 'processed_data', 'processed_titles.csv')
    if not os.path.exists(processed_data_path):
        return "Data not found"
    
    df = pd.read_csv(processed_data_path)
    return df.to_dict(orient="records")  # Menampilkan data dalam bentuk list of dicts
