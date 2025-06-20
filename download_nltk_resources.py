import nltk

resources = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet'
}

for resource_name, resource_path in resources.items():
    try:
        nltk.data.find(resource_path)
        print(f"Resource '{resource_name}' ({resource_path}) sudah ada.")
    except LookupError: # Langsung tangkap LookupError dari nltk.data
        print(f"Resource '{resource_name}' ({resource_path}) tidak ditemukan, mencoba mengunduh...")
        nltk.download(resource_name)
        print(f"Resource '{resource_name}' berhasil diunduh.")

print("\nProses pengecekan dan pengunduhan resource NLTK selesai.")