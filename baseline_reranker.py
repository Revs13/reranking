import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load synthetic dataset
with open("synthetic_green_ads.json", "r") as f:
    data = json.load(f)

vectorizer = TfidfVectorizer()

all_queries = []
all_preds = []
all_labels = []

for item in data:
    query = item["query"]
    docs = item["documents"]
    doc_texts = [d["text"] for d in docs]
    labels = [d["label"] for d in docs]

    # Fit vectorizer on both query and docs
    tfidf_matrix = vectorizer.fit_transform([query] + doc_texts)
    query_vec = tfidf_matrix[0]
    doc_vecs = tfidf_matrix[1:]

    # Compute cosine similarities
    similarities = cosine_similarity(query_vec, doc_vecs).flatten()
    ranked = sorted(zip(similarities, labels, doc_texts), reverse=True)

    print(f"\n🔍 Query: {query}")
    for i, (score, label, text) in enumerate(ranked):
        tag = "✅" if label == 1 else "❌"
        print(f"Rank {i+1}: [{score:.2f}] {tag} {text}")

    all_preds.extend([label for _, label, _ in ranked])
    all_labels.extend(labels)
