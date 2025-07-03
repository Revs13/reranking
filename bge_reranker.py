from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import json

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your query and documents
query = "How to treat the common cold?"
docs = [
    "Common cold is a viral infection of your nose and throat.",
    "Antibiotics are not effective against viruses like the cold.",
    "Drinking fluids and resting are effective treatments.",
    "Buy stock in cold medicine companies."
]

# Rerank: Get scores for each (query, doc) pair
def rerank(query, docs):
    scores = []
    for doc in docs:
        # Format as "[CLS] query [SEP] document [SEP]"
        encoded = tokenizer(query, doc, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
            score = logits[0].item()  # Probability of being relevant
        scores.append((doc, score))
    
    # Sort docs by descending score
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked

# Try reranking
#results = rerank(query, docs)

# Show results
#print("\nReranked Documents:")
#for i, (doc, score) in enumerate(results):
#    print(f"{i+1}. Score: {score:.4f} — {doc}")

with open("synthetic_green_ads.json", "r") as f:
    data = json.load(f)

top_1_accuracy = 0

for item in data:
    docs = []
    labels = {}

    for doc in item['documents']:
        docs.append(doc['text'])
        labels[doc['text']] = doc['label']
        
    reranked = rerank(item['query'], docs)
    print('\nReranked documents:')
    for i, (text, score) in enumerate(reranked):
        #tag = "✅" if score == labels[text] else "❌"
        tag = "❌" 
        if i == 0:
            if labels[text] == 1:
                tag = "✅"
                top_1_accuracy += 1
        print(f"Rank {i+1}: [{text}] {tag} {score}")

print(top_1_accuracy / len(data))