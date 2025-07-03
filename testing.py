import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import json
from datasets import load_dataset

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with open("synthetic_green_ads.json", "r") as f:
    data = json.load(f)

# Load the passage-ranking dataset (v1.1 or v2.1)
dataset = load_dataset("ms_marco", "v2.1", split="validation")

print(dataset[0])

subset = dataset.select(range(100))

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

#top 1 accuracy loop
accurate = 0

for item in subset:
    query = item['query']
    passages = item['passages']['passage_text']
    labels = item['passages']['is_selected']  # binary relevance

    reranked = rerank(query, passages)  # This gives list of (text, score)

    # Find the top ranked passage
    top_passage = reranked[0][0]  # Get just the text

    # Find its index in original passages
    top_index = passages.index(top_passage)

    if labels[top_index] == 1:
        accurate += 1

top_1_accuracy = accurate / len(subset)
print(f"Top-1 Accuracy: {top_1_accuracy:.4f}")
