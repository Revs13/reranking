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

# Load the passage-ranking dataset (v1.1 or v2.1)
dataset = load_dataset("ms_marco", "v2.1", split="validation")

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

#MRR@10 
mrr_total = 0

for item in subset:
    query = item['query']
    passages = item['passages']['passage_text']
    labels = item['passages']['is_selected']
    passage_labels = {}

    for passage, label in zip(passages, labels):
        passage_labels[passage] = label
    #print(passage_labels)

    reranked = rerank(query, passages)
    #print(query)
    #print(reranked)

    reciprocal_rank = 0
    for rank, (passage, _) in enumerate(reranked[:10], start=1):
        if passage_labels[passage] == 1:
            reciprocal_rank = 1 / rank
            break  # only first relevant doc counts
        
    mrr_total += reciprocal_rank

print(f"\nMRR@10: {mrr_total / len(subset):.4f}")
