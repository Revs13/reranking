import json
import random

queries = [
    "eco-friendly detergent",
    "sustainable packaging",
    "biodegradable cleaning products",
    "carbon-neutral shipping",
    "green energy alternatives",
    "plastic-free toothpaste",
    "compostable phone cases",
    "ethical clothing brands",
    "organic skincare",
    "zero waste lifestyle"
]

relevant_templates = [
    "Our {product} is made with 100% biodegradable ingredients.",
    "{product} is certified plastic-free and eco-friendly.",
    "This {product} reduces your carbon footprint by 70%.",
    "Customers love our sustainable {product} packaging.",
    "Made with compostable materials, this {product} is a zero-waste solution."
]

irrelevant_templates = [
    "Our {product} comes in a stylish plastic bottle.",
    "New formula for stronger cleaning — now with bleach!",
    "{product} includes synthetic fragrances for a fresh smell.",
    "Imported using air freight for faster delivery.",
    "{product} contains parabens and microbeads for deep cleaning."
]

def generate_documents(query, product, num_rel=2, num_irrel=4):
    relevant_docs = [{"text": random.choice(relevant_templates).format(product=product), "label": 1}
                     for _ in range(num_rel)]
    irrelevant_docs = [{"text": random.choice(irrelevant_templates).format(product=product), "label": 0}
                       for _ in range(num_irrel)]
    return relevant_docs + irrelevant_docs

dataset = []

for query in queries:
    product = query.split()[0]  # e.g., "eco-friendly"
    docs = generate_documents(query, product)
    random.shuffle(docs)
    dataset.append({"query": query, "documents": docs})

with open("synthetic_green_ads.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Synthetic dataset saved as 'synthetic_green_ads.json'")
