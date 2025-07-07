# Reranking Experiments with LLMs and Embedding Models

This repository contains three reranking algorithms, each with a different approach to reranking a candidate list of documents given a user query.

## Reranking Methods

### 1. LLM-Powered Reranker
- Uses OpenAI's LLM to semantically rerank the top-k documents.
- Reranking is framed as a prompt-based relevance assessment task.

### 2. BGE Reranker
- Uses a pretrained encoder model from the BGE family to score query-document pairs.
- Evaluated using Top-1 accuracy on a synthetic dataset.

### 3. HuggingFace AutoTokenizer-based Reranker
- Custom reranker built with HuggingFace's AutoTokenizer and AutoModel, pretrained on BGE-reranker-base.
- Evaluated using MRR@10 to measure the relevance of the top-10 results.

## Evaluation
The BGE Reranker achieved 0.5 top-1 accuracy.

The HuggingFace AutoTokenizer-based Reranker achieved an MRR@10 of 0.2296.
