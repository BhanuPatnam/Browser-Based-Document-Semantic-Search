---
project_name: Enterprise Retrieval-Augmented Generation Architecture
project_category: Generative AI / Information Retrieval
document_type: architecture_spec
tags: rag, langchain, vector-database, faiss, llamaindex, hallucination-mitigation, citations
url: https://github.com/BhanuPatnam/generative-ai-systems/tree/main/09-enterprise-rag-langchain
---
# Enterprise Retrieval-Augmented Generation Architecture

## Objective
Design and implement an enterprise-grade Retrieval-Augmented Generation (RAG) system for technical engineering manuals and standard operating procedures (SOPs). The goal is to provide accurate, grounded answers to complex multi-step queries while preventing factual hallucination.

## Retrieval pipeline and indexing
The ingestion engine applies recursive character and AST-aware chunking with metadata preservation (file paths, section headings, version hashes). Chunks are indexed using a hybrid vector store:
- Dense vector search powered by FAISS with Hierarchical Navigable Small World (HNSW) indexing.
- Sparse keyword indexing using BM25 with exact token preservation.
- Reciprocal Rank Fusion (RRF) with dynamic alpha score weighting to balance keyword matching and conceptual similarity.

## Evaluation and governance
RAG quality is systematically evaluated using Ragas and TruLens across four core dimensions:
- Faithfulness (percentage of generated claims substantiated by retrieved passages).
- Answer relevance (semantic alignment to query intent).
- Context recall (proportion of required ground-truth evidence successfully retrieved).
- Citation accuracy (verifiable passage provenance and character-level span attribution).
Deployment utilizes Docker containers orchestrated via Kubernetes, with Redis caching frequently retrieved query embeddings.
