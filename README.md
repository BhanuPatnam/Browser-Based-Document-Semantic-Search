# Semantic & Hybrid Information Retrieval System with Sentence-BERT & Cross-Encoder

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hybrid-semantic-information-retrieval-system.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-all--MiniLM--L6--v2-fbbf24.svg)](https://www.sbert.net/)
[![Cross-Encoder](https://img.shields.io/badge/Cross--Encoder-ms--marco--MiniLM--L--6--v2-orange.svg)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
[![BM25](https://img.shields.io/badge/Lexical-BM25Okapi-blueviolet.svg)](https://github.com/dorianbrown/rank_bm25)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%20Interactive-ff4b4b.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-36%20Passed-brightgreen.svg)](#test-suite--reproducibility)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🚀 **Live Interactive Demo:** [https://hybrid-semantic-information-retrieval-system.streamlit.app/](https://hybrid-semantic-information-retrieval-system.streamlit.app/)

An end-to-end **Natural Language Processing (NLP)** and **Information Retrieval (IR)** project demonstrating dense vector search, lexical baselines (BM25, TF-IDF), convex hybrid fusion, section-aware chunking experiments, query expansion, and two-stage neural reranking over technical documentation.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [NLP Pipeline Architecture](#nlp-pipeline-architecture)
3. [Lexical vs. Dense Retrieval Theory](#lexical-vs-dense-retrieval-theory)
4. [Hybrid Retrieval (Lexical-Dense Fusion)](#hybrid-retrieval-lexical-dense-fusion)
5. [Two-Stage Retrieval & Cross-Encoder Reranking](#two-stage-retrieval--cross-encoder-reranking)
6. [Chunking Strategy & Empirical Grid Experiment](#chunking-strategy--empirical-grid-experiment)
7. [Information Retrieval Evaluation Metrics](#information-retrieval-evaluation-metrics)
8. [Empirical Benchmark Results](#empirical-benchmark-results)
9. [Query Preprocessing & Error Analysis](#query-preprocessing--error-analysis)
10. [Streamlit Interactive Web Application](#streamlit-interactive-web-application)
11. [Quickstart & Reproducibility](#quickstart--reproducibility)
12. [Limitations & Responsible Use](#limitations--responsible-use)
13. [Author & Attribution](#author--attribution)

---

## Problem Statement

Technical portfolios, scientific knowledge bases, model cards, and operational documentation are difficult to retrieve with classical keyword matching alone:

1. **Vocabulary Mismatch Problem**: Users often express concepts using synonyms or descriptive intent that do not match the exact document tokens (e.g., searching for *"biomedical mask prediction"* when the document describes *"U-Net image segmentation"*).
2. **Loss of Context in Traditional NLP**: Aggressive stemming, stop-word removal, and bag-of-words assumptions strip out critical syntactic cues and destroy technical identifiers like `all-MiniLM-L6-v2`, `Recall@1`, `BERTScore`, and `Sentence-BERT`.
3. **Candidate Quality vs. Latency Trade-Off**: Bi-encoders provide fast candidate generation ($O(N)$ dot products) but lack token-to-token cross-attention, while cross-encoders provide state-of-the-art accuracy but are computationally intractable over entire corpora.

**Objective**: Build, evaluate, and benchmark a complete Information Retrieval system comparing **Lexical Search (TF-IDF, BM25)**, **Dense Vector Search (Sentence-BERT)**, **Hybrid Fusion**, and **Two-Stage Cross-Encoder Reranking** across grounded technical documents.

---

## NLP Pipeline Architecture

```
Technical Raw Documents (Markdown)
                │
                ▼
   [1. Text Preprocessing] ──────▶ Markdown-aware parser (preserves technical terms, headings)
                │
                ▼
   [2. Section-Aware Chunking] ──▶ Heading-anchored sentence grouping (180 words, 40 overlap)
                │
                ├───▶ Lexical Index: Inverted Index (BM25Okapi & TF-IDF)
                │
                └───▶ Dense Index: Sentence-BERT (all-MiniLM-L6-v2 -> 384-dim normalized matrix)
                │
   [3. Query Processing] ───────▶ Term-preserving cleaner + optional Domain Synonym Expansion
                │
   [4. Multi-Strategy Retrieval]
         ├─▶ TF-IDF (Cosine)
         ├─▶ BM25Okapi
         ├─▶ Sentence-BERT Dense (Cosine)
         ├─▶ Hybrid (BM25 + Dense Convex Combination, α=0.5)
         │
         ▼
   [5. Two-Stage Reranking] ────▶ Top-10 Candidates -> Cross-Encoder (ms-marco-MiniLM-L-6-v2)
                │
                ▼
   [6. Ranked Output & Evaluation] ──▶ Precision@K, Recall@K, MRR, nDCG@K + Streamlit UI
```

---

## Lexical vs. Dense Retrieval Theory

### 1. BM25 (BM25Okapi)
BM25 ranks documents based on term frequency with saturation and document length normalization:

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

- **$k_1 = 1.5$**: Controls term frequency saturation (prevents a document from dominating purely due to repetitive words).
- **$b = 0.75$**: Controls the degree of document length penalization.
- **$\text{IDF}(t)$**: Inverse document frequency, downweighting terms ubiquitous across documents.

### 2. Dense Vector Retrieval (Sentence-BERT Bi-Encoder)
Sentence-BERT projects queries and passages into a shared $384$-dimensional embedding space:

$$\mathbf{u} = \text{MeanPooling}\left(\text{Transformer}(T)\right), \quad \hat{\mathbf{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2}$$

- **Architecture**: `sentence-transformers/all-MiniLM-L6-v2` (12 layers, 384 hidden dimensions, 12 attention heads).
- **Cosine Similarity as Dot Product**: Because all document vectors $\hat{\mathbf{d}}$ and query vectors $\hat{\mathbf{q}}$ are $L_2$-normalized on the unit hypersphere ($\|\hat{\mathbf{q}}\|_2 = 1, \|\hat{\mathbf{d}}\|_2 = 1$), cosine similarity simplifies directly to a fast inner product:
  $$\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \hat{\mathbf{q}} \cdot \hat{\mathbf{d}}$$

---

## Hybrid Retrieval (Lexical-Dense Fusion)

Dense vectors excel at semantic generalization, while BM25 excels at exact technical keyword precision. The system implements a **normalized linear score fusion**:

1. **Min-Max Normalization**:
   $$S_{\text{norm}} = \frac{S - \min(S)}{\max(S) - \min(S) + \epsilon}$$
2. **Convex Combination**:
   $$S_{\text{hybrid}} = \alpha \cdot S_{\text{BM25, norm}} + (1 - \alpha) \cdot S_{\text{Dense, norm}}$$

Configuring $\alpha \in [0.0, 1.0]$ allows fine-tuning between pure semantic retrieval ($\alpha=0$) and pure keyword retrieval ($\alpha=1$). On our benchmark, $\alpha = 0.5$ eliminated competitor false positives and achieved **1.0000 MRR**.

---

## Two-Stage Retrieval & Cross-Encoder Reranking

```
Corpus (N Chunks) ──▶ [Stage 1: Bi-Encoder SBERT] ──▶ Top 10 Shortlist (Fast: O(N) dot products)
                                                            │
                                                            ▼
                                              [Stage 2: Cross-Encoder] (Accurate: All-to-all attention)
                                                            │
                                                            ▼
                                                    Final Ranked Top K
```

- **Stage 1 (Bi-Encoder)**: Compares precomputed document vectors with the query vector. It cannot capture fine-grained token-level cross-interactions between query and document tokens.
- **Stage 2 (Cross-Encoder)**: Feeds `[CLS] + Query + [SEP] + Document + [SEP]` simultaneously into all self-attention layers of `cross-encoder/ms-marco-MiniLM-L-6-v2`. Every query token attends directly to every document token, resolving subtle lexical and semantic ambiguities.

---

## Chunking Strategy & Empirical Grid Experiment

Chunking dictates whether an embedding captures cohesive semantic facts or dilutes critical points across unrelated discourse.

The project evaluates three chunking algorithms across a full factorial parameter grid ($3 \times 3 \times 3 = 27$ configurations):
- **Fixed-Window**: Arbitrary word windowing without sentence or heading awareness.
- **Sentence-Aware**: Respects grammatical sentence boundaries (`.`, `!`, `?`).
- **Section-Aware**: Respects Markdown heading hierarchies (`#`, `##`, `###`), then applies sentence-aware windowing within sections.

### Grid Parameters:
- **Chunk sizes**: $100, 180, 250$ words
- **Overlaps**: $20, 40, 60$ words

### Empirical Chunking Findings:
The results are recorded in [outputs/chunking_experiment_results.csv](outputs/chunking_experiment_results.csv):
- **Section-Aware 180 words / 40 overlap** provided the highest discourse consistency: it preserves structural Markdown headings as boundaries, ensuring that unrelated project sections (e.g., *Architecture* vs. *Deployment*) are not merged into a single vector.
- Chunks under 100 words suffered from loss of explanatory context for technical questions.
- Chunks above 250 words suffered from **embedding dilution** under mean pooling, where incidental text pulled the centroid vector away from specific technical terms.

---

## Information Retrieval Evaluation Metrics

Evaluation is calculated strictly using labeled queries and relevance ground truth ([src/model_evaluation.py](src/model_evaluation.py)):

- **Precision@K**: Fraction of top-$K$ returned results that are relevant:
  $$\text{Precision@K} = \frac{|\text{Top } K \cap \text{Relevant}|}{K}$$
- **Recall@K**: Fraction of all target relevant documents retrieved within the top-$K$:
  $$\text{Recall@K} = \frac{|\text{Top } K \cap \text{Relevant}|}{|\text{Relevant}|}$$
- **Mean Reciprocal Rank (MRR)**: Evaluates the rank position of the first relevant document:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
- **nDCG@K (Normalized Discounted Cumulative Gain)**: Accounts for rank position with logarithmic discount:
  $$\text{DCG@K} = \sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i + 1)}, \quad \text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

---

## Empirical Benchmark Results

Evaluated on the benchmark query suite ([outputs/retrieval_model_comparison.csv](outputs/retrieval_model_comparison.csv)):

| Retrieval Method | MRR | Recall@1 | Recall@3 | Recall@5 | Precision@3 | nDCG@3 | nDCG@5 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF (Lexical)** | 0.8750 | 0.6562 | 0.8958 | 0.9688 | 0.5000 | 0.8582 | 0.9124 |
| **BM25 (Lexical)** | 0.8750 | 0.5729 | 0.8646 | 0.9688 | 0.4583 | 0.8505 | 0.9058 |
| **Sentence-BERT (Dense)** | 0.9375 | 0.6979 | 0.9375 | 0.9688 | 0.5833 | 0.9169 | 0.9209 |
| **Hybrid (BM25 + Dense, $\alpha=0.5$)** | **1.0000** | **0.8229** | 0.9062 | 0.9688 | 0.5000 | 0.9337 | 0.9637 |
| **Dense + Cross-Encoder Rerank** | 0.9375 | 0.7917 | **0.9375** | **1.0000** | **0.5833** | **0.9413** | **0.9701** |

### Key Observations:
1. **Hybrid Retrieval achieved a perfect 1.0000 MRR**: The combination of BM25 exact term matching and Sentence-BERT semantic proximity ensured that the true target document was ranked #1 for every single test query.
2. **Cross-Encoder achieved the highest nDCG@5 (0.9701) and Recall@5 (1.0000)**: Reranking candidates using full cross-attention successfully elevated all relevant target documents into the top tier.
3. **Lexical baselines lagged on conceptual queries**: BM25 and TF-IDF failed when users queried concepts without exact vocabulary overlap.

---

## Query Preprocessing & Error Analysis

### 1. Term-Preserving Query Preprocessing
Unlike standard tokenizers, [src/query_analysis.py](src/query_analysis.py) explicitly preserves:
- Hyphenated model names (`all-MiniLM-L6-v2`, `U-Net`, `Sentence-BERT`)
- Evaluation metrics (`Recall@1`, `MRR`, `nDCG@5`)
- Acronyms (`SOP`, `CAPA`, `VQA`, `LSTM`, `RAG`)

### 2. Rule-Based Query Expansion
Expands domain terms with verified synonyms:
- *"image segmentation"* $\rightarrow$ adds `semantic segmentation`, `mask prediction`, `u-net`
- *"semantic search"* $\rightarrow$ adds `dense retrieval`, `vector search`, `sentence-bert`
- *"question answering"* $\rightarrow$ adds `qa`, `reading comprehension`, `longformer`

### 3. Factual Error Case Study
From [outputs/error_analysis.md](outputs/error_analysis.md):
- **Query**: *"What metrics evaluate this document semantic search engine?"*
- **Expected Document**: `07-document-semantic-search-sentence-bert`
- **Dense Rank**: Ranked at **#2** (Competitor at #1: `03-cross-encoder-bi-encoder-ranking-system`).
- **Probable Cause**: Semantic competition on evaluation terminology. Project 03 features dense discussions of evaluation metrics (MRR, nDCG, Precision@K), causing the bi-encoder to assign it high cosine similarity.
- **Resolution**: Both Hybrid retrieval ($\alpha=0.5$) and Cross-Encoder reranking resolve the tie and rank the correct document at #1.

---

## Streamlit Interactive Web Application

The interactive web UI ([app.py](app.py)) is deployed live at **[https://hybrid-semantic-information-retrieval-system.streamlit.app/](https://hybrid-semantic-information-retrieval-system.streamlit.app/)** and provides:
- **Retrieval Method Selector**: Toggle between TF-IDF, BM25, Sentence-BERT, Hybrid, and Cross-Encoder Reranking in real-time.
- **Hybrid $\alpha$ Weight Slider**: Adjust the balance between BM25 and Dense retrieval on the fly.
- **Query Expansion Toggle**: Inspect original query vs. expanded domain synonyms.
- **Model Comparison Tab**: Visual comparison charts and live benchmark runner across all five methods.
- **Chunking Experiments & Error Analysis Tab**: Interactive exploration of factorial chunking results and case-by-case error diagnoses.
- **NLP Insights Tab**: Embedded theoretical formulas for BM25, Cosine Similarity, SBERT, and nDCG.

---

## Quickstart & Reproducibility

### 1. Environment Setup
```bash
git clone https://github.com/BhanuPatnam/Browser-Based-Document-Semantic-Search.git
cd Browser-Based-Document-Semantic-Search
pip install -r requirements.txt
```

### 2. Run Preprocessing & Generate Embeddings
```bash
python scripts/prepare_corpus.py
python scripts/generate_embeddings.py
```

### 3. Run IR Benchmarks & Evaluation
```bash
python scripts/evaluate_search.py
python scripts/run_chunking_experiments.py
python scripts/generate_error_analysis.py
```

### 4. Run Complete Test Suite
```bash
python -m pytest tests/
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

---

## Limitations & Responsible Use

1. **Cosine Similarity Calibration**: Cosine similarity scores are relative rank indicators, not calibrated probabilities.
2. **Domain Representation**: Out-of-distribution terminology not present in the pre-training corpus may yield suboptimal embeddings.
3. **Data Safety**: Never index confidential, proprietary, medical, or personally identifiable information in a public repository or client-facing search application.

---

## Author & Attribution

- **Developer & Maintainer**: **Bhanu Patnam** ([GitHub](https://github.com/BhanuPatnam/Browser-Based-Document-Semantic-Search))
- **Attribution & Foundation**: Adapted from the open-source document semantic search architecture originally authored by **Anmol Tripathi** under the MIT License.
