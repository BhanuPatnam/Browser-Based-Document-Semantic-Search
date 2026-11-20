# Hybrid Neural Information Retrieval and Two-Stage Neural Reranking for Technical Knowledge Bases: An Empirical Study on Lexical, Dense, and Cross-Encoder Architectures

**P. Bhanu Prakash**  
*Roll No: 124CS0008*  
*Department of Computer Science and Engineering*  

---

## Abstract
Dense vector retrieval models, particularly Sentence-BERT (SBERT) bi-encoders, have advanced Natural Language Processing (NLP) and Information Retrieval (IR) by embedding text into continuous metric spaces, effectively mitigating the classical vocabulary mismatch problem of sparse inverted index models (such as TF-IDF and BM25). However, pure dense retrieval exhibits known degradation when resolving fine-grained technical identifiers, exact alphanumeric product tokens, and semantic competition between structurally similar documents. In this paper, we propose, implement, and benchmark a complete multi-paradigm Information Retrieval architecture combining sparse BM25Okapi lexical retrieval, dense Sentence-BERT bi-encoder representations (`all-MiniLM-L6-v2`), convex hybrid score fusion, and second-stage neural cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over technical documentation. Furthermore, we conduct a full-factorial chunking experiment across three segmentation strategies (fixed-window, sentence-aware, and section-aware) and nine parameter configurations (100–250 words, 20–60 word overlaps). Evaluated against labeled technical ground-truth queries, our empirical results show that convex hybrid fusion ($\alpha=0.5$) achieves a perfect Mean Reciprocal Rank ($\text{MRR} = 1.0000$) and Recall@1 of $82.29\%$, outperforming standalone BM25 ($\text{MRR} = 0.8750$) and standalone Sentence-BERT ($\text{MRR} = 0.9375$). Additionally, second-stage Cross-Encoder reranking achieves the highest ranking fidelity with an $\text{nDCG@5}$ of $0.9701$ and $\text{Recall@5}$ of $1.0000$. We provide formal mathematical formulations, latency profiling, error case diagnoses, and deployment through an interactive Streamlit framework.

**Index Terms**—Natural Language Processing (NLP), Information Retrieval (IR), Sentence-BERT, Bi-Encoder, Cross-Encoder, BM25Okapi, Hybrid Retrieval, Document Chunking, Mean Reciprocal Rank (MRR), nDCG.

---

## I. Introduction

Information Retrieval (IR) systems in technical domains—encompassing machine learning documentation, engineering specifications, model cards, and standard operating procedures (SOPs)—face a fundamental tension between lexical precision and semantic generalization. 

Classical search engines rely predominantly on sparse lexical models, such as Term Frequency-Inverse Document Frequency (TF-IDF) and BM25. While computationally efficient ($O(1)$ lookup per inverted posting list) and resilient on exact alphanumeric keywords, lexical matching suffers severely from the **vocabulary mismatch problem** [1]. Users querying conceptual intent (e.g., *"biomedical mask prediction"*) fail to retrieve relevant passages written using domain terminology (e.g., *"U-Net image segmentation"*), resulting in low retrieval recall.

Conversely, the advent of pre-trained Transformer backbones (e.g., BERT, RoBERTa) and contrastively trained Siamese bi-encoders (Sentence-BERT) has enabled **Dense Passage Retrieval (DPR)** [2]. By mapping queries and passages into a dense, continuous latent space $\mathbb{R}^{d}$, dense retrieval captures conceptual similarity independent of surface lexical overlap. Nevertheless, standalone dense retrieval has two critical vulnerabilities:
1. **Semantic Collision & Entity Drift**: Dense vectors struggle to prioritize exact technical codes, version metrics, and fine-grained distinctions when passages share high lexical or contextual similarity.
2. **Context Dilution**: Naive document ingestion exceeds the maximum positional embedding window of compact transformers ($256$ to $512$ tokens), causing mean pooling over long sequences to wash out specific factual details.

To address these challenges, this paper presents a comprehensive study and implementation of an end-to-end Neural Information Retrieval pipeline tailored for technical corpora. The core contributions of this work are:
- **Multi-Paradigm Retrieval**: Implementing and benchmarking TF-IDF, BM25Okapi, Sentence-BERT dense retrieval, and convex hybrid fusion.
- **Two-Stage Neural Architecture**: Developing a lightweight candidate-retrieval stage ($O(N)$ dot products) coupled with a Cross-Encoder second-stage reranker utilizing all-to-all token self-attention.
- **Factorial Chunking Grid Experiments**: Evaluating 27 permutations of chunking strategies (fixed-window, sentence-aware, and section-aware) and window sizes to ascertain the optimal discourse boundary.
- **Empirical IR Evaluation**: Measuring performance strictly on ground-truth labeled queries using Precision@K, Recall@K, MRR, and nDCG@K without synthetic or fabricated data.
- **Root-Cause Error Diagnostics & UI**: Providing an automated failure diagnosis module and an interactive Streamlit analytical system.

---

## II. Related Work

### A. Sparse Lexical Retrieval & Inverted Indexes
Lexical retrieval has formed the cornerstone of information retrieval for decades. Robertson et al. [3] introduced BM25, a non-linear term-frequency saturation model that normalizes document lengths. Despite its robustness for keyword search, BM25 cannot capture semantic relatedness, synonymy, or polysemy, motivating the development of neural semantic matching.

### B. Dense Representations & Siamese Networks
Devlin et al. [4] established contextualized language representations via BERT. However, computing pairwise semantic similarity using BERT requires feeding all possible document pairs into the model, incurring an $O(N)$ transformer inference cost per query that is computationally intractable for search. Reimers and Gurevych [2] introduced Sentence-BERT (SBERT), employing Siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared in milliseconds using cosine similarity.

### C. Two-Stage Retrieval Architectures
To reconcile the speed of bi-encoders with the precision of cross-attention, modern search architectures utilize multi-stage pipelines [5], [6]. In Stage 1, a bi-encoder or BM25 model extracts a top-$K$ candidate shortlist from the corpus. In Stage 2, a heavy Cross-Encoder scores the joint sequence `[CLS] + Query + [SEP] + Candidate + [SEP]`, computing token-level interactions that maximize rank fidelity.

---

## III. Theoretical Formulations & Mathematical Modeling

```
+-----------------------------------------------------------------------------------+
|                            THEORETICAL ARCHITECTURE                               |
|                                                                                   |
|  [Query q] ---> [BM25 Inverted Index] ---------> Raw Lexical Score S_BM25         |
|        |                                                    |                     |
|        |                                             [Min-Max Norm]               |
|        |                                                    v                     |
|        |                                            Normalized S_BM25             |
|        |                                                    |                     |
|        +------> [SBERT Bi-Encoder] -> Vector q              +-----> [Hybrid] ---> |
|                       |                                     |       Score         |
|                       v                                     v                     |
|                 [Matrix Dot Product] -------> Normalized S_Dense                  |
|                                                                                   |
|  [Top-10 Candidates] ---> [Cross-Encoder Joint Self-Attention] ---> [Reranked Top-K]|
+-----------------------------------------------------------------------------------+
```

### A. BM25Okapi Lexical Formulation
For a query $Q$ containing tokens $t \in Q$ and a document chunk $D$, the BM25Okapi score is formulated as:

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $f(t, D)$ represents the term frequency of token $t$ in passage $D$.
- $|D|$ and $\text{avgdl}$ denote the passage length and the corpus-wide average passage length in words, respectively.
- $k_1 = 1.5$ calibrates non-linear term frequency saturation.
- $b = 0.75$ controls document length normalization.
- $\text{IDF}(t)$ is the Inverse Document Frequency:

$$\text{IDF}(t) = \ln\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)$$

Where $N$ is the total number of chunks and $n(t)$ is the number of passages containing token $t$.

### B. Dense Bi-Encoder (Sentence-BERT) Formulation
The bi-encoder utilizes the `sentence-transformers/all-MiniLM-L6-v2` architecture (6 Transformer layers, hidden dimension $d = 384$, 12 multi-head self-attention mechanisms). The document chunk $D$ and query $Q$ are mapped independently:

$$\mathbf{H}_D = \text{TransformerEncoder}(D) \in \mathbb{R}^{L \times d}$$

$$\mathbf{u} = \text{MeanPooling}(\mathbf{H}_D) = \frac{\sum_{i=1}^L \mathbf{h}_{D, i} \cdot m_i}{\sum_{i=1}^L m_i}$$

Where $m_i \in \{0, 1\}$ represents the attention mask. To eliminate vector magnitude bias and allow fast inner-product retrieval, embeddings are strictly $L_2$-normalized onto the unit hypersphere:

$$\hat{\mathbf{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2}, \quad \hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

### C. Cosine Similarity in Normalized Metric Space
Because query vector $\hat{\mathbf{v}}$ and all passage vectors $\hat{\mathbf{u}}_j$ reside on the unit sphere ($\|\hat{\mathbf{v}}\|_2 = 1, \|\hat{\mathbf{u}}_j\|_2 = 1$), cosine similarity simplifies directly to the Euclidean inner product:

$$\text{Sim}(\mathbf{q}, \mathbf{d}_j) = \frac{\hat{\mathbf{v}} \cdot \hat{\mathbf{u}}_j}{\|\hat{\mathbf{v}}\|_2 \|\hat{\mathbf{u}}_j\|_2} = \hat{\mathbf{v}} \cdot \hat{\mathbf{u}}_j = \sum_{k=1}^{384} \hat{v}_k \cdot \hat{u}_{j, k}$$

### D. Convex Hybrid Score Fusion
Because BM25 scores are unbounded ($S_{\text{BM25}} \in [0, \infty)$) and cosine similarities occupy $[-1, 1]$, raw linear addition induces scale distortion. We apply dynamic Min-Max normalization:

$$S_{\text{norm}} = \frac{S - \min(S)}{\max(S) - \min(S) + \epsilon}$$

Where $\epsilon = 10^{-9}$. The convex hybrid score is computed as:

$$S_{\text{hybrid}} = \alpha \cdot S_{\text{BM25, norm}} + (1 - \alpha) \cdot S_{\text{Dense, norm}}, \quad \alpha \in [0.0, 1.0]$$

### E. Cross-Encoder Joint Self-Attention Reranking
In Stage 2, top-$M$ candidate passages ($M=10$) are reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Unlike the bi-encoder, the cross-encoder concatenates query and candidate into a single input sequence:

$$\mathbf{x} = [\text{CLS}] \circ q_1 \circ \dots \circ q_{|Q|} \circ [\text{SEP}] \circ d_1 \circ \dots \circ d_{|D|} \circ [\text{SEP}]$$

Every token attends to every other token across all cross-attention heads:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

The final representation of the `[CLS]` token is projected via a linear classification layer to output a scalar relevance logit:

$$S_{\text{rerank}} = \mathbf{W} \cdot \mathbf{h}_{[\text{CLS}]} + b$$

### F. Evaluation Metrics
1. **Precision@K**:

$$\text{Precision@K} = \frac{|\text{Top}_K \cap \text{Relevant}|}{K}$$

2. **Recall@K**:

$$\text{Recall@K} = \frac{|\text{Top}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$

3. **Mean Reciprocal Rank (MRR)**:

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

4. **Normalized Discounted Cumulative Gain (nDCG@K)**:

$$\text{DCG@K} = \sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i + 1)}, \quad \text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

---

## IV. System Architecture & Methodology

### A. Preprocessing Pipeline & Entity Preservation
Standard NLP tokenizers often strip punctuation and lowercase acronyms aggressively. Our preprocessor (`src/text_preprocessing.py`) parses Markdown structure, strips presentation delimiters, normalizes tables, and preserves retrieval-critical technical tokens (e.g., `all-MiniLM-L6-v2`, `U-Net`, `Recall@1`, `BERTScore`, `Sentence-BERT`, and `SOP`).

### B. Hierarchical Section-Aware Chunking
Documents are chunked according to heading hierarchies (`#`, `##`, `###`). Sentences within each section are segmented using regex lookarounds `(?<=[.!?])\s+(?=[A-Z0-9`])`. Chunks are packed into target word windows ($180$ words) with rolling overlap ($40$ words). Chunks are assigned deterministic hierarchical IDs (`{document_id}--{section_slug}--{chunk_index:03d}`).

### C. Query Processing & Domain Expansion
Query normalization (`src/query_analysis.py`) strips terminal query punctuation while preserving internal hyphens and metrics. An optional rule-based domain ontology expands technical terms with synonyms:
- `"image segmentation"` $\rightarrow$ `["semantic segmentation", "mask prediction", "u-net"]`
- `"semantic search"` $\rightarrow$ `["dense retrieval", "vector search", "sentence-bert"]`
- `"question answering"` $\rightarrow$ `["qa", "reading comprehension", "longformer"]`

---

## V. Experimental Setup & Implementation Details

- **Corpus**: 12 technical Markdown documents across 8 project domains, parsed into 34 section-aware chunks with an average word count of 89.58 words per document.
- **Dense Bi-Encoder**: `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional unit vectors.
- **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Lexical Retrievers**: `rank-bm25` (Okapi algorithm, $k_1=1.5, b=0.75$) and Scikit-Learn `TfidfVectorizer` (sublinear TF scaling).
- **Benchmark Suite**: 8 labeled ground-truth technical queries evaluating factual, conceptual, and multi-document retrieval tasks.
- **Environment**: Python 3.13 / PyTorch 2.6 / macOS ARM64 Metal acceleration.

---

## VI. Results, Empirical Analysis & Discussion

### A. Comparative Evaluation Across Retrieval Paradigms

```
+----------------------------------------------------------------------------------------------------+
|                                    RETRIEVAL BENCHMARK RESULTS                                     |
+------------------------------+--------+----------+----------+----------+-------------+-------------+
| Model / Paradigm             |  MRR   | Recall@1 | Recall@3 | Recall@5 | Precision@3 |   nDCG@5    |
+------------------------------+--------+----------+----------+----------+-------------+-------------+
| TF-IDF (Lexical)             | 0.8750 |  0.6562  |  0.8958  |  0.9688  |   0.5000    |   0.9124    |
| BM25 (Lexical)               | 0.8750 |  0.5729  |  0.8646  |  0.9688  |   0.4583    |   0.9058    |
| Sentence-BERT (Dense)        | 0.9375 |  0.6979  |  0.9375  |  0.9688  |   0.5833    |   0.9209    |
| Hybrid (BM25 + Dense, a=0.5) | 1.0000 |  0.8229  |  0.9062  |  0.9688  |   0.5000    |   0.9637    |
| Dense + Cross-Encoder Rerank | 0.9375 |  0.7917  |  0.9375  |  1.0000  |   0.5833    |   0.9701    |
+------------------------------+--------+----------+----------+----------+-------------+-------------+
```

#### Key Empirical Observations:
1. **Hybrid Retrieval Superiority**: Hybrid retrieval with $\alpha = 0.5$ attained **$\text{MRR} = 1.0000$** and **$\text{Recall@1} = 82.29\%$**. The fusion mechanism compensates for the dense bi-encoder's vulnerability to semantic competition by reinforcing exact lexical matches.
2. **Cross-Encoder Ranking Fidelity**: Two-stage Cross-Encoder reranking achieved the highest overall ranking fidelity with an **$\text{nDCG@5}$ of $0.9701$** and **$\text{Recall@5} = 1.0000$**, successfully ordering candidate passages by token-level semantic relevance.
3. **Lexical Baseline Bottleneck**: Pure BM25 achieved an MRR of $0.8750$ and Recall@1 of $57.29\%$, confirming that vocabulary mismatch impairs single-token inverted index matching on technical queries.

### B. Factorial Chunking Grid Analysis
We systematically evaluated 27 factorial combinations of chunking strategies, window sizes, and overlap parameters:

- **Section-Aware Chunking (180 words, 40 words overlap)** demonstrated the highest discourse consistency ($\text{MRR} = 0.9375, \text{nDCG@5} = 0.9209$). Preserving heading boundaries prevents unrelated technical sections from polluting a single embedding.
- **Short Chunks ($< 100$ words)** suffered from context starvation, failing to provide the bi-encoder with sufficient contextual information for accurate embedding generation.
- **Long Chunks ($> 250$ words)** exhibited embedding dilution under mean pooling, where the representation drifted toward generic topics rather than key technical details.

### C. Latency Profiling
- **Stage 1 (Sentence-BERT Matrix Dot Product)**: Average query latency of **$4.22\text{ ms}$** ($\approx 3.7\text{ ms}$ encoding + $0.5\text{ ms}$ inner product).
- **Stage 2 (Cross-Encoder Joint Evaluation)**: Reranking 10 candidates required **$48.3\text{ ms}$**.
- **Conclusion**: The two-stage architecture maintains interactive latency ($< 55\text{ ms}$ total response time) while achieving Cross-Encoder accuracy.

### D. Factual Error Diagnosis
An automated error analysis identified queries where dense retrieval did not position the target document at rank 1:
- **Query**: *"What metrics evaluate this document semantic search engine?"*
- **Expected Document**: `07-document-semantic-search-sentence-bert`
- **Dense Retrieval Rank**: `#2` (Rank #1 retrieved `03-cross-encoder-bi-encoder-ranking-system`).
- **Diagnosis**: Semantic competition on evaluation terminology. Project 03 features dense discussions of evaluation metrics (MRR, nDCG, Precision@K), which caused the bi-encoder to assign it high cosine similarity.
- **Resolution**: Both Hybrid retrieval ($\alpha = 0.5$) and Cross-Encoder reranking resolved this competition, promoting the correct document to **Rank #1**.

---

## VII. Conclusion & Future Work

In this paper, we engineered, analyzed, and benchmarked an end-to-end Neural Information Retrieval system for technical documentation. Our results substantiate that while dense bi-encoders substantially improve conceptual retrieval over lexical baselines, **convex hybrid fusion ($\alpha=0.5$) yields the highest early-precision retrieval ($\text{MRR} = 1.0000$)**, and **two-stage Cross-Encoder reranking delivers the highest overall ranking fidelity ($\text{nDCG@5} = 0.9701$)**.

Future extensions include scaling the index to millions of chunks using Hierarchical Navigable Small World (HNSW) graphs, evaluating late-interaction models such as ColBERT, and integrating the retriever into an end-to-end Retrieval-Augmented Generation (RAG) pipeline.

---

## References

1. A. Singhal, "Modern information retrieval: A brief overview," *IEEE Data Eng. Bull.*, vol. 24, no. 4, pp. 35–43, 2001.
2. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, 2019, pp. 3982–3992.
3. S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," *Found. Trends Inf. Retr.*, vol. 3, no. 4, pp. 333–389, 2009.
4. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics (NAACL)*, 2019, pp. 4171–4186.
5. V. Karpukhin et al., "Dense passage retrieval for open-domain question answering," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, 2020, pp. 6769–6781.
6. R. Nogueira and K. Cho, "Passage re-ranking with BERT," *arXiv preprint arXiv:1901.04085*, 2019.
