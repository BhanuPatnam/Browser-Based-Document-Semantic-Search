"""Interactive Document Semantic Search, Lexical Baselines, and Two-Stage Reranking.

An NLP and Information Retrieval application demonstrating dense vector retrieval,
BM25, TF-IDF, hybrid scoring, and Cross-Encoder reranking over technical documentation.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.hybrid_retrieval import HybridRetriever
from src.lexical_retrieval import BM25Retriever, TFIDFRetriever
from src.query_analysis import clean_query, expand_query
from src.reranking import CrossEncoderReranker
from src.semantic_search import search_with_vector

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHUNKS_PATH = DATA_DIR / "document_chunks.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.json"
CORPUS_PATH = DATA_DIR / "corpus.json"
EVAL_QUERIES_PATH = DATA_DIR / "evaluation_queries.json"

COMPARISON_CSV = OUTPUTS_DIR / "retrieval_model_comparison.csv"
CHUNKING_CSV = OUTPUTS_DIR / "chunking_experiment_results.csv"
CHUNKING_JSON = OUTPUTS_DIR / "chunking_experiment_results.json"
ERROR_ANALYSIS_JSON = OUTPUTS_DIR / "error_analysis.json"
ERROR_ANALYSIS_MD = OUTPUTS_DIR / "error_analysis.md"

st.set_page_config(
    page_title="NLP Semantic & Hybrid Information Retrieval",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .result-card {
        padding: 1.15rem;
        border-radius: 0.65rem;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.9rem;
    }
    .badge {
        display: inline-block;
        padding: 0.18rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .badge-category { background-color: #e0f2fe; color: #0369a1; }
    .badge-type { background-color: #fef3c7; color: #b45309; }
    .badge-method { background-color: #f3e8ff; color: #7e22ce; }
    .badge-rank {
        background-color: #2563eb;
        color: #ffffff;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 0.35rem;
    }
    .score-high { color: #15803d; font-weight: 700; font-size: 1rem; }
    .score-med { color: #b45309; font-weight: 700; font-size: 1rem; }
    .score-low { color: #475569; font-weight: 600; font-size: 1rem; }
    .expansion-box {
        padding: 0.75rem 1rem;
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 0.375rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Sentence-BERT Bi-Encoder...")
def get_bi_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner="Loading Cross-Encoder Reranker...")
def get_cross_encoder():
    try:
        reranker = CrossEncoderReranker()
        _ = reranker.model  # Trigger load
        return reranker
    except Exception as exc:
        st.warning(f"Cross-Encoder model could not be loaded: {exc}")
        return None


@st.cache_data(show_spinner="Loading document index...")
def get_index() -> tuple[list[dict[str, Any]], np.ndarray, str, list[dict[str, Any]]]:
    if not CHUNKS_PATH.is_file() or not EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError("Missing processed index data. Run scripts/generate_embeddings.py first.")

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8")) if CORPUS_PATH.is_file() else []
    payload = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))

    vector_map = {item["chunk_id"]: item["vector"] for item in payload.get("embeddings", [])}
    matrix = np.asarray([vector_map[c["chunk_id"]] for c in chunks], dtype=np.float32)
    model_name = payload.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    return chunks, matrix, model_name, corpus


@st.cache_resource(show_spinner="Initializing Lexical & Hybrid Retrievers...")
def get_retrievers(
    _chunks: list[dict[str, Any]],
    _matrix: np.ndarray,
) -> tuple[TFIDFRetriever, BM25Retriever, HybridRetriever]:
    tfidf = TFIDFRetriever().fit(_chunks)
    bm25 = BM25Retriever().fit(_chunks)
    hybrid = HybridRetriever(_chunks, _matrix, bm25_retriever=bm25, alpha=0.5)
    return tfidf, bm25, hybrid


# Load index and models
chunks, matrix, model_name, corpus = get_index()
bi_encoder = get_bi_encoder(model_name)
cross_encoder = get_cross_encoder()
tfidf_retriever, bm25_retriever, hybrid_retriever = get_retrievers(chunks, matrix)

all_categories = sorted(list({c.get("project_category", "") for c in chunks if c.get("project_category")}))
all_doc_types = sorted(list({c.get("document_type", "") for c in chunks if c.get("document_type")}))

# ----------------- Sidebar -----------------
with st.sidebar:
    st.title("⚙️ IR Configuration")

    st.subheader("Retrieval Strategy")
    retrieval_method = st.selectbox(
        "Select Retrieval Model:",
        options=[
            "Sentence-BERT (Dense)",
            "Hybrid (BM25 + Dense)",
            "Dense + Cross-Encoder Reranking",
            "BM25 (Lexical)",
            "TF-IDF (Lexical)",
        ],
        index=0,
    )

    hybrid_alpha = 0.5
    if retrieval_method == "Hybrid (BM25 + Dense)":
        hybrid_alpha = st.slider(
            "Hybrid Weight α (BM25 vs Dense)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="α = 1.0 is pure BM25 lexical; α = 0.0 is pure Sentence-BERT dense.",
        )

    st.subheader("Query Processing")
    enable_expansion = st.checkbox(
        "Enable Rule-Based Query Expansion",
        value=False,
        help="Expands queries with domain synonyms for technical terms (e.g. segmentation -> U-Net).",
    )

    st.subheader("Filters & Tuning")
    top_k = st.slider("Top Results (K)", min_value=1, max_value=10, value=5, step=1)
    min_score = st.slider("Min Score Threshold", min_value=0.0, max_value=1.0, value=0.10, step=0.05)

    selected_category = st.selectbox("Category Filter", options=["All"] + all_categories)
    selected_doc_type = st.selectbox("Document Type Filter", options=["All"] + all_doc_types)

    st.divider()
    st.subheader("Benchmark Test Queries")
    sample_queries = [
        "Which project uses Longformer for question answering?",
        "Find projects using U-Net for image segmentation.",
        "Which project uses a cross-encoder to rerank bi-encoder candidates?",
        "Where did I deploy a browser model with GitHub Pages?",
        "Which project evaluates summarization with BERTScore?",
        "How can semantic search support quality analytics and root-cause knowledge?",
        "Which project uses a bidirectional LSTM with attention?",
        "What metrics evaluate this document semantic search engine?",
    ]
    selected_sample = st.selectbox("Select a benchmark query:", ["-- Select a Query --"] + sample_queries)

    st.divider()
    st.caption("Developed by **Bhanu Patnam**")

# ----------------- Main App -----------------
st.title("🔍 Semantic & Hybrid Information Retrieval System")
st.markdown(
    "A comprehensive NLP evaluation platform comparing **Lexical Search (TF-IDF, BM25)**, "
    "**Dense Vector Search (Sentence-BERT)**, **Hybrid Fusion**, and **Two-Stage Cross-Encoder Reranking**."
)

tab_search, tab_comp, tab_exp, tab_corpus, tab_insights = st.tabs(
    [
        "🔎 Search & Retrieval",
        "📊 Model Comparison & Metrics",
        "🔬 Chunking Experiments & Errors",
        "📂 Document Corpus",
        "💡 NLP Insights & Theory",
    ]
)

# ----------------- Tab 1: Search & Retrieval -----------------
with tab_search:
    default_text = ""
    if selected_sample != "-- Select a Query --":
        default_text = selected_sample

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query_input = st.text_input(
            "Enter your technical search query:",
            value=default_text,
            placeholder="e.g. Which project implements bi-encoder candidate retrieval with cross-encoder reranking?",
            key="user_search_input",
        )
    with col_btn:
        st.write("")
        st.write("")
        search_clicked = st.button("Search 🚀", use_container_width=True, type="primary")

    if query_input:
        cleaned_q = clean_query(query_input)
        expanded_q, added_terms = expand_query(cleaned_q) if enable_expansion else (cleaned_q, [])

        if enable_expansion and added_terms:
            st.markdown(
                f"""
                <div class="expansion-box">
                    <strong>Original Query:</strong> {html.escape(cleaned_q)}<br>
                    <strong>Expanded Query:</strong> {html.escape(expanded_q)}<br>
                    <strong>Added Domain Synonyms:</strong> <em>{", ".join(html.escape(t) for t in added_terms)}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )

        category_arg = None if selected_category == "All" else selected_category
        doc_type_arg = None if selected_doc_type == "All" else selected_doc_type

        # Execute Retrieval
        t_start = time.perf_counter()
        results: list[dict[str, Any]] = []

        active_query = expanded_q if enable_expansion else cleaned_q

        if retrieval_method == "TF-IDF (Lexical)":
            results = tfidf_retriever.search(
                active_query, top_k=top_k, category=category_arg, document_type=doc_type_arg
            )
        elif retrieval_method == "BM25 (Lexical)":
            results = bm25_retriever.search(
                active_query, top_k=top_k, category=category_arg, document_type=doc_type_arg
            )
        elif retrieval_method == "Sentence-BERT (Dense)":
            q_vec = bi_encoder.encode([active_query], normalize_embeddings=True, convert_to_numpy=True)[0]
            results = search_with_vector(
                q_vec, chunks, matrix, top_k=top_k, category=category_arg, document_type=doc_type_arg
            )
        elif retrieval_method == "Hybrid (BM25 + Dense)":
            q_vec = bi_encoder.encode([active_query], normalize_embeddings=True, convert_to_numpy=True)[0]
            results = hybrid_retriever.search(
                active_query,
                q_vec,
                top_k=top_k,
                alpha=hybrid_alpha,
                category=category_arg,
                document_type=doc_type_arg,
            )
        elif retrieval_method == "Dense + Cross-Encoder Reranking":
            q_vec = bi_encoder.encode([active_query], normalize_embeddings=True, convert_to_numpy=True)[0]
            stage1_candidates = search_with_vector(
                q_vec, chunks, matrix, top_k=max(10, top_k), category=category_arg, document_type=doc_type_arg
            )
            if cross_encoder is not None:
                results = cross_encoder.rerank(active_query, stage1_candidates, top_k=top_k)
            else:
                st.warning("Cross-Encoder unavailable; showing stage-1 dense retrieval.")
                results = stage1_candidates[:top_k]

        total_latency_ms = (time.perf_counter() - t_start) * 1000

        # Filtering threshold
        filtered_results = [r for r in results if r["similarity_score"] >= min_score]

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Retrieved Passages", f"{len(filtered_results)} of {top_k}")
        col_m2.metric("Total Latency", f"{total_latency_ms:.2f} ms")
        col_m3.metric("Method", retrieval_method)

        st.divider()

        if not filtered_results:
            st.warning(f"No passages found matching threshold (>= {min_score:.2f}) and selected filters.")
        else:
            for item in filtered_results:
                score = float(item["similarity_score"])
                score_class = "score-high" if score >= 0.50 else ("score-med" if score >= 0.25 else "score-low")

                method_label = item.get("retrieval_method", retrieval_method)
                extra_score_info = ""
                if "hybrid_score" in item:
                    extra_score_info = f" | BM25: {item.get('bm25_score_raw', 0):.2f} (norm {item.get('bm25_score_norm', 0):.2f}) | Dense: {item.get('dense_score_raw', 0):.3f}"
                elif "rerank_score" in item:
                    extra_score_info = f" | Initial Dense Rank: #{item.get('initial_rank', '-')} (Dense Cosine: {item.get('dense_score', 0):.3f})"

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                            <div>
                                <span class="badge badge-rank">Rank #{item['rank']}</span>
                                <span class="badge badge-method">{method_label}</span>
                                <span class="badge badge-category">{html.escape(str(item.get('project_category', 'General')))}</span>
                                <span class="badge badge-type">{html.escape(str(item.get('document_type', 'doc')))}</span>
                            </div>
                            <div>
                                Score: <span class="{score_class}">{score:.4f}</span>
                                <small style="color: #64748b;">{extra_score_info}</small>
                            </div>
                        </div>
                        <h4 style="margin: 0.25rem 0;">{html.escape(str(item.get('project_name', '')))} — {html.escape(str(item.get('section_title', '')))}</h4>
                        <p style="color: #334155; line-height: 1.5; margin: 0.4rem 0;">{html.escape(item['text'])}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(f"View Chunk Details & Technical Metadata (#{item['chunk_id']})"):
                    st.json(
                        {
                            "chunk_id": item["chunk_id"],
                            "document_id": item["document_id"],
                            "source_file": item["source_file"],
                            "word_count": item.get("word_count"),
                            "tags": item.get("tags", []),
                            "url_or_local_path": item.get("url_or_local_path"),
                        }
                    )

# ----------------- Tab 2: Model Comparison & IR Metrics -----------------
with tab_comp:
    st.subheader("Retrieval Model Comparison on Benchmark Queries")
    st.markdown(
        "Empirical evaluation of **TF-IDF**, **BM25**, **Sentence-BERT Dense**, **Hybrid (BM25 + Dense)**, "
        "and **Two-Stage Cross-Encoder Reranking** across the labeled query benchmark."
    )

    if COMPARISON_CSV.is_file():
        df_comp = pd.read_csv(COMPARISON_CSV)
        st.dataframe(df_comp, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        best_mrr = df_comp.loc[df_comp["MRR"].idxmax()]
        best_r1 = df_comp.loc[df_comp["Recall@1"].idxmax()]
        best_ndcg = df_comp.loc[df_comp["nDCG@5"].idxmax()]

        c1.metric("Highest MRR", f"{best_mrr['MRR']:.4f}", f"{best_mrr['Method']}")
        c2.metric("Best Recall@1", f"{best_r1['Recall@1'] * 100:.1f}%", f"{best_r1['Method']}")
        c3.metric("Best Recall@3", f"{df_comp['Recall@3'].max() * 100:.1f}%")
        c4.metric("Best nDCG@5", f"{best_ndcg['nDCG@5']:.4f}", f"{best_ndcg['Method']}")

        st.divider()
        st.subheader("Metric Comparison Visualizer")
        chart_data = df_comp.set_index("Method")[["MRR", "Recall@1", "Recall@3", "nDCG@5"]]
        st.bar_chart(chart_data)
    else:
        st.info("Run `python scripts/evaluate_search.py` to generate the comparison report.")

    st.divider()
    st.subheader("Interactive Query Benchmark Runner")
    if st.button("Run Live Benchmark Across All 5 Methods", type="secondary"):
        eval_queries = json.loads(EVAL_QUERIES_PATH.read_text(encoding="utf-8")) if EVAL_QUERIES_PATH.is_file() else []
        live_rows = []
        for q in eval_queries:
            q_text = q["query"]
            q_vec = bi_encoder.encode([q_text], normalize_embeddings=True, convert_to_numpy=True)[0]
            target_ids = set(q.get("relevant_document_ids", []))

            res_tfidf = tfidf_retriever.search(q_text, top_k=3)
            res_bm25 = bm25_retriever.search(q_text, top_k=3)
            res_dense = search_with_vector(q_vec, chunks, matrix, top_k=3)
            res_hybrid = hybrid_retriever.search(q_text, q_vec, top_k=3, alpha=0.5)

            top10_dense = search_with_vector(q_vec, chunks, matrix, top_k=10)
            res_rerank = cross_encoder.rerank(q_text, top10_dense, top_k=3) if cross_encoder else res_dense

            live_rows.append(
                {
                    "Query": q_text,
                    "Target Doc(s)": ", ".join(target_ids),
                    "TF-IDF Top 1": res_tfidf[0]["document_id"] if res_tfidf else "-",
                    "BM25 Top 1": res_bm25[0]["document_id"] if res_bm25 else "-",
                    "Dense Top 1": res_dense[0]["document_id"] if res_dense else "-",
                    "Hybrid Top 1": res_hybrid[0]["document_id"] if res_hybrid else "-",
                    "Rerank Top 1": res_rerank[0]["document_id"] if res_rerank else "-",
                }
            )
        st.dataframe(pd.DataFrame(live_rows), use_container_width=True)

# ----------------- Tab 3: Chunking Experiments & Errors -----------------
with tab_exp:
    st.subheader("Chunking Strategy & Parameter Experiments")
    st.markdown(
        "Evaluating **Fixed-Window**, **Sentence-Aware**, and **Section-Aware** chunking strategies across "
        "multiple chunk word sizes ($100, 180, 250$) and overlap windows ($20, 40, 60$ words)."
    )

    if CHUNKING_JSON.is_file():
        chunk_data = json.loads(CHUNKING_JSON.read_text(encoding="utf-8"))
        best_cfg = chunk_data.get("best_configuration", {})
        st.success(
            f"**Best Configuration:** Strategy = `{best_cfg.get('strategy')}`, "
            f"Chunk Size = `{best_cfg.get('chunk_size_words')}` words, "
            f"Overlap = `{best_cfg.get('overlap_words')}` words "
            f"(MRR: `{best_cfg.get('mrr')}`, nDCG@5: `{best_cfg.get('ndcg_at_5')}`)"
        )
        st.info(chunk_data.get("explanation", ""))

    if CHUNKING_CSV.is_file():
        df_chunking = pd.read_csv(CHUNKING_CSV)
        st.dataframe(df_chunking, use_container_width=True)
    else:
        st.info("Run `python scripts/run_chunking_experiments.py` to generate chunking experiment results.")

    st.divider()
    st.subheader("Retrieval Error Analysis")
    if ERROR_ANALYSIS_JSON.is_file():
        error_records = json.loads(ERROR_ANALYSIS_JSON.read_text(encoding="utf-8"))
        st.markdown(f"Found **{len(error_records)}** suboptimal or competitor queries in dense retrieval.")

        for idx, err in enumerate(error_records, 1):
            with st.expander(f"Case {idx}: \"{err['query']}\" (First Relevant at Rank #{err['actual_first_relevant_rank']})"):
                st.markdown(f"**Expected Relevant Document(s):** `{', '.join(err['expected_document_ids'])}`")
                st.markdown("**Top Retrieved Passages:**")
                for r in err["top_retrieved_documents"][:3]:
                    st.write(f"- Rank #{r['rank']}: **{r['document_id']}** ({r['section_title']}) — Score: `{r['similarity_score']}`")
                st.markdown(f"**Probable Cause:** {err['probable_cause']}")
                st.markdown(f"**Suggested Improvement:** {err['suggested_improvement']}")
    elif ERROR_ANALYSIS_MD.is_file():
        st.markdown(ERROR_ANALYSIS_MD.read_text(encoding="utf-8"))
    else:
        st.info("Run `python scripts/generate_error_analysis.py` to generate error analysis reports.")

# ----------------- Tab 4: Document Corpus -----------------
with tab_corpus:
    st.subheader("Corpus Explorer")
    st.markdown(f"The index contains **{len(corpus)} documents** divided into **{len(chunks)} chunks**.")

    corpus_view = [
        {
            "Title": doc.get("project_name", doc.get("source_file")),
            "Category": doc.get("project_category"),
            "Document Type": doc.get("document_type"),
            "Word Count": len(doc.get("text", "").split()),
            "Tags": ", ".join(doc.get("tags", [])),
        }
        for doc in corpus
    ]
    st.dataframe(pd.DataFrame(corpus_view), use_container_width=True)

    st.divider()
    st.subheader("Chunk Inspector")
    selected_doc = st.selectbox(
        "Select Document to view chunks:",
        options=sorted(list({c.get("project_name", "") for c in chunks})),
    )
    doc_chunks = [c for c in chunks if c.get("project_name") == selected_doc]
    for idx, c in enumerate(doc_chunks, 1):
        with st.expander(f"Chunk #{idx} — Section: {c.get('section_title', 'General')} ({c.get('word_count')} words)"):
            st.write(c["text"])

# ----------------- Tab 5: NLP Insights & Theory -----------------
with tab_insights:
    st.subheader("Natural Language Processing & Information Retrieval Theory")

    with st.expander("1. Lexical Retrieval (TF-IDF & BM25) vs. Dense Retrieval (SBERT)", expanded=True):
        st.markdown(
            """
            - **Lexical Search (TF-IDF & BM25Okapi)**:
              - Operates on exact keyword tokens using an inverted index.
              - **Term Frequency (TF)** penalizes word repetition with saturation:
                $$\\text{TF}_{\\text{BM25}} = \\frac{f(t, d) \\cdot (k_1 + 1)}{f(t, d) + k_1 \\cdot (1 - b + b \\cdot \\frac{|d|}{\\text{avgdl}})}$$
              - **Inverse Document Frequency (IDF)** downweights common corpus terms:
                $$\\text{IDF}(t) = \\ln\\left(\\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\\right)$$
              - **Limitation**: Suffers from the *vocabulary mismatch problem*. Cannot bridge synonyms (e.g., 'segmentation' and 'mask prediction').
            - **Dense Vector Search (Sentence-BERT)**:
              - Projects queries and passages into a 384-dimensional continuous latent space.
              - Maps semantic concepts regardless of surface-level phrasing.
              - **Limitation**: May fail on rare alphanumeric identifiers, exact product codes, or acronyms not well-represented in the pre-training corpus.
            """
        )

    with st.expander("2. Hybrid Retrieval: Convex Lexical-Dense Fusion"):
        st.markdown(
            """
            Hybrid retrieval combines the precision of BM25 on exact keywords with the generalization of dense embeddings:
            $$S_{\\text{hybrid}} = \\alpha \\cdot \\text{BM25}_{\\text{norm}} + (1 - \\alpha) \\cdot \\text{Dense}_{\\text{norm}}$$
            - **Min-Max Normalization**: Scales both score sets to $[0, 1]$ before linear combination:
              $$\\text{Score}_{\\text{norm}} = \\frac{S - \\min(S)}{\\max(S) - \\min(S) + \\epsilon}$$
            - In empirical testing on this corpus, Hybrid achieved **1.0000 MRR**, eliminating dense false positives.
            """
        )

    with st.expander("3. Two-Stage Retrieval: Bi-Encoder + Cross-Encoder Reranking"):
        st.markdown(
            """
            ```
            Entire Corpus (N chunks)
                    │
                    ▼  [Stage 1: Bi-Encoder (SBERT)]
            Candidate Shortlist (Top 10)  (Fast: O(N) dot products)
                    │
                    ▼  [Stage 2: Cross-Encoder (Joint Transformer)]
            Reranked Final Top K  (Accurate: Full token-to-token cross-attention)
            ```
            - **Bi-Encoder**: Encodes $q$ and $d$ separately $\\rightarrow$ misses token-level cross-interaction.
            - **Cross-Encoder**: Feeds `[CLS] + Query + [SEP] + Document + [SEP]` simultaneously into self-attention layers. Token interactions are computed directly, yielding state-of-the-art ranking fidelity.
            """
        )

    with st.expander("4. Evaluation Metrics Formulation"):
        st.markdown(
            """
            - **MRR (Mean Reciprocal Rank)**:
              $$\\text{MRR} = \\frac{1}{|Q|} \\sum_{i=1}^{|Q|} \\frac{1}{\\text{rank}_i}$$
            - **Precision@K**: Fraction of top-$K$ passages that are relevant:
              $$\\text{Precision@K} = \\frac{|\\text{Top } K \\cap \\text{Relevant}|}{K}$$
            - **Recall@K**: Fraction of all target relevant documents retrieved in top-$K$:
              $$\\text{Recall@K} = \\frac{|\\text{Top } K \\cap \\text{Relevant}|}{|\\text{Relevant}|}$$
            - **nDCG@K (Normalized Discounted Cumulative Gain)**:
              $$\\text{DCG@K} = \\sum_{i=1}^K \\frac{\\text{rel}_i}{\\log_2(i + 1)}, \\quad \\text{nDCG@K} = \\frac{\\text{DCG@K}}{\\text{IDCG@K}}$$
            """
        )

    with st.expander("5. Limitations and Responsible Use"):
        st.markdown(
            """
            - **Cosine Similarity as Calibration**: Cosine similarity is not a probability; scores must be evaluated relatively.
            - **Domain Shifts**: Technical terms not seen during pre-training may receive suboptimal representations.
            - **Corpus Privacy**: Public deployment should never index confidential, personally identifiable, or proprietary information.
            """
        )
