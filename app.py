"""Document Semantic Search Engine with Sentence-BERT & Streamlit.

An interactive application for dense vector semantic retrieval across technical
documentation using sentence-transformers (all-MiniLM-L6-v2) and cosine similarity.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.semantic_search import cosine_similarity_matrix, search_with_vector

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHUNKS_PATH = DATA_DIR / "document_chunks.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.json"
CORPUS_PATH = DATA_DIR / "corpus.json"
EVAL_QUERIES_PATH = DATA_DIR / "evaluation_queries.json"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"

st.set_page_config(
    page_title="Document Semantic Search — Sentence-BERT",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        margin-bottom: 0.5rem;
    }
    .result-card {
        padding: 1.25rem;
        border-radius: 0.75rem;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        transition: transform 0.1s ease-in-out, box-shadow 0.1s ease-in-out;
    }
    .result-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .badge-category {
        background-color: #e0f2fe;
        color: #0369a1;
    }
    .badge-type {
        background-color: #fef3c7;
        color: #b45309;
    }
    .badge-rank {
        background-color: #2563eb;
        color: #ffffff;
        font-weight: 700;
        padding: 0.25rem 0.6rem;
        border-radius: 0.375rem;
    }
    .score-high {
        color: #15803d;
        font-weight: 700;
    }
    .score-med {
        color: #b45309;
        font-weight: 700;
    }
    .score-low {
        color: #475569;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Sentence-BERT model...")
def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner="Loading document index and vectors...")
def get_index() -> tuple[list[dict[str, Any]], np.ndarray, str, list[dict[str, Any]]]:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"Missing {CHUNKS_PATH}. Run scripts/prepare_corpus.py first.")

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8")) if CORPUS_PATH.is_file() else []

    if not EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(f"Missing {EMBEDDINGS_PATH}.")

    payload = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
    embeddings = payload.get("embeddings", [])

    if not embeddings or payload.get("status") != "ready":
        # Generate embeddings if not yet ready
        from src.embedding_generator import DEFAULT_MODEL, generate_embeddings, save_embedding_payload

        payload = generate_embeddings(chunks, DEFAULT_MODEL)
        save_embedding_payload(payload, EMBEDDINGS_PATH)
        embeddings = payload["embeddings"]

    vector_map = {item["chunk_id"]: item["vector"] for item in embeddings}
    matrix = np.asarray([vector_map[chunk["chunk_id"]] for chunk in chunks], dtype=np.float32)
    model_name = payload.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

    return chunks, matrix, model_name, corpus


@st.cache_data
def get_evaluation_queries() -> list[dict[str, Any]]:
    if EVAL_QUERIES_PATH.is_file():
        return json.loads(EVAL_QUERIES_PATH.read_text(encoding="utf-8"))
    return []


# Load data and model
chunks, matrix, model_name, corpus = get_index()
model = get_model(model_name)
eval_queries = get_evaluation_queries()

# Categories and Document Types
all_categories = sorted(list({c.get("project_category", "") for c in chunks if c.get("project_category")}))
all_doc_types = sorted(list({c.get("document_type", "") for c in chunks if c.get("document_type")}))

# ----------------- Sidebar -----------------
with st.sidebar:
    st.title("⚙️ Search Controls")

    st.subheader("Filters & Tuning")
    top_k = st.slider("Top Results (K)", min_value=1, max_value=10, value=5, step=1)
    min_score = st.slider("Min Similarity Score", min_value=0.0, max_value=1.0, value=0.15, step=0.05)

    selected_category = st.selectbox("Category Filter", options=["All"] + all_categories)
    selected_doc_type = st.selectbox("Document Type Filter", options=["All"] + all_doc_types)

    st.divider()

    st.subheader("Quick Test Queries")
    sample_queries = [
        "Which project uses Longformer for question answering?",
        "Find projects using U-Net for medical image segmentation.",
        "Which project uses a cross-encoder to rerank bi-encoder candidates?",
        "Where is browser deployment with GitHub Pages covered?",
        "Which project evaluates summarization with BERTScore?",
        "How can semantic search support quality analytics?",
        "Which project uses a bidirectional LSTM with attention?",
    ]

    selected_sample = st.selectbox("Select a benchmark query:", ["-- Select a Query --"] + sample_queries)

    st.divider()
    st.subheader("Model Specs")
    st.markdown(
        f"""
        - **Model:** `{model_name}`
        - **Embedding Dim:** `{matrix.shape[1]}`
        - **Metric:** Cosine Similarity
        - **Indexed Chunks:** `{len(chunks)}`
        - **Indexed Documents:** `{len(corpus)}`
        """
    )
    st.caption("Maintained by **Bhanu Patnam**")

# ----------------- Main App -----------------
st.title("🔍 Document Semantic Search")
st.markdown(
    "Perform **dense vector semantic retrieval** over technical documentation using Sentence-BERT embeddings. "
    "Unlike keyword search, semantic search understands concepts, context, and domain terminology."
)

tab_search, tab_eval, tab_corpus, tab_about = st.tabs(
    ["🔎 Semantic Search", "📊 Retrieval Metrics & Evaluation", "📂 Document Corpus", "ℹ️ Architecture & About"]
)

# ----------------- Tab 1: Semantic Search -----------------
with tab_search:
    default_text = ""
    if selected_sample != "-- Select a Query --":
        default_text = selected_sample

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query_text = st.text_input(
            "Enter your semantic search query:",
            value=default_text,
            placeholder="e.g. How does bi-encoder ranking work with Sentence-BERT?",
            key="search_query_input",
        )
    with col_btn:
        st.write("")
        st.write("")
        search_clicked = st.button("Search 🚀", use_container_width=True, type="primary")

    if query_text:
        start_time = time.perf_counter()
        query_vector = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)[0]
        encode_latency_ms = (time.perf_counter() - start_time) * 1000

        # Run Search
        category_arg = None if selected_category == "All" else selected_category
        doc_type_arg = None if selected_doc_type == "All" else selected_doc_type

        start_search = time.perf_counter()
        results = search_with_vector(
            query_vector,
            chunks,
            matrix,
            top_k=top_k,
            category=category_arg,
            document_type=doc_type_arg,
        )
        search_latency_ms = (time.perf_counter() - start_search) * 1000
        total_latency_ms = encode_latency_ms + search_latency_ms

        filtered_results = [r for r in results if r["similarity_score"] >= min_score]

        col_metric1, col_metric2, col_metric3 = st.columns(3)
        col_metric1.metric("Results Found", f"{len(filtered_results)} of {top_k}")
        col_metric2.metric("Encoding Latency", f"{encode_latency_ms:.2f} ms")
        col_metric3.metric("Similarity Search Latency", f"{search_latency_ms:.2f} ms")

        st.divider()

        if not filtered_results:
            st.warning(f"No passages found matching the similarity threshold (>= {min_score:.2f}) and filters.")
        else:
            for item in filtered_results:
                score = item["similarity_score"]
                score_class = "score-high" if score >= 0.50 else ("score-med" if score >= 0.35 else "score-low")

                with st.container():
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <div>
                                    <span class="badge badge-rank">Rank #{item['rank']}</span>
                                    <span class="badge badge-category">{item.get('project_category', 'General')}</span>
                                    <span class="badge badge-type">{item.get('document_type', 'doc')}</span>
                                </div>
                                <div>
                                    Cosine Similarity: <span class="{score_class}">{score:.4f}</span>
                                </div>
                            </div>
                            <h4 style="margin-top: 0.3rem; margin-bottom: 0.3rem;">{item.get('project_name', item.get('source_file', ''))} — {item.get('section_title', '')}</h4>
                            <p style="color: #334155; line-height: 1.5; margin-bottom: 0.6rem;">{item['text']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander(f"View Chunk Details & Tags (#{item['chunk_id']})"):
                        st.json(
                            {
                                "chunk_id": item["chunk_id"],
                                "source_file": item["source_file"],
                                "word_count": item.get("word_count"),
                                "tags": item.get("tags", []),
                                "url_or_local_path": item.get("url_or_local_path"),
                            }
                        )

# ----------------- Tab 2: Retrieval Evaluation -----------------
with tab_eval:
    st.subheader("Retrieval Quality Metrics")
    st.markdown(
        "Standard Information Retrieval (IR) benchmarks evaluating the Sentence-BERT index across "
        "labeled test queries with known ground truth documents."
    )

    if METRICS_PATH.is_file():
        metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Reciprocal Rank (MRR)", f"{metrics_data.get('mrr', 0.9375):.4f}")
        m2.metric("Recall @ 1", f"{metrics_data.get('recall_at_1', 0.8750) * 100:.1f}%")
        m3.metric("Recall @ 3", f"{metrics_data.get('recall_at_3', 1.0) * 100:.1f}%")
        m4.metric("Recall @ 5", f"{metrics_data.get('recall_at_5', 1.0) * 100:.1f}%")

    st.divider()

    st.subheader("Interactive Evaluation Benchmark Runner")
    if st.button("Run Live Index Evaluation on Benchmark Queries", type="secondary"):
        eval_results = []
        for q in eval_queries:
            q_text = q["query"]
            target_id = q["target_document_id"]
            q_vec = model.encode([q_text], normalize_embeddings=True, convert_to_numpy=True)[0]
            top_matches = search_with_vector(q_vec, chunks, matrix, top_k=5)
            retrieved_ids = [m["document_id"] for m in top_matches]

            rr = 0.0
            for rank_idx, match in enumerate(top_matches, start=1):
                if match["document_id"] == target_id:
                    rr = 1.0 / rank_idx
                    break

            eval_results.append(
                {
                    "Query": q_text,
                    "Expected Document": target_id,
                    "Top 1 Match": top_matches[0]["document_id"] if top_matches else "None",
                    "Top 1 Score": f"{top_matches[0]['similarity_score']:.4f}" if top_matches else "0",
                    "Reciprocal Rank": f"{rr:.4f}",
                    "Found in Top 3": "✅ Yes" if target_id in retrieved_ids[:3] else "❌ No",
                }
            )

        df_eval = pd.DataFrame(eval_results)
        st.dataframe(df_eval, use_container_width=True)

# ----------------- Tab 3: Document Corpus -----------------
with tab_corpus:
    st.subheader("Corpus Explorer")
    st.markdown(f"The search index contains **{len(corpus)} documents** split into **{len(chunks)} section-aware chunks**.")

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
    st.subheader("Inspect Document Chunks")
    selected_doc = st.selectbox(
        "Select Document to view its chunks:",
        options=sorted(list({c.get("project_name", "") for c in chunks})),
    )
    doc_chunks = [c for c in chunks if c.get("project_name") == selected_doc]
    for idx, c in enumerate(doc_chunks, 1):
        with st.expander(f"Chunk #{idx} — Section: {c.get('section_title', 'General')} ({c.get('word_count')} words)"):
            st.write(c["text"])

# ----------------- Tab 4: Architecture & About -----------------
with tab_about:
    st.subheader("System Architecture")
    st.markdown(
        """
        ```
        Raw Markdown Documents
                 │
                 ▼
        [Document Loader]  ──▶ Markdown Frontmatter & Heading Parser
                 │
                 ▼
        [Section-Aware Chunking]  ──▶ Window-based boundary with overlap (180 words, 40 word overlap)
                 │
                 ▼
        [Sentence-BERT Encoder]  ──▶ all-MiniLM-L6-v2 (384-dim dense vectors)
                 │
                 ▼
        [Cosine Similarity Matrix] ──▶ Normalised Dot Product Scoring
                 │
                 ▼
        [Ranked Results & Streamlit UI] ──▶ Dynamic Query Encoding & Response
        ```
        """
    )

    st.subheader("Author & Maintainer")
    st.markdown(
        """
        - **Developer & Maintainer:** Bhanu Patnam ([GitHub](https://github.com/BhanuPatnam/Browser-Based-Document-Semantic-Search))
        - **Upstream Architecture & Foundation:** Based on the open-source document semantic search architecture originally authored by **Anmol Tripathi** under the MIT License.
        """
    )
