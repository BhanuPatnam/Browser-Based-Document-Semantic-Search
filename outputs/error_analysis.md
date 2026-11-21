# Retrieval Error Analysis Report

This report identifies queries where dense retrieval did not rank a relevant target at rank 1, or where relevant documents ranked below the acceptable threshold.

**Total suboptimal queries identified:** 1

---
### Case 1: "What metrics evaluate this document semantic search engine?"
- **Expected Relevant Document(s):** `07-document-semantic-search-sentence-bert`
- **First Relevant Rank:** `#2`
- **Top Retrieved Passages:**
  - Rank #1: **03-cross-encoder-bi-encoder-ranking-system** (Portfolio value) — Score: `0.6149`
  - Rank #2: **07-document-semantic-search-sentence-bert** (Objective) — Score: `0.594`
  - Rank #3: **12-mlops-model-monitoring-evidently-prometheus** (Objective) — Score: `0.5164`
- **Probable Cause:** Semantic competition on evaluation terminology: Project 03 features dense discussions of evaluation metrics (MRR, nDCG, Precision@K, Cross-Encoder reranking scores), which causes the bi-encoder to assign it high cosine similarity alongside Project 07.
- **Suggested Improvement:** Implement two-stage Cross-Encoder reranking to resolve fine-grained distinction between project targets, or incorporate project-specific lexical filters.

---