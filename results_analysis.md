
# Results Analysis (starter)

This document will contain:
- Summary statistics for each chunking strategy
- Retrieval metrics: Hit Rate, MRR, Precision@K
- Answer quality metrics: ROUGE-L, BLEU, Cosine Similarity
- Per-question failure analysis
- Recommendations and optimal configuration

## How to produce the analysis
1. Run `python evaluation.py` to create `test_results.json`.
2. Use the saved `test_results_<config>.json` files to compute aggregated metrics (mean hit rate, mean MRR, etc).
3. Fill this document with charts/tables and observations.

## Quick recommendations (expected)
- Medium chunks (500-600 chars) often balance retrieval context vs noise.
- Small chunks increase recall for factual snippets but may lose coherence.
- Large chunks reduce retrieval splits but risk returning too much irrelevant context to the LLM.
