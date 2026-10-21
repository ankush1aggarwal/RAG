# E9 — Paragraph-aware chunking (vs E8 fixed 200w)

| Field | Value |
|-------|--------|
| **ID** | E9 |
| **Status** | **Complete** |
| **Primary variable** | Chunking: fixed 200w vs paragraph-aware (max 200w, 20% sub-split) |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §5 |

---

**Embedding (fixed):** `BAAI/bge-base-en-v1.5` (same as E8).

| Arm | Collection | Chunking |
|-----|------------|----------|
| Control (E8) | `fiqa_cs200_bge_base_en_v1_5` | Fixed 200 words / 20% overlap |
| Treatment | `fiqa_para_bge_base_en_v1_5` | Paragraph merge ≤200w; overlap on sub-split |

**Runner:** `src/scripts/run_week5_e9.py` · **Probe:** `week5_e9_probe.json`

**Normalization:** Option A — preserve `\n\n` paragraph boundaries (`normalize_whitespace_preserve_paragraphs`).

**Production:** E8 frozen index (`fiqa_cs200_bge_base_en_v1_5`); paragraph arm rejected.

## Results

| Arm | NDCG@10 (val) | NDCG@10 (test) | Points | Decision |
|-----|--------------:|---------------:|-------:|----------|
| Fixed 200w (E8) | **0.418** | **0.394** | 72,851 | **Keep** |
| Paragraph | **0.418** | — | 72,851 | **Reject** |

Test run for validation winner only (fixed 200w).

## Insights (interpretation)

- **Effect size:** **0** ΔNDCG@10 / MRR@10 / Recall@10 on validation; test **0.394** unchanged vs E8.
- **Mechanism:** Identical **72,851** chunk count → same chunk texts and embeddings; FiQA has few preserved paragraph breaks in practice.
- **Interaction:** Built on E8 BGE; does not change frozen profile (`fiqa_cs200_bge_base_en_v1_5`).
- **Caveats:** Tied winner defaulted to fixed arm; paragraph index **removed** from disk after eval.

## Decision

**Reject** paragraph collection for production/eval default; retain E8 chunking on `fiqa_cs200_bge_base_en_v1_5`.
