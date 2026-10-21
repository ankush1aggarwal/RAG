# Per-experiment result sheets

Each file follows [`EXPERIMENT_SHEET_TEMPLATE.md`](EXPERIMENT_SHEET_TEMPLATE.md), including a required **§ Insights (interpretation)** after results (effect size, mechanism, caveats). Measured values live here and in the consolidated [`rag_experiment_results.md`](../rag_experiment_results.md). Cumulative ladder: [`eval_protocol_reference.md`](../eval_protocol_reference.md). The master blank structure is [`../../RAG_EXPERIMENT_REPORT_TEMPLATE.md`](../../RAG_EXPERIMENT_REPORT_TEMPLATE.md).

| ID | Sheet | Status | Notes |
|----|--------|--------|--------|
| **E0** | [E0_unlabeled_latency_overlap.md](E0_unlabeled_latency_overlap.md) | Complete | [`week3_retrieval_comparison.md`](../week3_retrieval_comparison.md) |
| **E1** | [E1_labeled_retrieval_baseline.md](E1_labeled_retrieval_baseline.md) | Complete | Exp 2/4 reports |
| **E1b** | [E1b_weighted_hybrid_fusion.md](E1b_weighted_hybrid_fusion.md) | Complete | Exp 3/4 |
| **E2** | [E2_cross_encoder_rerank.md](E2_cross_encoder_rerank.md) | Complete | [`week4_exp5_test_rerank.md`](../week4_exp5_test_rerank.md) |
| **E2a** | [E2a_rerank_error_analysis.md](E2a_rerank_error_analysis.md) | Complete | Error analysis JSON |
| **E3** | [E3_candidate_diversity.md](E3_candidate_diversity.md) | Complete | Control pool frozen for rerank evals |
| **E5** | [E5_rerank_candidate_depth.md](E5_rerank_candidate_depth.md) | **Skipped** | E2 rejected CE @50; no new reranker |
| **E6** | [E6_chunk_size.md](E6_chunk_size.md) | Complete | Winner: 200w (`fiqa_cs200`) |
| **E7** | [E7_chunk_overlap.md](E7_chunk_overlap.md) | Complete | Keep 20% @ 200w |
| **E8** | [E8_embedding_model.md](E8_embedding_model.md) | Complete | BGE-base wins |
| **E9** | [E9_paragraph_chunking.md](E9_paragraph_chunking.md) | Complete | Tied E8 — reject paragraph |
| **E10** | [E10_generation_citations.md](E10_generation_citations.md) | Planned | Week 6–7 |
| **E11** | — | Planned | E2E performance / concurrency (Week 8–9); sheet TBD |

**Week 5 order:** E3 → **E6–E8** (E7 after E6); optional **E9**. **E5** skipped (see E2 + E3).

**Program overview:** [`rag_experiment_program_report.md`](../rag_experiment_program_report.md)  
**Teaching summary (E0–E9):** [`rag_retrieval_lessons.md`](../rag_retrieval_lessons.md)
