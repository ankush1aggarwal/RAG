# Per-experiment result sheets

Each file follows [`EXPERIMENT_SHEET_TEMPLATE.md`](EXPERIMENT_SHEET_TEMPLATE.md). Measured values live here and in the consolidated [`rag_experiment_results.md`](../rag_experiment_results.md). The master blank structure is [`../../RAG_EXPERIMENT_REPORT_TEMPLATE.md`](../../RAG_EXPERIMENT_REPORT_TEMPLATE.md).

| ID | Sheet | Status | Notes |
|----|--------|--------|--------|
| **E0** | [E0_unlabeled_latency_overlap.md](E0_unlabeled_latency_overlap.md) | Complete | [`week3_retrieval_comparison.md`](../week3_retrieval_comparison.md) |
| **E1** | [E1_labeled_retrieval_baseline.md](E1_labeled_retrieval_baseline.md) | Complete | Exp 2/4 reports |
| **E1b** | [E1b_weighted_hybrid_fusion.md](E1b_weighted_hybrid_fusion.md) | Complete | Exp 3/4 |
| **E2** | [E2_cross_encoder_rerank.md](E2_cross_encoder_rerank.md) | Complete | [`week4_exp5_test_rerank.md`](../week4_exp5_test_rerank.md) |
| **E2a** | [E2a_rerank_error_analysis.md](E2a_rerank_error_analysis.md) | Complete | Error analysis JSON |
| **E3** | [E3_candidate_diversity.md](E3_candidate_diversity.md) | Planned | Before reranker comparisons |
| **E4** | [E4_jev_rerank.md](E4_jev_rerank.md) | Planned | Jev `rerank()` only; hosted API audit |
| **E4b** | [E4b_jev_relevance_filtering.md](E4b_jev_relevance_filtering.md) | Optional | After E4 if Jev promising |
| **E5** | [E5_rerank_candidate_depth.md](E5_rerank_candidate_depth.md) | Planned | After E4; skip if no rerank wins |
| **E6** | [E6_chunk_size.md](E6_chunk_size.md) | Planned | Week 5 |
| **E7** | [E7_chunk_overlap.md](E7_chunk_overlap.md) | Planned | After E6 |
| **E8** | [E8_embedding_model.md](E8_embedding_model.md) | Planned | Week 5 |
| **E9** | [E9_hybrid_rrf_sensitivity.md](E9_hybrid_rrf_sensitivity.md) | Optional | Stretch |
| **E10** | [E10_generation_citations.md](E10_generation_citations.md) | Planned | Week 6–7 |
| **E11** | — | Planned | E2E performance / concurrency (Week 8–9); sheet TBD |

**Week 5 order:** E3 → **E4** (Jev ranking) → optional **E4b** → **E5** (depth) → E6–E8.

**Program overview:** [`rag_experiment_program_report.md`](../rag_experiment_program_report.md)
