# Ingest manifests for deployed API (dense `/ready`)

The API resolves **`{BM25_INDEX_DIR}/{collection}.manifest.json`** (same basename as the BM25 pickle path, but the **`.pkl` file is not required** when `API_ENFORCE_FROZEN_RETRIEVAL=true` — only the manifest is used for Qdrant alignment checks).

## Refresh after re-ingest

Copy the manifest from your local ingest output (not hand-edited):

```powershell
Copy-Item data\bm25_indexes\fiqa_cs200_bge_base_en_v1_5.manifest.json deploy\index_metadata\
```

```bash
cp data/bm25_indexes/fiqa_cs200_bge_base_en_v1_5.manifest.json deploy/index_metadata/
```

Commit the updated JSON, then rebuild and redeploy the container image.

Cloud Run / container default: `BM25_INDEX_DIR=deploy/index_metadata` (see `Dockerfile`, [`docs/cloud_run.md`](../../docs/cloud_run.md)).
