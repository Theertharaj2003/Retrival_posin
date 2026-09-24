# Phase completion and verification checklist

| Phase | Implementation | Verification |
|---|---|---|
| 1 Environment | config, seed-ready structure, GPU detector | compile + pytest |
| 2 Ingestion | txt/md/json/pdf readers + chunking | ingestion unit path |
| 3 Dense retrieval | SentenceTransformer + fallback | pipeline smoke test |
| 4 Hybrid retrieval | BM25 + dense fusion + reranker | pipeline smoke test |
| 5 Poisoning | factual/contradictory/indirect injection | poison-label test |
| 6 Detection | risk rules + DeBERTa hook | poison detection path |
| 7 Trust | configurable weighted trust | pipeline smoke test |
| 8 Verification | NLI + lexical fallback | pipeline smoke test |
| 9 Attribution | claim/evidence matching | attribution output |
| 10 Self-healing | quarantine + exclude + re-retrieve | healing unit test |
| 11 Generation | safe extractive fallback + local model hook | answer smoke test |
| 12 Evaluation | ASR, recovery, attribution and experiment logs | experiment runner |
| 13 UI/reproducibility | Streamlit, CSV/JSON, scripts, tests | compile + pytest + demo |

## Double verification protocol
Run the static/smoke checks, then run the experiment twice with the same configuration. If results differ, inspect model nondeterminism and environment state before using the results for research reporting.
