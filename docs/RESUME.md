# Resume and interview notes

## Bullets (pick 3)
- Built a warranty-claims analytics platform: DuckDB star schema with 120k synthetic claims and window-function SQL that scores claim-velocity fraud risk into ELEVATED / MODERATE / LOW tiers.
- Designed a triage agent that keeps the LLM out of the decision path: the LLM only extracts entities; coverage, deductible, waiting period and repair-vs-replace verdicts come from SQL rules and a cost model, with SHA-256 hashed, write-protected audit logs and an MCP tool server.
- Implemented hybrid RAG (BM25 + dense with reciprocal-rank fusion, cross-encoder reranking, Contextual Retrieval) and evaluated it on a 25-case golden set: 88% verdict accuracy; a Wilcoxon signed-rank test showed no significant gain over a dense-only baseline (p=0.745), reported as a negative result.
- Shipped a 3-tab Streamlit dashboard and a pytest suite; found and fixed a nondeterministic window-function ordering bug that made fraud-tier counts drift between runs.

## 30-second explanation
"It triages phone-insurance claims. An LLM reads the claim text, but every actual decision - is it covered, repair or replace - comes from SQL rules and a cost formula, so it's auditable, and every run is logged with a tamper-evident hash. I also measured whether a fancier retrieval method helped and found it didn't, so I reported that honestly."

## Likely questions
- Why not let the LLM decide? Auditability, reproducibility, works with no API key (rule-based fallback).
- Why did hybrid retrieval not win? ~41-chunk corpus, queries lack the carrier, situating prompt saw no section heading; next step is a larger held-out set.
- Limitations? Synthetic data, 25-case eval, SOP covers iPhone 15 only (others go to manual review), Anthropic path not live-tested.
