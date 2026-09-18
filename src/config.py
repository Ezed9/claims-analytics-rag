import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATA_DIR: Path = REPO_ROOT / "data"
DOCS_DIR: Path = DATA_DIR / "raw_documents"
DB_PATH: Path = DATA_DIR / "claims_warehouse.duckdb"
CHROMA_DIR: Path = DATA_DIR / "chroma"
CHUNK_CONTEXT_CACHE_PATH: Path = DATA_DIR / "chunk_context_cache.json"
SQL_DIR: Path = REPO_ROOT / "sql"
SCHEMA_DDL_PATH: Path = SQL_DIR / "01_schema_ddl.sql"
VELOCITY_SQL_PATH: Path = SQL_DIR / "02_velocity_anomaly_queries.sql"
AUDIT_DIR: Path = REPO_ROOT / "audit_logs"
EVAL_DIR: Path = REPO_ROOT / "eval"
GOLDEN_DATASET_PATH: Path = EVAL_DIR / "golden_dataset.json"
EVAL_RESULTS_PATH: Path = EVAL_DIR / "results.json"

RANDOM_SEED: int = 42
NUM_CLAIMS: int = 120_000
FRAUD_RING_RATE: float = 0.03

EMBED_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"

RRF_K: int = 60
TOP_K_CANDIDATES: int = 15
RERANK_TOP_N: int = 3
RERANK_THRESHOLD: float = 0.60
REPAIR_THRESHOLD: float = 0.65

GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
ANTHROPIC_MODEL_NAME: str = "claude-sonnet-5"


def resolve_provider() -> str:
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit in {"gemini", "anthropic"}:
        return explicit
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def has_active_llm() -> bool:
    provider = resolve_provider()
    if provider == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY"))
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return False


AUDIT_DIR.mkdir(parents=True, exist_ok=True)
