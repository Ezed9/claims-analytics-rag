import hashlib
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import AUDIT_DIR


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, default=str).encode("utf-8")


def write_audit_log(record: dict[str, Any], audit_dir: Path = AUDIT_DIR) -> Path:
    audit_id = str(uuid.uuid4())
    payload = {
        "audit_id": audit_id,
        "timestamp": datetime.now(UTC).isoformat(),
        **record,
    }
    payload["content_hash"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()

    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"{audit_id}.json"
    with path.open("x") as f:
        json.dump(payload, f, indent=2, default=str)
    path.chmod(0o444)
    return path


def verify_audit(path: Path) -> bool:
    payload = json.loads(path.read_text())
    recorded_hash = payload.pop("content_hash")
    recomputed_hash = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return recorded_hash == recomputed_hash


if __name__ == "__main__":
    written_path = write_audit_log({"claim_text": "example claim", "verdict": "MANUAL_REVIEW"})
    print(written_path, verify_audit(written_path))
