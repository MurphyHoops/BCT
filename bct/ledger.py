from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

class JsonlLedger:
    """Append-only audit ledger (jsonl)."""
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, record: Dict[str, Any]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
