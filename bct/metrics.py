from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Tuple

BANNED_IMPORTS = ("os", "subprocess", "socket", "requests")
BANNED_PATTERNS = (
    r"while\s+True\s*:",
    r"for\s*\(\s*;\s*;\s*\)",
    r"import\s+ctypes",
)

@dataclass
class RiskResult:
    risk: float
    hard_veto: bool
    reason: str

def risk_analysis(patch_code: str) -> RiskResult:
    """Static risk analysis. Hard veto on obviously dangerous patterns."""
    code = patch_code or ""
    # banned imports
    for mod in BANNED_IMPORTS:
        if re.search(rf"^\s*import\s+{re.escape(mod)}\b", code, flags=re.M):
            return RiskResult(1.0, True, f"banned_import:{mod}")
        if re.search(rf"^\s*from\s+{re.escape(mod)}\b", code, flags=re.M):
            return RiskResult(1.0, True, f"banned_from:{mod}")

    for pat in BANNED_PATTERNS:
        if re.search(pat, code, flags=re.M):
            return RiskResult(1.0, True, f"pattern:{pat}")

    # heuristic soft risk: long code, deep nesting, recursion
    lines = code.count("\n") + 1
    soft = min(1.0, lines / 200.0)
    if "def" in code and "return" not in code:
        soft = min(1.0, soft + 0.2)
    return RiskResult(max(0.05, soft), False, "heuristic")

def tax_calculation(patch_code: str, history: List[str], lambda_dup: float = 0.8) -> float:
    """Pigouvian externality tax: penalize redundancy via fast token similarity."""
    if not history:
        return 0.0
    code = patch_code or ""

    def _token_set(s: str, limit: int = 512) -> set[str]:
        tokens = re.findall(r"\w+", s.lower())
        return set(tokens[:limit])  # cap size to bound cost

    curr_tokens = _token_set(code)
    if not curr_tokens:
        return 0.0

    k = min(5, len(history))
    sims = []
    for prev in history[-k:]:
        prev_tokens = _token_set(prev)
        if not prev_tokens:
            continue
        inter = len(curr_tokens & prev_tokens)
        union = len(curr_tokens | prev_tokens)
        if union == 0:
            continue
        sims.append(inter / union)

    sim = max(sims) if sims else 0.0
    return float(lambda_dup * sim)
