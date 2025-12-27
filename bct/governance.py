from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class SafetyDecision:
    isolated: bool
    reason: str


class SafetyGovernor:
    """EWMA circuit breaker + cooldown; reputation ledger."""

    def __init__(self, alpha: float = 0.3, eta: float = 0.1, cooldown_period: int = 5):
        self.alpha = float(alpha)  # EWMA smoothing
        self.eta = float(eta)      # reputation update rate
        self.cooldown_period = int(cooldown_period)

        # agent_id -> index mapping for compact arrays
        self._idx_map = {}
        self._risks = np.zeros(0, dtype=float)
        self._cooldowns = np.zeros(0, dtype=np.int64)
        self._reputations = np.zeros(0, dtype=float)

    def _ensure_agent(self, agent_id: int, default_rep: float = 0.5) -> int:
        aid = int(agent_id)
        if aid in self._idx_map:
            return self._idx_map[aid]
        idx = len(self._idx_map)
        self._idx_map[aid] = idx
        self._risks = np.append(self._risks, 0.0)
        self._cooldowns = np.append(self._cooldowns, 0)
        self._reputations = np.append(self._reputations, float(default_rep))
        return idx

    def _theta(self, rho_b: float) -> float:
        """Dynamic tolerance θ(t): budget lower => tolerance lower."""
        theta_lo, theta_hi = 0.2, 0.7
        rho_b = max(0.0, min(1.0, float(rho_b)))
        return float(theta_lo + (theta_hi - theta_lo) * rho_b)

    def check_safety(self, agent_id: int, risk_val: float, hard_veto: bool, rho_b: float) -> SafetyDecision:
        """Update EWMA and return isolation decision."""
        idx = self._ensure_agent(agent_id)
        risk_val = max(0.0, min(1.0, float(risk_val)))

        # Update EWMA risk
        z_prev = float(self._risks[idx])
        z = (1.0 - self.alpha) * z_prev + self.alpha * risk_val
        self._risks[idx] = z

        # Maintain cooldown if active
        cd = int(self._cooldowns[idx])
        if cd > 0:
            self._cooldowns[idx] = cd - 1
            return SafetyDecision(True, f"cooldown({cd})")

        # Fresh decision
        theta_t = self._theta(rho_b)
        isolated = bool(hard_veto or (z > theta_t))

        if isolated:
            self._cooldowns[idx] = self.cooldown_period
            reason = "hard_veto" if hard_veto else f"ewma({z:.3f})>theta({theta_t:.3f})"
            return SafetyDecision(True, reason)

        return SafetyDecision(False, "ok")

    def reputation(self, agent_id: int, default: float = 0.5) -> float:
        idx = self._ensure_agent(agent_id, default_rep=default)
        return float(self._reputations[idx])

    def reputations(self, agent_ids: Iterable[int], default: float = 0.5) -> np.ndarray:
        ids = list(agent_ids)
        idxs: List[int] = [self._ensure_agent(aid, default_rep=default) for aid in ids]
        return self._reputations[np.array(idxs, dtype=int)]

    def update_reputation(self, agent_id: int, ig: float, risk: float, tax: float):
        """R_{t+1}=(1-η)R_t + η(IG - risk - tax)."""
        idx = self._ensure_agent(agent_id)
        ig = float(ig); risk = float(risk); tax = float(tax)
        r = float(self._reputations[idx])
        r = (1.0 - self.eta) * r + self.eta * (ig - risk - tax)
        self._reputations[idx] = max(0.0, r)

    def batch_update(self, agent_ids: Iterable[int], ig: Iterable[float], risk: Iterable[float], tax: Iterable[float]):
        """Vectorized reputation updates for multiple agents."""
        ids = list(agent_ids)
        if not ids:
            return
        ig_arr = np.asarray(list(ig), dtype=float)
        risk_arr = np.asarray(list(risk), dtype=float)
        tax_arr = np.asarray(list(tax), dtype=float)
        idxs = np.array([self._ensure_agent(aid) for aid in ids], dtype=int)

        current = self._reputations[idxs]
        updated = (1.0 - self.eta) * current + self.eta * (ig_arr - risk_arr - tax_arr)
        self._reputations[idxs] = np.maximum(0.0, updated)
