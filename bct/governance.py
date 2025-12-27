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
        self._risks = np.zeros(16, dtype=float)
        self._cooldowns = np.zeros(16, dtype=np.int64)
        self._reputations = np.full(16, 0.5, dtype=float)
        self._size = 0
        self._step = 0

    def _grow(self, new_cap: int) -> None:
        """Resize storage arrays to at least new_cap elements."""
        cap = max(new_cap, len(self._risks))
        if cap == len(self._risks):
            return
        new_r = np.zeros(cap, dtype=float)
        new_cd = np.zeros(cap, dtype=np.int64)
        new_rep = np.full(cap, 0.5, dtype=float)
        if self._size:
            new_r[: self._size] = self._risks[: self._size]
            new_cd[: self._size] = self._cooldowns[: self._size]
            new_rep[: self._size] = self._reputations[: self._size]
        self._risks = new_r
        self._cooldowns = new_cd
        self._reputations = new_rep

    def _advance_step(self, current_step: int | None) -> int:
        """Return the current step counter, optionally advancing to a provided step."""
        if current_step is None:
            self._step += 1
        else:
            self._step = max(self._step, int(current_step))
        return self._step

    def tick(self, current_step: int | None = None) -> int:
        """Advance internal step counter (useful when no safety checks are run)."""
        return self._advance_step(current_step)

    def _ensure_agent(self, agent_id: int, default_rep: float = 0.5) -> int:
        aid = int(agent_id)
        if aid in self._idx_map:
            return self._idx_map[aid]
        idx = self._size
        if idx >= len(self._risks):
            self._grow(max(1, len(self._risks) * 2))
        self._idx_map[aid] = idx
        self._risks[idx] = 0.0
        self._cooldowns[idx] = 0
        self._reputations[idx] = float(default_rep)
        self._size += 1
        return idx

    def _theta(self, rho_b: float) -> float:
        """Dynamic tolerance θ(t): budget lower => tolerance lower."""
        theta_lo, theta_hi = 0.2, 0.7
        rho_b = max(0.0, min(1.0, float(rho_b)))
        return float(theta_lo + (theta_hi - theta_lo) * rho_b)

    def check_safety(self, agent_id: int, risk_val: float, hard_veto: bool, rho_b: float, current_step: int | None = None) -> SafetyDecision:
        """Update EWMA and return isolation decision."""
        step = self._advance_step(current_step)
        idx = self._ensure_agent(agent_id)
        risk_val = max(0.0, min(1.0, float(risk_val)))

        # Update EWMA risk
        z_prev = float(self._risks[idx])
        z = (1.0 - self.alpha) * z_prev + self.alpha * risk_val
        self._risks[idx] = z

        # Maintain cooldown if active
        cd_exp = int(self._cooldowns[idx])
        if cd_exp > step:
            remaining = cd_exp - step
            return SafetyDecision(True, f"cooldown({remaining})")

        # Fresh decision
        theta_t = self._theta(rho_b)
        isolated = bool(hard_veto or (z > theta_t))

        if isolated:
            self._cooldowns[idx] = step + self.cooldown_period
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

    def batch_check_safety(
        self,
        agent_ids: Iterable[int],
        risk_vals: Iterable[float],
        hard_veto_mask: Iterable[bool] | None,
        rho_b: float,
        return_decisions: bool = True,
        current_step: int | None = None,
    ):
        """Vectorized safety check; returns (isolated_mask, decisions?)."""
        ids = list(agent_ids)
        if not ids:
            return np.zeros(0, dtype=bool), [] if return_decisions else None

        step = self._advance_step(current_step)
        risks = np.clip(np.asarray(list(risk_vals), dtype=float), 0.0, 1.0)
        if hard_veto_mask is None:
            hard = np.zeros(len(ids), dtype=bool)
        else:
            hard = np.asarray(list(hard_veto_mask), dtype=bool)
            if hard.shape[0] != len(ids):
                raise ValueError("hard_veto_mask length must match agent_ids length")
        idxs = np.array([self._ensure_agent(aid) for aid in ids], dtype=int)

        # Update EWMA risk
        z_prev = self._risks[idxs]
        z = (1.0 - self.alpha) * z_prev + self.alpha * risks
        self._risks[idxs] = z

        cds = self._cooldowns[idxs]
        active_cd = cds > step

        theta_t = self._theta(rho_b)
        isolated = hard | active_cd | (z > theta_t)

        # Start cooldown for newly isolated (not already in cooldown and not hard)
        new_iso = isolated & (~active_cd)
        if new_iso.any():
            self._cooldowns[idxs[new_iso]] = step + self.cooldown_period

        if not return_decisions:
            return isolated, None

        decisions = []
        updated_cd = self._cooldowns[idxs]
        for i, iso in enumerate(isolated):
            if hard[i]:
                reason = "hard_veto"
            elif active_cd[i]:
                remaining = max(0, int(updated_cd[i]) - step)
                reason = f"cooldown({remaining})"
            elif iso:
                reason = f"ewma({z[i]:.3f})>theta({theta_t:.3f})"
            else:
                reason = "ok"
            decisions.append(SafetyDecision(bool(iso), reason))
        return isolated, decisions
