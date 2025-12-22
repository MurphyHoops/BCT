from __future__ import annotations
import math
from dataclasses import dataclass

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

        self.risks = {}       # agent_id -> EWMA risk z_i
        self.cooldowns = {}   # agent_id -> remaining cooldown
        self.reputations = {} # agent_id -> reputation R_i

    def _theta(self, rho_b: float) -> float:
        """Dynamic tolerance θ(t): budget lower => tolerance lower."""
        theta_lo, theta_hi = 0.2, 0.7
        rho_b = max(0.0, min(1.0, float(rho_b)))
        return float(theta_lo + (theta_hi - theta_lo) * rho_b)

    def check_safety(self, agent_id: int, risk_val: float, hard_veto: bool, rho_b: float) -> SafetyDecision:
        """Update EWMA and return isolation decision."""
        aid = int(agent_id)
        risk_val = max(0.0, min(1.0, float(risk_val)))

        # Update EWMA risk
        z_prev = float(self.risks.get(aid, 0.0))
        z = (1.0 - self.alpha) * z_prev + self.alpha * risk_val
        self.risks[aid] = z

        # Maintain cooldown if active
        cd = int(self.cooldowns.get(aid, 0))
        if cd > 0:
            self.cooldowns[aid] = cd - 1
            return SafetyDecision(True, f"cooldown({cd})")

        # Fresh decision
        theta_t = self._theta(rho_b)
        isolated = bool(hard_veto or (z > theta_t))

        if isolated:
            self.cooldowns[aid] = self.cooldown_period
            reason = "hard_veto" if hard_veto else f"ewma({z:.3f})>theta({theta_t:.3f})"
            return SafetyDecision(True, reason)

        return SafetyDecision(False, "ok")

    def reputation(self, agent_id: int, default: float = 0.5) -> float:
        return float(self.reputations.get(int(agent_id), default))

    def update_reputation(self, agent_id: int, ig: float, risk: float, tax: float):
        """R_{t+1}=(1-η)R_t + η(IG - risk - tax)."""
        aid = int(agent_id)
        ig = float(ig); risk = float(risk); tax = float(tax)
        r = float(self.reputations.get(aid, 0.5))
        r = (1.0 - self.eta) * r + self.eta * (ig - risk - tax)
        self.reputations[aid] = max(0.0, r)
