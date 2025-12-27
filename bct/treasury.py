from __future__ import annotations

from typing import Optional

import numpy as np

def softmax_stable(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (log-sum-exp trick via max-shift)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    m = np.max(x)
    ex = np.exp(x - m)
    s = np.sum(ex)
    if s <= 0:
        # fallback: uniform
        return np.ones_like(x) / len(x)
    return ex / s

def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))

class BCTTreasury:
    """BCT Treasury implementing Φ(t), β(t), and softmax allocation."""

    def __init__(self, b0: int, rho_min: float = 0.1, phi_c: float = 2.5,
                 k_sig: float = 3.0, r_max: float = 1.0, eps: float = 1e-9,
                 top_k: Optional[int] = None):
        self.b0 = int(b0)
        self.b_rem = int(b0)
        self.rho_min = float(rho_min)
        self.phi_c = float(phi_c)
        self.k_sig = float(k_sig)
        self.r_max = float(r_max)
        self.eps = float(eps)
        self.top_k = top_k

    def rho_b(self) -> float:
        return float(self.b_rem) / float(self.b0)

    def calculate_phi(self, r_sys: float) -> float:
        """Log-barrier viability potential Φ(t)."""
        r_sys = float(np.clip(r_sys, 0.0, self.r_max - self.eps))
        rho_b = self.rho_b()

        term_budget = -np.log(max(self.eps, rho_b - self.rho_min))
        term_risk = -np.log(max(self.eps, 1.0 - r_sys / self.r_max))
        return float(term_budget + term_risk)

    def get_beta(self, phi: float) -> float:
        """Phase-transition sharpness β(t) as a sigmoid of Φ(t)."""
        beta_min, beta_max = 5.0, 25.0
        phi = float(phi)
        return float(beta_min + (beta_max - beta_min) / (1.0 + np.exp(-self.k_sig * (phi - self.phi_c))))

    def allocate(self, scores: np.ndarray, r_sys: float, step_frac: float = 0.2):
        """Allocate integer NTU quotas based on scores.
        Returns: allocations(int[N]), beta, phi, weights(float[N]), H(weights).
        """
        if self.b_rem <= 0:
            n = len(scores)
            w = np.ones(n) / max(1, n)
            return np.zeros(n, dtype=int), 0.0, float('inf'), w, entropy(w)

        scores = np.asarray(scores, dtype=float)
        phi = self.calculate_phi(r_sys)
        beta = self.get_beta(phi)

        # Isolated agents can be encoded as -inf scores; clamp for numerical safety.
        safe_scores = np.where(np.isfinite(scores), scores, -1e12)

        # Optional top-k softmax to trim long tails for large N.
        n = len(safe_scores)
        if self.top_k is not None and n > self.top_k:
            k = max(1, min(int(self.top_k), n))
            top_idx = np.argpartition(safe_scores, -k)[-k:]
            top_scores = safe_scores[top_idx]
            top_weights = softmax_stable(beta * (top_scores - np.max(top_scores)))
            weights = np.zeros_like(safe_scores, dtype=float)
            weights[top_idx] = top_weights
        else:
            weights = softmax_stable(beta * (safe_scores - np.max(safe_scores)))
        H = entropy(weights)

        # allocate a fraction of remaining budget per round
        b_step = int(max(1, int(self.b_rem * float(step_frac))))
        alloc = np.rint(b_step * weights).astype(int)

        # minimum-execution safeguard: top-1 gets at least 1 if budget allows
        if alloc.sum() == 0 and self.b_rem >= 1 and len(alloc) > 0:
            alloc[int(np.argmax(safe_scores))] = 1

        # enforce conservation
        alloc_sum = int(alloc.sum())
        alloc_sum = min(alloc_sum, self.b_rem)
        if alloc_sum < int(alloc.sum()):
            # scale down if somehow exceeded (rare with flooring, but keep strict)
            idx = np.argsort(-alloc)
            over = int(alloc.sum()) - alloc_sum
            for j in idx:
                d = min(over, int(alloc[j]))
                alloc[j] -= d
                over -= d
                if over <= 0:
                    break

        self.b_rem -= int(alloc.sum())
        return alloc, beta, phi, weights, H
