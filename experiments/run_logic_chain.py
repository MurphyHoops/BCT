from __future__ import annotations
import argparse
import csv
import os
import shutil
import time
import math
import sys
from pathlib import Path

import numpy as np

# Ensure local package imports work when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bct.treasury import BCTTreasury, entropy
from bct.governance import SafetyGovernor
from bct.metrics import risk_analysis, tax_calculation
from bct.sandbox import run_in_sandbox
from bct.ledger import JsonlLedger

from agents.heuristic_agents import GoodAgent, NoisyAgent, SpamAgent

def ensure_work_repo(repo_template: Path, work_dir: Path) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(repo_template, work_dir)
    return work_dir

def read_current_solution(work_repo: Path) -> str:
    return (work_repo / "solution.py").read_text(encoding="utf-8")

def apply_solution(work_repo: Path, code: str):
    (work_repo / "solution.py").write_text(code, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=int, default=5)
    args = ap.parse_args()

    np.random.seed(args.seed)

    project_root = PROJECT_ROOT
    repo_template = project_root / "env" / "repo_template"
    runs_dir = project_root / "runs" / time.strftime("%Y%m%d_%H%M%S")
    runs_dir.mkdir(parents=True, exist_ok=True)

    ledger = JsonlLedger(str(runs_dir / "run_log.jsonl"))
    csv_path = runs_dir / "metrics.csv"

    # Work repo is the "current best" state.
    work_repo = ensure_work_repo(repo_template, runs_dir / "work_repo")

    # Initialize
    treasury = BCTTreasury(b0=args.budget, rho_min=0.1, phi_c=2.5, k_sig=3.0, r_max=1.0)
    governor = SafetyGovernor(alpha=0.3, eta=0.1, cooldown_period=5)

    agents = [GoodAgent("good"), NoisyAgent("noisy"), SpamAgent("spam")]
    n = len(agents)

    # Per-agent performance belief: EMA of observed IG (proxy)
    ig_ema = {i: 0.0 for i in range(n)}
    r_sys_ema = 0.0

    # Baseline observation for current work_repo state
    baseline_obs = run_in_sandbox(-1, read_current_solution(work_repo), str(work_repo), timeout_s=args.timeout)
    best_pass, best_cov = baseline_obs.pass_rate, baseline_obs.coverage

    history = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "b_rem", "rho_b", "phi", "beta", "H_w", "best_pass", "best_cov", "r_sys_ema", "winner_agent"])

        for step in range(args.steps):
            rho_b = treasury.rho_b()

            # Build proposals
            proposals = [a.propose().code for a in agents]

            # Score each agent using historical IG belief + reputation, penalize risk/tax; isolate via governor
            scores = np.zeros(n, dtype=float)
            per_agent = {}
            for i in range(n):
                rr = risk_analysis(proposals[i])
                tax = tax_calculation(proposals[i], history)
                rep = governor.reputation(i)
                ig_hat = ig_ema[i]

                decision = governor.check_safety(i, rr.risk, rr.hard_veto, rho_b)
                if decision.isolated:
                    scores[i] = -1e12
                else:
                    # S_i = IG_hat - γ*Risk - τ*Tax + κ*Rep
                    scores[i] = ig_hat - 2.0 * rr.risk - 1.0 * tax + 0.5 * rep

                per_agent[i] = {"risk": rr.risk, "hard_veto": rr.hard_veto, "tax": tax, "rep": rep, "ig_hat": ig_hat, "isolated": decision.isolated, "iso_reason": decision.reason}

            # Allocate budgets
            allocs, beta, phi, weights, H_w = treasury.allocate(scores, r_sys=r_sys_ema, step_frac=0.2)

            # Evaluate only those with alloc > 0 and not isolated
            evals = []
            for i in range(n):
                if allocs[i] <= 0:
                    continue
                if per_agent[i]["isolated"]:
                    continue
                obs = run_in_sandbox(i, proposals[i], str(work_repo), timeout_s=args.timeout)
                evals.append((i, obs))

                # Update system risk EWMA: timeout as high-risk event
                event_risk = 1.0 if obs.hard_veto else 0.0
                r_sys_ema = 0.8 * r_sys_ema + 0.2 * event_risk

                # Observed IG proxy is delta relative to current best
                delta_pass = obs.pass_rate - best_pass
                delta_cov = obs.coverage - best_cov
                ig_obs = 0.7 * delta_pass + 0.3 * delta_cov

                # Update agent IG belief
                ig_ema[i] = 0.8 * ig_ema[i] + 0.2 * ig_obs

                # Update reputation using observed IG and current penalties
                governor.update_reputation(i, ig=ig_obs, risk=per_agent[i]["risk"], tax=per_agent[i]["tax"])

                per_agent[i]["obs_pass"] = obs.pass_rate
                per_agent[i]["obs_cov"] = obs.coverage
                per_agent[i]["obs_veto"] = obs.hard_veto
                per_agent[i]["obs_reason"] = obs.reason
                per_agent[i]["ig_obs"] = ig_obs

            # Select winner patch and apply if improves
            winner = None
            if evals:
                # maximize (pass_rate, coverage)
                winner = max(evals, key=lambda t: (t[1].pass_rate, t[1].coverage))[0]
                win_obs = dict(evals)[winner]
                if (win_obs.pass_rate, win_obs.coverage) > (best_pass, best_cov):
                    apply_solution(work_repo, proposals[winner])
                    history.append(proposals[winner])
                    best_pass, best_cov = win_obs.pass_rate, win_obs.coverage

            # Audit log
            ledger.append({
                "step": step,
                "b_rem": treasury.b_rem,
                "rho_b": rho_b,
                "phi": phi,
                "beta": beta,
                "weights": weights.tolist(),
                "H_w": H_w,
                "scores": scores.tolist(),
                "allocs": allocs.tolist(),
                "best_pass": best_pass,
                "best_cov": best_cov,
                "r_sys_ema": r_sys_ema,
                "per_agent": per_agent,
                "winner_agent": winner,
            })

            w.writerow([step, treasury.b_rem, rho_b, phi, beta, H_w, best_pass, best_cov, r_sys_ema, winner if winner is not None else ""])
            f.flush()

            # Stop if fully solved
            if best_pass >= 1.0:
                break

    print(f"Run complete. Logs at: {runs_dir}")

if __name__ == "__main__":
    main()
