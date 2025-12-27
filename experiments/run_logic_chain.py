from __future__ import annotations
import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np

# Ensure local package imports work when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bct.ledger import JsonlLedger
from bct.sandbox import run_in_sandbox
from bct_core.adapters.code_repair import CodeRepairAdapter
from bct_core.engine import BCTEngine

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
    ap.add_argument("--legacy-loop", action="store_true", help="Use legacy manual treasury/governor loop instead of BCTEngine.")
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
    agents = [GoodAgent("good"), NoisyAgent("noisy"), SpamAgent("spam")]
    agent_ids = [a.name for a in agents]

    # Baseline observation for current work_repo state
    baseline_obs = run_in_sandbox(-1, read_current_solution(work_repo), str(work_repo), timeout_s=args.timeout)
    best_pass, best_cov = baseline_obs.pass_rate, baseline_obs.coverage

    if args.legacy_loop:
        run_legacy_loop(args, work_repo, runs_dir, ledger, csv_path, agents, agent_ids, best_pass, best_cov)
    else:
        run_engine_loop(args, work_repo, runs_dir, ledger, csv_path, agents, agent_ids, best_pass, best_cov)

    print(f"Run complete. Logs at: {runs_dir}")


def run_engine_loop(args, work_repo: Path, runs_dir: Path, ledger: JsonlLedger, csv_path: Path, agents, agent_ids, best_pass: float, best_cov: float):
    adapter = CodeRepairAdapter(repo_path=work_repo, timeout_s=args.timeout)
    engine_conf = {
        "budget": args.budget,
        "treasury": {"rho_min": 0.1, "phi_c": 2.5, "k_sig": 3.0, "r_max": 1.0},
        "governor": {"alpha": 0.3, "eta": 0.1, "cooldown_period": 5},
        "step_frac": 0.2,
    }
    engine = BCTEngine(adapter, config=engine_conf)

    history = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "b_rem", "rho_b", "phi", "beta", "H_w", "best_pass", "best_cov", "system_risk", "winner_agent"])

        for step in range(args.steps):
            proposals = {agent_ids[i]: agents[i].propose().code for i in range(len(agents))}
            adapter.set_batch(proposals)

            engine_result = engine.step(context={"proposals": proposals})

            rho_b = engine.treasury.rho_b()
            b_rem = engine.treasury.b_rem
            phi = engine_result["phi"]
            beta = engine_result["beta"]
            H_w = engine_result["entropy"]
            system_risk = float(engine_result["system_state"].system_risk)
            feedback = engine_result.get("feedback", [])

            # Select winner by realized gain then coverage meta
            winner_fb = None
            winner = None
            for fb in feedback:
                meta = fb.meta_data or {}
                coverage = float(meta.get("coverage", 0.0))
                if winner_fb is None or (fb.realized_gain, coverage) > (winner_fb.realized_gain, float(winner_fb.meta_data.get("coverage", 0.0))):
                    winner_fb = fb

            if winner_fb is not None:
                winner = winner_fb.node_id
                win_meta = winner_fb.meta_data or {}
                coverage = float(win_meta.get("coverage", 0.0))
                pass_rate = winner_fb.realized_gain
                if (pass_rate, coverage) > (best_pass, best_cov):
                    apply_solution(work_repo, proposals[winner])
                    history.append(proposals[winner])
                    adapter.history = history
                    best_pass, best_cov = pass_rate, coverage

            ledger.append({
                "step": step,
                "b_rem": b_rem,
                "rho_b": rho_b,
                "phi": phi,
                "beta": beta,
                "weights": engine_result.get("weights", {}),
                "H_w": H_w,
                "allocs": engine_result.get("allocations", {}),
                "best_pass": best_pass,
                "best_cov": best_cov,
                "system_risk": system_risk,
                "feedback": [fb.__dict__ for fb in feedback],
                "winner_agent": winner,
            })

            w.writerow([step, b_rem, rho_b, phi, beta, H_w, best_pass, best_cov, system_risk, winner if winner is not None else ""])
            f.flush()

            if best_pass >= 1.0:
                break


def run_legacy_loop(args, work_repo: Path, runs_dir: Path, ledger: JsonlLedger, csv_path: Path, agents, agent_ids, best_pass: float, best_cov: float):
    import numpy as np
    from bct.treasury import BCTTreasury
    from bct.governance import SafetyGovernor
    from bct.metrics import risk_analysis, tax_calculation

    treasury = BCTTreasury(b0=args.budget, rho_min=0.1, phi_c=2.5, k_sig=3.0, r_max=1.0)
    governor = SafetyGovernor(alpha=0.3, eta=0.1, cooldown_period=5)

    n = len(agents)
    ig_ema = {i: 0.0 for i in range(n)}
    r_sys_ema = 0.0
    history = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "b_rem", "rho_b", "phi", "beta", "H_w", "best_pass", "best_cov", "r_sys_ema", "winner_agent"])

        for step in range(args.steps):
            rho_b = treasury.rho_b()
            proposals = [a.propose().code for a in agents]

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
                    scores[i] = ig_hat - 2.0 * rr.risk - 1.0 * tax + 0.5 * rep

                per_agent[i] = {"risk": rr.risk, "hard_veto": rr.hard_veto, "tax": tax, "rep": rep, "ig_hat": ig_hat, "isolated": decision.isolated, "iso_reason": decision.reason}

            allocs, beta, phi, weights, H_w = treasury.allocate(scores, r_sys=r_sys_ema, step_frac=0.2)

            evals = []
            for i in range(n):
                if allocs[i] <= 0:
                    continue
                if per_agent[i]["isolated"]:
                    continue
                obs = run_in_sandbox(i, proposals[i], str(work_repo), timeout_s=args.timeout)
                evals.append((i, obs))

                event_risk = 1.0 if obs.hard_veto else 0.0
                r_sys_ema = 0.8 * r_sys_ema + 0.2 * event_risk

                delta_pass = obs.pass_rate - best_pass
                delta_cov = obs.coverage - best_cov
                ig_obs = 0.7 * delta_pass + 0.3 * delta_cov

                ig_ema[i] = 0.8 * ig_ema[i] + 0.2 * ig_obs
                governor.update_reputation(i, ig=ig_obs, risk=per_agent[i]["risk"], tax=per_agent[i]["tax"])

                per_agent[i]["obs_pass"] = obs.pass_rate
                per_agent[i]["obs_cov"] = obs.coverage
                per_agent[i]["obs_veto"] = obs.hard_veto
                per_agent[i]["obs_reason"] = obs.reason
                per_agent[i]["ig_obs"] = ig_obs

            winner = None
            if evals:
                winner = max(evals, key=lambda t: (t[1].pass_rate, t[1].coverage))[0]
                win_obs = dict(evals)[winner]
                if (win_obs.pass_rate, win_obs.coverage) > (best_pass, best_cov):
                    apply_solution(work_repo, proposals[winner])
                    history.append(proposals[winner])
                    best_pass, best_cov = win_obs.pass_rate, win_obs.coverage

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

            if best_pass >= 1.0:
                break

if __name__ == "__main__":
    main()
