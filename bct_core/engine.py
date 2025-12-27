from __future__ import annotations

from typing import Any, Dict

import numpy as np

from bct import BCTTreasury, SafetyGovernor

from .interfaces import BCTAdapter


class BCTEngine:
    """Core loop binding a domain adapter to the BCT treasury/governor primitives."""

    def __init__(self, adapter: BCTAdapter, config: Dict[str, Any]):
        self.adapter = adapter
        self.config = config or {}
        self._node_index: Dict[str, int] = {}

        treasury_conf = dict(self.config.get("treasury", {}))
        if "b0" not in treasury_conf:
            if "budget" in self.config:
                treasury_conf["b0"] = self.config["budget"]
        if "b0" not in treasury_conf:
            raise ValueError("config must provide 'treasury.b0' or top-level 'budget'.")

        governor_conf = dict(self.config.get("governor", {}))

        self.treasury = BCTTreasury(**treasury_conf)
        self.governor = SafetyGovernor(**governor_conf) if governor_conf else SafetyGovernor()
        self.step_frac = float(self.config.get("step_frac", 0.2))

    def step(self, context: Any) -> Dict[str, Any]:
        """Run a full sense->score->allocate->act->learn loop."""
        system_state = self.adapter.get_system_state()
        candidates = list(self.adapter.get_candidates())

        metrics_result = self.adapter.evaluate_nodes(candidates, context)
        if isinstance(metrics_result, dict):
            metrics_by_id = metrics_result
        elif metrics_result is None:
            metrics_by_id = {}
        else:
            metrics_by_id = {m.node_id: m for m in metrics_result}

        scores = np.zeros(len(candidates), dtype=float)
        id_to_index = {cid: idx for idx, cid in enumerate(candidates)}
        id_to_agent = {cid: self._node_index.setdefault(cid, len(self._node_index)) for cid in candidates}
        per_node = {}

        for idx, node_id in enumerate(candidates):
            agent_id = id_to_agent[node_id]
            metric = metrics_by_id.get(node_id)
            if metric is None:
                metric = self.adapter.evaluate_node(node_id, context)
            reputation = self.governor.reputation(agent_id)
            decision = self.governor.check_safety(
                agent_id=agent_id,
                risk_val=metric.static_risk,
                hard_veto=metric.hard_veto,
                rho_b=self.treasury.rho_b(),
            )

            if decision.isolated:
                score = -1e12
            else:
                score = (
                    metric.predicted_gain
                    + 0.5 * reputation
                    - 2.0 * metric.static_risk
                    - 1.0 * metric.static_tax
                )

            scores[idx] = score
            per_node[node_id] = {
                "metric": metric,
                "reputation": reputation,
                "decision": decision,
                "score": score,
                "agent_id": agent_id,
            }

        allocations_arr, beta, phi, weights, entropy_val = self.treasury.allocate(
            scores=scores,
            r_sys=system_state.system_risk,
            step_frac=self.step_frac,
        )

        allocations: Dict[str, int] = {
            node_id: int(allocations_arr[idx])
            for node_id, idx in id_to_index.items()
            if allocations_arr[idx] > 0
        }

        execution_results = self.adapter.execute_allocation(allocations)
        feedback_list = self.adapter.collect_feedback_batch(execution_results)

        agent_ids_fb = []
        ig_vals = []
        risk_vals = []
        tax_vals = []
        for fb in feedback_list:
            agent_id = id_to_agent.get(fb.node_id, self._node_index.get(fb.node_id))
            if agent_id is None:
                continue
            agent_ids_fb.append(agent_id)
            ig_vals.append(fb.realized_gain)
            risk_vals.append(fb.realized_risk)
            tax_vals.append(fb.cost_incurred)

        self.governor.batch_update(agent_ids_fb, ig_vals, risk_vals, tax_vals)

        return {
            "system_state": system_state,
            "candidates": candidates,
            "scores": {nid: per_node[nid]["score"] for nid in candidates},
            "allocations": allocations,
            "beta": beta,
            "phi": phi,
            "weights": {nid: float(weights[id_to_index[nid]]) for nid in candidates},
            "entropy": entropy_val,
            "execution_results": execution_results,
            "feedback": feedback_list,
            "per_node": per_node,
        }
