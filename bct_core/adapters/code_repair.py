from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from bct.metrics import CodeMetrics
from bct.sandbox import PytestEnvironment, SandboxObs
from bct_core.interfaces import BCTAdapter, ExecutionFeedback, NodeMetric, SystemState


class CodeRepairAdapter(BCTAdapter):
    """Adapter that evaluates code patches via a pytest-based environment."""

    def __init__(
        self,
        repo_path: str | Path,
        timeout_s: int = 5,
        target_file: str = "solution.py",
        metrics: Optional[CodeMetrics] = None,
    ):
        self.repo_path = Path(repo_path)
        self.timeout_s = int(timeout_s)
        self.target_file = target_file
        self.metrics = metrics or CodeMetrics()
        self.history: List[str] = []
        self._proposals: Dict[str, str] = {}
        self._last_metrics: Dict[str, NodeMetric] = {}
        self._system_risk_ema = 0.0

    def set_batch(self, proposals: Dict[str, str]) -> None:
        """Provide proposals for the next engine step."""
        self._proposals = dict(proposals or {})

    def get_system_state(self) -> SystemState:
        return SystemState(budget_remaining=1.0, system_risk=self._system_risk_ema)

    def get_candidates(self) -> List[str]:
        return list(self._proposals.keys())

    def evaluate_node(self, node_id: str, context: Any) -> NodeMetric:
        code = self._proposals.get(node_id, "")
        rr = self.metrics.risk_analysis(code)
        tax = self.metrics.tax_calculation(code, self.history)

        metric = NodeMetric(
            node_id=node_id,
            static_risk=rr.risk,
            static_tax=tax,
            predicted_gain=max(0.0, 1.0 - tax),
            hard_veto=rr.hard_veto,
        )
        self._last_metrics[node_id] = metric
        return metric

    def evaluate_nodes(self, node_ids: List[str], context: Any) -> Dict[str, NodeMetric]:
        metrics: Dict[str, NodeMetric] = {}
        for nid in node_ids:
            metrics[nid] = self.evaluate_node(nid, context)
        return metrics

    def execute_allocation(self, allocations: Dict[str, int]) -> Dict[str, Any]:
        feedback: List[ExecutionFeedback] = []
        exec_meta: Dict[str, Any] = {}

        for node_id, alloc in allocations.items():
            if alloc <= 0:
                continue

            metric = self._last_metrics.get(node_id)
            if metric and metric.hard_veto:
                fb = ExecutionFeedback(
                    node_id=node_id,
                    realized_gain=0.0,
                    realized_risk=1.0,
                    cost_incurred=metric.static_tax,
                    meta_data={"reason": "hard_veto"},
                )
                feedback.append(fb)
                continue

            code = self._proposals.get(node_id, "")
            env = PytestEnvironment(repo_path=self.repo_path, target_file=self.target_file, timeout_s=self.timeout_s)
            env.setup(code)
            obs: SandboxObs
            try:
                env.execute()
                obs = env.observe()
            finally:
                env.teardown()

            realized_risk = 1.0 if obs.hard_veto else max(0.0, 1.0 - obs.pass_rate)
            cost_incurred = max(0.0, 1.0 - obs.coverage)
            fb = ExecutionFeedback(
                node_id=node_id,
                realized_gain=obs.pass_rate,
                realized_risk=realized_risk,
                cost_incurred=cost_incurred,
                meta_data={"coverage": obs.coverage, "reason": obs.reason},
            )
            feedback.append(fb)
            exec_meta[node_id] = obs

            event_risk = 1.0 if obs.hard_veto else 0.0
            self._system_risk_ema = 0.8 * self._system_risk_ema + 0.2 * event_risk

        return {"feedback": feedback, "observations": exec_meta}

    def collect_feedback(self, execution_results: Dict[str, Any]) -> List[ExecutionFeedback]:
        raw_feedback = execution_results.get("feedback", [])
        return list(raw_feedback)

    def collect_feedback_batch(self, execution_results: Dict[str, Any]) -> List[ExecutionFeedback]:
        return self.collect_feedback(execution_results)
