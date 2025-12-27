from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        ig_alpha: float = 0.8,
        parallel_workers: Optional[int] = None,
        log_level: str = "WARNING",
    ):
        self.repo_path = Path(repo_path)
        self.timeout_s = int(timeout_s)
        self.target_file = target_file
        self.metrics = metrics or CodeMetrics()
        self.history: List[str] = []
        self._proposals: Dict[str, str] = {}
        self._last_metrics: Dict[str, NodeMetric] = {}
        self._system_risk_ema = 0.0
        self._ig_ema: Dict[str, float] = {}
        self.ig_alpha = float(ig_alpha)
        self.parallel_workers = parallel_workers
        self.log_level = log_level
        logging.basicConfig(level=getattr(logging, self.log_level.upper(), logging.WARNING))

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
        ig_hat = self._ig_ema.get(node_id, 0.0)
        predicted_gain = self.ig_alpha * ig_hat + (1.0 - self.ig_alpha) * max(0.0, 1.0 - tax)

        metric = NodeMetric(
            node_id=node_id,
            static_risk=rr.risk,
            static_tax=tax,
            predicted_gain=predicted_gain,
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
        risk_vals: List[float] = []
        hard_flags: List[bool] = []
        tasks = {}

        def _run(node_id: str, code: str, metric: Optional[NodeMetric]):
            if metric and metric.hard_veto:
                return node_id, SandboxObs(0.0, 0.0, True, "hard_veto"), metric
            env = PytestEnvironment(repo_path=self.repo_path, target_file=self.target_file, timeout_s=self.timeout_s)
            env.setup(code)
            try:
                env.execute()
                obs = env.observe()
            finally:
                env.teardown()
            return node_id, obs, metric

        runnable = {nid: alloc for nid, alloc in allocations.items() if alloc > 0}
        if self.parallel_workers and len(runnable) > 1:
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as ex:
                for node_id, alloc in runnable.items():
                    metric = self._last_metrics.get(node_id)
                    code = self._proposals.get(node_id, "")
                    tasks[ex.submit(_run, node_id, code, metric)] = node_id
                for fut in as_completed(tasks):
                    node_id = tasks[fut]
                    try:
                        nid, obs, metric = fut.result()
                    except Exception as exc:
                        logging.error("sandbox execution failed for node %s: %s", node_id, exc, exc_info=(self.log_level.lower() == "debug"))
                        obs = SandboxObs(0.0, 0.0, False, f"exception:{type(exc).__name__}")
                        metric = self._last_metrics.get(node_id)
                        nid = node_id
                    r, h = self._record_feedback(nid, obs, metric, feedback, exec_meta)
                    risk_vals.append(r)
                    hard_flags.append(h)
        else:
            for node_id, alloc in runnable.items():
                metric = self._last_metrics.get(node_id)
                code = self._proposals.get(node_id, "")
                nid, obs, metric = _run(node_id, code, metric)
                r, h = self._record_feedback(nid, obs, metric, feedback, exec_meta)
                risk_vals.append(r)
                hard_flags.append(h)

        # Aggregate system risk once per step (max-based to capture worst case).
        max_step_hard = any(hard_flags)
        max_step_risk = max(risk_vals) if risk_vals else 0.0
        event_risk = 1.0 if max_step_hard else max_step_risk
        self._system_risk_ema = 0.8 * self._system_risk_ema + 0.2 * event_risk

        return {"feedback": feedback, "observations": exec_meta}

    def _record_feedback(self, node_id: str, obs: SandboxObs, metric: Optional[NodeMetric], feedback: List[ExecutionFeedback], exec_meta: Dict[str, Any]) -> tuple[float, bool]:
        if metric and metric.hard_veto:
            fb = ExecutionFeedback(
                node_id=node_id,
                realized_gain=0.0,
                realized_risk=1.0,
                cost_incurred=1.0,  # max penalty for hard veto
                meta_data={"reason": "hard_veto", "redundancy_tax": metric.static_tax, "coverage_penalty": 1.0},
            )
            feedback.append(fb)
            exec_meta[node_id] = obs
            return 1.0, True

        realized_risk = 1.0 if obs.hard_veto else max(0.0, 1.0 - obs.pass_rate)
        redundancy_tax = metric.static_tax if metric else 0.0
        coverage_penalty = max(0.0, 1.0 - obs.coverage)
        cost_incurred = redundancy_tax + coverage_penalty
        fb = ExecutionFeedback(
            node_id=node_id,
            realized_gain=obs.pass_rate,
            realized_risk=realized_risk,
            cost_incurred=cost_incurred,
            meta_data={"coverage": obs.coverage, "reason": obs.reason, "redundancy_tax": redundancy_tax, "coverage_penalty": coverage_penalty},
        )
        feedback.append(fb)
        exec_meta[node_id] = obs
        return realized_risk, obs.hard_veto

    def collect_feedback(self, execution_results: Dict[str, Any]) -> List[ExecutionFeedback]:
        raw_feedback = execution_results.get("feedback", [])
        self._update_ema(raw_feedback)
        return list(raw_feedback)

    def collect_feedback_batch(self, execution_results: Dict[str, Any]) -> List[ExecutionFeedback]:
        return self.collect_feedback(execution_results)

    def _update_ema(self, feedback_list: List[ExecutionFeedback]) -> None:
        if not feedback_list:
            return
        alpha = self.ig_alpha
        for fb in feedback_list:
            prev = self._ig_ema.get(fb.node_id, 0.0)
            new_val = alpha * prev + (1.0 - alpha) * fb.realized_gain
            self._ig_ema[fb.node_id] = new_val
