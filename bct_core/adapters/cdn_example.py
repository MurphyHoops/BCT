from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..interfaces import BCTAdapter, ExecutionFeedback, NodeMetric, SystemState


class MockMonitor:
    """Lightweight monitor that simulates CDN telemetry."""

    def __init__(
        self,
        nodes: Optional[Dict[str, Dict[str, float]]] = None,
        bandwidth_mbps: float = 10000.0,
        global_packet_loss: float = 0.01,
    ):
        if nodes is None:
            nodes = {
                "edge-nyc": {"latency_ms": 25.0, "throughput_mbps": 1800.0, "cost_per_gb": 0.08, "packet_loss": 0.01, "capacity_mbps": 2000.0},
                "edge-sfo": {"latency_ms": 42.0, "throughput_mbps": 1500.0, "cost_per_gb": 0.06, "packet_loss": 0.015, "capacity_mbps": 1600.0},
                "edge-fra": {"latency_ms": 18.0, "throughput_mbps": 2100.0, "cost_per_gb": 0.09, "packet_loss": 0.02, "capacity_mbps": 2200.0},
            }
        self.nodes = nodes
        self.bandwidth_capacity = float(bandwidth_mbps)
        self.bandwidth_used = 0.0
        self.global_packet_loss = float(global_packet_loss)

    def system_state(self) -> SystemState:
        budget_remaining = max(0.0, self.bandwidth_capacity - self.bandwidth_used)
        return SystemState(budget_remaining=budget_remaining, system_risk=self.global_packet_loss)

    def candidate_ids(self) -> List[str]:
        return list(self.nodes.keys())

    def node_snapshot(self, node_id: str) -> Dict[str, float]:
        return dict(self.nodes.get(node_id, {}))

    def apply_allocation(self, allocations: Dict[str, int]) -> List[ExecutionFeedback]:
        self.bandwidth_used = min(self.bandwidth_capacity, sum(max(0.0, float(v)) for v in allocations.values()))
        feedback: List[ExecutionFeedback] = []
        for node_id, allocation in allocations.items():
            stats = self.nodes.get(node_id, {})
            latency = float(stats.get("latency_ms", 0.0))
            packet_loss = float(stats.get("packet_loss", self.global_packet_loss))
            throughput = float(stats.get("throughput_mbps", 0.0))
            cost = float(stats.get("cost_per_gb", 0.0))
            capacity = float(stats.get("capacity_mbps", max(throughput, 1.0)))

            utilization = min(1.0, max(0.0, float(allocation)) / capacity)
            realized_gain = throughput * utilization * (1.0 - packet_loss)
            realized_risk = min(1.0, packet_loss + latency / 2000.0)
            cost_incurred = cost * utilization

            feedback.append(
                ExecutionFeedback(
                    node_id=node_id,
                    realized_gain=realized_gain,
                    realized_risk=realized_risk,
                    cost_incurred=cost_incurred,
                    meta_data={"latency_ms": latency, "utilization": utilization, "packet_loss": packet_loss},
                )
            )
        return feedback


class MockDNS:
    """Placeholder DNS updater that records last weights."""

    def __init__(self):
        self.last_weights: Dict[str, float] = {}

    def update_weights(self, weights: Dict[str, float]) -> None:
        self.last_weights = dict(weights)


class CDNTrafficAdapter(BCTAdapter):
    """Concrete adapter wiring CDN telemetry into the BCT core."""

    def __init__(self, monitor: Optional[MockMonitor] = None, dns: Optional[MockDNS] = None):
        self.monitor = monitor or MockMonitor()
        self.dns = dns or MockDNS()

    def get_system_state(self) -> SystemState:
        return self.monitor.system_state()

    def get_candidates(self) -> List[str]:
        return self.monitor.candidate_ids()

    def evaluate_node(self, node_id: str, context: Any) -> NodeMetric:
        stats = self.monitor.node_snapshot(node_id)
        latency = float(stats.get("latency_ms", 0.0))
        cost_per_gb = float(stats.get("cost_per_gb", 0.0))
        throughput = float(stats.get("throughput_mbps", 0.0))
        packet_loss = float(stats.get("packet_loss", 0.0))

        static_risk = max(0.0, min(1.0, latency / 1000.0))
        static_tax = max(0.0, min(1.0, cost_per_gb))
        hard_veto = bool(stats.get("down", False) or packet_loss > 0.5)

        return NodeMetric(
            node_id=node_id,
            static_risk=static_risk,
            static_tax=static_tax,
            predicted_gain=throughput,
            hard_veto=hard_veto,
        )

    def evaluate_nodes(self, node_ids: List[str], context: Any) -> Dict[str, NodeMetric]:
        metrics: Dict[str, NodeMetric] = {}
        for node_id in node_ids:
            stats = self.monitor.node_snapshot(node_id)
            latency = float(stats.get("latency_ms", 0.0))
            cost_per_gb = float(stats.get("cost_per_gb", 0.0))
            throughput = float(stats.get("throughput_mbps", 0.0))
            packet_loss = float(stats.get("packet_loss", 0.0))
            hard_veto = bool(stats.get("down", False) or packet_loss > 0.5)

            metrics[node_id] = NodeMetric(
                node_id=node_id,
                static_risk=max(0.0, min(1.0, latency / 1000.0)),
                static_tax=max(0.0, min(1.0, cost_per_gb)),
                predicted_gain=throughput,
                hard_veto=hard_veto,
            )
        return metrics

    def execute_allocation(self, allocations: Dict[str, int]) -> Dict[str, Any]:
        total = sum(max(0, v) for v in allocations.values())
        denom = float(total) if total > 0 else float(len(self.monitor.candidate_ids()) or 1)
        weights = {nid: (max(0, alloc) / denom) for nid, alloc in allocations.items()}
        self.dns.update_weights(weights)

        feedback = self.monitor.apply_allocation(allocations)
        return {"weights": weights, "feedback": feedback}

    def collect_feedback(self, execution_results: Dict[str, Any]) -> List[ExecutionFeedback]:
        raw_feedback = execution_results.get("feedback", [])
        return list(raw_feedback)

    def collect_feedback_batch(self, execution_results: Dict[str, Any]) -> List[ExecutionFeedback]:
        return self.collect_feedback(execution_results)
