from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(slots=True)
class SystemState:
    """Snapshot of the global system for a decision step."""

    budget_remaining: float
    system_risk: float


@dataclass(slots=True)
class NodeMetric:
    """Static or slowly-varying properties about a candidate node."""

    node_id: str
    static_risk: float
    static_tax: float
    predicted_gain: float
    hard_veto: bool


@dataclass(slots=True)
class ExecutionFeedback:
    """Observed outcomes after an allocation is executed."""

    node_id: str
    realized_gain: float
    realized_risk: float
    cost_incurred: float


class BCTAdapter(ABC):
    """Abstract adapter boundary for plugging domain-specific logic into the BCT core."""

    @abstractmethod
    def get_system_state(self) -> SystemState:
        """Return global snapshot for a decision step."""
        ...

    @abstractmethod
    def get_candidates(self) -> List[str]:
        """Return candidate node IDs."""
        ...

    @abstractmethod
    def evaluate_node(self, node_id: str, context: Any) -> NodeMetric:
        """Evaluate a single node. Fallback when batch evaluation is unavailable."""
        ...

    @abstractmethod
    def execute_allocation(self, allocations: Dict[str, int]) -> Dict[str, Any]:
        """Apply allocations to the domain and return any execution artifacts."""
        ...

    @abstractmethod
    def collect_feedback(self, execution_results: Dict[str, Any]) -> List[ExecutionFeedback]:
        """Collect feedback for allocations. Fallback when batch feedback is unavailable."""
        ...

    # Optional batch paths -------------------------------------------------
    def evaluate_nodes(self, node_ids: List[str], context: Any) -> Dict[str, NodeMetric]:
        """Batch evaluation; default falls back to per-node evaluation."""
        return {nid: self.evaluate_node(nid, context) for nid in node_ids}

    def collect_feedback_batch(self, execution_results: Dict[str, Any]) -> List[ExecutionFeedback]:
        """Batch feedback collection; default falls back to per-node collection."""
        return self.collect_feedback(execution_results)
