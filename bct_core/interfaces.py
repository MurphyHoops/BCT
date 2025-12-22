from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SystemState:
    """Snapshot of the global system for a decision step."""

    budget_remaining: float
    system_risk: float


@dataclass
class NodeMetric:
    """Static or slowly-varying properties about a candidate node."""

    node_id: str
    static_risk: float
    static_tax: float
    predicted_gain: float
    hard_veto: bool


@dataclass
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
        ...

    @abstractmethod
    def get_candidates(self) -> List[str]:
        ...

    @abstractmethod
    def evaluate_node(self, node_id: str, context: Any) -> NodeMetric:
        ...

    @abstractmethod
    def execute_allocation(self, allocations: Dict[str, int]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def collect_feedback(self, execution_results: Dict[str, Any]) -> List[ExecutionFeedback]:
        ...
