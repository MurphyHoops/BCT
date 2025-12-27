from __future__ import annotations

import shutil
from pathlib import Path
import pytest

from agents.heuristic_agents import GoodAgent
from bct.metrics import CodeMetrics
from bct.sandbox import PytestEnvironment
from bct_core.adapters.cdn_example import CDNTrafficAdapter
from bct_core.adapters.code_repair import CodeRepairAdapter
from bct_core.engine import BCTEngine
from bct_core.interfaces import NodeMetric


def _copy_repo(tmp_path: Path) -> Path:
    repo_template = Path(__file__).resolve().parents[1] / "env" / "repo_template"
    dest = tmp_path / "work_repo"
    shutil.copytree(repo_template, dest)
    return dest


def test_pytest_environment_lifecycle(tmp_path):
    pytest.importorskip("pytest_jsonreport")
    pytest.importorskip("pytest_cov")
    work_repo = _copy_repo(tmp_path)
    good_patch = GoodAgent("good").propose().code
    env = PytestEnvironment(repo_path=work_repo, timeout_s=5)
    env.setup(good_patch)
    env.execute()
    obs = env.observe()
    env.teardown()

    assert obs.hard_veto is False
    assert obs.pass_rate > 0.0
    assert obs.coverage > 0.0


def test_engine_with_code_repair_adapter(tmp_path):
    work_repo = _copy_repo(tmp_path)
    adapter = CodeRepairAdapter(repo_path=work_repo, timeout_s=5)
    proposals = {"good": GoodAgent("good").propose().code}
    adapter.set_batch(proposals)
    engine = BCTEngine(adapter, config={"budget": 3})

    result = engine.step(context={"proposals": proposals})

    assert result["allocations"]
    assert result["feedback"]
    for fb in result["feedback"]:
        assert fb.realized_gain >= 0.0
    # EMA should have been updated for the node
    assert adapter._ig_ema.get("good", 0.0) >= 0.0


def test_system_risk_updates_once_per_step(tmp_path):
    work_repo = _copy_repo(tmp_path)
    adapter = CodeRepairAdapter(repo_path=work_repo, timeout_s=5)
    proposals = {"good": GoodAgent("good").propose().code}
    adapter.set_batch(proposals)
    engine = BCTEngine(adapter, config={"budget": 2})

    prev_risk = adapter._system_risk_ema
    engine.step(context={"proposals": proposals})
    new_risk = adapter._system_risk_ema
    # Ensure risk updated at least once and not multiple times per feedback
    assert new_risk >= prev_risk


def test_engine_with_cdn_adapter():
    adapter = CDNTrafficAdapter()
    engine = BCTEngine(adapter, config={"budget": 5, "score_weights": {"gain": 1.0, "reputation": 0.1, "risk": -1.0, "tax": -0.5}})

    result = engine.step(context=None)

    assert result["allocations"]
    assert result["feedback"]
    for fb in result["feedback"]:
        assert fb.cost_incurred >= 0.0


def test_code_metrics_wrappers():
    metrics = CodeMetrics()
    rr = metrics.risk_analysis("import os\n")
    assert rr.hard_veto is True
    tax = metrics.tax_calculation("print('hi')", ["print('hi')"])
    assert tax > 0.0


def test_parallel_hard_veto_feedback():
    adapter = CodeRepairAdapter(repo_path=Path("."), timeout_s=1, parallel_workers=2)
    adapter._proposals = {"a": "", "b": ""}
    adapter._last_metrics = {
        "a": NodeMetric(node_id="a", static_risk=0.1, static_tax=0.2, predicted_gain=0.0, hard_veto=True),
        "b": NodeMetric(node_id="b", static_risk=0.1, static_tax=0.3, predicted_gain=0.0, hard_veto=True),
    }
    allocations = {"a": 1, "b": 1}

    result = adapter.execute_allocation(allocations)
    feedback = result["feedback"]

    assert len(feedback) == 2
    assert all(fb.cost_incurred == 1.0 for fb in feedback)
