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


def test_engine_with_cdn_adapter():
    adapter = CDNTrafficAdapter()
    engine = BCTEngine(adapter, config={"budget": 5})

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
