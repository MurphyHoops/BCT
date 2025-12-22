from __future__ import annotations
import os
import json
import shutil
import subprocess
import tempfile
import sys
from dataclasses import dataclass
from typing import Optional

@dataclass
class SandboxObs:
    pass_rate: float
    coverage: float
    hard_veto: bool
    reason: str

def run_in_sandbox(agent_id: int, patch_code: str, repo_path: str, timeout_s: int = 5) -> SandboxObs:
    """Run pytest+coverage in an isolated temp dir and return observations.
    Requires: pytest-json-report, pytest-cov
    """
    tmp = tempfile.mkdtemp(prefix=f"bct_sandbox_{agent_id}_")
    try:
        shutil.copytree(repo_path, os.path.join(tmp, "repo"), dirs_exist_ok=True)
        sandbox_dir = os.path.join(tmp, "repo")

        # Apply patch (overwrite target file for MVP)
        with open(os.path.join(sandbox_dir, "solution.py"), "w", encoding="utf-8") as f:
            f.write(patch_code)

        report_file = ".report.json"
        cov_file = "coverage.json"

        cmd = [
            sys.executable, "-m", "pytest", "-q", "--maxfail=1",
            "--json-report", f"--json-report-file={report_file}",
            "--cov=.", f"--cov-report=json:{cov_file}"
        ]

        result = subprocess.run(
            cmd, cwd=sandbox_dir, capture_output=True, text=True,
            timeout=timeout_s
        )

        # Parse json-report
        if not os.path.exists(os.path.join(sandbox_dir, report_file)):
            return SandboxObs(0.0, 0.0, False, "missing_json_report")

        with open(os.path.join(sandbox_dir, report_file), "r", encoding="utf-8") as f:
            report = json.load(f)

        summ = report.get("summary", {})
        total = float(summ.get("total", 0) or 0)
        passed = float(summ.get("passed", 0) or 0)
        pass_rate = (passed / total) if total > 0 else 0.0

        # Parse coverage
        coverage = 0.0
        cov_path = os.path.join(sandbox_dir, cov_file)
        if os.path.exists(cov_path):
            with open(cov_path, "r", encoding="utf-8") as f:
                cov = json.load(f)
            totals = cov.get("totals", {})
            coverage = float(totals.get("percent_covered", 0.0) or 0.0) / 100.0

        # Hard veto if test process non-zero due to crash? For MVP keep as soft.
        return SandboxObs(pass_rate, coverage, False, "ok")

    except subprocess.TimeoutExpired:
        return SandboxObs(0.0, 0.0, True, "timeout")
    except Exception as e:
        return SandboxObs(0.0, 0.0, False, f"exception:{type(e).__name__}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
