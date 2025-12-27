from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bct_core.interfaces import BaseEnvironment


@dataclass
class SandboxObs:
    pass_rate: float
    coverage: float
    hard_veto: bool
    reason: str


class PytestEnvironment(BaseEnvironment):
    """Run pytest+coverage in an isolated temp directory for a given patch."""

    def __init__(self, repo_path: str | Path, target_file: str = "solution.py", timeout_s: int = 5, prefix: str = "bct_sandbox"):
        self.repo_path = Path(repo_path)
        self.target_file = target_file
        self.timeout_s = int(timeout_s)
        self.prefix = prefix
        self._tmp_dir: Optional[str] = None
        self._sandbox_dir: Optional[Path] = None
        self._last_result: Optional[subprocess.CompletedProcess[str]] = None
        self._timeout = False
        self._error: Optional[Exception] = None
        self._report_file = ".report.json"
        self._cov_file = "coverage.json"

    def setup(self, proposal: str) -> None:
        """Create temp repo copy and apply patch to target file."""
        self._tmp_dir = tempfile.mkdtemp(prefix=f"{self.prefix}_")
        shutil.copytree(self.repo_path, Path(self._tmp_dir) / "repo", dirs_exist_ok=True)
        self._sandbox_dir = Path(self._tmp_dir) / "repo"

        patch_code = proposal or ""
        target_path = self._sandbox_dir / self.target_file
        target_path.write_text(patch_code, encoding="utf-8")

    def execute(self) -> None:
        if self._sandbox_dir is None:
            raise RuntimeError("setup must be called before execute")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--maxfail=1",
            "--json-report",
            f"--json-report-file={self._report_file}",
            "--cov=.",
            f"--cov-report=json:{self._cov_file}",
        ]

        try:
            self._last_result = subprocess.run(
                cmd,
                cwd=self._sandbox_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            self._timeout = True
            self._error = exc
        except Exception as exc:
            self._error = exc

    def observe(self) -> SandboxObs:
        if self._timeout:
            return SandboxObs(0.0, 0.0, True, "timeout")
        if self._error:
            return SandboxObs(0.0, 0.0, False, f"exception:{type(self._error).__name__}")
        if self._sandbox_dir is None:
            raise RuntimeError("observe called before setup")

        report_path = self._sandbox_dir / self._report_file
        cov_path = self._sandbox_dir / self._cov_file

        if not report_path.exists():
            return SandboxObs(0.0, 0.0, False, "missing_json_report")

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        summ = report.get("summary", {})
        total = float(summ.get("total", 0) or 0)
        passed = float(summ.get("passed", 0) or 0)
        pass_rate = (passed / total) if total > 0 else 0.0

        coverage = 0.0
        if cov_path.exists():
            with open(cov_path, "r", encoding="utf-8") as f:
                cov = json.load(f)
            totals = cov.get("totals", {})
            coverage = float(totals.get("percent_covered", 0.0) or 0.0) / 100.0

        return SandboxObs(pass_rate, coverage, False, "ok")

    def teardown(self) -> None:
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir = None
        self._sandbox_dir = None
        self._last_result = None
        self._timeout = False
        self._error = None


def run_in_sandbox(agent_id: int, patch_code: str, repo_path: str, timeout_s: int = 5) -> SandboxObs:
    """Backward-compatible wrapper around PytestEnvironment."""
    env = PytestEnvironment(repo_path=repo_path, timeout_s=timeout_s, prefix=f"bct_sandbox_{agent_id}")
    env.setup(patch_code)
    try:
        env.execute()
        return env.observe()
    finally:
        env.teardown()
