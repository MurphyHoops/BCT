# BCT MVP — Logic Chain Economy (pytest + subprocess sandbox)

This is a minimal, verifiable prototype for the **Biomorphic Cybernetic Treasury (BCT)** protocol:
- Treasury: log-barrier viability potential Φ and phase-transition sharpness β
- Governance: EWMA circuit breaker + cooldown; reputation ledger
- Sandbox: subprocess + pytest-json-report + pytest-cov (JSON)

## Quickstart

```bash
python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# macOS/Linux:
#   source .venv/bin/activate

pip install -r requirements.txt

python experiments/run_logic_chain.py --budget 300 --steps 30
```

Outputs:
- `runs/<timestamp>/run_log.jsonl` : append-only audit log (per step)
- `runs/<timestamp>/metrics.csv`   : tabular metrics for plotting

## Security note
The sandbox executes code patches. Do **NOT** run untrusted patches on a machine with sensitive data.
For stronger isolation, containerize each sandbox run (Docker) or use OS-level sandboxing.
