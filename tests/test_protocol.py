import numpy as np
from bct.treasury import BCTTreasury
from bct.governance import SafetyGovernor

def test_budget_conservation():
    t = BCTTreasury(b0=10)
    scores = np.array([1.0, 0.0, -1.0])
    alloc, beta, phi, w, H = t.allocate(scores, r_sys=0.0, step_frac=0.5)
    assert alloc.sum() <= 10
    assert t.b_rem == 10 - alloc.sum()

def test_cooldown_enforced():
    g = SafetyGovernor(alpha=1.0, cooldown_period=3)  # EWMA = risk instantly
    # trigger
    d1 = g.check_safety(0, risk_val=1.0, hard_veto=False, rho_b=1.0)
    assert d1.isolated
    # cooldown next rounds
    d2 = g.check_safety(0, risk_val=0.0, hard_veto=False, rho_b=1.0)
    assert d2.isolated
    d3 = g.check_safety(0, risk_val=0.0, hard_veto=False, rho_b=1.0)
    assert d3.isolated
