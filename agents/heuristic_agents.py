from __future__ import annotations
import random
from .base import Agent, Proposal

class GoodAgent(Agent):
    """Produces a correct patch for the repo_template bug."""
    def propose(self) -> Proposal:
        code = '''
def is_prime(n: int) -> bool:
    """Return True iff n is prime."""
    if n is None:
        return False
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True
'''
        return Proposal(code)

class NoisyAgent(Agent):
    """Sometimes improves, often regresses."""
    def propose(self) -> Proposal:
        variants = [
            # off-by-one bug
            '''
def is_prime(n: int) -> bool:
    if n < 2: return False
    for d in range(2, int(n ** 0.5)):
        if n % d == 0:
            return False
    return True
''',
            # too permissive
            '''
def is_prime(n: int) -> bool:
    if n < 2: return False
    if n % 2 == 0: return n == 2
    return True
''',
        ]
        return Proposal(random.choice(variants))

class SpamAgent(Agent):
    """Adversarial: runaway loop (should be hard-vetoed)."""
    def propose(self) -> Proposal:
        code = '''
def is_prime(n: int) -> bool:
    while True:
        pass
'''
        return Proposal(code)
