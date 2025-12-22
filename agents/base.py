from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Proposal:
    code: str

class Agent:
    def __init__(self, name: str):
        self.name = name

    def propose(self) -> Proposal:
        raise NotImplementedError
