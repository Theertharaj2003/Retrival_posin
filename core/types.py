from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class Chunk:
    document_id: str
    chunk_id: str
    text: str
    source: str
    is_poisoned: bool = False
    attack_type: str = 'clean'
    source_reliability: float = 0.8
    metadata: dict[str, Any] | None = None

    def to_dict(self): return asdict(self)

@dataclass
class Evidence:
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    poison_risk: float = 0.0
    trust_score: float = 0.0
    verification: str = 'unknown'
    attribution_score: float = 0.0
    quarantined: bool = False

    def to_dict(self):
        d = asdict(self); d['chunk'] = self.chunk.to_dict(); return d
