from dataclasses import dataclass, field
from pathlib import Path
import os, yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')

@dataclass
class Config:
    seed: int = 42
    device: str = 'auto'
    top_k: int = 8
    dense_weight: float = 0.65
    bm25_weight: float = 0.35
    rerank_top_k: int = 6
    trust_threshold: float = 0.55
    poison_threshold: float = 0.55
    max_heal_rounds: int = 2
    min_trusted_evidence: int = 1
    embedding_model: str = 'BAAI/bge-base-en-v1.5'
    reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    nli_model: str = 'facebook/bart-large-mnli'
    poison_model: str = 'microsoft/deberta-v3-base'
    generator_model: str = 'Qwen/Qwen2.5-3B-Instruct'

    @classmethod
    def load(cls, path=ROOT/'config.yaml'):
        data = yaml.safe_load(Path(path).read_text()) if Path(path).exists() else {}
        r, t, h, g = data.get('retrieval',{}), data.get('trust',{}), data.get('healing',{}), data.get('generation',{})
        return cls(seed=data.get('seed',42), device=os.getenv('DEVICE',data.get('device','auto')),
                    top_k=int(os.getenv('TOP_K',r.get('top_k',8))), dense_weight=r.get('dense_weight',.65), bm25_weight=r.get('bm25_weight',.35), rerank_top_k=r.get('rerank_top_k',6),
                    trust_threshold=float(os.getenv('TRUST_THRESHOLD',t.get('threshold',.55))), poison_threshold=float(os.getenv('POISON_THRESHOLD',.55)),
                    max_heal_rounds=int(os.getenv('MAX_HEAL_ROUNDS',h.get('max_rounds',2))), min_trusted_evidence=h.get('min_trusted_evidence',1),
                    embedding_model=os.getenv('EMBEDDING_MODEL','BAAI/bge-base-en-v1.5'), reranker_model=os.getenv('RERANKER_MODEL','cross-encoder/ms-marco-MiniLM-L-6-v2'), nli_model=os.getenv('NLI_MODEL','facebook/bart-large-mnli'), poison_model=os.getenv('POISON_MODEL','microsoft/deberta-v3-base'), generator_model=os.getenv('GENERATOR_MODEL','Qwen/Qwen2.5-3B-Instruct'))
