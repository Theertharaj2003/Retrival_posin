from core.dataset import demo_chunks
from core.poisoning import poison_chunks
from core.config import Config
from core.pipeline import TrustRAG

def test_poison_generation_labels():
    p=poison_chunks(demo_chunks(),'factual'); assert any(x.is_poisoned for x in p)

def test_pipeline_returns_answer():
    r=TrustRAG(demo_chunks(),Config.load()).run('What is knowledge poisoning in RAG?'); assert isinstance(r['answer'],str) and r['answer']

def test_self_healing_removes_marked_poison():
    p=poison_chunks(demo_chunks(),'indirect_injection'); r=TrustRAG(p,Config.load()).run('What is knowledge poisoning in RAG?',heal=True); assert all(not e.chunk.is_poisoned for e in r['evidence'] if not e.quarantined)
