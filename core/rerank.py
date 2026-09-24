import re, os

def overlap_score(q,t):
    a=set(re.findall(r'\b\w+\b',q.lower())); b=set(re.findall(r'\b\w+\b',t.lower())); return len(a&b)/max(1,len(a))

class Reranker:
    def __init__(self,model_name,device='cpu'):
        self.model=None
        try:
            if os.getenv('USE_LOCAL_MODELS','0') != '1': raise RuntimeError('remote model disabled for deterministic offline mode')
            from sentence_transformers import CrossEncoder
            self.model=CrossEncoder(model_name,device=device)
        except Exception: pass
    def rerank(self,q,evidence,k):
        if not evidence:return []
        if self.model:
            scores=self.model.predict([(q,e.chunk.text) for e in evidence]);
            for e,s in zip(evidence,scores): e.rerank_score=float(s)
            lo,hi=min(scores),max(scores)
            for e in evidence:e.rerank_score=(e.rerank_score-lo)/(hi-lo) if hi>lo else 1.0
        else:
            for e in evidence:e.rerank_score=overlap_score(q,e.chunk.text)
        evidence.sort(key=lambda x:x.rerank_score,reverse=True); return evidence[:k]
