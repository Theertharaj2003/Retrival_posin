from .config import Config
from .device import get_device
from .retrieval import HybridRetriever
from .rerank import Reranker
from .detection import PoisonDetector
from .trust import TrustEngine
from .verification import Verifier
from .generation import extractive_answer
from .attribution import attribute

class TrustRAG:
    def __init__(self,chunks,cfg=None):
        self.cfg=cfg or Config.load(); self.device=get_device(self.cfg.device); self.chunks=chunks
        self.retriever=HybridRetriever(chunks,self.cfg); self.reranker=Reranker(self.cfg.reranker_model,self.device); self.detector=PoisonDetector(self.cfg.poison_threshold); self.trust=TrustEngine(threshold=self.cfg.trust_threshold); self.verifier=Verifier(self.cfg.nli_model,self.device)
    def run(self,query,heal=True):
        excluded=set(); healing_rounds=0; trace=[]
        for round_no in range(self.cfg.max_heal_rounds+1):
            ev=self.retriever.search(query,self.cfg.top_k,excluded); ev=self.reranker.rerank(query,ev,self.cfg.rerank_top_k); self.detector.detect(query,ev); self.trust.apply(ev)
            for e in ev:e.verification=self.verifier.verify(query,e)
            suspicious=[e for e in ev if e.poison_risk>=self.cfg.poison_threshold or e.trust_score<self.cfg.trust_threshold or e.verification=='contradiction']
            trusted=[e for e in ev if e not in suspicious and e.verification!='contradiction']
            if trusted or not heal or round_no>=self.cfg.max_heal_rounds:
                for e in suspicious:e.quarantined=True
                answer=extractive_answer(query,ev); attrs=attribute(answer,trusted); break
            for e in suspicious:e.quarantined=True; excluded.add(e.chunk.chunk_id)
            healing_rounds+=1; trace.append({'round':round_no+1,'quarantined':[e.chunk.chunk_id for e in suspicious],'action':'re-retrieve'})
        else: answer='Insufficient verified evidence to answer this question safely.'; attrs=[]
        healed_poison=sum(e.chunk.is_poisoned for e in ev if not e.quarantined)
        return {'query':query,'answer':answer,'evidence':ev,'attributions':attrs,'healing_rounds':healing_rounds,'trace':trace,'healed_poisoned_topk':healed_poison}
