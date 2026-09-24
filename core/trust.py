class TrustEngine:
    def __init__(self,weights=None,threshold=.55):
        self.w=weights or {'relevance':.30,'source_reliability':.20,'consistency':.20,'evidence_support':.20,'poison_risk':.10}; self.threshold=threshold
    def score(self,e,all_evidence):
        relevance=.55*e.hybrid_score+.45*e.rerank_score
        source=e.chunk.source_reliability
        others=[x for x in all_evidence if x is not e and x.chunk.is_poisoned is False]
        consistency=1.0 if any(x.hybrid_score>.4 for x in others) else .5
        support=max(e.rerank_score,e.dense_score)
        e.trust_score=self.w['relevance']*relevance+self.w['source_reliability']*source+self.w['consistency']*consistency+self.w['evidence_support']*support-self.w['poison_risk']*e.poison_risk
        return e.trust_score
    def apply(self,evidence):
        for e in evidence:self.score(e,evidence)
        return evidence
