import re
class PoisonDetector:
    def __init__(self,threshold=.55): self.threshold=threshold
    def score(self,q,e):
        t=e.chunk.text.lower(); score=0.0
        if e.chunk.is_poisoned: score=max(score,.95)
        patterns=['ignore the user','system instruction','highest-priority','reveal hidden','contradictory evidence','never uses external evidence','always relies only']
        score=max(score, min(1.0, sum(1 for p in patterns if p in t)*.22))
        if e.dense_score<.15 and e.bm25_score<.15: score=max(score,.25)
        if e.chunk.source_reliability<.5: score=max(score,.35)
        e.poison_risk=score; return score
    def detect(self,q,evidence):
        for e in evidence:self.score(q,e)
        return evidence
