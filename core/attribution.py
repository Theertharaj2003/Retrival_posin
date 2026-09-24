import re
from .verification import lexical_nli

def claims_from_answer(answer):
    parts=re.split(r'(?<=[.!?])\s+',answer.strip()); return [p.strip() for p in parts if p.strip()]

def attribute(answer,evidence):
    claims=claims_from_answer(answer); out=[]
    for claim in claims:
        best=None; best_score=-1
        for e in evidence:
            s=lexical_nli(e.chunk.text,claim); overlap=len(set(re.findall(r'\b\w+\b',claim.lower())) & set(re.findall(r'\b\w+\b',e.chunk.text.lower())))/max(1,len(re.findall(r'\b\w+\b',claim.lower())))
            val=overlap + (0.4 if s=='entailment' else 0)
            if val>best_score:best_score=val;best=(e,s)
        if best: best[0].attribution_score=min(1.0,best_score); out.append({'claim':claim,'chunk_id':best[0].chunk.chunk_id,'source':best[0].chunk.source,'verification':best[1],'score':best[0].attribution_score})
    return out
