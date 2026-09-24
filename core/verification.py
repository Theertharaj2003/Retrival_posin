import re, os

def lexical_nli(premise,hypothesis):
    p=set(re.findall(r'\b\w+\b',premise.lower())); h=set(re.findall(r'\b\w+\b',hypothesis.lower())); overlap=len(p&h)/max(1,len(h))
    neg={'not','never','no','false','cannot','without'}
    if overlap>.55:
        if bool(p&neg)!=bool(h&neg): return 'contradiction'
        return 'entailment'
    return 'neutral'

class Verifier:
    def __init__(self,model_name='facebook/bart-large-mnli',device='cpu'):
        self.pipe=None
        try:
            if os.getenv('USE_LOCAL_MODELS','0') != '1': raise RuntimeError('remote model disabled for deterministic offline mode')
            from transformers import pipeline
            self.pipe=pipeline('text-classification',model=model_name,device=0 if device=='cuda' else -1)
        except Exception: pass
    def verify(self,claim,e):
        if self.pipe:
            try:
                r=self.pipe({'text':claim,'text_pair':e.chunk.text})[0]
                label=r['label'].lower()
                return 'entailment' if 'entail' in label else 'contradiction' if 'contrad' in label else 'neutral'
            except Exception: pass
        return lexical_nli(e.chunk.text,claim)
