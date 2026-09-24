import re, math, os
from collections import Counter, defaultdict
import numpy as np
from .types import Chunk, Evidence

class BM25:
    def __init__(self, chunks):
        self.chunks=chunks; self.docs=[self.tok(c.text) for c in chunks]; self.df=Counter(); self.tf=[]
        for d in self.docs:
            c=Counter(d); self.tf.append(c); self.df.update(c.keys())
        self.avg=sum(map(len,self.docs))/max(1,len(self.docs)); self.N=len(self.docs); self.k1=1.5; self.b=.75
    @staticmethod
    def tok(x): return re.findall(r'\b[a-zA-Z0-9_]+\b',x.lower())
    def score(self,q,i):
        terms=self.tok(q); tf=self.tf[i]; dl=len(self.docs[i]); s=0
        for t in terms:
            if t not in tf: continue
            idf=math.log(1+(self.N-self.df[t]+.5)/(self.df[t]+.5)); f=tf[t]
            s += idf*f*(self.k1+1)/(f+self.k1*(1-self.b+self.b*dl/max(1,self.avg)))
        return s
    def search(self,q,k):
        vals=[self.score(q,i) for i in range(self.N)]; order=np.argsort(vals)[::-1][:k]
        return [(self.chunks[i],float(vals[i])) for i in order]

class DenseRetriever:
    def __init__(self,chunks,model_name='BAAI/bge-base-en-v1.5',device='cpu'):
        self.chunks=chunks; self.device=device; self.backend='tfidf'
        self.model=None; self.matrix=None
        try:
            if os.getenv('USE_LOCAL_MODELS','0') != '1': raise RuntimeError('remote model disabled for deterministic offline mode')
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(model_name,device=device); self.matrix=self.model.encode([c.text for c in chunks],normalize_embeddings=True,show_progress_bar=False); self.backend='sentence-transformers'
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize
            self.vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2)); self.matrix=normalize(self.vectorizer.fit_transform([c.text for c in chunks]))
    def search(self,q,k):
        if self.backend=='sentence-transformers':
            qv=self.model.encode([q],normalize_embeddings=True)[0]; vals=np.asarray(self.matrix)@qv
        else:
            qv=self.vectorizer.transform([q]); vals=(self.matrix@qv.T).toarray().ravel()
        order=np.argsort(vals)[::-1][:k]; return [(self.chunks[i],float(vals[i])) for i in order]

class HybridRetriever:
    def __init__(self,chunks,cfg):
        self.chunks=chunks; self.bm=BM25(chunks); self.dense=DenseRetriever(chunks,cfg.embedding_model,'cuda' if cfg.device=='cuda' else 'cpu'); self.dw=cfg.dense_weight; self.bw=cfg.bm25_weight
    @staticmethod
    def norm(vals):
        if not vals: return {}
        a=np.array(list(vals.values()),float); lo,hi=a.min(),a.max();
        if hi-lo<1e-9: return {k:1.0 for k in vals}
        return {k:(v-lo)/(hi-lo) for k,v in vals.items()}
    def search(self,q,k=8,exclude=None):
        exclude=exclude or set(); d=dict((c.chunk_id,s) for c,s in self.dense.search(q,min(len(self.chunks),k*3)) if c.chunk_id not in exclude); b=dict((c.chunk_id,s) for c,s in self.bm.search(q,min(len(self.chunks),k*3)) if c.chunk_id not in exclude); dn,bn=self.norm(d),self.norm(b); by={c.chunk_id:c for c in self.chunks}
        rows=[]
        for cid in set(dn)|set(bn): rows.append(Evidence(by[cid],dn.get(cid,0),bn.get(cid,0),self.dw*dn.get(cid,0)+self.bw*bn.get(cid,0)))
        rows.sort(key=lambda x:x.hybrid_score,reverse=True); return rows[:k]
