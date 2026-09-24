import json
from pathlib import Path
from .types import Chunk

def read_documents(path):
    path=Path(path); docs=[]
    for p in sorted(path.rglob('*')):
        if not p.is_file(): continue
        try:
            if p.suffix.lower() in {'.txt','.md'}: text=p.read_text(encoding='utf-8',errors='ignore')
            elif p.suffix.lower()=='.json':
                obj=json.loads(p.read_text(encoding='utf-8')); text=json.dumps(obj, ensure_ascii=False)
            elif p.suffix.lower()=='.pdf':
                try:
                    from pypdf import PdfReader
                    text='\n'.join(page.extract_text() or '' for page in PdfReader(str(p)).pages)
                except Exception: continue
            else: continue
            if text.strip(): docs.append((p.stem,text,p.name))
        except Exception: continue
    return docs

def chunk_text(text, size=120, overlap=25):
    words=text.split(); out=[]; start=0
    while start<len(words):
        end=min(len(words),start+size); out.append(' '.join(words[start:end]));
        if end==len(words): break
        start=max(end-overlap,start+1)
    return out

def build_chunks(documents, size=120, overlap=25):
    chunks=[]
    for doc_id,text,source in documents:
        for i,part in enumerate(chunk_text(text,size,overlap)):
            chunks.append(Chunk(doc_id,f'{doc_id}_c{i+1}',part,source))
    return chunks
