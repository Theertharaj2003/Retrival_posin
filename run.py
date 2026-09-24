import argparse,json
from core.config import Config
from core.dataset import demo_chunks
from core.pipeline import TrustRAG
from experiments.run_experiment import run

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['demo','pipeline','experiment','gpu'],default='demo'); ap.add_argument('--query',default='What is knowledge poisoning in RAG?'); a=ap.parse_args(); cfg=Config.load()
    if a.mode=='gpu':
        from core.device import gpu_info; print(json.dumps(gpu_info(),indent=2)); return
    if a.mode=='experiment': run(); return
    rag=TrustRAG(demo_chunks(),cfg); out=rag.run(a.query,heal=True)
    if a.mode=='demo': print('TrustRAG demo OK'); print(out['answer']); print('healing_rounds=',out['healing_rounds'])
    else:
        print(json.dumps({'query':a.query,'answer':out['answer'],'healing_rounds':out['healing_rounds'],'trace':out['trace'],'evidence':[e.to_dict() for e in out['evidence']],'attributions':out['attributions']},indent=2))
if __name__=='__main__': main()
