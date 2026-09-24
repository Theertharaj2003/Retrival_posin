import streamlit as st
from core.config import Config
from core.dataset import demo_chunks
from core.pipeline import TrustRAG
from core.device import gpu_info

st.set_page_config(page_title='TrustRAG Research Console',page_icon='🛡️',layout='wide')
st.title('🛡️ TrustRAG Research Console')
st.caption('Trust-aware + self-healing retrieval, evidence verification, attribution and secure grounding')
info=gpu_info(); st.sidebar.subheader('System'); st.sidebar.write(f"GPU: {info.get('name') or 'CPU'}"); st.sidebar.write(f"CUDA: {info.get('cuda')}"); st.sidebar.write(f"VRAM: {info.get('vram_gb',0)} GB")
q=st.text_area('Research query','What is knowledge poisoning in RAG?',height=90)
if st.button('Run TrustRAG',type='primary'):
    result=TrustRAG(demo_chunks(),Config.load()).run(q,heal=True)
    a,b,c,d=st.columns(4); a.metric('Evidence',len(result['evidence'])); b.metric('Quarantined',sum(e.quarantined for e in result['evidence'])); c.metric('Healing rounds',result['healing_rounds']); d.metric('Attributions',len(result['attributions']))
    st.subheader('Grounded Answer'); st.write(result['answer'])
    st.subheader('Healing Trace'); st.json(result['trace'])
    st.subheader('Evidence Inspector')
    for e in result['evidence']:
        with st.expander(f"{e.chunk.chunk_id} — trust {e.trust_score:.3f}"):
            st.write(e.chunk.text); st.write({'dense':e.dense_score,'bm25':e.bm25_score,'hybrid':e.hybrid_score,'rerank':e.rerank_score,'poison_risk':e.poison_risk,'verification':e.verification,'quarantined':e.quarantined,'poison_label':e.chunk.is_poisoned})
    st.subheader('Claim Attribution'); st.dataframe(result['attributions'],use_container_width=True)
