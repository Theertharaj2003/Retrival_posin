import hashlib
import html

import streamlit as st
from core.config import Config
from core.dataset import demo_chunks
from core.ingestion import build_chunks, read_uploaded_document
from core.pipeline import TrustRAG
from core.device import gpu_info

st.set_page_config(page_title='TrustRAG | Research Console', page_icon='T', layout='wide')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root {
  --ink: #18251f;
  --muted: #66756c;
  --paper: #f4f6f1;
  --surface: #ffffff;
  --line: #dfe6df;
  --green: #176b4d;
  --green-soft: #e7f2eb;
  --amber: #ad5a24;
  --amber-soft: #fbefe4;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
.block-container { max-width: 1280px; padding-top: 4.5rem; padding-bottom: 4rem; }
[data-testid="stSidebar"] { background: #e9eee8; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--ink); letter-spacing: 0 !important; }
h1 { font-size: 2rem !important; line-height: 1.15 !important; }
h2 { font-size: 1.2rem !important; }
p, label, .stMarkdown { color: var(--ink); }
.eyebrow { font: 500 11px 'DM Mono', monospace; color: var(--green); letter-spacing: 0; text-transform: uppercase; }
.masthead { display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; border-bottom: 1px solid var(--line); padding-bottom: 1.25rem; margin-bottom: 1.5rem; }
.brand { display: flex; align-items: center; gap: 13px; }
.brand-mark { display: grid; place-items: center; width: 42px; height: 42px; background: var(--green); color: white; font: 800 19px 'Manrope', sans-serif; }
.brand-title { margin: 0; font: 800 23px 'Manrope', sans-serif; letter-spacing: 0; color: var(--ink); }
.brand-subtitle { margin-top: 3px; font-size: 12px; color: var(--muted); }
.status { display: flex; align-items: center; gap: 8px; font: 500 11px 'DM Mono', monospace; color: var(--muted); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #4b936d; }
.section-kicker { display: flex; align-items: center; gap: 9px; margin: 1.6rem 0 .75rem; font: 500 11px 'DM Mono', monospace; text-transform: uppercase; color: var(--muted); }
.section-kicker span { color: var(--green); }
.answer-panel { background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--green); padding: 1.35rem 1.5rem; }
.answer-label { margin-bottom: 9px; font: 500 10px 'DM Mono', monospace; color: var(--green); text-transform: uppercase; }
.answer-text { font-size: 16px; line-height: 1.7; color: var(--ink); }
.metric { height: 100%; background: var(--surface); border-top: 2px solid var(--green); border-bottom: 1px solid var(--line); padding: 13px 15px; }
.metric.warn { border-top-color: var(--amber); }
.metric-label { font: 500 10px 'DM Mono', monospace; text-transform: uppercase; color: var(--muted); }
.metric-value { margin-top: 5px; font: 700 25px 'Manrope', sans-serif; color: var(--ink); }
.metric-note { margin-top: 2px; font-size: 11px; color: var(--muted); }
.evidence-meta { display: flex; flex-wrap: wrap; gap: 8px 16px; margin: 10px 0; font: 400 11px 'DM Mono', monospace; color: var(--muted); }
.evidence-copy { padding: 12px 14px; background: #f6f8f4; border-left: 2px solid var(--line); font-size: 13px; line-height: 1.6; }
.evidence-copy.evidence-risk { background: #fff3ee; border-left: 4px solid #c45d37; }
.evidence-risk mark { background: #ffd7c2; color: #71351f; }
.tag-safe { color: var(--green); }
.tag-risk { color: var(--amber); }
div[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--line); border-radius: 3px; margin-bottom: 8px; }
div[data-testid="stExpander"] summary { font-size: 13px; }
.stButton > button[kind="primary"] { background: var(--green); border: 1px solid var(--green); color: white; border-radius: 3px; min-height: 43px; font-weight: 700; }
.stButton > button[kind="primary"]:hover { background: #10573e; border-color: #10573e; color: white; }
div[data-testid="stBaseButton-primary"] { color: white !important; }
.stButton > button p { color: white !important; }
.stTextArea textarea { border: 1px solid #cbd6cc; border-radius: 3px; background: white; color: var(--ink) !important; -webkit-text-fill-color: var(--ink); font-size: 14px; line-height: 1.55; }
.stTextArea textarea::placeholder { color: var(--muted) !important; opacity: 1; }
div[data-testid="stDataFrame"] { border: 1px solid var(--line); }
@media (max-width: 700px) {
  .block-container { padding: 1rem 1rem 3rem; }
  .masthead { align-items: flex-start; }
  .status { margin-top: 4px; }
  .brand-title { font-size: 20px; }
}
</style>
""", unsafe_allow_html=True)

info = gpu_info()
st.markdown(f"""
<header class="masthead">
  <div class="brand">
    <div class="brand-mark">T</div>
    <div>
      <div class="brand-title">TrustRAG</div>
      <div class="brand-subtitle">Evidence integrity research console</div>
    </div>
  </div>
  <div class="status"><span class="status-dot"></span> SYSTEM READY <span style="color:#c3cec4">/</span> {info.get('name') or 'CPU'}</div>
</header>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="eyebrow">01 / Investigation</div>', unsafe_allow_html=True)
    st.markdown("### Ask a question")
    query = st.text_area(
        'Research query',
        'What is knowledge poisoning in RAG?',
        height=125,
        label_visibility='collapsed',
        placeholder='Enter a question to investigate...',
    )
    uploaded_file = st.file_uploader(
        'Analyze a document',
        type=['pdf', 'docx'],
        help='Upload a PDF or Word (.docx) file to use it as the retrieval corpus.',
    )
    run_query = st.button('Run TrustRAG', type='primary', use_container_width=True)
    st.markdown('---')
    st.markdown('<div class="eyebrow">02 / Runtime</div>', unsafe_allow_html=True)
    st.caption(f"Device: {info.get('name') or 'CPU'}")
    st.caption(f"CUDA available: {'Yes' if info.get('cuda') else 'No'}")
    st.caption(f"VRAM: {info.get('vram_gb', 0)} GB")
    st.markdown('---')
    st.caption('Trust-aware retrieval · poison detection · self-healing · claim attribution')

uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
active_corpus_key = (
    hashlib.sha256(uploaded_bytes).hexdigest() if uploaded_bytes is not None else 'demo'
)
corpus_label = html.escape(uploaded_file.name) if uploaded_file is not None else 'DEMO CORPUS'

st.markdown(
    f'<div class="eyebrow">RESEARCH WORKSPACE / {corpus_label}</div>',
    unsafe_allow_html=True,
)
st.markdown('## Inspect an answer, not just a response')
st.markdown(
    'Run a query to retrieve evidence, verify claims, and quarantine suspicious sources. '
    'Suspicious passages are highlighted in the evidence inspector.'
)

if run_query:
    if not query.strip():
        st.warning('Enter a research question before running the pipeline.')
    else:
        try:
            chunks = (
                build_chunks([
                    read_uploaded_document(uploaded_file.name, uploaded_bytes)
                ])
                if uploaded_file is not None and uploaded_bytes is not None
                else demo_chunks()
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            with st.spinner('Retrieving and verifying evidence...'):
                result = TrustRAG(chunks, Config.load()).run(query.strip(), heal=True)
            st.session_state['trustrag_result'] = result
            st.session_state['trustrag_corpus_key'] = active_corpus_key

has_current_result = (
    'trustrag_result' in st.session_state
    and st.session_state.get('trustrag_corpus_key') == active_corpus_key
)

if has_current_result:
    result = st.session_state['trustrag_result']
    evidence = result['evidence']
    quarantined_count = sum(item.quarantined for item in evidence)

    st.markdown('<div class="section-kicker"><span>03</span> Run summary</div>', unsafe_allow_html=True)
    metric_columns = st.columns(4)
    metric_data = [
        ('Evidence retrieved', len(evidence), 'ranked passages', False),
        ('Quarantined', quarantined_count, 'flagged as unsafe', quarantined_count > 0),
        ('Healing rounds', result['healing_rounds'], 're-retrieval cycles', False),
        ('Claim links', len(result['attributions']), 'attributed claims', False),
    ]
    for column, (label, value, note, warning) in zip(metric_columns, metric_data):
        variant = ' warn' if warning else ''
        column.markdown(
            f'<div class="metric{variant}"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-kicker"><span>04</span> Grounded response</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer-panel"><div class="answer-label">Answer / {len(result["attributions"])} claim links</div>'
        f'<div class="answer-text">{html.escape(result["answer"])}</div></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.55, 1], gap='large')
    with left:
        st.markdown('<div class="section-kicker"><span>05</span> Evidence inspector</div>', unsafe_allow_html=True)
        for item in evidence:
            state = 'QUARANTINED' if item.quarantined else 'RETAINED'
            label = f"{item.chunk.chunk_id}  /  trust {item.trust_score:.2f}  /  {state}"
            with st.expander(label, expanded=False):
                evidence_class = 'evidence-copy evidence-risk' if item.quarantined else 'evidence-copy'
                evidence_text = html.escape(item.chunk.text)
                if item.quarantined:
                    evidence_text = f'<mark>{evidence_text}</mark>'
                st.markdown(
                    f'<div class="{evidence_class}">{evidence_text}</div>',
                    unsafe_allow_html=True,
                )
                tag_class = 'tag-risk' if item.quarantined else 'tag-safe'
                st.markdown(
                    f'<div class="evidence-meta"><span>Source: {html.escape(item.chunk.source)}</span>'
                    f'<span class="{tag_class}">Status: {state.lower()}</span>'
                    f'<span>Verification: {item.verification}</span>'
                    f'<span>Poison risk: {item.poison_risk:.3f}</span></div>',
                    unsafe_allow_html=True,
                )
                st.json({
                    'dense': item.dense_score,
                    'bm25': item.bm25_score,
                    'hybrid': item.hybrid_score,
                    'rerank': item.rerank_score,
                    'trust': item.trust_score,
                    'poison_label': item.chunk.is_poisoned,
                })

    with right:
        st.markdown('<div class="section-kicker"><span>06</span> Healing trace</div>', unsafe_allow_html=True)
        if result['trace']:
            for event in result['trace']:
                st.markdown(
                    f"**Round {event['round']}** · {event['action']}  \n"
                    f"Quarantined: {', '.join(event['quarantined']) or 'none'}"
                )
        else:
            st.caption('No re-retrieval was needed for this query.')

        st.markdown('<div class="section-kicker"><span>07</span> Claim attribution</div>', unsafe_allow_html=True)
        if result['attributions']:
            st.dataframe(result['attributions'], use_container_width=True, hide_index=True)
        else:
            st.caption('No trusted evidence was available to attribute claims.')
else:
    st.markdown('<div class="section-kicker"><span>03</span> Awaiting investigation</div>', unsafe_allow_html=True)
    if uploaded_file is not None:
        st.info('Submit a question to analyze the uploaded document and inspect highlighted evidence.')
    else:
        st.info('Submit a question from the left panel to inspect retrieved evidence and its trust signals.')
