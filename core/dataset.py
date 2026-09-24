from pathlib import Path
from .types import Chunk

def demo_chunks():
    rows=[
    ('doc_rag_001','Retrieval-Augmented Generation combines a retrieval component with a language model. The retrieval corpus can contain documents, web pages, or enterprise records.'),
    ('doc_security_001','Knowledge poisoning occurs when malicious or misleading content is inserted into a retrieval corpus so that a RAG system retrieves manipulated evidence.'),
    ('doc_healing_001','A self-healing RAG workflow can quarantine suspicious passages, re-retrieve alternatives, verify replacement evidence, and regenerate the answer. It should abstain when verified evidence is insufficient.'),
    ('doc_trust_001','Trust-aware RAG assigns different trust levels to evidence using relevance, source reliability, consistency, evidence support, and poisoning risk.'),
    ('doc_verify_001','Evidence verification checks whether retrieved passages support a claim. Contradictory evidence should reduce trust and may trigger recovery.'),
    ('doc_attr_001','Source attribution maps answer claims to supporting evidence passages so that users can inspect the provenance of important statements.'),
    ]
    return [Chunk(d,f'{d}_c1',t,d+'.txt') for d,t in rows]
