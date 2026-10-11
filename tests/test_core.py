from io import BytesIO
from zipfile import ZipFile

import pytest

from core.dataset import demo_chunks
from core.ingestion import build_chunks, read_documents, read_uploaded_document
from core.poisoning import poison_chunks
from core.config import Config
from core.pipeline import TrustRAG


def test_poison_generation_labels():
    p=poison_chunks(demo_chunks(),'factual'); assert any(x.is_poisoned for x in p)


def test_pipeline_returns_answer():
    r=TrustRAG(demo_chunks(),Config.load()).run('What is knowledge poisoning in RAG?'); assert isinstance(r['answer'],str) and r['answer']


def test_self_healing_removes_marked_poison():
    p=poison_chunks(demo_chunks(),'indirect_injection'); r=TrustRAG(p,Config.load()).run('What is knowledge poisoning in RAG?',heal=True); assert all(not e.chunk.is_poisoned for e in r['evidence'] if not e.quarantined)


def test_uploaded_docx_extracts_text_for_chunking(tmp_path):
    document_xml = b'''<?xml version="1.0" encoding="UTF-8"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>First paragraph.</w:t></w:r></w:p>
        <w:p><w:r><w:t>Second paragraph.</w:t></w:r></w:p>
      </w:body>
    </w:document>'''
    content = BytesIO()
    with ZipFile(content, 'w') as archive:
        archive.writestr('word/document.xml', document_xml)

    document_bytes = content.getvalue()
    document = read_uploaded_document('research.docx', document_bytes)
    (tmp_path / 'research.docx').write_bytes(document_bytes)
    chunks = build_chunks([document])

    assert document == ('research', 'First paragraph.\nSecond paragraph.', 'research.docx')
    assert read_documents(tmp_path) == [document]
    assert len(chunks) == 1
    assert chunks[0].source == 'research.docx'
    assert chunks[0].text == 'First paragraph. Second paragraph.'


def test_uploaded_pdf_extracts_text():
    objects = [
        b'<< /Type /Catalog /Pages 2 0 R >>',
        b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] '
        b'/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    ]
    stream = b'BT /F1 12 Tf 40 50 Td (Readable PDF text.) Tj ET'
    objects.append(
        b'<< /Length ' + str(len(stream)).encode()
        + b' >>\nstream\n' + stream + b'\nendstream'
    )
    pdf = b'%PDF-1.4\n'
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += f'{index} 0 obj\n'.encode() + obj + b'\nendobj\n'
    xref_offset = len(pdf)
    pdf += f'xref\n0 {len(objects) + 1}\n'.encode() + b'0000000000 65535 f \n'
    for offset in offsets[1:]:
        pdf += f'{offset:010} 00000 n \n'.encode()
    pdf += (
        f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n'
        f'startxref\n{xref_offset}\n%%EOF'
    ).encode()

    assert read_uploaded_document('manual.pdf', pdf) == (
        'manual',
        'Readable PDF text.',
        'manual.pdf',
    )


def test_uploaded_document_rejects_unsupported_and_invalid_files():
    with pytest.raises(ValueError, match='Unsupported file type'):
        read_uploaded_document('research.doc', b'content')

    with pytest.raises(ValueError, match='valid .docx'):
        read_uploaded_document('research.docx', b'not a Word document')
