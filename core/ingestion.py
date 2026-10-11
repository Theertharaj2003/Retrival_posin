import json
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile
from xml.etree import ElementTree

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .types import Chunk


def read_uploaded_document(filename: str, content: bytes) -> tuple[str, str, str]:
    """Extract searchable text from an uploaded PDF or Word document."""
    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix not in {'.pdf', '.docx'}:
        raise ValueError('Unsupported file type. Upload a PDF or Word (.docx) file.')
    if not content:
        raise ValueError('The uploaded file is empty.')

    if suffix == '.pdf':
        try:
            reader = PdfReader(BytesIO(content), strict=False)
            if reader.is_encrypted:
                raise ValueError('Password-protected PDFs are not supported.')
            text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        except (OSError, PdfReadError, ValueError) as exc:
            raise ValueError(f'Unable to read PDF: {exc}') from exc
    else:
        try:
            with ZipFile(BytesIO(content)) as archive:
                document_xml = archive.read('word/document.xml')
            root = ElementTree.fromstring(document_xml)
        except (BadZipFile, KeyError, ElementTree.ParseError) as exc:
            raise ValueError('Unable to read Word document. Make sure it is a valid .docx file.') from exc

        word_namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        paragraphs = []
        for paragraph in root.iter(f'{word_namespace}p'):
            parts = []
            for element in paragraph.iter():
                if element.tag == f'{word_namespace}t':
                    parts.append(element.text or '')
                elif element.tag == f'{word_namespace}tab':
                    parts.append('\t')
                elif element.tag in {f'{word_namespace}br', f'{word_namespace}cr'}:
                    parts.append('\n')
            paragraph_text = ''.join(parts).strip()
            if paragraph_text:
                paragraphs.append(paragraph_text)
        text = '\n'.join(paragraphs)

    if not text.strip():
        raise ValueError('No extractable text was found in the uploaded file.')
    return path.stem, text, path.name


def read_documents(path):
    path=Path(path); docs=[]
    for p in sorted(path.rglob('*')):
        if not p.is_file(): continue
        try:
            if p.suffix.lower() in {'.txt','.md'}: text=p.read_text(encoding='utf-8',errors='ignore')
            elif p.suffix.lower()=='.json':
                obj=json.loads(p.read_text(encoding='utf-8')); text=json.dumps(obj, ensure_ascii=False)
            elif p.suffix.lower() in {'.pdf', '.docx'}:
                _, text, _ = read_uploaded_document(p.name, p.read_bytes())
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
