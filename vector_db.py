import requests
import json

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

pdfs_paths = [
    'pdfs/accident.pdf',
    'pdfs/AccidentConditions.pdf',
    'pdfs/AnnuityGeneralCondition.pdf',
    'pdfs/CancerConditions.pdf',
    'pdfs/DiseasesConditions.pdf',
    'pdfs/ExemptionConditions.pdf',
    'pdfs/InstantAnnuityOfferGeneralConditions.pdf',
    'pdfs/UniversalGeneralCondition.pdf',
    'pdfs/UniversalGeneralConditionGOLD_.pdf',
]

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=1000,
    chunk_overlap=10,
    length_function=len
)


def read_pdf_to_text(path: str) -> str:
    reader = PdfReader(f'{path}')
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text


def get_faq():
    res = requests.post('https://melico.ir/web_service/get_faq')
    raw_text = ''
    for i in res.json()['list']:
        raw_text += json.dumps(i)
    return raw_text


def get_vector_db(embeddings, create_new=False) -> FAISS:
    if not create_new:
        return FAISS.load_local("faiss_melico_conditions", embeddings)
    raw_text = ''
    for path in pdfs_paths:
        raw_text += read_pdf_to_text(path)
    raw_text += get_faq()
    texts = text_splitter.split_text(raw_text)

    vectordb = FAISS.from_texts(texts, embeddings)
    vectordb.save_local("faiss_melico_conditions")
    return vectordb

