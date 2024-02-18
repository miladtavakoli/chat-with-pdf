import requests
import json

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

pdfs_paths = [
    'pdfs/1565066687.45_WPBLMIZ9.pdf',
    'pdfs/1565066860.09_LS7ORFN4.pdf',
    'pdfs/1565066883.24_COAO2QV7.pdf',
    'pdfs/1565066914.13_J1DY65CW.pdf',
    'pdfs/1565069940.23_DOV75GXW.pdf',
    'pdfs/1565070036.81_QY4NH7QC.pdf',
    'pdfs/1566192090.7_YUJZAXZB.pdf',
    'pdfs/1573369544.2_EHX9Y4S5.pdf',
    'pdfs/1573369883.41_LRCJFK3J.pdf',
    'pdfs/1589348670.54_XHQMBKYW.pdf',
    'pdfs/1591602815.52_PPP711J3.pdf',
    'pdfs/1651312981.18_A7TE4G57.pdf',
    'pdfs/1651313096.25_AZ9P794C.pdf',
    'pdfs/1657698022.04_AGLKGCFO.pdf',
    'pdfs/1658824039.66_BX2OAMNS.pdf',
    'pdfs/1662277086.51_ZGKHGRY7.pdf',
    'pdfs/1664187747.84_2TWIS6XW.pdf',
    'pdfs/1669190912.22_FHGWVRB6.pdf',
    'pdfs/1670739488.23_VUWWFDPC.pdf',
    'pdfs/1670925247.87_L0WDS2IF.pdf',
    'pdfs/1673946631.59_D7HI67TC.pdf',
    'pdfs/1685433098.27_JNRBC4GF.pdf',
    'pdfs/1687067777.08_539Z86VX.pdf',
    'pdfs/1687350440.94_O2OAZQAX.pdf',
    'pdfs/1694348094.89_X6SDP8ZL.V03.1402.04.06.pdf',
    'pdfs/1694348634.39_FWQFO1KP.pdf',
    'pdfs/1694349231.11_W63ZM84T.pdf',
    'pdfs/1698827772.68_TQNBXU3G.1 - 14020810 (2).pdf',
    'pdfs/1698827875.65_L5ORF74E.pdf',
    'pdfs/1700658145.66_XJ2A38AM.pdf',
    'pdfs/1700661672.86_6VGTCQ0J.pdf',
    'pdfs/1702108653.27_DZXSPS12.pdf',
    'pdfs/1705498098.33_XWTS4V09.pdf',
    'pdfs/1705999665.14_P7I2ACCA.pdf',
    'pdfs/1706610580.68_XACMZFPH.pdf',
    'pdfs/accident.pdf',
    'pdfs/AccidentConditions.pdf',
    'pdfs/AnnuityGeneralCondition.pdf',
    'pdfs/CancerConditions.pdf',
    'pdfs/DefannuityGeneralConditions.pdf',
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
