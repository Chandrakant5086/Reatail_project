from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from utils.logger import get_logger
import os

logger=get_logger(__name__)

DOCUMENT_PATHS=[
    "documents/pricing_policy.pdf",
    "documents/store_operations_sop.pdf"
]

VECTOR_DB_PATH="vbl_model/faiss_index"

def ingest_documents():
    logger.info("starting document  ingestion")

    all_docs=[]
    for path  in  DOCUMENT_PATHS:
        loader=PyPDFLoader(path)
        docs=loader.load()
        all_docs.extend(docs)

        splitter=RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunk=splitter.split_documents(all_docs)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")

     

        vector_db = FAISS.from_documents(chunk, embeddings)
        vector_db.save_local(VECTOR_DB_PATH)

        logger.info("FAISS vector store  created succedfully")

if  __name__=="__main__":
 ingest_documents()