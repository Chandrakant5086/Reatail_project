import os
try:
    from langchain.chains import RetrievalQA
except Exception:
    try:
        from langchain.chains.qa import RetrievalQA
    except Exception:
        RetrievalQA = None
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

from utils.logger import get_logger

logger = get_logger(__name__)

VECTOR_DB_PATH = "vbl_model/faiss_index"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

def load_qa_chain():
    """
    Load a QA chain. Prefer FAISS + Ollama when available; otherwise
    fall back to a TF-IDF retriever built from files listed in
    `rag_model/ingest_document.DOCUMENT_PATHS`.
    """
    # Lazy-import the Ollama integration to avoid importing heavy deps
    global ChatOllama, OllamaEmbeddings
    try:
        from langchain_ollama import ChatOllama as _ChatOllama, OllamaEmbeddings as _OllamaEmbeddings
        ChatOllama = _ChatOllama
        OllamaEmbeddings = _OllamaEmbeddings
    except Exception as e:
        logger.warning("langchain_ollama not available: %s", e)
        ChatOllama = None
        OllamaEmbeddings = None

    # Try to use FAISS if it's present and usable
    if FAISS is not None and os.path.isdir(VECTOR_DB_PATH) and OllamaEmbeddings is not None:
        try:
            logger.info("Loading existing FAISS vector store")
            embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            vector_db = FAISS.load_local(
                VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )

            # Initialize LLM if we have it
            llm = None
            if ChatOllama is not None:
                try:
                    llm = ChatOllama(model=LLM_MODEL)
                except Exception as e:
                    logger.warning("Failed to initialize ChatOllama: %s", e)

            if RetrievalQA is not None:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
                    return_source_documents=False,
                )
                logger.info("RAG QA chain loaded (FAISS)")
                return qa_chain

            # Fallback wrapper for when RetrievalQA isn't present
            class SimpleRetrievalQA:
                def __init__(self, llm, retriever):
                    self.llm = llm
                    self.retriever = retriever

                def run(self, query: str, k: int = 3):
                    docs = self.retriever.get_relevant_documents(query, k=k) if hasattr(self.retriever, 'get_relevant_documents') else self.retriever.get_relevant_documents(query)
                    contents = [getattr(d, 'page_content', str(d)) for d in docs[:k]]
                    return "\n\n".join(contents)

            return SimpleRetrievalQA(llm=llm, retriever=vector_db.as_retriever())

        except ImportError as ie:
            logger.warning("FAISS or its dependencies not available: %s", ie)
        except Exception as e:
            logger.warning("Failed to load FAISS index: %s", e)

    # FAISS path unavailable or failed; fall back to TF-IDF over local documents
    logger.info("Using TF-IDF fallback retriever built from local documents")

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from dataclasses import dataclass
    from rag_model.ingest_document import DOCUMENT_PATHS

    def _load_documents(paths):
        texts = []
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except Exception:
            PyPDFLoader = None

        for p in paths:
            if PyPDFLoader is not None and p.lower().endswith('.pdf'):
                try:
                    loader = PyPDFLoader(p)
                    docs = loader.load()
                    for d in docs:
                        texts.append(getattr(d, 'page_content', str(d)))
                    continue
                except Exception as e:
                    logger.warning('Failed to load PDF %s: %s', p, e)
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except Exception as e:
                logger.warning('Failed to read document %s: %s', p, e)

        return texts

    texts = _load_documents(DOCUMENT_PATHS)
    if not texts:
        raise RuntimeError('No documents available to build TF-IDF retriever. Run document ingestion first.')

    vectorizer = TfidfVectorizer().fit(texts)
    matrix = vectorizer.transform(texts)

    @dataclass
    class SimpleDoc:
        page_content: str

    class TFIDFRetriever:
        def __init__(self, texts, vectorizer, matrix):
            self.texts = texts
            self.vectorizer = vectorizer
            self.matrix = matrix

        def get_relevant_documents(self, query, k: int = 3):
            qv = self.vectorizer.transform([query])
            sims = (self.matrix @ qv.T).toarray().ravel()
            idx = np.argsort(-sims)[:k]
            return [SimpleDoc(self.texts[i]) for i in idx]

    retriever = TFIDFRetriever(texts, vectorizer, matrix)

    class SimpleRetrievalQA:
        def __init__(self, llm, retriever):
            self.llm = llm
            self.retriever = retriever

        def run(self, query: str, k: int = 3):
            docs = self.retriever.get_relevant_documents(query, k=k)
            contents = [getattr(d, 'page_content', str(d)) for d in docs[:k]]
            context = "\n\n".join(contents)
            if not self.llm:
                return context
            # Try to call LLM if available; wrap in try/except
            try:
                gen = self.llm.generate([context + "\n\nQuestion: " + query])
                if hasattr(gen, 'generations'):
                    return gen.generations[0][0].text
                return str(gen)
            except Exception:
                try:
                    res = self.llm(context + "\n\nQuestion: " + query)
                    return str(res)
                except Exception:
                    logger.warning('LLM call failed; returning retrieved context')
                    return context

    # attempt to initialize LLM if available
    llm = None
    if ChatOllama is not None:
        try:
            llm = ChatOllama(model=LLM_MODEL)
        except Exception as e:
            logger.warning('Failed to initialize LLM for TF-IDF retriever: %s', e)

    logger.info('TF-IDF retriever built and loaded')
    return SimpleRetrievalQA(llm=llm, retriever=retriever)
