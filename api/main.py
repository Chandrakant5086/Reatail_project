from fastapi import  FastAPI,HTTPException
from pydantic import BaseModel
from utils.logger import get_logger
from ml_model.inference import predict_sales
from rag_model.qa_chain import load_qa_chain



logger=get_logger(__name__)

app = FastAPI(title="Sales Prediction + RAG API")

# --------------------------------------------------
# Load RAG once at startup (IMPORTANT)
# --------------------------------------------------
try:
    qa_chain = load_qa_chain()
    logger.info("RAG QA chain loaded at startup")
except Exception as e:
    qa_chain = None
    logger.error("Failed to load RAG QA chain", exc_info=True)

# --------------------------------------------------
# ML PREDICTION MODELS
# --------------------------------------------------
class SalesRequest(BaseModel):
    promotion: int
    holiday: int


class SalesResponse(BaseModel):
    predicted_sales: float


# --------------------------------------------------
# RAG REQUEST MODEL
# --------------------------------------------------
class QuestionRequest(BaseModel):
    question: str


# --------------------------------------------------
# ML PREDICTION ENDPOINT
# --------------------------------------------------
@app.post("/predict", response_model=SalesResponse)
def predict(request: SalesRequest):
    try:
        logger.info(f"Received prediction request: {request}")

        prediction = predict_sales(
            promotion=request.promotion,
            holiday=request.holiday
        )

        return SalesResponse(predicted_sales=prediction)

    except Exception:
        logger.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction error")


# --------------------------------------------------
# RAG QUESTION-ANSWERING ENDPOINT
# --------------------------------------------------
@app.post("/ask")
def ask(req: QuestionRequest):
    if qa_chain is None:
        raise HTTPException(
            status_code=500,
            detail="RAG system not initialized"
        )

    try:
        logger.info(f"RAG question received: {req.question}")

        answer = qa_chain.run(req.question)

        return {
            "question": req.question,
            "answer": answer
        }

    except Exception:
        logger.error("RAG failed", exc_info=True)
        raise HTTPException(status_code=500, detail="RAG error")
