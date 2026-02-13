from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict
import os

from src.workflow import questionAgent

app = FastAPI(title="Entropia AI Agent", version="1.0.0")


class QuestionRequest(BaseModel):
    user_question: str = Field(min_length=1, description="Pregunta del usuario en español")


class QuestionResponse(BaseModel):
    original_answer: str
    dry_answer: str
    funny_answer: str
    _trace: Dict[str, Any] | None = None


@app.post("/question_agent", response_model=QuestionResponse)
def question_agent(req: QuestionRequest):
    try:
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        out = questionAgent(req.user_question, model_name=model_name, include_trace=True)

        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
