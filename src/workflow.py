from __future__ import annotations

from typing import Any, Dict
from .llm import getChatModel
from .router import decideFile
from .retriever import retrieveAndAnswer
from .stylist import styleAnswer


def questionAgent(
    user_question: str,
    model_name: str,
    include_trace: bool = True,
) -> Dict[str, Any]:

    llm = getChatModel(model_name=model_name)

    # Router
    decision = decideFile(llm, user_question)

    # Retriever
    original = retrieveAndAnswer(llm, decision.filename, user_question)

    # Stylist
    styled = styleAnswer(llm, original)

    # Trace info
    if include_trace:
        styled["_trace"] = {
            "user_question": user_question,
            "selected_file": decision.filename,
            "router_reason": decision.reason,
        }

    return styled
