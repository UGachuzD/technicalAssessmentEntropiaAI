from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

class StyledAnswers(BaseModel):
    original_answer: str = Field(description="The original answer generated in step 7")
    dry_answer: str = Field(description="Very concise and direct version in Spanish")
    funny_answer: str = Field(description="Humorous version in Spanish, still faithful to original answer")

def styleAnswer(llm, original_answer: str) -> dict:
    parser = PydanticOutputParser(pydantic_object=StyledAnswers)

    system = SystemMessage(content=(
        "Reescribe la respuesta en 3 tonos. "
        "IMPORTANTE: No inventes info nueva, no agregues datos. "
        "Solo reescribe lo ya dicho."
    ))

    human = HumanMessage(content=(
        f"RESPUESTA ORIGINAL:\n{original_answer}\n\n"
        f"{parser.get_format_instructions()}"
    ))

    msg = llm.invoke([system, human])
    content = msg.content
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and "text" in part
        ).strip()

    return parser.parse(content).model_dump()

