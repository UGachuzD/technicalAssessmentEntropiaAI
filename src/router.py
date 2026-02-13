from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

class RouteDecision(BaseModel):
    filename: str = Field(description="Uno de: datos_clima_mexico.csv, GPT-41_PromptingGuide.txt, maiz_info.jpg, NONE")
    reason: str = Field(description="Razón breve EN ESPAÑOL de por qué ese archivo es el adecuado")

def decideFile(llm, user_question: str) -> RouteDecision:
    parser = PydanticOutputParser(pydantic_object=RouteDecision)

    system = SystemMessage(content=(
        "Eres un router determinista. Debes elegir SOLO un archivo para responder.\n"
        "Archivos disponibles:\n"
        "1) datos_clima_mexico.csv -> preguntas de clima/temperaturas/estados/meses/años.\n"
        "2) GPT-41_PromptingGuide.txt -> preguntas sobre prompting, XML tags, guía GPT-4.1.\n"
        "3) maiz_info.jpg -> preguntas que se respondan observando el contenido de la imagen.\n"
        "Si la pregunta no se puede responder con ninguno, responde filename=NONE.\n"
        "Responde EXCLUSIVAMENTE en el formato solicitado."
    ))

    human = HumanMessage(content=(
        f"Pregunta del usuario: {user_question}\n\n"
        f"{parser.get_format_instructions()}"
    ))

    msg = llm.invoke([system, human])

    content = msg.content
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and "text" in part
        ).strip()

    return parser.parse(content)
