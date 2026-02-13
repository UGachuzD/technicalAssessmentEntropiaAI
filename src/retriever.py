from pathlib import Path
import re
import pandas as pd
import base64
from langchain_core.messages import SystemMessage, HumanMessage

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def loadTXTContext(path: Path, user_question: str, max_chars: int = 12000) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")

    keywords = [w.lower() for w in re.findall(r"[A-Za-z0-9\-\.]+", user_question) if len(w) >= 4]
    lines = text.splitlines()
    scored = []
    for i, line in enumerate(lines):
        l = line.lower()
        score = sum(1 for k in keywords if k in l)
        if score > 0:
            scored.append((score, i, line))

    scored.sort(reverse=True)
    picked = [f"L{idx+1}: {line}" for _, idx, line in scored[:120]]
    ctx = "\n".join(picked) if picked else text[:max_chars]
    return ctx[:max_chars]

def loadCSVContext(path: Path, user_question: str, max_rows: int = 400) -> str:
    df = pd.read_csv(path)

    # Parse fecha
    df["PERIODO"] = pd.to_datetime(df["PERIODO"], errors="coerce")

    # Extraer año
    year = None
    m = re.search(r"(19|20)\d{2}", user_question)
    if m:
        year = int(m.group(0))

    # Extraer mes
    month = None
    months = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
        "noviembre": 11, "diciembre": 12
    }
    ql = user_question.lower()
    for name, num in months.items():
        if name in ql:
            month = num
            break

    # Filtrado determinista
    df_f = df.copy()
    if year is not None:
        df_f = df_f[df_f["PERIODO"].dt.year == year]
    if month is not None:
        df_f = df_f[df_f["PERIODO"].dt.month == month]

    if df_f.empty:
        return (
            "CSV_SCHEMA:\n"
            f"columns={list(df.columns)}\n\n"
            f"NO_HAY_FILAS para year={year}, month={month}.\n"
            "Instrucción: responde explícitamente que el dataset no contiene esos registros.\n"
        )

    cols = ["PERIODO", "ENTIDAD", "TEMP_MINIMA", "TEMP_MEDIA", "TEMP_MAXIMA", "PRECIPITACION"]
    sample = df_f[cols].head(max_rows)

    return (
        "CSV_SCHEMA:\n"
        f"columns={cols}\n\n"
        f"FILTRADO: year={year}, month={month}\n"
        f"TOTAL_FILAS_FILTRADAS: {len(df_f)}\n\n"
        "CSV_ROWS_FILTRADAS:\n"
        f"{sample.to_csv(index=False)}\n"
        "INSTRUCCIÓN: Responde SOLO con base en estas filas.\n"
    )

def imageToDataURL(image_path: str) -> str:
    path = Path(image_path)
    mime = "image/jpeg" 
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def answerFromImageLangChain(llm, user_question: str, image_path: str) -> str:
    data_url = imageToDataURL(image_path)

    system = SystemMessage(content=(
        "Responde SOLO usando lo que se ve/lee en la imagen. "
        "Si la respuesta no está en la imagen, responde exactamente: "
        "'No tengo información suficiente en la imagen.'"
    ))

    human = HumanMessage(content=[
        {"type": "text", "text": f"PREGUNTA: {user_question}"},
        {"type": "image_url", "image_url": data_url},
    ])

    return llm.invoke([system, human]).content


def retrieveAndAnswer(llm, filename: str, user_question: str) -> str:
    """
    Loads file content, creates a "contexted" LLM call, returns grounded answer.
    """
    if filename == "NONE":
        return "No puedo responder porque no hay un archivo aplicable en data/ para esta pregunta."

    path = DATA_DIR / filename
    if not path.exists():
        return f"No encontré el archivo esperado: {filename}"

    system = SystemMessage(content=(
        "Responde SOLO usando el CONTEXTO proporcionado. "
        "Si el contexto no contiene la respuesta, di: "
        "'No tengo información suficiente en el archivo proporcionado.' "
        "No uses conocimiento externo."
    ))

    if filename.endswith(".txt"):
        ctx = loadTXTContext(path, user_question)
        human = HumanMessage(content=(
            f"CONTEXTO (extraído del TXT):\n{ctx}\n\n"
            f"PREGUNTA: {user_question}"
        ))
        return llm.invoke([system, human]).content

    if filename.endswith(".csv"):
        ctx = loadCSVContext(path, user_question)
        human = HumanMessage(content=(
            f"CONTEXTO (derivado del CSV):\n{ctx}\n\n"
            f"PREGUNTA: {user_question}\n"
            "Si la pregunta pide 'top N', explica el criterio y muestra N resultados."
        ))
        return llm.invoke([system, human]).content

    if filename.endswith(".jpg"):
        return answerFromImageLangChain(llm, user_question, str(path))



    return "Formato de archivo no soportado."
