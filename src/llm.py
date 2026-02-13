import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

def getChatModel(model_name: str | None = None):
    model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"Falta GOOGLE_API_KEY (o GEMINI_API_KEY) en {ROOT/'.env'}")

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,       
        temperature=0,
    )
