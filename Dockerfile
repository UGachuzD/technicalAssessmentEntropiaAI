FROM python:3.11-slim

WORKDIR /app

# Instalar uv
RUN pip install --no-cache-dir uv

# Copiar archivos de dependencias primero
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

# Instalar dependencias
RUN uv sync --frozen --no-dev

# Copiar todo el código fuente
COPY src/ /app/src/
COPY data/ /app/data/
COPY main.py /app/main.py

EXPOSE 8000

# Ejecutar usando uv
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
