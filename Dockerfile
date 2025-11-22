FROM python:3.11-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY whisper_api_fastapi.py .

# Expor porta
EXPOSE 8000

# Comando para rodar
CMD ["python", "whisper_api_fastapi.py"]