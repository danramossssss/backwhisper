from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os
import uvicorn

app = FastAPI(title="Whisper API", version="1.0.0")

# ✅ CORS para produção - especifique seus domínios
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Desenvolvimento
    "https://learnfun-sigma.vercel.app/",  # Produção frontend
    "https://seu-dominio43414123.com",  # Seu domínio
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Cache do modelo - carregar uma vez
_model = None

def get_model():
    global _model
    if _model is None:
        print("🔄 Carregando modelo Whisper...")
        _model = whisper.load_model("small")
        print("✅ Modelo carregado!")
    return _model


@app.get("/")
async def root():
    return {"status": "online", "service": "Whisper API"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "whisper-small"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcrever áudio"""

    # ✅ Validação de tamanho (max 25MB)
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
    file_size = 0

    try:
        # Validar extensão
        allowed_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm']
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Formato não suportado. Use: {', '.join(allowed_extensions)}"
            )

        # Salvar arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Ler em chunks para controlar tamanho
            while chunk := await file.read(8192):  # 8KB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    os.unlink(temp_file.name)
                    raise HTTPException(
                        status_code=413,
                        detail="Arquivo muito grande. Máximo: 25MB"
                    )
                temp_file.write(chunk)
            temp_path = temp_file.name

        print(f"📁 Arquivo: {file.filename} ({file_size/1024:.1f}KB)")

        # Transcrever
        model = get_model()
        result = model.transcribe(temp_path)

        # Limpar arquivo temporário
        os.unlink(temp_path)

        print(f"✅ Transcrito: {result['text'][:50]}...")

        return {
            "success": True,
            "text": result["text"],
            "language": result["language"]
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)