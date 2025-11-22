from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from datetime import datetime

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


API_KEY = os.getenv("API_KEY", "dhsauidhasdhu123hh34i3h4hiu12hdsuh")

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.get("/")
async def root():
    return {"status": "online", "service": "Whisper API"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "whisper-small"}


@app.post("/transcribe")
@limiter.limit("10/minute")
async def transcribe_audio(file: UploadFile = File(...)):
    logger.info(f"Transcription request: {file.filename}")
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
        logger.info(f"Transcription completed: {len(result['text'])} chars")

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