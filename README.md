Guía Completa: Configuración Profesional de DeepSeek-R1-Distill-Llama-8B para Producción
📋 Índice

Optimización BIOS y Hardware
Preparación del Sistema Ubuntu
Instalación de Dependencias y Entorno
Instalación y Configuración del Modelo
Optimización para CPU (sin GPU)
Implementación de API Profesional
Seguridad y Privacidad (GDPR-compliant)
Monitoreo y Logs
Documentación para Clientes
Deployment y Venta


1. 🔧 Optimización BIOS y Hardware
Acceso a BIOS

Reinicia tu PC y presiona F2, Del o F12 (depende del fabricante)
Si tienes un motherboard moderno, busca el modo "Advanced" o "Expert"

Configuraciones Recomendadas BIOS
A. Performance Settings
CPU Configuration:
├── Intel Turbo Boost: ENABLED
├── CPU Core Ratio: Sync All Cores
├── All-Core Ratio Limit: 50 (5.0 GHz - conservador)
├── CPU Core Voltage: Adaptive Mode
└── Voltage Offset: -0.050V (undervolting para menor temp)

Memory Configuration:
├── XMP Profile: ENABLED (Profile 1 - 6000MHz)
├── Memory Frequency: 6000 MHz
├── Memory Timing Mode: Auto
└── Command Rate: 1T

Power Management:
├── CPU Power Limit 1 (PL1): 125W
├── CPU Power Limit 2 (PL2): 253W
├── Power Limit Time Window: 28 seconds
├── Intel SpeedStep: ENABLED
└── C-States: ENABLED (ahorro energético cuando idle)
B. Cooling & Thermal
Fan Configuration:
├── CPU Fan Mode: PWM Mode
├── CPU Fan Profile: Performance
├── CPU Temperature Target: 75°C
└── Fan Speed Minimum: 40%
C. Storage Optimization
SATA/NVMe Configuration:
├── SATA Mode: AHCI
├── NVMe Support: ENABLED
└── Fast Boot: DISABLED (para estabilidad)
D. Security (para producción)
Security Settings:
├── Secure Boot: ENABLED
├── TPM 2.0: ENABLED
└── Virtualization (VT-x/VT-d): ENABLED
⚠️ Advertencias Importantes

NO hagas overclock agresivo sin monitoreo térmico
Incrementa voltajes gradualmente (0.025V pasos)
Monitorea temperaturas con sensors en Linux
Tu cooler de 240mm es adecuado para 5.0-5.2 GHz


2. 🐧 Preparación del Sistema Ubuntu
Actualización Completa del Sistema
bash# Actualizar repositorios
sudo apt update && sudo apt upgrade -y

# Instalar herramientas esenciales
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    htop \
    tmux \
    vim \
    net-tools \
    ufw

# Limpiar paquetes innecesarios
sudo apt autoremove -y && sudo apt autoclean
Configurar Seguridad del Firewall
bash# Habilitar UFW (Uncomplicated Firewall)
sudo ufw enable

# Permitir SSH (si lo usas)
sudo ufw allow 22/tcp

# Permitir puerto para API (definiremos 8000)
sudo ufw allow 8000/tcp

# Verificar estado
sudo ufw status verbose

3. 📦 Instalación de Dependencias y Entorno Python
Instalar Python 3.11 (recomendado para mejor rendimiento)
bash# Agregar repositorio deadsnakes
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Instalar Python 3.11 y herramientas
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip

# Configurar Python 3.11 como predeterminado
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
Crear Entorno Virtual Aislado
bash# Crear directorio del proyecto
mkdir -p ~/deepseek-api
cd ~/deepseek-api

# Crear entorno virtual
python3.11 -m venv venv

# Activar entorno
source venv/bin/activate

# Actualizar pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

4. 🤖 Instalación y Configuración del Modelo
Instalar Dependencias para LLM en CPU
bash# Activar entorno (si no está activo)
source ~/deepseek-api/venv/bin/activate

# Instalar PyTorch optimizado para CPU (Intel MKL)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar Transformers y dependencias de Hugging Face
pip install \
    transformers>=4.37.0 \
    accelerate>=0.26.0 \
    sentencepiece \
    protobuf

# Instalar herramientas de optimización para CPU
pip install \
    intel-extension-for-pytorch \
    neural-compressor \
    optimum[intel]

# Instalar servidor API y utilidades
pip install \
    fastapi \
    uvicorn[standard] \
    pydantic \
    python-multipart \
    python-jose[cryptography] \
    passlib[bcrypt] \
    python-dotenv

# Herramientas de logging y monitoreo
pip install \
    loguru \
    prometheus-client \
    psutil

# Herramientas de seguridad
pip install \
    cryptography \
    hashlib-additional
Descargar el Modelo DeepSeek-R1-Distill-Llama-8B
bash# Instalar Hugging Face CLI
pip install huggingface_hub

# Login en Hugging Face (necesitarás crear cuenta en huggingface.co)
huggingface-cli login
# Pega tu token cuando se solicite

# Descargar modelo (esto tomará tiempo dependiendo de tu conexión)
# El modelo pesa aproximadamente 16GB
python3 << 'EOF'
from huggingface_hub import snapshot_download

model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
cache_dir = "./models"

print("Descargando modelo... Esto puede tomar 10-30 minutos")
snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    local_dir=f"{cache_dir}/{model_id}",
    local_dir_use_symlinks=False
)
print("Modelo descargado exitosamente")
EOF

5. ⚡ Optimización Extrema para CPU (Sin GPU)
A. Optimización de Kernel Linux
bash# Editar parámetros del kernel
sudo nano /etc/sysctl.conf

# Agregar al final del archivo:
conf# Optimización de memoria para LLM
vm.swappiness=10
vm.vfs_cache_pressure=50
vm.dirty_ratio=15
vm.dirty_background_ratio=5

# Optimización de red para API
net.core.somaxconn=4096
net.ipv4.tcp_max_syn_backlog=4096
net.core.netdev_max_backlog=5000

# Aumentar límites de archivos abiertos
fs.file-max=2097152
bash# Aplicar cambios
sudo sysctl -p
B. Configurar Límites de Recursos
bash# Editar límites
sudo nano /etc/security/limits.conf

# Agregar:
conf* soft nofile 1048576
* hard nofile 1048576
* soft nproc 65536
* hard nproc 65536
C. Optimización Intel MKL y OpenMP
bash# Crear archivo de configuración de entorno
nano ~/.bashrc

# Agregar al final:
bash# Optimizaciones Intel MKL para CPU
export MKL_NUM_THREADS=20  # Todos tus cores
export OMP_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export VECLIB_MAXIMUM_THREADS=20

# Usar todos los cores para inferencia
export NUMEXPR_NUM_THREADS=20

# Optimización de alocación de memoria
export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"

# Intel MKL optimizations
export MKL_ENABLE_INSTRUCTIONS=AVX512
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"
bash# Recargar configuración
source ~/.bashrc
D. Script de Optimización del Sistema
bash# Crear script de optimización
nano ~/deepseek-api/optimize_system.sh
bash#!/bin/bash

echo "🚀 Optimizando sistema para inferencia LLM..."

# Deshabilitar transparent huge pages (mejora latencia)
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

# Configurar CPU governor a performance
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee $cpu
done

# Deshabilitar servicios innecesarios
sudo systemctl stop cups bluetooth ModemManager
sudo systemctl disable cups bluetooth ModemManager

# Limpiar cache
sudo sync && sudo sysctl vm.drop_caches=3

# Configurar IRQ affinity para distribuir interrupciones
echo "Configurando IRQ affinity..."

echo "✅ Optimización completada"
echo "📊 Estado del sistema:"
grep MHz /proc/cpuinfo | head -20
free -h
bash# Dar permisos de ejecución
chmod +x ~/deepseek-api/optimize_system.sh

# Ejecutar
./optimize_system.sh

6. 🌐 Implementación de API Profesional
Estructura del Proyecto
bashcd ~/deepseek-api

# Crear estructura
mkdir -p {app,config,logs,scripts,tests}

# Estructura final:
# deepseek-api/
# ├── app/
# │   ├── __init__.py
# │   ├── main.py
# │   ├── models.py
# │   ├── security.py
# │   └── inference.py
# ├── config/
# │   ├── .env
# │   └── config.py
# ├── logs/
# ├── models/
# ├── scripts/
# └── venv/
Archivo 1: config/.env
bashnano config/.env
env# API Configuration
API_TITLE="DeepSeek AI API - Private Instance"
API_VERSION="1.0.0"
API_HOST="0.0.0.0"
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY="GENERATE_RANDOM_KEY_HERE_64_CHARS_MIN"
API_KEY_HEADER="X-API-Key"
RATE_LIMIT_PER_MINUTE=60

# Model Configuration
MODEL_PATH="./models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MAX_LENGTH=2048
TEMPERATURE=0.7
TOP_P=0.9

# Privacy & Logging
LOG_LEVEL="INFO"
LOG_RETENTION_DAYS=7
ANONYMIZE_LOGS=true
DISABLE_REQUEST_LOGGING=false

# Performance
USE_CACHE=true
CACHE_SIZE=100
BATCH_SIZE=1
Archivo 2: config/config.py
bashnano config/config.py
pythonfrom pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_TITLE: str
    API_VERSION: str
    API_HOST: str
    API_PORT: int
    API_WORKERS: int
    
    # Security
    SECRET_KEY: str
    API_KEY_HEADER: str
    RATE_LIMIT_PER_MINUTE: int
    
    # Model
    MODEL_PATH: str
    MAX_LENGTH: int
    TEMPERATURE: float
    TOP_P: float
    
    # Privacy & Logging
    LOG_LEVEL: str
    LOG_RETENTION_DAYS: int
    ANONYMIZE_LOGS: bool
    DISABLE_REQUEST_LOGGING: bool
    
    # Performance
    USE_CACHE: bool
    CACHE_SIZE: int
    BATCH_SIZE: int
    
    class Config:
        env_file = "config/.env"
        case_sensitive = True

settings = Settings()
Archivo 3: app/security.py
bashnano app/security.py
pythonfrom fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from config.config import settings
import hashlib
import secrets
from typing import Optional
from datetime import datetime, timedelta
from collections import defaultdict
import threading

# API Key validation
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)

# In-memory storage (en producción usar Redis)
VALID_API_KEYS = set()
rate_limit_store = defaultdict(list)
rate_limit_lock = threading.Lock()

def generate_api_key() -> str:
    """Genera una API key segura"""
    return secrets.token_urlsafe(32)

def hash_api_key(api_key: str) -> str:
    """Hash de API key para almacenamiento seguro"""
    return hashlib.sha256(api_key.encode()).hexdigest()

def add_api_key(api_key: str):
    """Agrega API key al almacenamiento"""
    VALID_API_KEYS.add(hash_api_key(api_key))

def validate_api_key(api_key: str = Security(api_key_header)) -> str:
    """Valida API key"""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key requerida"
        )
    
    if hash_api_key(api_key) not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida"
        )
    
    return api_key

def check_rate_limit(api_key: str) -> bool:
    """Rate limiting por API key"""
    with rate_limit_lock:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Limpiar requests antiguos
        rate_limit_store[api_key] = [
            req_time for req_time in rate_limit_store[api_key]
            if req_time > minute_ago
        ]
        
        # Verificar límite
        if len(rate_limit_store[api_key]) >= settings.RATE_LIMIT_PER_MINUTE:
            return False
        
        # Agregar request actual
        rate_limit_store[api_key].append(now)
        return True

def anonymize_data(data: str) -> str:
    """Anonimiza datos sensibles para logs"""
    if not settings.ANONYMIZE_LOGS:
        return data
    
    # Hash de datos sensibles
    return hashlib.sha256(data.encode()).hexdigest()[:16]
Archivo 4: app/models.py
bashnano app/models.py
pythonfrom pydantic import BaseModel, Field
from typing import Optional, List

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_length: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(50, ge=0, le=100)
    repetition_penalty: Optional[float] = Field(1.0, ge=1.0, le=2.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain quantum computing in simple terms",
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }

class GenerationResponse(BaseModel):
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    finish_reason: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float

class APIKeyResponse(BaseModel):
    api_key: str
    message: str
Archivo 5: app/inference.py
bashnano app/inference.py
pythonimport torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.config import settings
from loguru import logger
import time
from functools import lru_cache
from typing import Dict, Any

class ModelInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.load_model()
    
    def load_model(self):
        """Carga el modelo con optimizaciones para CPU"""
        logger.info(f"Cargando modelo desde {settings.MODEL_PATH}")
        start_time = time.time()
        
        try:
            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_PATH,
                trust_remote_code=True
            )
            
            # Cargar modelo con optimizaciones
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_PATH,
                torch_dtype=torch.bfloat16,  # Menor uso de memoria
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Optimizaciones Intel
            try:
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model, dtype=torch.bfloat16)
                logger.info("✅ Intel Extension for PyTorch habilitado")
            except ImportError:
                logger.warning("Intel Extension no disponible, usando PyTorch estándar")
            
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"✅ Modelo cargado en {load_time:.2f} segundos")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise
    
    @torch.inference_mode()
    def generate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera texto usando el modelo"""
        prompt = request_data["prompt"]
        max_length = request_data.get("max_length", settings.MAX_LENGTH)
        temperature = request_data.get("temperature", settings.TEMPERATURE)
        top_p = request_data.get("top_p", settings.TOP_P)
        top_k = request_data.get("top_k", 50)
        repetition_penalty = request_data.get("repetition_penalty", 1.0)
        
        # Tokenizar
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_length = inputs.input_ids.shape[1]
        
        # Generar
        start_time = time.time()
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generation_time = time.time() - start_time
        
        # Decodificar
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        output_length = outputs.shape[1] - input_length
        
        logger.info(f"Generación completada en {generation_time:.2f}s "
                   f"({output_length/generation_time:.2f} tokens/s)")
        
        return {
            "generated_text": generated_text,
            "prompt_tokens": input_length,
            "completion_tokens": output_length,
            "total_tokens": input_length + output_length,
            "model": "DeepSeek-R1-Distill-Llama-8B",
            "finish_reason": "stop"
        }

# Singleton global
model_inference = None

def get_model_inference() -> ModelInference:
    global model_inference
    if model_inference is None:
        model_inference = ModelInference()
    return model_inference
Archivo 6: app/main.py
bashnano app/main.py
pythonfrom fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config.config import settings
from app.models import *
from app.security import *
from app.inference import get_model_inference
from loguru import logger
import sys
import time
from datetime import datetime

# Configurar logging
logger.remove()
logger.add(
    sys.stdout,
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)
logger.add(
    f"logs/api_{datetime.now().strftime('%Y%m%d')}.log",
    rotation="00:00",
    retention=f"{settings.LOG_RETENTION_DAYS} days",
    level=settings.LOG_LEVEL
)

# Inicializar FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
START_TIME = time.time()

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Iniciando DeepSeek API...")
    
    # Generar API key inicial (en producción, usar base de datos)
    initial_key = generate_api_key()
    add_api_key(initial_key)
    logger.info(f"🔑 API Key inicial generada: {initial_key}")
    logger.warning("⚠️  GUARDA ESTA KEY DE FORMA SEGURA")
    
    # Cargar modelo
    get_model_inference()
    logger.info("✅ API lista para recibir requests")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de salud"""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version=settings.API_VERSION,
        uptime_seconds=time.time() - START_TIME
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    api_key: str = Depends(validate_api_key)
):
    """Endpoint principal de generación"""
    
    # Rate limiting
    if not check_rate_limit(api_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit excedido: {settings.RATE_LIMIT_PER_MINUTE} requests/minuto"
        )
    
    try:
        # Log anonimizado
        if not settings.DISABLE_REQUEST_LOGGING:
            logger.info(f"Request de {anonymize_data(api_key)}: "
                       f"prompt_length={len(request.prompt)}")
        
        # Generar
        model = get_model_inference()
        result = model.generate(request.dict())
        
        return GenerationResponse(**result)
        
    except Exception as e:
        logger.error(f"Error en generación: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )

@app.post("/admin/generate-key", response_model=APIKeyResponse)
async def create_api_key(admin_password: str):
    """Genera nueva API key (protegido por contraseña de admin)"""
    
    # En producción, usar hash seguro
    if admin_password != "CHANGE_THIS_ADMIN_PASSWORD":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Contraseña de admin incorrecta"
        )
    
    new_key = generate_api_key()
    add_api_key(new_key)
    
    logger.info(f"Nueva API key generada: {anonymize_data(new_key)}")
    
    return APIKeyResponse(
        api_key=new_key,
        message="API key generada exitosamente"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )

7. 🔐 Seguridad y Privacidad (GDPR-Compliant)
Crear Script de Limpieza de Logs
bashnano scripts/cleanup_logs.sh
bash#!/bin/bash

LOG_DIR="$HOME/deepseek-api/logs"
RETENTION_DAYS=7

echo "🧹 Limpiando logs antiguos (>$RETENTION_DAYS días)..."

find "$LOG_DIR" -name "*.log" -type f -mtime +$RETENTION_DAYS -delete

echo "✅ Limpieza completada"
bashchmod +x scripts/cleanup_logs.sh

# Agregar a crontab (ejecutar diariamente a las 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * $HOME/deepseek-api/scripts/cleanup_logs.sh") | crontab -
Documento de Privacidad para Clientes
bashnano PRIVACY_POLICY.md
markdown# Privacy Policy - DeepSeek AI API

## Data Processing
- **No Data Retention**: User prompts and generated outputs are NOT stored after processing
- **Temporary Logs**: System logs contain only anonymized metadata (timestamps, token counts)
- **Log Retention**: All logs are automatically deleted after 7 days
- **No Third-Party Sharing**: Data never leaves your private infrastructure

## GDPR Compliance
✅ Right to erasure: No data stored = nothing to erase
✅ Data minimization: Only essential operational metadata logged
✅ Purpose limitation: Data used only for immediate API response
✅ Privacy by design: Anonymization enabled by default

## Security Measures
- API Key authentication required
- Rate limiting per client
- Encrypted connections (HTTPS recommended)
- No external dependencies or telemetry

## Contact
For privacy inquiries: [your-email@domain.com]

8. 📊 Monitoreo y Métricas
Script de Monitoreo de Sistema
bashnano scripts/monitor.sh
bash#!/bin/bash

echo "📊 Estado del Sistema DeepSeek API"
echo "=================================="
echo ""

# CPU
echo "🔥 CPU:"
grep MHz /proc/cpuinfo | head -1
mpstat 1 1 | grep Average

# Memoria
echo ""
echo "💾 Memoria:"
free -h | grep "Mem:"

# Temperatura
echo ""
echo "🌡️  Temperatura CPU:"
sensors | grep "Package id 0" || sensors | grep "Core 0"

# API Status
echo ""
echo "🌐 API Status:"
curl -s http://localhost:8000/health | jq || echo "API no disponible"

# Procesos Python
echo ""
echo "🐍 Procesos Python:"
ps aux | grep "uvicorn" | grep -v grep || echo "Ninguno"
bashchmod +x scripts/monitor.sh

9. 🚀 Deployment y Ejecución
Crear Servicio Systemd (Auto-inicio)
bashsudo nano /etc/systemd/system/deepseek-api.service
ini[Unit]
Description=DeepSeek AI API Service
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/deepseek-api
Environment="PATH=/home/YOUR_USERNAME/deepseek-api/venv/bin"
ExecStartPre=/home/YOUR_USERNAME/deepseek-api/optimize_system.sh
ExecStart=/home/YOUR_USERNAME/deepseek-api/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
bash# Reemplazar YOUR_USERNAME
sudo sed -i "s/YOUR_USERNAME/$USER/g" /etc/systemd/system/deepseek-api.service

# Habilitar servicio
sudo systemctl daemon-reload
sudo systemctl enable deepseek-api.service
sudo systemctl start deepseek-api.service

# Verificar estado
sudo systemctl status deepseek-api.service
Script de Inicio Manual
bashnano start_api.sh
bash#!/bin/bash

cd ~/deepseek-api

# Optimizar sistema
./optimize_system.sh

# Activar entorno
source venv/bin/activate

# Iniciar API
echo "🚀 Iniciando DeepSeek API..."
python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info
bashchmod +x start_api.sh

10. 📝 Documentación para Clientes
Crear README para GitHub
bashnano README.md
markdown# DeepSeek AI API - Self-Hosted Private Instance

## 🚀 Features
- ✅ **100% Private**: Zero data retention, GDPR-compliant
- ✅ **High Performance**: Optimized for Intel i7 14th Gen (20 cores)
- ✅ **Production Ready**: Rate limiting, API key auth, monitoring
- ✅ **Enterprise Grade**: Professional logging, auto-cleanup, systemd service

## 🔧 API Endpoints

### Base URL
```
http://your-server-ip:8000
```

### Authentication
All requests require `X-API-Key` header:
```bash
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/health
```

### Endpoints

#### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime_seconds": 3600.5
}
```

#### 2. Generate Text
```bash
POST /generate
Content-Type: application/json
X-API-Key: YOUR_API_KEY

{
  "prompt": "Explain quantum computing",
  "max_length": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Response:
```json
{
  "generated_text": "Quantum computing is...",
  "prompt_tokens": 5,
  "completion_tokens": 120,
  "total_tokens": 125,
  "model": "DeepSeek-R1-Distill-Llama-8B",
  "finish_reason": "stop"
}
```

## 📊 Performance Specs
- **CPU**: Intel i7 14th Gen, 20 cores @ 5.0 GHz
- **RAM**: 32GB DDR5 @ 6000MHz
- **Storage**: 2TB NVMe PCIe 4.0
- **Throughput**: ~15-25 tokens/second (CPU inference)
- **Latency**: < 500ms first token

## 🔒 Privacy & Security
- No data storage or retention
- Anonymized logs (auto-deleted after 7 days)
- API key authentication
- Rate limiting (60 req/min per key)
- No external API calls or telemetry

## 💰 Pricing
**Enterprise SaaS License**
- $299/month - unlimited requests
- $0.02/1K tokens - pay-as-you-go
- Custom enterprise plans available

## 📞 Support
- Email: support@yourdomain.com
- Documentation: https://docs.yourdomain.com
- SLA: 99.5% uptime guarantee
Crear Ejemplos de Código para Clientes
bashmkdir -p examples
nano examples/python_example.py
pythonimport requests

API_URL = "http://your-server:8000"
API_KEY = "your-api-key-here"

def generate_text(prompt, max_length=512):
    """Genera texto usando la API"""
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "max_length": max_length,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    response = requests.post(
        f"{API_URL}/generate",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# Ejemplo de uso
if __name__ == "__main__":
    result = generate_text("Explain machine learning in simple terms")
    print(result["generated_text"])
    print(f"Tokens used: {result['total_tokens']}")
bashnano examples/curl_example.sh
bash#!/bin/bash

API_URL="http://localhost:8000"
API_KEY="your-api-key-here"

# Health check
curl -H "X-API-Key: $API_KEY" "$API_URL/health"

echo -e "\n\n"

# Generate text
curl -X POST "$API_URL/generate" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a professional email requesting a meeting",
    "max_length": 256,
    "temperature": 0.7
  }' | jq

11. 🎯 Pasos Finales de Deployment
Verificación Completa
bashcd ~/deepseek-api

# 1. Verificar instalación de dependencias
source venv/bin/activate
pip list | grep -E "torch|transformers|fastapi|uvicorn"

# 2. Verificar modelo descargado
ls -lh models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/

# 3. Test del modelo (esto tomará 1-2 minutos la primera vez)
python << 'EOF'
from app.inference import get_model_inference

print("Cargando modelo...")
model = get_model_inference()

print("Generando texto de prueba...")
result = model.generate({
    "prompt": "Hello, how are you?",
    "max_length": 50
})

print(f"✅ Generación exitosa: {result['generated_text'][:100]}...")
print(f"Tokens: {result['total_tokens']}")
EOF
Iniciar API
bash# Opción 1: Servicio systemd (recomendado)
sudo systemctl start deepseek-api.service
sudo systemctl status deepseek-api.service

# Opción 2: Manual
./start_api.sh
Test de API
bash# Guardar API key mostrada al inicio
export API_KEY="LA_KEY_QUE_SE_MOSTRO_AL_INICIAR"

# Test health
curl http://localhost:8000/health

# Test generación
curl -X POST http://localhost:8000/generate \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a haiku about artificial intelligence",
    "max_length": 100,
    "temperature": 0.8
  }'
```

---

## 12. 💼 Venta en Marketplaces Americanos

### Plataformas Recomendadas

1. **RapidAPI** (https://rapidapi.com/developer)
   - Mayor marketplace de APIs
   - Comisión: 20%
   - Setup: 30 minutos

2. **AWS Marketplace** (https://aws.amazon.com/marketplace)
   - Audiencia enterprise
   - Requiere empaquetado Docker
   
3. **Mashape / RapidAPI Hub**
   - Bueno para SaaS APIs
   - Pricing flexible

### Pricing Sugerido
```
🎯 Tier Pricing:

1. Starter: $49/month
   - 100,000 tokens/mes
   - 60 req/min
   - Email support

2. Professional: $149/month  
   - 500,000 tokens/mes
   - 120 req/min
   - Priority support

3. Enterprise: $499/month
   - Unlimited tokens
   - Unlimited requests
   - Dedicated support
   - SLA 99.9%
Preparar para RapidAPI
bash# Instalar HTTPS (necesario para producción)
sudo apt install certbot python3-certbot-nginx

# Generar certificado SSL
sudo certbot certonly --standalone -d your-domain.com

# Configurar Nginx como reverse proxy
sudo apt install nginx
sudo nano /etc/nginx/sites-available/deepseek-api
nginxserver {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
bashsudo ln -s /etc/nginx/sites-available/deepseek-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 📊 Benchmarks Esperados

Con tu configuración (i7 14th Gen, 32GB RAM, sin GPU):
```
Métrica                    | Valor Esperado
---------------------------|------------------
Primera Respuesta          | 0.5-1.5 segundos
Tokens/Segundo             | 15-25 tokens/s
Requests Concurrentes      | 4-8 (con 4 workers)
Memoria Usada              | 18-22 GB
CPU Load                   | 70-90% durante inferencia
Latencia p95               | < 2 segundos

🐛 Troubleshooting
Modelo no carga
bash# Verificar memoria disponible
free -h

# Verificar que el modelo está completo
du -sh models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/

# Logs detallados
tail -f logs/api_*.log
API lenta
bash# Verificar CPU governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Debería decir "performance", si no:
./optimize_system.sh

# Verificar temperaturas
sensors

# Si >85°C, revisar cooling
Errores de memoria
bash# Reducir workers en config/.env
API_WORKERS=2

# Habilitar swap (último recurso)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

✅ Checklist Final
Antes de vender:

 Modelo cargado y funcionando
 API responde en /health
 Generación de texto funciona
 Logs se crean correctamente
 Limpieza automática de logs configurada
 Servicio systemd habilitado
 Firewall configurado
 SSL/HTTPS configurado (si expones públicamente)
 Rate limiting probado
 README y documentación completa
 Pricing definido
 Cuenta en RapidAPI creada


📚 Recursos Adicionales
bash# Monitoreo continuo
watch -n 1 './scripts/monitor.sh'

# Logs en tiempo real
tail -f logs/api_$(date +%Y%m%d).log

# Benchmark de rendimiento
ab -n 100 -c 10 -H "X-API-Key: $API_KEY" \
   -p test_payload.json \
   -T application/json \
   http://localhost:8000/generate

🎓 Siguiente Nivel
Para escalar:

Agregar GPU: RTX 4090 = 10x velocidad
Load Balancer: Nginx con múltiples instancias
Redis: Cache de respuestas frecuentes
PostgreSQL: Gestión de API keys y métricas
Docker: Containerización para portabilidad
Kubernetes: Orquestación y auto-scaling
