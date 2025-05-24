FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install (WITHOUT bitsandbytes initially)
COPY requirements.txt .

# Install base packages first
RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.2 \
    pydantic==2.11.4 \
    pydantic-settings==2.0.0 \
    elasticsearch>=8.8.0 \
    sentence-transformers>=2.2.2 \
    transformers>=4.35.0 \
    torch>=2.0.0 \
    accelerate>=0.20.0 \
    optimum>=1.13.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    python-dotenv>=1.0.0 \
    httpx>=0.24.0 \
    typing-extensions==4.13.2

# PRE-DOWNLOAD BASE MODELS (WITHOUT QUANTIZATION)
RUN python -c "import os; \
os.environ['TRANSFORMERS_CACHE'] = '/app/model_cache'; \
os.environ['HF_HOME'] = '/app/model_cache'; \
\
print('🔄 Downloading Qwen2.5-0.5B base model...'); \
from transformers import AutoModelForCausalLM, AutoTokenizer; \
import torch; \
\
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'; \
\
# Download without quantization (build-time safe) \
tokenizer = AutoTokenizer.from_pretrained( \
    model_name, \
    cache_dir='/app/model_cache' \
); \
\
# Download model in fp16 (no bitsandbytes needed) \
model = AutoModelForCausalLM.from_pretrained( \
    model_name, \
    torch_dtype=torch.float16, \
    cache_dir='/app/model_cache' \
); \
print('✅ Base model downloaded (quantization will happen at runtime)!') \
del model, tokenizer \
"

RUN python -c " \
print('🔄 Downloading Sentence Transformer...') \
from sentence_transformers import SentenceTransformer \
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') \
print('✅ Sentence Transformer downloaded!') \
del model \
" 

# Copy application code
COPY . .

# Create startup script for runtime GPU detection
RUN echo '#!/bin/bash\n\
echo \"🔍 Detecting runtime environment...\"\n\
if command -v nvidia-smi >/dev/null 2>&1; then\n\
    echo \"🚀 GPU detected, installing bitsandbytes...\"\n\
    pip install bitsandbytes>=0.41.0 --no-cache-dir\n\
    echo \"✅ GPU packages installed\"\n\
else\n\
    echo \"⚠️ No GPU detected, running CPU-only mode\"\n\
fi\n\
echo \"🎬 Starting Movie Search API...\"\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1' > /app/start.sh

# Make startup script executable
RUN chmod +x /app/start.sh

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV PYTHONPATH=/app

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use startup script instead of direct uvicorn
CMD ["/app/start.sh"]