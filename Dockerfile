FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PRE-DOWNLOAD MODELS FOR GPU OPTIMIZATION
RUN python -c "import os; \
os.environ['TRANSFORMERS_CACHE'] = '/app/model_cache'; \
os.environ['HF_HOME'] = '/app/model_cache'; \
print('🔄 Downloading Qwen2.5-0.5B for GPU inference...'); \
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig; \
import torch; \
quantization_config = BitsAndBytesConfig( \
    load_in_4bit=True, \
    bnb_4bit_compute_dtype=torch.float16, \
    bnb_4bit_use_double_quant=True, \
    bnb_4bit_quant_type='nf4' \
); \
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'; \
tokenizer = AutoTokenizer.from_pretrained( \
    model_name, \
    cache_dir='/app/model_cache' \
); \
model = AutoModelForCausalLM.from_pretrained( \
    model_name, \
    quantization_config=quantization_config, \
    torch_dtype=torch.float16, \
    device_map='auto', \
    cache_dir='/app/model_cache' \
); \
print('✅ GPU-optimized Qwen model downloaded!'); \
del model, tokenizer"

# Copy application code
RUN python -c "print('🔄 Downloading Sentence Transformer...'); \
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('✅ Sentence Transformer downloaded!'); \
del model"
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start with GPU support
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]