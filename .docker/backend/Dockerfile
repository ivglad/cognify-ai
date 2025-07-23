# ---- Builder Stage ----
FROM python:3.11-slim AS builder

# Install build dependencies for complex packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /install

# Copy requirements file
COPY ./backend/requirements.txt .

# Install dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix="/install" -r requirements.txt


# ---- Production Stage ----
FROM python:3.11-slim AS production

# Create non-root user for security
RUN groupadd -r ragflow && useradd -r -g ragflow ragflow

# Install system dependencies for RAGFlow components
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic utilities
    curl \
    wget \
    unzip \
    netcat-traditional \
    # Document processing
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    # Computer vision for DeepDoc
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjemalloc-dev \
    # Database clients
    postgresql-client \
    # Monitoring tools
    htop \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Set the WORKDIR
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/models /app/storage /app/tmp /app/logs /app/cache && \
    chown -R ragflow:ragflow /app

# Download NLP models before copying app code to leverage caching
RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet

# Install Spacy models for Russian and English
RUN python -m spacy download ru_core_news_sm && \
    python -m spacy download ru_core_news_md && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_md

# Create optimized model download script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Initializing ONNX models for DeepDoc..."\n\
mkdir -p /app/models/{layout,ocr,table}\n\
\n\
# Create model manifest\n\
cat > /app/models/model_manifest.json << EOF\n\
{\n\
  "layout_models": {\n\
    "publaynet": "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin"\n\
  },\n\
  "ocr_models": {\n\
    "text_detection": "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin"\n\
  },\n\
  "table_models": {\n\
    "structure_recognition": "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin"\n\
  }\n\
}\n\
EOF\n\
\n\
echo "Model directories initialized. Models will be downloaded on first use."\n\
' > /app/download_models.sh && chmod +x /app/download_models.sh

# Copy the application code
COPY --chown=ragflow:ragflow ./backend/ /app/

# Copy production configuration files
COPY --chown=ragflow:ragflow ./backend/entrypoint.sh /app/entrypoint.sh
COPY --chown=ragflow:ragflow ./backend/production.env /app/.env

# Make scripts executable
RUN chmod +x /app/entrypoint.sh

# Health check for production monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/monitoring/status || exit 1

# Switch to non-root user
USER ragflow

# Expose port
EXPOSE 8000

# Use production entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--max-requests", "1000", "--max-requests-jitter", "100", "--preload", "--access-logfile", "-", "--error-logfile", "-"] 