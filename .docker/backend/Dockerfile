# ---- Builder Stage ----
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /install

# Copy requirements file
COPY ./backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix="/install" -r requirements.txt


# ---- Final Stage ----
FROM python:3.11-slim

# Install system dependencies required by 'unstructured' and other tools
# - netcat is for the entrypoint script to wait for Postgres
# - poppler-utils is for PDF processing by 'unstructured'
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-traditional \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Set the WORKDIR
WORKDIR /app

# Download models before copying app code to leverage caching
# This layer will be cached unless requirements.txt changes
RUN python -m nltk.downloader punkt punkt_tab stopwords

# Install both Spacy models for flexibility via environment variable
# This allows switching between models without rebuilding the container
RUN python -m spacy download ru_core_news_sm && \
    python -m spacy download ru_core_news_md

# Copy the application code
COPY ./backend/ /app/

# Copy the entrypoint script
COPY ./backend/entrypoint.sh /app/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint to be executed with sh
ENTRYPOINT ["sh", "/app/entrypoint.sh"] 