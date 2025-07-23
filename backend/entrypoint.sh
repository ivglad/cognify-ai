#!/bin/bash
set -e

echo "Starting RAGFlow implementation with trio async..."

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
while ! nc -z postgres 5432; do
  sleep 1
done
echo "PostgreSQL is ready!"

# Wait for Elasticsearch to be ready
echo "Waiting for Elasticsearch..."
while ! curl -f http://elasticsearch:9200/_cluster/health >/dev/null 2>&1; do
  sleep 2
done
echo "Elasticsearch is ready!"

# Wait for Redis to be ready
echo "Waiting for Redis..."
while ! nc -z redis 6379; do
  sleep 1
done
echo "Redis is ready!"

# Wait for Infinity DB to be ready
echo "Waiting for Infinity DB..."
while ! nc -z infinity 23817; do
  sleep 1
done
echo "Infinity DB is ready!"

# Create necessary directories
mkdir -p /app/data /app/models /app/storage /app/tmp /app/logs

# Download ONNX models if needed (placeholder for now)
if [ "$DEEPDOC_ENABLED" = "true" ]; then
    echo "DeepDoc is enabled. ONNX models will be downloaded on first use."
    /app/download_models.sh
fi

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Start the FastAPI application with trio support
echo "Starting FastAPI server with trio async backend..."
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload