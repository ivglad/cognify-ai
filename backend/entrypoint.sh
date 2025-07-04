#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# The 'depends_on' with 'service_healthy' in docker-compose.yml now handles waiting.
# The previous loop is no longer necessary.
echo "Waiting for PostgreSQL is now handled by Docker Compose healthcheck..."

# Apply database migrations, specifying the config file path
echo "Applying database migrations..."
alembic -c alembic/alembic.ini upgrade head

# Start the Uvicorn server
echo "Starting Uvicorn server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 