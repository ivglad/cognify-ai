#!/bin/bash

# RAGFlow Production Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE="backend/production.env"
BACKUP_DIR="backups"
LOG_FILE="deploy.log"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running. Please start Docker service."
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        error "Environment file $ENV_FILE not found. Please create it first."
    fi
    
    # Check if required environment variables are set
    if [ -z "$DB_PASSWORD" ]; then
        error "DB_PASSWORD environment variable is not set."
    fi
    
    if [ -z "$SECRET_KEY" ]; then
        error "SECRET_KEY environment variable is not set."
    fi
    
    if [ -z "$JWT_SECRET_KEY" ]; then
        error "JWT_SECRET_KEY environment variable is not set."
    fi
    
    success "Prerequisites check passed."
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p "$BACKUP_DIR"
    mkdir -p "monitoring/prometheus"
    mkdir -p "monitoring/grafana/dashboards"
    mkdir -p "monitoring/grafana/datasources"
    mkdir -p "monitoring/fluentd"
    mkdir -p "certs"
    mkdir -p "scripts"
    
    success "Directories created."
}

# Generate SSL certificates (self-signed for development)
generate_ssl_certs() {
    log "Generating SSL certificates..."
    
    if [ ! -f "certs/cert.pem" ] || [ ! -f "certs/key.pem" ]; then
        openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        success "SSL certificates generated."
    else
        log "SSL certificates already exist."
    fi
}

# Create monitoring configuration
create_monitoring_config() {
    log "Creating monitoring configuration..."
    
    # Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'ragflow-backend'
    static_configs:
      - targets: ['ragflow-backend:8000']
    metrics_path: '/api/v1/monitoring/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'elasticsearch'
    static_configs:
      - targets: ['elasticsearch:9200']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
EOF

    # Grafana datasource configuration
    mkdir -p monitoring/grafana/datasources
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Fluentd configuration
    cat > monitoring/fluentd/fluent.conf << EOF
<source>
  @type tail
  path /app/logs/ragflow.log
  pos_file /fluentd/log/ragflow.log.pos
  tag ragflow.app
  format json
</source>

<source>
  @type tail
  path /var/log/nginx/access.log
  pos_file /fluentd/log/nginx.access.log.pos
  tag nginx.access
  format nginx
</source>

<match **>
  @type stdout
</match>
EOF

    success "Monitoring configuration created."
}

# Backup existing data
backup_data() {
    if [ "$1" = "--skip-backup" ]; then
        log "Skipping backup as requested."
        return
    fi
    
    log "Creating backup of existing data..."
    
    BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="$BACKUP_DIR/backup_$BACKUP_TIMESTAMP"
    
    mkdir -p "$BACKUP_PATH"
    
    # Backup database if running
    if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up"; then
        log "Backing up PostgreSQL database..."
        docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U ragflow_user ragflow_prod > "$BACKUP_PATH/database.sql"
    fi
    
    # Backup volumes
    if docker volume ls | grep -q ragflow; then
        log "Backing up Docker volumes..."
        docker run --rm -v ragflow_data:/data -v "$(pwd)/$BACKUP_PATH":/backup alpine tar czf /backup/ragflow_data.tar.gz -C /data .
        docker run --rm -v ragflow_models:/data -v "$(pwd)/$BACKUP_PATH":/backup alpine tar czf /backup/ragflow_models.tar.gz -C /data .
    fi
    
    success "Backup created at $BACKUP_PATH"
}

# Build and deploy
deploy() {
    log "Starting deployment..."
    
    # Pull latest images
    log "Pulling latest images..."
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Build application image
    log "Building application image..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache ragflow-backend
    
    # Start services
    log "Starting services..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    check_service_health
    
    success "Deployment completed successfully!"
}

# Check service health
check_service_health() {
    log "Checking service health..."
    
    local services=("postgres" "redis" "elasticsearch" "minio" "ragflow-backend")
    local failed_services=()
    
    for service in "${services[@]}"; do
        log "Checking $service..."
        if ! docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "healthy\|Up"; then
            failed_services+=("$service")
            warning "$service is not healthy"
        else
            success "$service is healthy"
        fi
    done
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        error "The following services are not healthy: ${failed_services[*]}"
    fi
    
    # Test API endpoint
    log "Testing API endpoint..."
    if curl -f http://localhost:8000/api/v1/monitoring/status &> /dev/null; then
        success "API endpoint is responding"
    else
        warning "API endpoint is not responding yet"
    fi
}

# Rollback deployment
rollback() {
    log "Rolling back deployment..."
    
    # Stop current services
    docker-compose -f "$COMPOSE_FILE" down
    
    # Restore from latest backup
    LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | head -n1)
    if [ -n "$LATEST_BACKUP" ]; then
        log "Restoring from backup: $LATEST_BACKUP"
        
        # Restore database
        if [ -f "$BACKUP_DIR/$LATEST_BACKUP/database.sql" ]; then
            docker-compose -f "$COMPOSE_FILE" up -d postgres
            sleep 10
            docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U ragflow_user -d ragflow_prod < "$BACKUP_DIR/$LATEST_BACKUP/database.sql"
        fi
        
        # Restore volumes
        if [ -f "$BACKUP_DIR/$LATEST_BACKUP/ragflow_data.tar.gz" ]; then
            docker run --rm -v ragflow_data:/data -v "$(pwd)/$BACKUP_DIR/$LATEST_BACKUP":/backup alpine tar xzf /backup/ragflow_data.tar.gz -C /data
        fi
        
        success "Rollback completed"
    else
        error "No backup found for rollback"
    fi
}

# Show logs
show_logs() {
    local service=${1:-"ragflow-backend"}
    docker-compose -f "$COMPOSE_FILE" logs -f "$service"
}

# Show status
show_status() {
    log "Service Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    log "\nResource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

# Cleanup old images and volumes
cleanup() {
    log "Cleaning up old images and volumes..."
    
    docker system prune -f
    docker volume prune -f
    
    success "Cleanup completed"
}

# Main script logic
case "$1" in
    "deploy")
        check_prerequisites
        create_directories
        generate_ssl_certs
        create_monitoring_config
        backup_data "$2"
        deploy
        ;;
    "rollback")
        rollback
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "$2"
        ;;
    "backup")
        backup_data
        ;;
    "cleanup")
        cleanup
        ;;
    "health")
        check_service_health
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|status|logs|backup|cleanup|health}"
        echo ""
        echo "Commands:"
        echo "  deploy [--skip-backup]  - Deploy the application"
        echo "  rollback               - Rollback to previous version"
        echo "  status                 - Show service status"
        echo "  logs [service]         - Show logs for service"
        echo "  backup                 - Create backup"
        echo "  cleanup                - Clean up old images and volumes"
        echo "  health                 - Check service health"
        exit 1
        ;;
esac