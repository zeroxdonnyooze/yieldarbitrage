#!/bin/bash
# Production deployment script for yield arbitrage system
set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${ENVIRONMENT:-production}"
VERSION="${VERSION:-latest}"
DEPLOY_USER="${DEPLOY_USER:-arbitrage}"
BACKUP_DIR="/backups/yieldarbitrage"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if running as correct user
check_user() {
    if [[ "$ENVIRONMENT" == "production" ]] && [[ "$(whoami)" != "$DEPLOY_USER" ]]; then
        error "Production deployment must be run as user: $DEPLOY_USER"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Check environment file
    local env_file="$PROJECT_DIR/config/$ENVIRONMENT.env"
    if [[ ! -f "$env_file" ]]; then
        error "Environment file not found: $env_file"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Load environment variables
load_environment() {
    log "Loading environment configuration..."
    
    local env_file="$PROJECT_DIR/config/$ENVIRONMENT.env"
    if [[ -f "$env_file" ]]; then
        # Export variables from env file
        set -a
        source "$env_file"
        set +a
        log "Loaded environment from: $env_file"
    else
        error "Environment file not found: $env_file"
        exit 1
    fi
}

# Backup database
backup_database() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Creating database backup..."
        
        # Create backup directory
        mkdir -p "$BACKUP_DIR"
        
        # Get current timestamp
        local timestamp=$(date +"%Y%m%d_%H%M%S")
        local backup_file="$BACKUP_DIR/yieldarbitrage_${timestamp}.sql"
        
        # Create backup
        docker-compose -f "$PROJECT_DIR/docker-compose.prod.yml" exec -T postgres \
            pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" > "$backup_file"
        
        # Compress backup
        gzip "$backup_file"
        
        success "Database backup created: ${backup_file}.gz"
        
        # Keep only last 7 days of backups
        find "$BACKUP_DIR" -name "*.sql.gz" -mtime +7 -delete
    else
        log "Skipping database backup for $ENVIRONMENT environment"
    fi
}

# Pull latest images
pull_images() {
    log "Pulling latest Docker images..."
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    elif [[ "$ENVIRONMENT" == "staging" ]]; then
        compose_file="docker-compose.staging.yml"
    fi
    
    cd "$PROJECT_DIR"
    docker-compose -f "$compose_file" pull
    
    success "Docker images pulled successfully"
}

# Build application
build_application() {
    log "Building application..."
    
    cd "$PROJECT_DIR"
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    elif [[ "$ENVIRONMENT" == "staging" ]]; then
        compose_file="docker-compose.staging.yml"
    fi
    
    # Build the application image
    docker-compose -f "$compose_file" build app
    
    success "Application built successfully"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    cd "$PROJECT_DIR"
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    elif [[ "$ENVIRONMENT" == "staging" ]]; then
        compose_file="docker-compose.staging.yml"
    fi
    
    # Start database if not running
    docker-compose -f "$compose_file" up -d postgres redis
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    sleep 10
    
    # Run migrations
    docker-compose -f "$compose_file" run --rm app python -m alembic upgrade head
    
    success "Database migrations completed"
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    cd "$PROJECT_DIR"
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    elif [[ "$ENVIRONMENT" == "staging" ]]; then
        compose_file="docker-compose.staging.yml"
    fi
    
    # Deploy with zero-downtime strategy
    docker-compose -f "$compose_file" up -d --remove-orphans
    
    success "Services deployed successfully"
}

# Health check
health_check() {
    log "Performing health check..."
    
    local max_attempts=30
    local attempt=1
    local health_url="http://localhost:8000/health"
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$health_url" > /dev/null 2>&1; then
            success "Health check passed"
            return 0
        else
            log "Health check attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
            sleep 10
            ((attempt++))
        fi
    done
    
    error "Health check failed after $max_attempts attempts"
    return 1
}

# Cleanup old images
cleanup() {
    log "Cleaning up old Docker images..."
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (be careful in production)
    if [[ "$ENVIRONMENT" != "production" ]]; then
        docker volume prune -f
    fi
    
    success "Cleanup completed"
}

# Show deployment status
show_status() {
    log "Deployment status:"
    
    cd "$PROJECT_DIR"
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    elif [[ "$ENVIRONMENT" == "staging" ]]; then
        compose_file="docker-compose.staging.yml"
    fi
    
    docker-compose -f "$compose_file" ps
    
    echo ""
    log "Application logs (last 20 lines):"
    docker-compose -f "$compose_file" logs --tail=20 app
}

# Rollback function
rollback() {
    error "Deployment failed. Initiating rollback..."
    
    cd "$PROJECT_DIR"
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    elif [[ "$ENVIRONMENT" == "staging" ]]; then
        compose_file="docker-compose.staging.yml"
    fi
    
    # Stop current deployment
    docker-compose -f "$compose_file" down
    
    # Restore from backup if production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        warn "Manual database restoration may be required"
        warn "Latest backup location: $BACKUP_DIR"
    fi
    
    error "Rollback completed. Please investigate the issue."
}

# Main deployment function
main() {
    log "Starting deployment for environment: $ENVIRONMENT"
    
    # Set up error handling
    trap rollback ERR
    
    # Run deployment steps
    check_user
    check_prerequisites
    load_environment
    backup_database
    pull_images
    build_application
    run_migrations
    deploy_services
    
    # Health check with retry
    if ! health_check; then
        rollback
        exit 1
    fi
    
    cleanup
    show_status
    
    success "Deployment completed successfully!"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Production deployment checklist:"
        log "1. Verify application is responding at configured endpoints"
        log "2. Check monitoring dashboards for any alerts"
        log "3. Verify trading functionality is working correctly"
        log "4. Monitor logs for any errors or warnings"
        log "5. Backup verification: Latest backup in $BACKUP_DIR"
    fi
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "health")
        health_check
        ;;
    "status")
        show_status
        ;;
    "backup")
        load_environment
        backup_database
        ;;
    "rollback")
        rollback
        ;;
    *)
        echo "Usage: $0 {deploy|health|status|backup|rollback}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Full deployment (default)"
        echo "  health   - Run health check"
        echo "  status   - Show deployment status"
        echo "  backup   - Create database backup"
        echo "  rollback - Rollback deployment"
        exit 1
        ;;
esac