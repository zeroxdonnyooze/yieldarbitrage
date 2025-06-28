#!/bin/bash
# Production deployment validation script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test health endpoints
test_health_endpoints() {
    log "Testing health endpoints..."
    
    local base_url="http://localhost:8000"
    local timeout=10
    
    # Test basic health endpoint
    if curl -f -s -m $timeout "$base_url/health" > /dev/null; then
        success "Basic health endpoint responding"
    else
        error "Basic health endpoint not responding"
        return 1
    fi
    
    # Test detailed health endpoint
    local health_response=$(curl -f -s -m $timeout "$base_url/health/detailed" 2>/dev/null || echo "")
    if [[ -n "$health_response" ]]; then
        local status=$(echo "$health_response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
        if [[ "$status" == "healthy" ]]; then
            success "Detailed health check: healthy"
        elif [[ "$status" == "degraded" ]]; then
            warn "Detailed health check: degraded"
        else
            error "Detailed health check: $status"
            return 1
        fi
    else
        error "Detailed health endpoint not responding"
        return 1
    fi
    
    # Test Kubernetes probes
    for probe in "live" "ready" "startup"; do
        if curl -f -s -m $timeout "$base_url/health/$probe" > /dev/null; then
            success "Kubernetes $probe probe responding"
        else
            warn "Kubernetes $probe probe not responding"
        fi
    done
}

# Test container health
test_container_health() {
    log "Testing container health..."
    
    cd "$PROJECT_DIR"
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    elif [[ "$ENVIRONMENT" == "staging" ]]; then
        compose_file="docker-compose.staging.yml"
    fi
    
    # Check if containers are running
    local running_containers=$(docker-compose -f "$compose_file" ps --services --filter "status=running" | wc -l)
    local total_containers=$(docker-compose -f "$compose_file" ps --services | wc -l)
    
    if [[ $running_containers -eq $total_containers ]]; then
        success "All containers running ($running_containers/$total_containers)"
    else
        error "Some containers not running ($running_containers/$total_containers)"
        docker-compose -f "$compose_file" ps
        return 1
    fi
    
    # Check container health status
    local unhealthy_containers=$(docker ps --filter "health=unhealthy" --format "table {{.Names}}" | grep -v NAMES | wc -l)
    if [[ $unhealthy_containers -eq 0 ]]; then
        success "All containers healthy"
    else
        error "$unhealthy_containers containers unhealthy"
        docker ps --filter "health=unhealthy" --format "table {{.Names}}\t{{.Status}}"
        return 1
    fi
}

# Test database connectivity
test_database() {
    log "Testing database connectivity..."
    
    cd "$PROJECT_DIR"
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    elif [[ "$ENVIRONMENT" == "staging" ]]; then
        compose_file="docker-compose.staging.yml"
    fi
    
    # Test database connection
    if docker-compose -f "$compose_file" exec -T postgres pg_isready > /dev/null 2>&1; then
        success "Database connection healthy"
    else
        error "Database connection failed"
        return 1
    fi
    
    # Test basic query
    local table_count=$(docker-compose -f "$compose_file" exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | tr -d ' ' || echo "0")
    
    if [[ $table_count -gt 0 ]]; then
        success "Database tables present ($table_count tables)"
    else
        warn "No database tables found (may be normal for fresh deployment)"
    fi
}

# Test Redis connectivity
test_redis() {
    log "Testing Redis connectivity..."
    
    cd "$PROJECT_DIR"
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    elif [[ "$ENVIRONMENT" == "staging" ]]; then
        compose_file="docker-compose.staging.yml"
    fi
    
    # Test Redis connection
    if docker-compose -f "$compose_file" exec -T redis redis-cli ping | grep -q "PONG"; then
        success "Redis connection healthy"
    else
        error "Redis connection failed"
        return 1
    fi
    
    # Test basic Redis operations
    docker-compose -f "$compose_file" exec -T redis redis-cli set test_key "test_value" > /dev/null
    local test_value=$(docker-compose -f "$compose_file" exec -T redis redis-cli get test_key | tr -d '\r')
    docker-compose -f "$compose_file" exec -T redis redis-cli del test_key > /dev/null
    
    if [[ "$test_value" == "test_value" ]]; then
        success "Redis operations working"
    else
        error "Redis operations failed"
        return 1
    fi
}

# Test application logs
test_application_logs() {
    log "Testing application logs..."
    
    cd "$PROJECT_DIR"
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    elif [[ "$ENVIRONMENT" == "staging" ]]; then
        compose_file="docker-compose.staging.yml"
    fi
    
    # Check for recent logs
    local recent_logs=$(docker-compose -f "$compose_file" logs --tail=10 app 2>/dev/null | wc -l)
    
    if [[ $recent_logs -gt 0 ]]; then
        success "Application generating logs"
    else
        warn "No recent application logs found"
    fi
    
    # Check for error patterns in logs
    local error_count=$(docker-compose -f "$compose_file" logs --tail=100 app 2>/dev/null | grep -c "ERROR\|CRITICAL\|FATAL" || echo "0")
    
    if [[ $error_count -eq 0 ]]; then
        success "No errors in recent logs"
    else
        warn "$error_count errors found in recent logs"
        echo "Recent errors:"
        docker-compose -f "$compose_file" logs --tail=100 app 2>/dev/null | grep "ERROR\|CRITICAL\|FATAL" | tail -5
    fi
}

# Test resource usage
test_resource_usage() {
    log "Testing resource usage..."
    
    # Check system resources
    local memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    
    log "Memory usage: ${memory_usage}%"
    log "Disk usage: ${disk_usage}%"
    
    # Check Docker resource usage
    local container_stats=$(docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | tail -n +2)
    
    if [[ -n "$container_stats" ]]; then
        success "Container resource stats available"
        echo "$container_stats"
    else
        warn "No container stats available"
    fi
    
    # Warning thresholds
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        warn "High memory usage: ${memory_usage}%"
    fi
    
    if [[ $disk_usage -gt 90 ]]; then
        warn "High disk usage: ${disk_usage}%"
    fi
}

# Test network connectivity
test_network() {
    log "Testing network connectivity..."
    
    # Test external connectivity
    if curl -f -s -m 5 "https://httpbin.org/get" > /dev/null; then
        success "External network connectivity working"
    else
        warn "External network connectivity issues"
    fi
    
    # Test DNS resolution
    if nslookup google.com > /dev/null 2>&1; then
        success "DNS resolution working"
    else
        warn "DNS resolution issues"
    fi
}

# Run production validation tests
run_validation_tests() {
    log "Running production validation tests..."
    
    cd "$PROJECT_DIR"
    
    # Run pytest validation tests if available
    if command -v pytest &> /dev/null && [[ -f "tests/test_production_validation.py" ]]; then
        log "Running pytest validation tests..."
        if pytest tests/test_production_validation.py -v; then
            success "Production validation tests passed"
        else
            error "Production validation tests failed"
            return 1
        fi
    else
        warn "Pytest validation tests not available"
    fi
}

# Generate validation report
generate_report() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    local report_file="validation_report_$(date +'%Y%m%d_%H%M%S').txt"
    
    log "Generating validation report: $report_file"
    
    cat > "$report_file" << EOF
Deployment Validation Report
Environment: $ENVIRONMENT
Timestamp: $timestamp

=== Validation Summary ===
Health Endpoints: $1
Container Health: $2
Database: $3
Redis: $4
Application Logs: $5
Resource Usage: $6
Network: $7
Validation Tests: $8

=== System Information ===
$(uname -a)
$(docker --version)
$(docker-compose --version)

=== Container Status ===
$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")

=== Resource Usage ===
Memory: $(free -h | grep Mem)
Disk: $(df -h /)

=== Recent Application Logs ===
$(docker-compose logs --tail=20 app 2>/dev/null || echo "No logs available")
EOF

    success "Validation report generated: $report_file"
}

# Main validation function
main() {
    log "Starting deployment validation for environment: $ENVIRONMENT"
    
    local results=()
    local overall_status="PASSED"
    
    # Run validation tests
    test_health_endpoints && results+=("PASS") || { results+=("FAIL"); overall_status="FAILED"; }
    test_container_health && results+=("PASS") || { results+=("FAIL"); overall_status="FAILED"; }
    test_database && results+=("PASS") || { results+=("FAIL"); overall_status="FAILED"; }
    test_redis && results+=("PASS") || { results+=("FAIL"); overall_status="FAILED"; }
    test_application_logs && results+=("PASS") || { results+=("FAIL"); overall_status="FAILED"; }
    test_resource_usage && results+=("PASS") || results+=("PASS") # Always pass resource check
    test_network && results+=("PASS") || results+=("PASS") # Always pass network check
    run_validation_tests && results+=("PASS") || { results+=("FAIL"); overall_status="FAILED"; }
    
    # Generate report
    generate_report "${results[@]}"
    
    echo ""
    if [[ "$overall_status" == "PASSED" ]]; then
        success "🎉 Deployment validation PASSED"
        exit 0
    else
        error "❌ Deployment validation FAILED"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-validate}" in
    "validate")
        main
        ;;
    "health")
        test_health_endpoints
        ;;
    "containers")
        test_container_health
        ;;
    "database")
        test_database
        ;;
    "redis")
        test_redis
        ;;
    "logs")
        test_application_logs
        ;;
    "resources")
        test_resource_usage
        ;;
    "network")
        test_network
        ;;
    "tests")
        run_validation_tests
        ;;
    *)
        echo "Usage: $0 {validate|health|containers|database|redis|logs|resources|network|tests}"
        echo ""
        echo "Commands:"
        echo "  validate    - Full validation (default)"
        echo "  health      - Test health endpoints"
        echo "  containers  - Test container health"
        echo "  database    - Test database connectivity"
        echo "  redis       - Test Redis connectivity"
        echo "  logs        - Test application logs"
        echo "  resources   - Test resource usage"
        echo "  network     - Test network connectivity"
        echo "  tests       - Run validation tests"
        exit 1
        ;;
esac