#!/bin/bash
# Monitoring setup script for yield arbitrage system
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MONITORING_DIR="$PROJECT_DIR/monitoring"

# Colors for output
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

# Create monitoring directory structure
create_monitoring_structure() {
    log "Creating monitoring directory structure..."
    
    mkdir -p "$MONITORING_DIR"/{dashboards,config}
    
    success "Monitoring directories created"
}

# Create Prometheus configuration
create_prometheus_config() {
    log "Creating Prometheus configuration..."
    
    cat > "$MONITORING_DIR/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'yieldarbitrage'
    static_configs:
      - targets: ['app:9090']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
EOF

    success "Prometheus configuration created"
}

# Create Grafana datasource configuration
create_grafana_datasources() {
    log "Creating Grafana datasource configuration..."
    
    cat > "$MONITORING_DIR/grafana-datasources.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
EOF

    success "Grafana datasources configuration created"
}

# Create Grafana dashboard provisioning
create_grafana_dashboards_config() {
    log "Creating Grafana dashboard configuration..."
    
    cat > "$MONITORING_DIR/grafana-dashboards.yml" << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

    success "Grafana dashboard configuration created"
}

# Create application dashboard
create_application_dashboard() {
    log "Creating application dashboard..."
    
    cat > "$MONITORING_DIR/dashboards/yield-arbitrage-overview.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Yield Arbitrage Overview",
    "tags": ["yield-arbitrage"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Active Edges",
        "type": "stat",
        "targets": [
          {
            "expr": "arbitrage_active_edges_count",
            "legendFormat": "Active Edges"
          }
        ],
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Total Profit (24h)",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(arbitrage_total_profit_usd[24h])",
            "legendFormat": "Profit USD"
          }
        ],
        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
      },
      {
        "id": 3,
        "title": "Transaction Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(arbitrage_transactions_successful[5m]) / rate(arbitrage_transactions_total[5m])",
            "legendFormat": "Success Rate"
          }
        ],
        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0}
      },
      {
        "id": 4,
        "title": "Edge Update Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "arbitrage_edge_update_duration_seconds",
            "legendFormat": "Update Duration"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      }
    ],
    "time": {"from": "now-1h", "to": "now"},
    "refresh": "5s"
  }
}
EOF

    success "Application dashboard created"
}

# Create Loki configuration
create_loki_config() {
    log "Creating Loki configuration..."
    
    cat > "$MONITORING_DIR/loki-config.yml" << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 1h
  max_chunk_age: 1h
  chunk_target_size: 1048576
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s
EOF

    success "Loki configuration created"
}

# Create Promtail configuration
create_promtail_config() {
    log "Creating Promtail configuration..."
    
    cat > "$MONITORING_DIR/promtail-config.yml" << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: arbitrage-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: yieldarbitrage
          __path__: /var/log/app/*.log
    pipeline_stages:
      - json:
          expressions:
            timestamp: timestamp
            level: level
            message: message
            component: logger_name
      - timestamp:
          source: timestamp
          format: RFC3339
      - labels:
          level:
          component:
EOF

    success "Promtail configuration created"
}

# Create alerting rules
create_alerting_rules() {
    log "Creating alerting rules..."
    
    cat > "$MONITORING_DIR/alerts.yml" << 'EOF'
groups:
  - name: yield-arbitrage
    rules:
      - alert: HighErrorRate
        expr: rate(arbitrage_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: LowProfitability
        expr: rate(arbitrage_total_profit_usd[1h]) < 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low profitability detected"
          description: "Profit rate is {{ $value }} USD per hour"

      - alert: EdgeUpdateLatency
        expr: arbitrage_edge_update_duration_seconds > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High edge update latency"
          description: "Edge update taking {{ $value }} seconds"

      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is down"
          description: "PostgreSQL database is not responding"

      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis is down"
          description: "Redis cache is not responding"
EOF

    success "Alerting rules created"
}

# Main function
main() {
    log "Setting up monitoring infrastructure..."
    
    create_monitoring_structure
    create_prometheus_config
    create_grafana_datasources
    create_grafana_dashboards_config
    create_application_dashboard
    create_loki_config
    create_promtail_config
    create_alerting_rules
    
    success "Monitoring setup completed!"
    
    echo ""
    log "Next steps:"
    log "1. Review configurations in $MONITORING_DIR"
    log "2. Update docker-compose files with monitoring services"
    log "3. Deploy with: ./deploy.sh"
    log "4. Access Grafana at http://localhost:3000"
    log "5. Configure alerting endpoints (Slack, email, etc.)"
}

# Run main function
main "$@"