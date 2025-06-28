# Multi-stage production Dockerfile for yield arbitrage system
# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production runtime
FROM python:3.11-slim as production

# Create non-root user for security
RUN groupadd -r arbitrage && useradd -r -g arbitrage arbitrage

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/arbitrage/.local

# Copy source code
COPY --chown=arbitrage:arbitrage . .

# Add Python packages to PATH
ENV PATH=/home/arbitrage/.local/bin:$PATH
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs \
    && chown -R arbitrage:arbitrage /app

# Switch to non-root user
USER arbitrage

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Default command
CMD ["python", "-m", "yield_arbitrage.main"]

# Development stage for local development
FROM production as development

USER root

# Install development dependencies
COPY --from=builder /root/.local /root/.local
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

USER arbitrage

# Override for development
CMD ["python", "-m", "yield_arbitrage.main", "--reload"]