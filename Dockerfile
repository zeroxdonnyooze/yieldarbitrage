FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy source code
COPY . .

# Add src to Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

EXPOSE 8000

CMD ["python", "-m", "yield_arbitrage.main"]