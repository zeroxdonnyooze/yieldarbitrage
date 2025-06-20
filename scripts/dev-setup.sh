#!/bin/bash

# Development environment setup script

echo "🚀 Setting up Yield Arbitrage development environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Copy environment file
if [ ! -f .env ]; then
    echo "📋 Creating .env file from .env.docker..."
    cp .env.docker .env
else
    echo "📋 .env file already exists, skipping..."
fi

# Start services
echo "🔄 Starting PostgreSQL and Redis services..."
docker-compose up -d postgres redis

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
timeout=60
counter=0

while [ $counter -lt $timeout ]; do
    if docker-compose ps postgres | grep -q "healthy" && docker-compose ps redis | grep -q "healthy"; then
        echo "✅ Services are ready!"
        break
    fi
    
    echo "⏳ Waiting for services... ($counter/$timeout)"
    sleep 2
    counter=$((counter + 2))
done

if [ $counter -ge $timeout ]; then
    echo "❌ Services failed to start within $timeout seconds"
    docker-compose logs postgres redis
    exit 1
fi

# Run database migrations
echo "📊 Running database migrations..."
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
    alembic upgrade head 2>/dev/null || echo "ℹ️  No migrations to run yet"
else
    echo "⚠️  Virtual environment not found. Please run migrations manually."
fi

echo "🎉 Development environment is ready!"
echo ""
echo "📋 Services running:"
echo "   📊 PostgreSQL: localhost:5432"
echo "   📦 Redis: localhost:6379"
echo ""
echo "🚀 To start the application:"
echo "   source venv/bin/activate"
echo "   python -m yield_arbitrage.main"
echo ""
echo "🧪 To run tests with real services:"
echo "   source venv/bin/activate"
echo "   python -m pytest tests/"
echo ""
echo "🛑 To stop services:"
echo "   docker-compose down"