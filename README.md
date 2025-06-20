# Yield Arbitrage System

AI-driven DeFi yield arbitrage through graph-based strategy discovery.

## Overview

This system discovers and executes profitable yield arbitrage opportunities across DeFi protocols using a graph-based approach. Instead of hard-coding specific strategies, we model the entire DeFi ecosystem as a directed graph where nodes represent assets and edges represent transformations (trades, lending, staking, etc.). Profitable opportunities emerge naturally as paths through this graph where the output value exceeds the input value.

## Setup

### Local Development with Docker (Recommended)

1. **Quick Start**: `./scripts/dev-setup.sh`
   - Starts PostgreSQL and Redis in Docker
   - Sets up environment variables
   - Runs database migrations

2. **Manual Setup**:
   ```bash
   # Start services
   docker-compose up -d postgres redis
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt -r requirements-dev.txt
   
   # Set up environment
   cp .env.docker .env
   
   # Run migrations (when available)
   alembic upgrade head
   ```

### Local Development without Docker

1. Create virtual environment: `python3 -m venv venv`
2. Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt -r requirements-dev.txt`
4. Install PostgreSQL and Redis locally
5. Copy `.env.example` to `.env` and configure your settings
6. Run tests: `pytest`

## Development

### With Docker Services

Start services first:
```bash
docker-compose up -d postgres redis
```

Run the development server:
```bash
source venv/bin/activate
python -m yield_arbitrage.main
```

### Full Docker Development

```bash
# Build and run everything
docker-compose up --build

# Or run in development mode with file watching
docker-compose up app
```

### Testing

With services running:
```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/unit/
pytest tests/integration/

# Test with real services
python validate_database.py
python validate_redis.py
```

## Architecture

- **Graph Engine**: Core graph data structure representing the DeFi ecosystem
- **Protocol Integration**: Adapters for different DeFi protocols
- **Data Collection**: Smart data collection and caching system
- **Pathfinding**: ML-guided beam search for profitable opportunities
- **Risk Management**: Delta tracking and position monitoring
- **Execution**: Transaction submission and monitoring

## License

Private - All rights reserved