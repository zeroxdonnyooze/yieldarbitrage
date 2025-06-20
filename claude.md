# Yield Arbitrage Graph System

## Project Overview

This project implements a sophisticated yield arbitrage system that models the entire DeFi ecosystem as a directed graph. Instead of hard-coding trading strategies, profitable opportunities emerge naturally as paths through the graph where output value exceeds input value. The system uses machine learning to accelerate pathfinding while remaining open to discovering novel strategies.

## Core Concept

- **Nodes**: Represent any asset (tokens, PT/YT tokens, collateral positions)
- **Edges**: Represent transformations (trades, lending, staking, bridges, etc.)
- **Opportunities**: Profitable paths through the graph discovered by ML-guided search
- **Learning**: System improves over time by learning from successful patterns

## Key Technical Decisions

1. **Language**: Python 3.11+ with async/await throughout
2. **Graph Search**: ML-guided beam search (not A*) with probabilistic pruning
3. **Data Storage**: PostgreSQL for historical data, Redis for real-time cache
4. **ML Framework**: PyTorch for neural networks, scikit-learn for simpler models
5. **User Interface**: Telegram bot for control and monitoring

## Architecture Guidelines

- **No Over-Engineering**: Each component is self-contained with built-in health monitoring
- **Async First**: All I/O operations use async/await
- **Smart Caching**: Paths cached with dependency tracking for intelligent invalidation
- **Position Monitoring**: Active positions monitored more frequently than potential ones
- **Delta Awareness**: All paths track market exposure for risk management

## Component Structure

```
yield-arbitrage-system/
├── graph_engine/
│   ├── __init__.py
│   ├── universal_graph.py      # Core graph data structure
│   ├── edge_types.py           # Edge type definitions and calculations
│   └── edge_state.py           # Dynamic edge state management
├── protocol_integration/
│   ├── __init__.py
│   ├── base_adapter.py         # Base class for protocol adapters
│   ├── ai_analyzer.py          # AI-powered protocol discovery
│   ├── token_filter.py         # Quality control for tokens
│   └── adapters/               # Protocol-specific adapters
│       ├── pendle_adapter.py
│       ├── aave_adapter.py
│       └── uniswap_adapter.py
├── search/
│   ├── __init__.py
│   ├── beam_search.py          # Main pathfinding algorithm
│   ├── ml_scorer.py            # ML models for edge/path scoring
│   └── pattern_learner.py      # Learn from successful patterns
├── data_collection/
│   ├── __init__.py
│   ├── hybrid_collector.py     # Orchestrates data sources
│   ├── event_monitor.py        # On-chain event listening
│   └── api_clients.py          # Protocol-specific APIs
├── execution/
│   ├── __init__.py
│   ├── simulator.py            # Simulate paths before execution
│   ├── delta_tracker.py        # Track market exposure
│   ├── executor.py             # Submit transactions
│   └── position_monitor.py     # Monitor active positions
├── cache/
│   ├── __init__.py
│   ├── path_cache.py           # Smart caching with invalidation
│   └── edge_cache.py           # Real-time edge state cache
├── telegram_bot/
│   ├── __init__.py
│   ├── bot.py                  # Command handlers
│   └── alerts.py               # Push notifications
├── config/
│   ├── __init__.py
│   ├── settings.py             # Configuration management
│   └── chains.py               # Chain-specific settings
├── tests/
│   ├── unit/                   # Unit tests for each component
│   ├── integration/            # End-to-end tests
│   └── simulation/             # Historical backtesting
├── scripts/
│   ├── setup_db.py             # Database initialization
│   └── populate_graph.py       # Initial graph population
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Local development setup
├── .env.example                # Environment variables template
└── README.md                   # Setup and usage instructions
```

## Development Workflow

1. **Start Simple**: Begin with basic TRADE edges between major tokens
2. **Add Protocols Incrementally**: Start with Uniswap, then add lending, then yield tokenization
3. **Test Each Edge Type**: Ensure calculations are correct before adding to graph
4. **Monitor Performance**: Built-in metrics in each component
5. **Learn and Iterate**: Use ML feedback to improve search algorithms

## Edge Type Implementations

When implementing edges, each type has specific requirements:

- **TRADE**: Must account for AMM fees, slippage, gas costs
- **SPLIT**: Must maintain value conservation (1 stETH = 1 PT + 1 YT)
- **LEND**: Must track collateral ratios and liquidation thresholds
- **BRIDGE**: Must include bridge fees and time delays
- **SHORT**: Must track funding rates and liquidation prices

## ML Model Training

The ML components should:
- Start with simple features (conversion rate, liquidity, gas cost)
- Use online learning to adapt quickly
- Balance exploration (finding new patterns) vs exploitation (using known patterns)
- Track confidence scores for uncertain predictions

## Testing Requirements

Every component needs:
- Unit tests for core logic
- Integration tests with mock data
- Simulation tests with historical data
- Error handling for network failures
- Performance benchmarks

## Security Considerations

- Never store private keys in code or logs
- Validate all external data before use
- Use read-only blockchain connections where possible
- Implement slippage protection in execution
- Monitor for abnormal price movements

## Supported Chains and Protocols

### Chains
- Ethereum, Arbitrum, Base, Sonic, Berachain

### Initial Protocols
- **DEXs**: Uniswap V3, Curve, Camelot, Aerodrome, BEX, Kodiak
- **Lending**: Aave V3, Compound V3, Morpho, Silo
- **Yield**: Pendle, Spectra
- **Staking**: Lido, Rocket Pool, Infrared

## Configuration

Key settings to externalize:
```python
# Pathfinding
MAX_PATH_LENGTH = 8
MIN_PROFIT_RATIO = 1.005
MAX_TIME_HORIZON = 30 * 86400  # 30 days
BEAM_WIDTH = 50

# Risk Management
MAX_DELTA_EXPOSURE = 0.1  # 10% of capital
MAX_GAS_PERCENTAGE = 0.02  # 2% of profit

# Data Collection
CRITICAL_UPDATE_INTERVAL = 30  # seconds
NORMAL_UPDATE_INTERVAL = 300  # 5 minutes

# ML Parameters
EDGE_SCORE_THRESHOLD = 0.1
EXPLORATION_RATE = 0.2
RETRAIN_INTERVAL = 7 * 86400  # weekly
```

## Common Pitfalls to Avoid

1. **Don't Over-Optimize Early**: Get basic arbitrage working before complex strategies
2. **Handle Split Complexity**: PT/YT splits create multiple assets to track
3. **Account for Gas**: Many profitable paths become unprofitable after gas
4. **Respect Rate Limits**: Both blockchain RPCs and APIs have limits
5. **Monitor Liquidations**: Leveraged positions need constant monitoring

## Performance Targets

- Find top 10 opportunities in < 1 second
- Update critical edges every 30 seconds
- Maintain > 80% cache hit rate
- Execute trades with < 1% slippage
- Zero liquidations on managed positions

## Next Steps

1. Set up development environment with Docker
2. Initialize database schema
3. Implement basic graph structure
4. Add first protocol adapter (suggest Uniswap V3)
5. Build simple pathfinding without ML
6. Add ML scoring once basic system works
7. Integrate Telegram bot for monitoring
8. Begin paper trading to validate strategies
9. Add more protocols incrementally
10. Deploy with small capital for live testing

Remember: Start simple, test thoroughly, and let the graph discover opportunities rather than hard-coding strategies.

## Debugging Guidelines

When encountering unexpected behavior, Claude Code should:

1. **Check whether the AI understands the system being used.**
   - If a tool/library/API is new, read its documentation before guessing.

2. **Validate assumptions.**
   - Log the actual values or outcomes before using them in logic.
   - Confirm contract ABI/function call structure matches assumptions.

3. **Log raw blockchain data.**
   - Before decoding or transforming it, log raw logs/traces to validate assumptions.

4. **Run small, isolated unit tests.**
   - Isolate suspicious parts of logic and write local reproducible tests.

5. **Do not fabricate unknown parameters.**
   - If contract ABIs or addresses are missing, do not guess. Flag and request.

---

## 🐞 Debugging Protocol

### 1. Understand the Problem Before Acting

- Begin by reviewing the code and understanding its intended functionality.
- Consider the overall goal of the project and how the current issue is impacting that goal.

### 2. Consider the Context of the Error

- Ask: Did I write this code? Is this newly written code or something that was changed recently?
- Determine if the error is new or if it existed before any recent changes. This helps identify whether the bug is related to new code, old code, or external systems.
- If using a new system or library, consider that the error might be due to a misunderstanding of how to use that system. In such cases, refer to the documentation.

### 3. Avoid Quick Fixes and Iterative Guessing

- Avoid adding logs or making changes without first attempting to understand the issue thoroughly.
- Simplification, like commenting out code or turning off features, should be a last resort and done with user approval.

### 4. Ask Clarifying Questions

- If the AI is uncertain about the context of the code or the error, it should ask the user for clarification. Questions like “Did you write this code?” or “When did this error first appear?” help provide essential context.

### 5. Summarize Understanding Before Proceeding

- After reviewing the code and context, the AI should provide a summary of its understanding and confirm it with the user before suggesting any fixes.

### 6. Use Logging Judiciously

- Only add logs if, after attempting to understand the code and context, more information is still needed.

## Task Management with Taskmaster

This project uses **Taskmaster AI** for structured task management. All tasks are defined in `.taskmaster/tasks/tasks.json` with detailed subtasks for complex components.

### Essential Commands
- `task-master list` - View all tasks and progress
- `task-master next` - Get the next task to work on
- `task-master show <id>` - View task details
- `task-master set-status --id=<id> --status=done` - Mark task complete

### Work Flow
1. **Check current task**: Always run `task-master next` to see what to work on
2. **Mark in-progress**: `task-master set-status --id=<id> --status=in-progress`
3. **Complete work**: Implement the task according to its detailed requirements
4. **Mark done**: `task-master set-status --id=<id> --status=done` when fully complete
5. **Update if needed**: Use `task-master update-task --id=<id> --prompt="..."` for changes

### Key Rules
- **ALWAYS** mark tasks as `done` when completed - don't leave them in-progress
- **NEVER** skip ahead - follow dependency order shown by `task-master next`
- **VERIFY** task completion matches the testStrategy requirements before marking done
- **UPDATE** tasks if requirements change during implementation
