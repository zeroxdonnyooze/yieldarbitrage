# Yield Arbitrage Graph System

## Project Overview

This project implements a production-ready yield arbitrage system that models the entire DeFi ecosystem as a directed graph. The system captures MEV opportunities through atomic smart contract execution, real-time data processing, and ML-guided pathfinding. Based on expert feedback, the architecture now includes sophisticated MEV protection, smart contract routers, and advanced execution safety mechanisms.

## Core Concept

- **Nodes**: Represent any asset (tokens, PT/YT tokens, collateral positions)
- **Edges**: Represent transformations (trades, lending, staking, bridges, etc.)
- **Opportunities**: Profitable paths through the graph discovered by ML-guided search
- **Learning**: System improves over time by learning from successful patterns

## Key Technical Decisions

1. **Language**: Python 3.11+ with async/await throughout
2. **Execution**: Smart contract routers for atomic multi-protocol transactions
3. **Data Processing**: WebSocket event subscriptions with state delta processing
4. **MEV Protection**: Flashbots/private relay integration with optimal bidding
5. **ML Pipeline**: Continuous online learning with market microstructure features
6. **Graph Search**: ML-guided beam search with uncertainty quantification
7. **Data Storage**: PostgreSQL for historical data, Redis for real-time cache
8. **User Interface**: Telegram bot for control and monitoring

## Architecture Guidelines

- **Atomic Execution**: All arbitrage paths execute atomically in smart contract routers
- **MEV Competitive**: System designed to compete effectively in MEV-protected environments
- **Real-Time First**: WebSocket event processing with sub-second latency for critical edges
- **Smart Caching**: Predictive caching with time-series models for less critical edges
- **Risk Aware**: Comprehensive slippage prediction and circuit breaker protection
- **Async First**: All I/O operations use async/await
- **Delta Awareness**: All paths track market exposure for risk management

## Component Structure

```
yield-arbitrage-system/
├── contracts/                  # Smart contract router system
│   ├── __init__.py
│   ├── router_deployer.py      # Deploy routers on each chain
│   ├── router_manager.py       # Manage router contracts
│   ├── calldata_generator.py   # Generate dynamic calldata
│   ├── flash_loan_wrapper.py   # Flash loan integration
│   └── solidity/               # Solidity contracts
│       ├── ArbitrageRouter.sol
│       ├── FlashLoanRouter.sol
│       └── interfaces/
├── websocket_manager/          # Real-time data streaming
│   ├── __init__.py
│   ├── event_subscriber.py     # WebSocket event subscriptions
│   ├── state_delta_processor.py # Process state changes
│   ├── latency_monitor.py      # Sub-second latency tracking
│   └── predictive_cache.py     # Time-series based caching
├── mev_protection/             # Private mempool integration
│   ├── __init__.py
│   ├── flashbots_client.py     # Flashbots integration
│   ├── private_relay.py        # Private relay management
│   ├── bid_calculator.py       # Optimal bid calculation
│   └── mev_risk_assessor.py    # MEV risk evaluation
├── risk_analysis/              # Advanced risk metrics
│   ├── __init__.py
│   ├── slippage_predictor.py   # Liquidity-based slippage prediction
│   ├── circuit_breaker.py      # Abnormal condition detection
│   ├── gas_estimator.py        # Volatility-aware gas estimation
│   └── uncertainty_quantifier.py # ML confidence scoring
├── graph_engine/
│   ├── __init__.py
│   ├── universal_graph.py      # Core graph data structure
│   ├── edge_types.py           # Edge type definitions (+ back-running)
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
│   ├── beam_search.py          # ML-guided pathfinding with uncertainty
│   ├── ml_scorer.py            # Continuous online learning models
│   ├── pattern_learner.py      # Market microstructure features
│   └── exploration_manager.py  # Dynamic exploration rate
├── data_collection/
│   ├── __init__.py
│   ├── hybrid_collector.py     # Orchestrates data sources
│   ├── event_monitor.py        # WebSocket event processing
│   └── api_clients.py          # Protocol-specific APIs
├── execution/
│   ├── __init__.py
│   ├── atomic_validator.py     # Validate atomic execution feasibility
│   ├── simulator.py            # Simulate paths before execution
│   ├── delta_tracker.py        # Track market exposure
│   ├── executor.py             # Submit transactions via MEV protection
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

1. **Smart Contract First**: Deploy router contracts before implementing arbitrage logic
2. **Real-Time Data**: Set up WebSocket event processing for critical pools
3. **MEV Protection**: Integrate Flashbots before any mainnet execution
4. **Atomic Validation**: Ensure all paths can execute atomically before adding to graph
5. **Risk Management**: Implement circuit breakers and slippage prediction
6. **Start Simple**: Begin with basic TRADE edges between major tokens
7. **Add Protocols Incrementally**: Start with Uniswap, then add lending, then yield tokenization
8. **Continuous Learning**: Use online ML models that adapt to market conditions

## Edge Type Implementations

When implementing edges, each type has specific requirements:

- **TRADE**: Must account for AMM fees, slippage, gas costs, atomic execution feasibility
- **SPLIT**: Must maintain value conservation (1 stETH = 1 PT + 1 YT) and atomic execution
- **LEND**: Must track collateral ratios and liquidation thresholds with real-time monitoring
- **BRIDGE**: Must include bridge fees, time delays, and cross-chain execution safety
- **SHORT**: Must track funding rates and liquidation prices with position monitoring
- **BACK_RUN**: New edge type for MEV back-running opportunities with optimal timing

## ML Model Training

The enhanced ML pipeline includes:
- **Continuous Learning**: Replace weekly retraining with online learning algorithms
- **Market Microstructure**: Add volatility, liquidity depth, and order flow features
- **Dynamic Exploration**: Adjust exploration rate based on market conditions
- **Uncertainty Quantification**: Track confidence scores with statistical bounds
- **Predictive Models**: Time-series forecasting for less critical edge caching
- **Risk Features**: Include MEV risk, slippage prediction, and gas volatility

## Testing Requirements

Every component needs:
- Unit tests for core logic and smart contract interactions
- Integration tests with real blockchain data and MEV simulation
- Atomic execution tests to ensure transaction feasibility
- Stress tests for WebSocket event processing under high load
- Slippage prediction accuracy tests with historical data
- Circuit breaker tests for abnormal market conditions
- Performance benchmarks for sub-second latency requirements

## Security Considerations

- **Smart Contract Security**: Audit all router contracts for reentrancy and flash loan attacks
- **MEV Protection**: Use private mempools to prevent frontrunning and sandwich attacks
- **Key Management**: Never store private keys in code or logs, use secure key management
- **Atomic Execution**: Ensure all transactions can revert safely if any step fails
- **Slippage Protection**: Implement comprehensive slippage bounds with circuit breakers
- **Real-Time Monitoring**: Monitor for abnormal price movements and market manipulation
- **Gas Safety**: Implement gas estimation with volatility protection to prevent failures

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

- **Latency**: Process WebSocket events with sub-second latency for critical edges
- **Discovery**: Find top 10 opportunities in < 500ms using ML-guided search
- **Execution**: Atomic transaction confirmation within 12 seconds on mainnet
- **Accuracy**: Slippage prediction within 0.1% of actual execution
- **Reliability**: > 99% atomic execution success rate with proper gas estimation
- **MEV Protection**: Successfully submit > 90% of transactions via private relays
- **Cache Performance**: > 95% cache hit rate with predictive models

## Next Steps

1. **Infrastructure Setup**: Set up development environment with Docker and databases
2. **Smart Contract Development**: Deploy router contracts on testnets
3. **WebSocket Integration**: Implement real-time event processing for critical pools
4. **MEV Protection Setup**: Integrate Flashbots and private relay infrastructure
5. **Risk Management**: Implement circuit breakers and slippage prediction
6. **Graph Structure**: Implement enhanced graph with atomic execution validation
7. **ML Pipeline**: Build continuous learning models with uncertainty quantification
8. **Protocol Integration**: Add Uniswap V3 with atomic execution support
9. **Testing Framework**: Comprehensive testing including atomic execution validation
10. **Mainnet Deployment**: Deploy with MEV protection and small capital for live testing

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

- If the AI is uncertain about the context of the code or the error, it should ask the user for clarification. Questions like "Did you write this code?" or "When did this error first appear?" help provide essential context.

### 5. Summarize Understanding Before Proceeding

- After reviewing the code and context, the AI should provide a summary of its understanding and confirm it with the user before suggesting any fixes.

### 6. Use Logging Judiciously

- Only add logs if, after attempting to understand the code and context, more information is still needed.

## Enhanced Edge Model and Execution Architecture

### Edge Execution Properties

Every edge in the graph declares its execution requirements through properties:

```python
class EdgeExecutionProperties(BaseModel):
    # Core execution constraints
    supports_synchronous: bool = True  # Can execute in same transaction
    requires_time_delay: Optional[int] = None  # Seconds needed between steps
    requires_bridge: bool = False  # Needs cross-chain bridge
    requires_capital_holding: bool = False  # E.g., lending positions, yield farming
    
    # Risk and MEV properties
    max_slippage: float = 0.05  # Maximum acceptable slippage
    mev_sensitivity: float = 0.5  # 0-1 scale of frontrun risk
    supports_private_mempool: bool = True  # Can use Flashbots etc
    
    # Gas and cost properties
    gas_estimate: int = 100000  # Estimated gas for this edge
    requires_approval: bool = True  # Needs token approval
    
    # Liquidity and market properties
    min_liquidity_required: float = 10000  # Minimum USD liquidity
    max_impact_allowed: float = 0.01  # Max price impact

class YieldGraphEdge(BaseModel):
    # ... existing fields ...
    execution_properties: EdgeExecutionProperties = EdgeExecutionProperties()
```

### Special Edge Types

#### Flash Loan Edges
Flash loans are modeled as edges that provide capital but require synchronous execution:

```python
class FlashLoanEdge(YieldGraphEdge):
    def __init__(self, chain_name: str, provider: str, asset: str, max_amount: float, fee: float):
        super().__init__(
            edge_type=EdgeType.FLASH_LOAN,
            source_asset_id=f"{chain_name}_FLASH_{asset}",
            target_asset_id=f"{chain_name}_{asset}",
            execution_properties=EdgeExecutionProperties(
                supports_synchronous=True,
                requires_capital_holding=False,
                gas_estimate=150000,
                mev_sensitivity=0.3  # Lower MEV risk for flash loans
            )
        )
        self.provider = provider  # AAVE, Balancer, etc
        self.fee_percentage = fee
```

#### Back-Running Edges
MEV opportunities modeled as edges:

```python
class BackRunEdge(YieldGraphEdge):
    def __init__(self, target_tx: str, impact: Dict):
        super().__init__(
            edge_type=EdgeType.BACK_RUN,
            execution_properties=EdgeExecutionProperties(
                supports_synchronous=True,
                mev_sensitivity=0.0,  # No frontrun risk
                requires_capital_holding=False
            )
        )
```

### Path Analysis and Execution

#### Path Feasibility Analysis
System analyzes complete paths to determine execution requirements and identifies synchronous segments for batching:

```python
class PathExecutionAnalyzer:
    async def analyze_path(self, path: List[YieldGraphEdge]) -> PathAnalysis:
        # Identify synchronous segments that can be batched together
        synchronous_segments = self._identify_synchronous_segments(path)
        
        return PathAnalysis(
            is_fully_atomic=all(e.execution_properties.supports_synchronous for e in path),
            synchronous_segments=synchronous_segments,  # List of (start_idx, end_idx) tuples
            total_time_required=sum(e.execution_properties.requires_time_delay or 0 for e in path),
            chains_involved=list(set(e.chain_name for e in path)),
            requires_capital=any(e.execution_properties.requires_capital_holding for e in path),
            max_slippage=max(e.execution_properties.max_slippage for e in path),
            total_gas=sum(e.execution_properties.gas_estimate for e in path),
            mev_risk=max(e.execution_properties.mev_sensitivity for e in path)
        )
    
    def _identify_synchronous_segments(self, path: List[YieldGraphEdge]) -> List[Tuple[int, int]]:
        """Identify consecutive edges that can be executed synchronously in batches."""
        segments = []
        current_start = 0
        
        for i, edge in enumerate(path):
            # Check if this edge breaks synchronous execution
            if (not edge.execution_properties.supports_synchronous or 
                edge.execution_properties.requires_time_delay or
                (i > 0 and edge.chain_name != path[i-1].chain_name)):
                
                # Close current segment if it has multiple edges
                if i > current_start:
                    segments.append((current_start, i - 1))
                
                # Handle non-synchronous edge as single segment
                if not edge.execution_properties.supports_synchronous:
                    segments.append((i, i))  # Individual execution required
                
                current_start = i + 1
        
        # Close final segment
        if current_start < len(path):
            segments.append((current_start, len(path) - 1))
            
        return segments
```

#### Execution Routing
Based on path analysis, execute synchronous segments in batches and coordinate multi-step execution:

```python
class ExecutionRouter:
    async def execute_path(self, path: List[YieldGraphEdge], analysis: PathAnalysis):
        """Execute path by batching synchronous segments and coordinating multi-step execution."""
        execution_results = []
        
        for segment_start, segment_end in analysis.synchronous_segments:
            segment = path[segment_start:segment_end + 1]
            segment_analysis = await self.analyze_segment(segment)
            
            # Route segment execution based on properties
            if len(segment) == 1:
                # Single edge - choose appropriate executor
                result = await self._execute_single_edge(segment[0])
            else:
                # Multiple edges - batch in smart contract
                result = await self._execute_batched_segment(segment, segment_analysis)
            
            execution_results.append(result)
            
            # Handle any required delays between segments
            if segment_end < len(path) - 1:  # Not the last segment
                next_edge = path[segment_end + 1]
                if next_edge.execution_properties.requires_time_delay:
                    await self._handle_time_delay(next_edge.execution_properties.requires_time_delay)
        
        return execution_results
    
    async def _execute_batched_segment(self, segment: List[YieldGraphEdge], analysis):
        """Execute multiple synchronous edges in a single transaction."""
        # Route based on MEV risk and value
        segment_value = sum(edge.state.liquidity_usd or 0 for edge in segment)
        max_mev_risk = max(edge.execution_properties.mev_sensitivity for edge in segment)
        
        if segment_value > 100_000 and max_mev_risk > 0.7:
            # High-value, high-risk -> Flashbots
            return await self.flashbots_executor.execute_batch(segment)
        elif max_mev_risk > 0.3:
            # Medium risk -> Private relay
            return await self.private_relay_executor.execute_batch(segment)
        else:
            # Low risk -> Public mempool
            return await self.atomic_executor.execute_batch(segment)
    
    async def _execute_single_edge(self, edge: YieldGraphEdge):
        """Execute single edge with appropriate method."""
        if edge.execution_properties.requires_capital_holding:
            return await self.position_manager.execute_edge(edge)
        elif edge.execution_properties.requires_bridge:
            return await self.bridge_executor.execute_edge(edge)
        else:
            return await self.atomic_executor.execute_edge(edge)
```

### Smart Contract Router Architecture

Router contracts support batched execution of synchronous edge segments:

```solidity
// ArbitrageRouter.sol
contract ArbitrageRouter {
    // Execute batched synchronous edges in a single transaction
    function executeBatchedSegment(
        address[] calldata targets,
        bytes[] calldata calldatas,
        uint256[] calldata values,
        address inputToken,
        uint256 inputAmount,
        address outputToken,
        uint256 minOutputAmount
    ) external payable returns (uint256 outputAmount) {
        // Transfer input tokens to router
        IERC20(inputToken).transferFrom(msg.sender, address(this), inputAmount);
        
        uint256 currentAmount = inputAmount;
        address currentToken = inputToken;
        
        // Execute all calls in sequence
        for (uint i = 0; i < targets.length; i++) {
            // Approve tokens if needed
            if (currentAmount > 0) {
                IERC20(currentToken).approve(targets[i], currentAmount);
            }
            
            uint256 balanceBefore = IERC20(outputToken).balanceOf(address(this));
            (bool success, bytes memory result) = targets[i].call{value: values[i]}(calldatas[i]);
            require(success, "Batched call failed");
            
            // Update current token/amount for next step
            currentAmount = IERC20(outputToken).balanceOf(address(this)) - balanceBefore;
            currentToken = outputToken;
        }
        
        // Verify minimum output and transfer to sender
        require(currentAmount >= minOutputAmount, "Insufficient output");
        IERC20(outputToken).transfer(msg.sender, currentAmount);
        
        return currentAmount;
    }
    
    // Execute with flash loan for capital-efficient batched arbitrage
    function executeBatchedWithFlashLoan(
        address flashLoanProvider,
        uint256 flashLoanAmount,
        address[] calldata targets,
        bytes[] calldata calldatas,
        uint256[] calldata values
    ) external {
        // Initiate flash loan and execute batched calls in callback
        // Repay flash loan with profits
    }
    
    // Flash loan callback handlers
    function onFlashLoan(...) external { ... }
}
```

### Real-Time Data Architecture

#### Edge Priority Classification
```python
class EdgePriorityClassifier:
    def classify(self, edge: YieldGraphEdge) -> Priority:
        score = (
            edge.state.liquidity_usd * 0.3 +
            edge.state.daily_volume * 0.3 +
            edge.profitability_score * 0.4
        )
        
        if score > 1_000_000: return Priority.CRITICAL  # WebSocket
        elif score > 100_000: return Priority.HIGH      # 5 second updates  
        else: return Priority.NORMAL                   # Predictive cache
```

#### WebSocket Event Processing
Different subscription strategies per edge type:
- **TRADE edges**: Subscribe to Swap events
- **LEND edges**: Subscribe to Deposit/Withdraw/Liquidation events
- **YIELD edges**: Subscribe to PT/YT price updates

### ML Pipeline Architecture

#### Edge-Specific Models
```python
edge_models = {
    EdgeType.TRADE: {
        'features': ['liquidity_depth', 'volume_24h', 'price_volatility'],
        'model': OnlineLSTM(hidden_size=128)
    },
    EdgeType.LEND: {
        'features': ['utilization_rate', 'borrow_apy', 'collateral_factor'],
        'model': OnlineGradientBoosting()
    },
    EdgeType.YIELD: {
        'features': ['time_to_maturity', 'implied_yield', 'pt_yt_ratio'],
        'model': OnlineRandomForest()
    }
}
```

#### Dynamic Exploration
```python
exploration_rate = base_rate * (1 + volatility_factor + uncertainty_factor)
# Higher exploration when:
# - Market volatility increases
# - Model uncertainty is high
# - New edge types discovered
```

### Safety and Risk Management

#### Edge-Specific Circuit Breakers
```python
circuit_breakers = {
    EdgeType.TRADE: {
        'max_slippage': 0.02,
        'min_liquidity': 100_000,
        'max_price_impact': 0.01
    },
    EdgeType.LEND: {
        'max_utilization': 0.95,
        'min_health_factor': 1.1,
        'max_ltv': 0.8
    },
    EdgeType.BRIDGE: {
        'max_bridge_fee': 0.01,
        'max_wait_time': 3600,
        'min_bridge_liquidity': 1_000_000
    }
}
```

## Simulation & Validation Layer

### Tenderly Integration

We will use the Tenderly API to perform off-chain transaction simulations and extract call graphs that represent edge operations in our DeFi graph. Each edge in the graph corresponds to a state-transforming contract call (e.g., a swap, a borrow, an LP addition). Tenderly will be used in the following ways:

- **Edge Simulation & Validation**: Before executing a path through the graph, simulate it on a forked chain state using Tenderly to detect failures (e.g., reverts, slippage) and ensure viability.
- **Edge Extraction & Typing**: Use Tenderly's call trace API to extract call graphs from historical or hypothetical transactions. These graphs help us classify and define reusable edge templates.
- **Fallback Planning**: We may support fallback to local EVM forks (e.g., Hardhat or Anvil) for simulation to reduce cost and avoid rate limits.
- **Phase Scope**: Initially used for development and testing. Future phases may include live path validation before transaction submission.

#### Tenderly Architecture Integration

```python
class TenderlySimulator:
    """Tenderly-based transaction simulation and call graph extraction."""
    
    async def simulate_edge(self, edge: YieldGraphEdge, state: ChainState) -> SimulationResult:
        """Simulate single edge execution and return detailed results."""
        
    async def simulate_path(self, path: List[YieldGraphEdge], state: ChainState) -> PathSimulationResult:
        """Simulate entire arbitrage path with fork state management."""
        
    async def extract_call_graph(self, tx_hash: str, chain_id: int) -> CallGraphResult:
        """Extract call graph from historical transaction for edge template creation."""
        
    async def validate_with_fallback(self, operation: Operation) -> ValidationResult:
        """Try Tenderly first, fallback to local simulation if needed."""

class EdgeValidator:
    """Enhanced edge validation using simulation results."""
    
    async def validate_edge_viability(self, edge: YieldGraphEdge) -> ValidationResult:
        """Check if edge will execute successfully with current market conditions."""
        return ValidationResult(
            will_revert=False,
            expected_output=1500.0,
            gas_estimate=150000,
            price_impact=0.002,
            slippage_estimate=0.001,
            warnings=["High gas cost relative to profit"]
        )
```

#### Simulation-Enhanced Execution Pipeline

```python
class SimulationEnhancedExecutor:
    """Executor that validates paths via simulation before live execution."""
    
    async def execute_with_validation(self, path: List[YieldGraphEdge]) -> ExecutionResult:
        # 1. Simulate path on current chain state
        simulation = await self.tenderly.simulate_path(path, await self.get_current_state())
        
        # 2. Check simulation results
        if simulation.will_revert:
            return ExecutionResult(success=False, reason=simulation.revert_reason)
        
        # 3. Validate profitability after simulation adjustments
        adjusted_profit = simulation.expected_output - simulation.gas_cost_usd
        if adjusted_profit < self.min_profit_threshold:
            return ExecutionResult(success=False, reason="Insufficient profit after simulation")
        
        # 4. Execute on mainnet with simulation-informed parameters
        return await self.atomic_executor.execute_path(path, simulation.optimized_params)
```

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

---

# Production Deployment Plan: Yield Arbitrage System

## 🔍 Current System Analysis

**Mocked/Incomplete Components Identified:**
- Graph Engine (TODO in main.py)
- Position Monitor (Mock implementations)
- Delta Tracker (Mock implementations) 
- Execution Logger (Mock implementations)
- Pathfinding components (Some incomplete)
- MEV Protection (Stub implementations)
- ML Models (Empty module)
- Telegram Bot (Using database adapters/mocks)

**Working Components:**
- Database layer with Alembic migrations
- Blockchain connectivity (AsyncWeb3, multicall)
- FastAPI application structure
- Redis caching
- Telegram bot framework
- Protocol adapters (Uniswap V3, others)

## 🎯 Production Deployment Strategy

### Phase 1: Infrastructure Setup (Week 1)

**1.1 Cloud Hosting Selection**
- **Free Option:** Railway (free tier: 512MB RAM, 1GB storage)
- **Paid Option:** DigitalOcean App Platform ($12/month for 1GB RAM, 25GB storage)
- **Alternative:** AWS ECS Fargate ($20-30/month estimated)

**1.2 Database Hosting**
- **Free Option:** Supabase (500MB free PostgreSQL)
- **Paid Option:** Railway PostgreSQL ($5/month for 1GB)
- **Alternative:** AWS RDS PostgreSQL ($15-25/month)

**1.3 Cache/Redis Hosting**
- **Free Option:** Railway Redis (256MB free)
- **Paid Option:** Redis Cloud ($5/month for 30MB)
- **Alternative:** AWS ElastiCache ($10-15/month)

### Phase 2: Blockchain Infrastructure (Week 1-2)

**2.1 RPC Provider Setup**
- **Free Option:** Alchemy (300M compute units/month free)
- **Paid Option:** Alchemy Growth Plan ($49/month for higher limits)
- **Fallback:** Infura ($50/month for Growth)

**2.2 Multi-chain Support**
- Configure Ethereum mainnet (required)
- Configure Arbitrum (required for arbitrage)
- Configure Base (high volume DEX activity)
- Configure Sonic/Berachain (emerging opportunities)

### Phase 3: Complete Mock Replacements (Week 2-3)

**3.1 Graph Engine Implementation**
- Replace TODO with real graph initialization
- Implement edge discovery from DEX contracts
- Add real-time price feed integration
- Connect to blockchain data sources

**3.2 Position Monitor**
- Replace mock with real position tracking
- Implement portfolio health calculations
- Add liquidation risk monitoring
- Create alert system for pT/yT positions

**3.3 Delta Tracker**
- Implement real P&L calculations
- Add position exposure tracking
- Create risk metrics dashboard
- Connect to price oracles

**3.4 Execution Logger**
- Replace mock with real transaction logging
- Add performance analytics
- Implement execution success/failure tracking
- Create audit trails

### Phase 4: Advanced Features (Week 3-4)

**4.1 MEV Protection**
- Implement Flashbots integration
- Add private mempool routing
- Create MEV risk assessment
- Add frontrunning protection

**4.2 ML Integration**
- Implement ML model serving
- Add profit prediction models
- Create market condition analysis
- Implement dynamic strategy selection

**4.3 Real-time Data Streams**
- WebSocket price feeds
- Mempool monitoring
- Block event streaming
- Market data aggregation

### Phase 5: Production Testing (Week 4-5)

**5.1 Unit Test Migration**
- Replace mocked unit tests with live integrations
- Add integration tests with real blockchain data
- Create end-to-end testing suite
- Implement chaos engineering tests

**5.2 Performance Testing**
- Load testing with realistic data volumes
- Latency optimization for arbitrage execution
- Memory and CPU profiling
- Database query optimization

## 💰 Cost Breakdown

### Minimal Setup (Free Tier)
- **Hosting:** Railway (Free)
- **Database:** Supabase (Free 500MB)
- **Cache:** Railway Redis (Free 256MB)
- **RPC:** Alchemy (Free 300M CU/month)
- **Total:** $0/month (limited scalability)

### Recommended Production Setup
- **Hosting:** DigitalOcean App Platform ($12/month)
- **Database:** Railway PostgreSQL ($5/month)
- **Cache:** Redis Cloud ($5/month)
- **RPC:** Alchemy Growth ($49/month)
- **Monitoring:** Built-in (included)
- **Total:** ~$71/month

### High-Performance Setup
- **Hosting:** AWS ECS Fargate ($25/month)
- **Database:** AWS RDS ($20/month)
- **Cache:** AWS ElastiCache ($15/month)
- **RPC:** Alchemy Scale ($199/month)
- **Monitoring:** AWS CloudWatch ($10/month)
- **CDN:** CloudFlare (Free)
- **Total:** ~$269/month

## 🔧 Implementation Priorities

### High Priority (Must Complete)
1. Database hosting and migration
2. Real blockchain RPC connections
3. Graph engine implementation
4. Position monitoring system
5. Telegram bot cloud deployment

### Medium Priority (Should Complete)
1. MEV protection implementation
2. Real-time price oracles
3. Performance monitoring
4. Alert system enhancement
5. Unit test live integration

### Low Priority (Nice to Have)
1. ML model integration
2. Advanced analytics
3. Multi-region deployment
4. Automated scaling
5. Advanced security features

## 📋 Technical Migration Steps

### 1. Infrastructure Setup
- Set up cloud hosting accounts
- Configure database connections
- Deploy initial application version
- Set up monitoring and logging

### 2. Service Integration
- Replace mock implementations systematically
- Test each component in isolation
- Integrate components step by step
- Validate system performance

### 3. Security Hardening
- Implement API rate limiting
- Add authentication layers
- Secure environment variables
- Enable HTTPS/SSL

### 4. Testing & Validation
- Run comprehensive test suites
- Validate with real market data
- Perform stress testing
- Monitor for edge cases

This plan provides a systematic approach to move from development/mock setup to production-ready deployment with real blockchain integration and live trading capabilities.