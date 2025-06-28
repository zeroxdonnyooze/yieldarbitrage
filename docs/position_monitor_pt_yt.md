# Principal Token (pT) and Yield Token (yT) Position Monitor

## Overview

The `PrincipalYieldPositionMonitor` is a specialized monitoring system for positions involving Principal Tokens (pT) and Yield Tokens (yT) on Pendle-like protocols. These instruments represent time-decay sensitive derivatives of yield-bearing assets and require unique monitoring approaches.

## Key Concepts

### Principal Tokens (pT)
- Represent the principal portion of a yield-bearing asset
- Trade at a discount to the underlying asset
- Converge to face value at maturity
- Benefit from decreasing interest rates
- Subject to minimal time decay

### Yield Tokens (yT)
- Represent the yield portion of a yield-bearing asset
- Capture yield accrual over time
- Benefit from increasing interest rates
- Subject to significant time decay as maturity approaches
- Value approaches zero at maturity if no yield accrues

## Monitoring Categories

### 1. Maturity Risk Monitoring

#### Critical Alerts (≤1 day to maturity)
- **Alert Type**: `maturity_critical`
- **Severity**: `CRITICAL`
- **Action**: Immediate position closure or settlement preparation
- **Timeline**: <24 hours

#### Warning Alerts (≤7 days to maturity)
- **Alert Type**: `maturity_warning`
- **Severity**: `WARNING`
- **Action**: Plan position closure or settlement
- **Timeline**: Within 7 days

#### Informational Updates (≤30 days to maturity)
- **Alert Type**: `maturity_info`
- **Severity**: `INFO`
- **Purpose**: Timeline awareness for medium-term positions

### 2. Time Decay (Theta) Monitoring

Time decay affects pT and yT differently:

```python
# Simplified time decay calculation
if position_side in ['yield', 'both']:
    # Yield tokens experience more time decay
    estimated_decay = time_elapsed_ratio * 0.3  # Up to 30% decay
else:
    # Principal tokens less affected by time decay
    estimated_decay = time_elapsed_ratio * 0.1  # Up to 10% decay
```

#### Time Decay Alerts
- **Threshold**: 15% estimated decay
- **Severity**: WARNING (15-25%) or ERROR (>25%)
- **Recommendation**: Consider position closure to avoid further decay

### 3. Implied Yield Deviation Monitoring

Monitors changes in implied yield rates that affect position values:

#### Yield Impact Assessment
- **Principal Token Positions**: Higher yields are favorable (discount increases)
- **Yield Token Positions**: Lower yields are favorable (more yield capture)
- **Balanced Positions**: Neutral assessment

#### Alert Thresholds
- **Minor Deviation**: 25-50% change → WARNING
- **Major Deviation**: >50% change → ERROR
- **Favorable Changes**: INFO alerts for profit opportunities

### 4. pT/yT Balance Monitoring

For positions holding both pT and yT tokens:

```python
current_ratio = pt_amount / yt_amount
ratio_deviation = abs(current_ratio - target_ratio) / target_ratio

if ratio_deviation > 0.15:  # 15% deviation threshold
    severity = WARNING if ratio_deviation < 0.3 else ERROR
```

#### Balance Alerts
- **Alert Type**: `pt_yt_imbalance`
- **Threshold**: 15% deviation from target ratio
- **Action**: Rebalance token exposures

### 5. Underlying Asset Exposure

Monitors price movements of the underlying yield-bearing asset:

#### Volatility Thresholds
- **20-30% movement**: INFO alert
- **30-40% movement**: WARNING alert
- **>40% movement**: ERROR alert

## Configuration Parameters

```python
@dataclass
class MonitoringConfig:
    # pT/yT specific thresholds
    maturity_warning_days: int = 7       # Warn X days before maturity
    maturity_critical_days: int = 1      # Critical alert X days before maturity
    time_decay_threshold: float = 0.15   # 15% time decay warning
    yield_rate_deviation_threshold_pt: float = 30.0  # 30% yield rate change
    implied_yield_deviation_threshold: float = 25.0  # 25% implied yield change
```

## Position Metadata Requirements

For proper monitoring, positions should include:

```python
metadata = {
    "maturity_date": "2024-12-31T23:59:59Z",  # ISO format or timestamp
    "position_side": "both",                   # 'principal', 'yield', or 'both'
    "underlying_asset": "ETH_MAINNET_STETH",   # Underlying asset identifier
    "initial_implied_yield": 0.055,            # Initial implied yield rate
    "current_implied_yield": 0.048,            # Current implied yield rate
    "initial_days_to_maturity": 365,           # Days at position creation
    "pt_amount": 25.0,                         # Principal token amount
    "yt_amount": 25.0,                         # Yield token amount (if applicable)
    "target_pt_yt_ratio": 1.0,                 # Target pT:yT ratio
    "initial_underlying_price": 2000.0,        # Underlying price at creation
    "protocol": "pendle"                       # Protocol identifier
}
```

## Alert Examples

### 1. Critical Maturity Alert
```json
{
    "position_id": "PT_YT_CRITICAL_001",
    "position_type": "principal_yield",
    "alert_type": "maturity_critical",
    "severity": "critical",
    "message": "Position expires in 18.0 hours",
    "recommended_action": "URGENT: Close position or prepare for settlement immediately"
}
```

### 2. Time Decay Warning
```json
{
    "position_id": "PT_YT_DECAY_002",
    "alert_type": "time_decay",
    "severity": "error",
    "message": "Significant time decay: 27.2%",
    "recommended_action": "Consider closing position to avoid further time decay"
}
```

### 3. Yield Deviation Alert
```json
{
    "position_id": "PT_YT_YIELD_003",
    "alert_type": "yield_deviation",
    "severity": "warning",
    "message": "Implied yield decreased by 37.8% (unfavorable)",
    "recommended_action": "Consider rebalancing position based on yield environment changes"
}
```

### 4. Balance Imbalance Alert
```json
{
    "position_id": "PT_YT_IMBALANCE_004",
    "alert_type": "pt_yt_imbalance",
    "severity": "error",
    "message": "pT/yT ratio imbalance: 1.67 (target: 1.00)",
    "recommended_action": "Rebalance by adjusting principal token exposure"
}
```

## Risk Assessment Framework

### Portfolio-Level Analysis

1. **Maturity Distribution**
   - Expired Soon (≤1 day): Critical risk
   - Short Term (1-7 days): High attention needed
   - Medium Term (7-30 days): Regular monitoring
   - Long Term (>30 days): Standard monitoring

2. **Risk Score Calculation**
   ```python
   risk_factors = {
       "critical_alerts": count * 40,
       "error_alerts": count * 20,
       "warning_alerts": count * 10,
       "expiring_soon": count * 30,
       "short_term_positions": count * 15
   }
   ```

3. **Risk Status Classification**
   - **≥70% Risk Score**: 🔴 HIGH RISK - Immediate attention required
   - **40-69% Risk Score**: 🟡 MEDIUM RISK - Monitor closely
   - **<40% Risk Score**: 🟢 LOW RISK - Continue regular monitoring

## Integration Points

### 1. Asset Oracle Integration
- Real-time underlying asset price feeds
- Implied yield rate calculations
- Market condition assessment

### 2. Delta Tracker Integration
- Position value tracking
- Exposure calculations
- Portfolio aggregation

### 3. Execution Logger Integration
- Alert persistence
- Historical trend analysis
- Performance metrics

## Best Practices

### 1. Position Management
- Set maturity reminders well in advance
- Monitor time decay acceleration near maturity
- Maintain balanced pT/yT ratios when applicable
- Consider yield environment changes

### 2. Risk Management
- Avoid holding positions too close to maturity
- Monitor underlying asset volatility
- Set position size limits based on maturity
- Plan exit strategies early

### 3. Monitoring Frequency
- **High-frequency**: Positions <7 days to maturity
- **Regular**: Positions with significant time decay
- **Standard**: Long-term positions with stable yields

## Testing and Validation

The system includes comprehensive testing scenarios:

1. **Critical maturity positions** (18 hours to expiry)
2. **High time decay positions** (27% decay)
3. **Unfavorable yield environments** (37% yield decrease)
4. **Imbalanced pT/yT ratios** (67% over target)
5. **Healthy warning positions** (5 days to maturity)
6. **Long-term stable positions** (11+ months)

## Future Enhancements

1. **Advanced Time Decay Models**
   - Option Greeks integration
   - Volatility-adjusted decay rates
   - Protocol-specific decay curves

2. **Yield Curve Analysis**
   - Forward rate implications
   - Curve steepening/flattening effects
   - Cross-asset yield correlations

3. **Automated Rebalancing**
   - Dynamic pT/yT ratio adjustment
   - Maturity ladder optimization
   - Yield environment adaptation

4. **Cross-Protocol Analytics**
   - Pendle vs. other pT/yT protocols
   - Arbitrage opportunity detection
   - Liquidity analysis across venues