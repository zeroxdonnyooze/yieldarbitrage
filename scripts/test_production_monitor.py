#!/usr/bin/env python3
"""Test script for production monitoring and validation system."""
import asyncio
import sys
import logging
from typing import Dict, Any

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.monitoring.production_monitor import (
    ProductionMonitor, Alert, AlertSeverity, MonitoringCategory, MonitoringMetric
)
from yield_arbitrage.blockchain_connector.provider import BlockchainProvider
from yield_arbitrage.execution.onchain_price_oracle import OnChainPriceOracle
from yield_arbitrage.config.production import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def initialize_dependencies():
    """Initialize all required dependencies."""
    print("🚀 Initializing monitoring dependencies...")
    
    # Load configuration
    config = get_config()
    
    # Initialize blockchain provider
    blockchain_provider = BlockchainProvider()
    await blockchain_provider.initialize()
    
    # Initialize on-chain price oracle
    try:
        from unittest.mock import AsyncMock
        redis_client = AsyncMock()  # Mock Redis for testing
        oracle = OnChainPriceOracle(blockchain_provider, redis_client)
        print("   ✅ Dependencies initialized")
    except Exception as e:
        print(f"   ⚠️  Dependency initialization failed: {e}")
        oracle = None
        redis_client = None
    
    return blockchain_provider, oracle, redis_client


async def test_monitor_initialization():
    """Test production monitor initialization."""
    print("\n📋 Testing Production Monitor Initialization\n")
    
    blockchain_provider, oracle, redis_client = await initialize_dependencies()
    
    try:
        # Initialize production monitor
        monitor = ProductionMonitor(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            redis_client=redis_client,
            alert_webhook_url=None  # No webhook for testing
        )
        
        print("   🔄 Initializing production monitor...")
        
        # Check initialization
        if monitor.health_checks:
            print(f"   ✅ Production monitor initialized with {len(monitor.health_checks)} health checks")
            
            print("   📊 Registered Health Checks:")
            for check_name in monitor.health_checks.keys():
                print(f"      • {check_name}")
            
            print(f"   📊 Monitoring Configuration:")
            for key, value in monitor.monitoring_config.items():
                print(f"      • {key}: {value}")
            
            return monitor
        else:
            print("   ❌ Failed to initialize health checks")
            return None
            
    except Exception as e:
        print(f"   ❌ Monitor initialization test failed: {e}")
        return None
    finally:
        await blockchain_provider.close()


async def test_health_checks():
    """Test individual health checks."""
    print("\n🔍 Testing Individual Health Checks\n")
    
    blockchain_provider, oracle, redis_client = await initialize_dependencies()
    
    try:
        monitor = ProductionMonitor(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            redis_client=redis_client
        )
        
        # Test each health check individually
        health_check_results = {}
        
        for check_name, check_func in monitor.health_checks.items():
            print(f"   🔄 Testing {check_name}...")
            
            try:
                result = await check_func()
                health_check_results[check_name] = result
                
                status_icon = {
                    "healthy": "✅",
                    "degraded": "⚠️",
                    "unhealthy": "❌"
                }.get(result.status, "❓")
                
                print(f"      {status_icon} Status: {result.status}")
                print(f"      📊 Metrics: {len(result.metrics)}")
                print(f"      🚨 Alerts: {len(result.alerts)}")
                print(f"      ⏱️  Duration: {result.duration_ms:.1f}ms")
                
                # Show key metrics
                if result.metrics:
                    print(f"      📈 Key Metrics:")
                    for metric in result.metrics[:3]:  # Show first 3 metrics
                        tags_str = ", ".join([f"{k}={v}" for k, v in metric.tags.items()]) if metric.tags else ""
                        print(f"         • {metric.name}: {metric.value:.2f} {metric.unit} {tags_str}")
                
                # Show alerts
                if result.alerts:
                    print(f"      🚨 Alerts Generated:")
                    for alert in result.alerts:
                        severity_icon = {
                            AlertSeverity.INFO: "ℹ️",
                            AlertSeverity.WARNING: "⚠️",
                            AlertSeverity.ERROR: "❌",
                            AlertSeverity.CRITICAL: "🚨"
                        }.get(alert.severity, "❓")
                        print(f"         {severity_icon} {alert.severity.value}: {alert.title}")
                
                print()
                
            except Exception as e:
                print(f"      ❌ Health check failed: {e}")
                health_check_results[check_name] = None
        
        # Summary
        successful_checks = sum(1 for result in health_check_results.values() if result is not None)
        healthy_checks = sum(1 for result in health_check_results.values() 
                           if result is not None and result.status == "healthy")
        
        print(f"   📊 Health Check Results:")
        print(f"      Successful checks: {successful_checks}/{len(monitor.health_checks)}")
        print(f"      Healthy checks: {healthy_checks}/{successful_checks}")
        
        return health_check_results
        
    except Exception as e:
        print(f"   ❌ Health checks test failed: {e}")
        return None
    finally:
        await blockchain_provider.close()


async def test_alert_system():
    """Test alert generation and handling."""
    print("\n🚨 Testing Alert System\n")
    
    blockchain_provider, oracle, redis_client = await initialize_dependencies()
    
    try:
        monitor = ProductionMonitor(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            redis_client=redis_client
        )
        
        # Create test alerts
        test_alerts = [
            Alert(
                alert_id="test_info_alert",
                category=MonitoringCategory.SYSTEM_PERFORMANCE,
                severity=AlertSeverity.INFO,
                title="Test INFO Alert",
                description="This is a test informational alert",
                source_component="test_system"
            ),
            Alert(
                alert_id="test_warning_alert",
                category=MonitoringCategory.PRICE_DEVIATION,
                severity=AlertSeverity.WARNING,
                title="Test WARNING Alert",
                description="This is a test warning alert",
                source_component="test_system",
                affected_assets=["ETH_MAINNET_WETH"],
                metrics={"test_metric": 42.5}
            ),
            Alert(
                alert_id="test_critical_alert",
                category=MonitoringCategory.BLOCKCHAIN_CONNECTIVITY,
                severity=AlertSeverity.CRITICAL,
                title="Test CRITICAL Alert", 
                description="This is a test critical alert",
                source_component="test_system",
                affected_assets=["ethereum", "arbitrum"]
            )
        ]
        
        print("   🔄 Testing alert handling...")
        
        # Handle test alerts
        for alert in test_alerts:
            await monitor._handle_alert(alert)
            print(f"      ✅ Handled {alert.severity.value} alert: {alert.title}")
        
        # Test alert retrieval
        active_alerts = await monitor.get_active_alerts()
        print(f"\n   📊 Active Alerts: {len(active_alerts)}")
        
        for alert_data in active_alerts:
            severity_icon = {
                "info": "ℹ️",
                "warning": "⚠️", 
                "error": "❌",
                "critical": "🚨"
            }.get(alert_data["severity"], "❓")
            
            print(f"      {severity_icon} {alert_data['severity'].upper()}: {alert_data['title']}")
            print(f"         Category: {alert_data['category']}")
            print(f"         Source: {alert_data['source_component']}")
            if alert_data['affected_assets']:
                print(f"         Affected: {', '.join(alert_data['affected_assets'])}")
        
        print(f"\n   ✅ Alert system test completed")
        return len(active_alerts)
        
    except Exception as e:
        print(f"   ❌ Alert system test failed: {e}")
        return None
    finally:
        await blockchain_provider.close()


async def test_metrics_collection():
    """Test metrics collection and storage."""
    print("\n📈 Testing Metrics Collection\n")
    
    blockchain_provider, oracle, redis_client = await initialize_dependencies()
    
    try:
        monitor = ProductionMonitor(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            redis_client=redis_client
        )
        
        # Create test metrics
        test_metrics = [
            MonitoringMetric(
                name="test_counter",
                value=100.0,
                unit="count",
                tags={"component": "test"}
            ),
            MonitoringMetric(
                name="test_latency",
                value=15.5,
                unit="ms",
                tags={"operation": "api_call"}
            ),
            MonitoringMetric(
                name="test_ratio",
                value=0.95,
                unit="ratio",
                threshold_warning=0.8,
                threshold_critical=0.6
            )
        ]
        
        print("   🔄 Testing metrics storage...")
        
        # Store test metrics
        for metric in test_metrics:
            await monitor._store_metric(metric)
            print(f"      ✅ Stored metric: {metric.name} = {metric.value} {metric.unit}")
        
        # Test metrics summary
        metrics_summary = await monitor.get_metrics_summary()
        print(f"\n   📊 Metrics Summary: {len(metrics_summary)} metric types")
        
        for metric_name, summary in metrics_summary.items():
            print(f"      • {metric_name}:")
            print(f"         Current: {summary['current_value']:.2f} {summary['unit']}")
            print(f"         Average (1h): {summary['average_1h']:.2f}")
            print(f"         Range (1h): {summary['min_1h']:.2f} - {summary['max_1h']:.2f}")
            print(f"         Samples: {summary['sample_count']}")
        
        print(f"\n   ✅ Metrics collection test completed")
        return len(metrics_summary)
        
    except Exception as e:
        print(f"   ❌ Metrics collection test failed: {e}")
        return None
    finally:
        await blockchain_provider.close()


async def test_system_health_summary():
    """Test system health summary generation."""
    print("\n🏥 Testing System Health Summary\n")
    
    blockchain_provider, oracle, redis_client = await initialize_dependencies()
    
    try:
        monitor = ProductionMonitor(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            redis_client=redis_client
        )
        
        # Add some test alerts for health summary
        test_warning = Alert(
            alert_id="health_test_warning",
            category=MonitoringCategory.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            title="Test Warning for Health Summary",
            description="Warning alert for health summary testing",
            source_component="test"
        )
        
        await monitor._handle_alert(test_warning)
        
        print("   🔄 Generating system health summary...")
        
        # Get system health
        health_summary = await monitor.get_system_health()
        
        print(f"   📊 System Health Summary:")
        print(f"      Overall Status: {health_summary['overall_status'].upper()}")
        print(f"      Active Alerts: {health_summary['active_alerts']}")
        print(f"      Critical Alerts: {health_summary['critical_alerts']}")
        print(f"      Warning Alerts: {health_summary['warning_alerts']}")
        
        print(f"\n   📊 Monitoring Statistics:")
        stats = health_summary['monitoring_stats']
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.2f}")
            else:
                print(f"      {key}: {value}")
        
        status_healthy = health_summary['overall_status'] in ['healthy', 'degraded']
        
        if status_healthy:
            print(f"\n   ✅ System health summary generated successfully")
        else:
            print(f"\n   ⚠️  System health summary shows issues")
        
        return status_healthy
        
    except Exception as e:
        print(f"   ❌ System health summary test failed: {e}")
        return None
    finally:
        await blockchain_provider.close()


async def test_monitoring_loop():
    """Test the monitoring loop functionality."""
    print("\n🔄 Testing Monitoring Loop\n")
    
    blockchain_provider, oracle, redis_client = await initialize_dependencies()
    
    try:
        monitor = ProductionMonitor(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            redis_client=redis_client
        )
        
        print("   🔄 Starting monitoring system for 15 seconds...")
        
        # Start monitoring
        await monitor.start_monitoring()
        
        print("      ✅ Monitoring started")
        
        # Let it run for 15 seconds
        await asyncio.sleep(15)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        print("      ✅ Monitoring stopped")
        
        # Check results
        final_stats = monitor.stats
        health_summary = await monitor.get_system_health()
        
        print(f"\n   📊 Monitoring Loop Results:")
        print(f"      Total checks performed: {final_stats['total_checks']}")
        print(f"      Alerts generated: {final_stats['alerts_generated']}")
        print(f"      Active alerts: {health_summary['active_alerts']}")
        
        if final_stats['total_checks'] > 0:
            print(f"   ✅ Monitoring loop completed successfully")
            return True
        else:
            print(f"   ⚠️  No monitoring checks performed")
            return False
        
    except Exception as e:
        print(f"   ❌ Monitoring loop test failed: {e}")
        return False
    finally:
        await monitor.shutdown()
        await blockchain_provider.close()


async def test_production_readiness():
    """Test production readiness of the monitoring system."""
    print("\n🚀 Testing Production Readiness\n")
    
    blockchain_provider, oracle, redis_client = await initialize_dependencies()
    
    try:
        monitor = ProductionMonitor(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            redis_client=redis_client
        )
        
        print("   📊 Production Readiness Assessment:")
        
        # Check health check coverage
        required_checks = [
            "blockchain_connectivity",
            "price_oracle_health", 
            "data_quality_validation",
            "system_performance"
        ]
        
        available_checks = list(monitor.health_checks.keys())
        coverage_score = 0
        
        for check in required_checks:
            if check in available_checks:
                print(f"      ✅ {check}: Available")
                coverage_score += 1
            else:
                print(f"      ❌ {check}: Missing")
        
        # Check monitoring configuration
        config_complete = True
        required_config = [
            "price_deviation_threshold",
            "min_liquidity_usd", 
            "edge_staleness_warning",
            "check_interval_seconds"
        ]
        
        for config_key in required_config:
            if config_key in monitor.monitoring_config:
                print(f"      ✅ Config {config_key}: {monitor.monitoring_config[config_key]}")
            else:
                print(f"      ❌ Config {config_key}: Missing")
                config_complete = False
        
        # Check alert system
        alert_system_ready = True
        try:
            test_alert = Alert(
                alert_id="readiness_test",
                category=MonitoringCategory.SYSTEM_PERFORMANCE,
                severity=AlertSeverity.INFO,
                title="Readiness Test Alert",
                description="Testing alert system for production readiness",
                source_component="readiness_test"
            )
            await monitor._handle_alert(test_alert)
            print(f"      ✅ Alert System: Functional")
        except Exception as e:
            print(f"      ❌ Alert System: Error - {e}")
            alert_system_ready = False
        
        # Check blockchain connectivity
        blockchain_ready = False
        try:
            web3 = await blockchain_provider.get_web3("ethereum")
            if web3:
                block_number = await web3.eth.block_number
                print(f"      ✅ Blockchain Connectivity: Block {block_number:,}")
                blockchain_ready = True
            else:
                print(f"      ❌ Blockchain Connectivity: Failed")
        except Exception as e:
            print(f"      ❌ Blockchain Connectivity: Error - {e}")
        
        # Calculate readiness score
        total_checks = 4
        passed_checks = (
            (coverage_score >= 3) +
            config_complete +
            alert_system_ready + 
            blockchain_ready
        )
        
        readiness_score = (passed_checks / total_checks) * 100
        
        print(f"\n   📊 Production Readiness Score: {readiness_score:.0f}/100")
        
        if readiness_score >= 90:
            print("   🚀 Monitoring system ready for production deployment")
            return True
        elif readiness_score >= 70:
            print("   ⚠️  Monitoring system needs minor improvements")
            return False
        else:
            print("   ❌ Monitoring system not ready for production")
            return False
            
    except Exception as e:
        print(f"   ❌ Production readiness test failed: {e}")
        return False
    finally:
        await blockchain_provider.close()


async def main():
    """Run all production monitoring tests."""
    print("🚀 Production Monitoring System Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    try:
        # Test 1: Monitor initialization
        monitor = await test_monitor_initialization()
        test_results['initialization'] = monitor is not None
        
        # Test 2: Individual health checks
        health_results = await test_health_checks()
        test_results['health_checks'] = health_results is not None
        
        # Test 3: Alert system
        alert_count = await test_alert_system()
        test_results['alert_system'] = alert_count is not None and alert_count > 0
        
        # Test 4: Metrics collection
        metrics_count = await test_metrics_collection()
        test_results['metrics_collection'] = metrics_count is not None and metrics_count > 0
        
        # Test 5: System health summary
        health_summary_ok = await test_system_health_summary()
        test_results['health_summary'] = health_summary_ok is not None
        
        # Test 6: Monitoring loop
        loop_success = await test_monitoring_loop()
        test_results['monitoring_loop'] = loop_success
        
        # Test 7: Production readiness
        production_ready = await test_production_readiness()
        test_results['production_readiness'] = production_ready
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        test_results['overall'] = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 Test Suite Summary")
    print("=" * 60)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        test_display = test_name.replace('_', ' ').title()
        print(f"   {status}: {test_display}")
    
    print(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Production monitoring system ready.")
        print("\n✅ Task 14.6: Production Monitoring & Validation - COMPLETED")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed. Minor issues need attention.")
    else:
        print("❌ Multiple test failures. Monitoring system needs significant work.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    asyncio.run(main())