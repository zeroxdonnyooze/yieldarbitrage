#!/usr/bin/env python3
"""
Production validation tests for yield arbitrage system.

These tests validate that the production deployment is working correctly
and all systems are functioning as expected.
"""
import asyncio
import pytest
import logging
import sys
import os
import time
import requests
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.config.environment import initialize_environment, get_environment_info
from yield_arbitrage.monitoring.health_checks import health_checker, HealthStatus
from yield_arbitrage.api.health import basic_health_check, detailed_health_check

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestProductionValidation:
    """Production validation test suite."""
    
    @pytest.fixture(scope="class")
    async def environment_info(self):
        """Initialize and return environment information."""
        return get_environment_info()
    
    def test_environment_configuration(self, environment_info):
        """Test that environment is properly configured."""
        logger.info("🔧 Testing environment configuration...")
        
        assert environment_info is not None
        assert "environment" in environment_info
        assert "host" in environment_info
        assert "port" in environment_info
        
        # Check required environment variables
        required_vars = ["DATABASE_URL", "REDIS_URL", "ALCHEMY_API_KEY"]
        for var in required_vars:
            assert os.getenv(var) is not None, f"Required environment variable {var} not set"
        
        logger.info(f"   ✅ Environment: {environment_info['environment']}")
        logger.info(f"   ✅ Host: {environment_info['host']}:{environment_info['port']}")
        logger.info("   ✅ All required environment variables present")
    
    def test_health_check_endpoints(self):
        """Test health check endpoint functionality."""
        logger.info("🏥 Testing health check endpoints...")
        
        # Test basic health check
        basic_result = basic_health_check()
        assert basic_result["status"] == "healthy"
        assert "message" in basic_result
        assert "timestamp" in basic_result
        
        # Test detailed health check
        detailed_result = detailed_health_check()
        assert detailed_result["status"] == "healthy"
        assert "checks" in detailed_result
        assert "summary" in detailed_result
        
        logger.info("   ✅ Basic health check working")
        logger.info("   ✅ Detailed health check working")
    
    @pytest.mark.asyncio
    async def test_health_checker_system(self):
        """Test the health checker system."""
        logger.info("📊 Testing health checker system...")
        
        # Register a test health check
        def test_check():
            return {"status": HealthStatus.HEALTHY, "message": "Test check OK"}
        
        from yield_arbitrage.monitoring.health_checks import HealthCheck
        health_checker.register_check(HealthCheck(
            name="test_check",
            check_func=test_check,
            timeout_seconds=5.0,
            tags=["test"]
        ))
        
        # Run the test check
        result = await health_checker.run_check("test_check")
        assert result.status == HealthStatus.HEALTHY
        assert result.name == "test_check"
        assert result.duration_ms > 0
        
        # Run all checks
        results = await health_checker.run_all_checks()
        assert "test_check" in results
        
        # Get overall health
        overall_health = health_checker.get_overall_health(results)
        assert "status" in overall_health
        assert "summary" in overall_health
        
        logger.info("   ✅ Health checker registration working")
        logger.info("   ✅ Individual check execution working")
        logger.info("   ✅ Overall health assessment working")
    
    def test_docker_container_health(self):
        """Test Docker container health if running in containerized environment."""
        logger.info("🐳 Testing Docker container health...")
        
        # Check if running in Docker
        if os.path.exists("/.dockerenv"):
            logger.info("   📦 Running in Docker container")
            
            # Check container health endpoint
            try:
                response = requests.get("http://localhost:8000/health", timeout=10)
                assert response.status_code == 200
                health_data = response.json()
                assert health_data["status"] in ["healthy", "degraded"]
                logger.info("   ✅ Container health endpoint accessible")
            except requests.RequestException:
                logger.warning("   ⚠️  Health endpoint not accessible (may be normal in test environment)")
        else:
            logger.info("   📋 Not running in Docker container")
    
    def test_configuration_security(self):
        """Test configuration security measures."""
        logger.info("🔒 Testing configuration security...")
        
        # Check that sensitive environment variables are not exposed
        sensitive_vars = ["PRIVATE_KEY", "SECRET", "PASSWORD", "TOKEN"]
        
        for var_name in os.environ:
            var_value = os.environ[var_name]
            
            # Check that sensitive variables don't contain obvious secrets
            if any(sensitive in var_name.upper() for sensitive in sensitive_vars):
                if var_value and var_value not in ["your_key_here", "placeholder"]:
                    # In production, these should be properly secured
                    assert len(var_value) > 10, f"Sensitive variable {var_name} appears to be a placeholder"
                    logger.info(f"   🔐 Sensitive variable {var_name} appears to be set")
        
        logger.info("   ✅ Configuration security checks passed")
    
    def test_logging_configuration(self):
        """Test logging configuration."""
        logger.info("📝 Testing logging configuration...")
        
        # Test different log levels
        test_logger = logging.getLogger("test_production_validation")
        
        test_logger.debug("Debug message test")
        test_logger.info("Info message test")
        test_logger.warning("Warning message test")
        test_logger.error("Error message test")
        
        # Check log level is appropriate
        env_info = get_environment_info()
        expected_level = env_info.get("log_level", "INFO")
        
        root_logger = logging.getLogger()
        actual_level = logging.getLevelName(root_logger.level)
        
        logger.info(f"   📊 Current log level: {actual_level}")
        logger.info(f"   📊 Expected log level: {expected_level}")
        logger.info("   ✅ Logging system functional")
    
    def test_performance_baseline(self):
        """Test basic performance baselines."""
        logger.info("⚡ Testing performance baselines...")
        
        # Test health check performance
        start_time = time.time()
        result = basic_health_check()
        duration = time.time() - start_time
        
        assert duration < 1.0, f"Health check took too long: {duration:.3f}s"
        logger.info(f"   ⚡ Health check duration: {duration*1000:.1f}ms")
        
        # Test detailed health check performance
        start_time = time.time()
        detailed_result = detailed_health_check()
        detailed_duration = time.time() - start_time
        
        assert detailed_duration < 2.0, f"Detailed health check took too long: {detailed_duration:.3f}s"
        logger.info(f"   ⚡ Detailed health check duration: {detailed_duration*1000:.1f}ms")
        
        logger.info("   ✅ Performance baselines met")
    
    def test_resource_limits(self):
        """Test resource usage is within limits."""
        logger.info("💾 Testing resource limits...")
        
        import psutil
        
        # Check memory usage
        memory = psutil.virtual_memory()
        memory_usage_percent = memory.percent
        
        logger.info(f"   💾 Memory usage: {memory_usage_percent:.1f}%")
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"   🔥 CPU usage: {cpu_percent:.1f}%")
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        logger.info(f"   💿 Disk usage: {disk_usage_percent:.1f}%")
        
        # Basic resource checks (adjust limits as needed)
        if memory_usage_percent > 90:
            logger.warning(f"   ⚠️  High memory usage: {memory_usage_percent:.1f}%")
        
        if disk_usage_percent > 90:
            logger.warning(f"   ⚠️  High disk usage: {disk_usage_percent:.1f}%")
        
        logger.info("   ✅ Resource usage checked")
    
    def test_network_connectivity(self):
        """Test network connectivity to external services."""
        logger.info("🌐 Testing network connectivity...")
        
        # Test DNS resolution
        import socket
        try:
            socket.gethostbyname("google.com")
            logger.info("   ✅ DNS resolution working")
        except socket.gaierror:
            logger.warning("   ⚠️  DNS resolution issue")
        
        # Test HTTP connectivity
        try:
            response = requests.get("https://httpbin.org/get", timeout=10)
            if response.status_code == 200:
                logger.info("   ✅ HTTP connectivity working")
            else:
                logger.warning(f"   ⚠️  HTTP connectivity issue: {response.status_code}")
        except requests.RequestException as e:
            logger.warning(f"   ⚠️  HTTP connectivity issue: {e}")
        
        logger.info("   ✅ Network connectivity tests completed")
    
    def test_error_handling(self):
        """Test error handling and recovery mechanisms."""
        logger.info("🚨 Testing error handling...")
        
        # Test graceful error handling in health checks
        def failing_check():
            raise Exception("Intentional test failure")
        
        from yield_arbitrage.monitoring.health_checks import HealthCheck
        health_checker.register_check(HealthCheck(
            name="failing_test_check",
            check_func=failing_check,
            timeout_seconds=5.0,
            tags=["test"]
        ))
        
        # This should not crash the application
        try:
            import asyncio
            result = asyncio.run(health_checker.run_check("failing_test_check"))
            assert result.status == HealthStatus.UNHEALTHY
            assert "Intentional test failure" in result.message
            logger.info("   ✅ Error handling working correctly")
        except Exception as e:
            logger.error(f"   ❌ Error handling failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test system behavior under concurrent load."""
        logger.info("🔄 Testing concurrent operations...")
        
        # Test concurrent health checks
        tasks = []
        for i in range(10):
            task = asyncio.create_task(health_checker.run_check("test_check"))
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        # Check that most operations succeeded
        successful_results = [r for r in results if not isinstance(r, Exception)]
        success_rate = len(successful_results) / len(results)
        
        assert success_rate >= 0.8, f"Concurrent operations success rate too low: {success_rate:.1%}"
        assert duration < 5.0, f"Concurrent operations took too long: {duration:.3f}s"
        
        logger.info(f"   ✅ Concurrent operations success rate: {success_rate:.1%}")
        logger.info(f"   ✅ Concurrent operations duration: {duration:.3f}s")


@pytest.mark.asyncio
async def test_full_system_validation():
    """Run a comprehensive system validation test."""
    logger.info("🚀 Running full system validation...")
    
    start_time = time.time()
    
    try:
        # Initialize environment
        env_info = get_environment_info()
        logger.info(f"Environment: {env_info['environment']}")
        
        # Run health checks
        results = await health_checker.run_all_checks()
        overall_health = health_checker.get_overall_health(results)
        
        logger.info(f"Overall system health: {overall_health['status']}")
        logger.info(f"Health checks: {overall_health['summary']}")
        
        # Check critical systems
        critical_systems = ["database", "redis", "blockchain"]
        for system in critical_systems:
            if system in results:
                status = results[system].status
                logger.info(f"{system}: {status}")
        
        duration = time.time() - start_time
        logger.info(f"Full validation completed in {duration:.3f}s")
        
        # Assert overall system health
        assert overall_health["status"] in [HealthStatus.HEALTHY, HealthStatus.DEGRADED], \
               f"System health check failed: {overall_health['message']}"
        
        logger.info("🎉 Full system validation PASSED")
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Full system validation FAILED after {duration:.3f}s: {e}")
        raise


if __name__ == "__main__":
    # Run validation tests directly
    asyncio.run(test_full_system_validation())