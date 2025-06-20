"""Test basic project setup."""
import sys
from pathlib import Path


def test_python_version():
    """Test that Python version is 3.11+."""
    assert sys.version_info >= (3, 11), f"Python version is {sys.version_info}, expected >= 3.11"


def test_project_structure():
    """Test that basic project structure exists."""
    project_root = Path(__file__).parent.parent.parent
    src_dir = project_root / "src" / "yield_arbitrage"
    
    # Check main source directory exists
    assert src_dir.exists(), "Source directory should exist"
    
    # Check core modules exist
    expected_modules = [
        "graph_engine",
        "blockchain_connector", 
        "protocols",
        "data_collector",
        "pathfinding",
        "execution",
        "cache",
        "telegram_interface",
        "config",
        "db",
        "ml_models",
        "risk"
    ]
    
    for module in expected_modules:
        module_dir = src_dir / module
        assert module_dir.exists(), f"Module {module} should exist"
        assert (module_dir / "__init__.py").exists(), f"Module {module} should have __init__.py"


def test_imports():
    """Test that we can import basic modules."""
    # Add src to path for imports
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / "src"))
    
    try:
        import yield_arbitrage
        assert yield_arbitrage is not None
    except ImportError as e:
        assert False, f"Could not import yield_arbitrage: {e}"