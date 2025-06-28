"""Unit tests for Alembic configuration and setup."""
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestAlembicConfiguration:
    """Test Alembic configuration and setup."""
    
    def test_alembic_ini_exists(self):
        """Test that alembic.ini file exists and is properly configured."""
        project_root = Path(__file__).parent.parent.parent
        alembic_ini_path = project_root / "alembic.ini"
        
        assert alembic_ini_path.exists(), "alembic.ini file not found"
        
        # Read and check basic configuration
        content = alembic_ini_path.read_text()
        assert "[alembic]" in content
        assert "script_location = alembic" in content
        assert "sqlalchemy.url" in content
    
    def test_alembic_env_py_exists(self):
        """Test that alembic/env.py exists and has proper imports."""
        project_root = Path(__file__).parent.parent.parent
        env_py_path = project_root / "alembic" / "env.py"
        
        assert env_py_path.exists(), "alembic/env.py file not found"
        
        # Check that env.py has required imports
        content = env_py_path.read_text()
        assert "from yield_arbitrage.database.connection import Base" in content
        assert "from yield_arbitrage.database.models import ExecutedPath, TokenMetadata" in content
        assert "target_metadata = Base.metadata" in content
    
    def test_alembic_script_template_exists(self):
        """Test that script template exists."""
        project_root = Path(__file__).parent.parent.parent
        template_path = project_root / "alembic" / "script.py.mako"
        
        assert template_path.exists(), "alembic/script.py.mako file not found"
        
        # Check template content
        content = template_path.read_text()
        assert "def upgrade() -> None:" in content
        assert "def downgrade() -> None:" in content
    
    def test_metadata_contains_models(self):
        """Test that metadata contains our defined models."""
        from yield_arbitrage.database.connection import Base
        from yield_arbitrage.database.models import ExecutedPath, TokenMetadata
        
        # Check that tables are in metadata
        table_names = list(Base.metadata.tables.keys())
        assert 'executed_paths' in table_names
        assert 'token_metadata' in table_names
        
        # Check table structure
        executed_paths_table = Base.metadata.tables['executed_paths']
        token_metadata_table = Base.metadata.tables['token_metadata']
        
        # Check key columns exist
        assert 'id' in executed_paths_table.columns
        assert 'path_hash' in executed_paths_table.columns
        assert 'profit_usd' in executed_paths_table.columns
        
        assert 'id' in token_metadata_table.columns
        assert 'asset_id' in token_metadata_table.columns
        assert 'symbol' in token_metadata_table.columns
    
    def test_alembic_help_command(self):
        """Test that alembic help command works."""
        project_root = Path(__file__).parent.parent.parent
        
        try:
            result = subprocess.run(
                ['python', '-m', 'alembic', '--help'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            assert result.returncode == 0, f"Alembic help failed: {result.stderr}"
            assert "usage: alembic" in result.stdout
            assert "revision" in result.stdout
            assert "upgrade" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("Alembic help command timed out")
    
    def test_alembic_current_command(self):
        """Test that alembic current command works."""
        project_root = Path(__file__).parent.parent.parent
        
        try:
            result = subprocess.run(
                ['python', '-m', 'alembic', 'current'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            # Should return 0 even if no migrations have been run
            assert result.returncode == 0, f"Alembic current failed: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Alembic current command timed out")
    
    def test_alembic_history_command(self):
        """Test that alembic history command works."""
        project_root = Path(__file__).parent.parent.parent
        
        try:
            result = subprocess.run(
                ['python', '-m', 'alembic', 'history'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            assert result.returncode == 0, f"Alembic history failed: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Alembic history command timed out")
    
    def test_alembic_heads_command(self):
        """Test that alembic heads command works."""
        project_root = Path(__file__).parent.parent.parent
        
        try:
            result = subprocess.run(
                ['python', '-m', 'alembic', 'heads'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            assert result.returncode == 0, f"Alembic heads failed: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Alembic heads command timed out")
    
    def test_file_template_configuration(self):
        """Test that file template is properly configured."""
        project_root = Path(__file__).parent.parent.parent
        alembic_ini_path = project_root / "alembic.ini"
        
        content = alembic_ini_path.read_text()
        
        # Check that file template includes date/time (using %% escaping in ini files)
        assert "file_template" in content
        assert "%%(year)d_%%(month).2d_%%(day).2d" in content
    
    def test_database_url_configuration(self):
        """Test that database URL can be overridden from settings."""
        from yield_arbitrage.config.settings import settings
        
        # Test that settings has database_url attribute
        assert hasattr(settings, 'database_url')
        
        # The URL might be None in test environment, which is fine
        # The important thing is that the attribute exists
    
    def test_alembic_imports_work(self):
        """Test that all Alembic-related imports work correctly."""
        try:
            from alembic import command
            from alembic.config import Config
            assert command is not None
            assert Config is not None
            
        except ImportError as e:
            pytest.fail(f"Failed to import Alembic components: {e}")
    
    def test_versions_directory_exists(self):
        """Test that versions directory exists."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        assert versions_dir.exists(), "alembic/versions directory not found"
        assert versions_dir.is_dir(), "alembic/versions is not a directory"