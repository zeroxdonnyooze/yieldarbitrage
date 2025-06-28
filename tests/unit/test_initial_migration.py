"""Unit tests for the initial database migration."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestInitialMigration:
    """Test the initial database migration file."""
    
    @pytest.fixture
    def migration_file(self):
        """Get the path to the initial migration file."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        # Find the migration file by name pattern
        migration_files = list(versions_dir.glob("*_create_initial_tables.py"))
        assert len(migration_files) == 1, "Should have exactly one initial migration file"
        
        return migration_files[0]
    
    def test_migration_file_exists(self, migration_file):
        """Test that migration file exists and is readable."""
        assert migration_file.exists(), "Migration file should exist"
        assert migration_file.is_file(), "Migration file should be a file"
        
        # Test that file is readable
        content = migration_file.read_text()
        assert len(content) > 0, "Migration file should not be empty"
    
    def test_migration_file_syntax(self, migration_file):
        """Test that migration file has valid Python syntax."""
        content = migration_file.read_text()
        
        # Should compile without syntax errors
        compile(content, str(migration_file), 'exec')
    
    def test_migration_has_required_functions(self, migration_file):
        """Test that migration has upgrade and downgrade functions."""
        content = migration_file.read_text()
        
        assert "def upgrade() -> None:" in content, "Migration should have upgrade function"
        assert "def downgrade() -> None:" in content, "Migration should have downgrade function"
    
    def test_migration_has_correct_metadata(self, migration_file):
        """Test that migration has correct revision metadata."""
        content = migration_file.read_text()
        
        # Check revision identifiers
        assert "revision = " in content, "Migration should have revision ID"
        assert "down_revision = None" in content, "Initial migration should have None down_revision"
        assert "branch_labels = None" in content, "Should have branch_labels"
        assert "depends_on = None" in content, "Should have depends_on"
    
    def test_migration_creates_executed_paths_table(self, migration_file):
        """Test that migration creates executed_paths table."""
        content = migration_file.read_text()
        
        # Check for table creation
        assert "op.create_table('executed_paths'" in content
        
        # Check for key columns
        expected_columns = [
            "id",
            "path_hash", 
            "transaction_hash",
            "profit_usd",
            "profit_percentage",
            "ml_confidence_score",
            "discovered_at",
            "executed_at"
        ]
        
        for column in expected_columns:
            assert f"'{column}'" in content or f'"{column}"' in content, f"Should have {column} column"
    
    def test_migration_creates_token_metadata_table(self, migration_file):
        """Test that migration creates token_metadata table."""
        content = migration_file.read_text()
        
        # Check for table creation
        assert "op.create_table('token_metadata'" in content
        
        # Check for key columns
        expected_columns = [
            "id",
            "asset_id",
            "symbol",
            "name",
            "price_usd",
            "price_eth",
            "security_score",
            "is_verified"
        ]
        
        for column in expected_columns:
            assert f"'{column}'" in content or f'"{column}"' in content, f"Should have {column} column"
    
    def test_migration_creates_indexes(self, migration_file):
        """Test that migration creates necessary indexes."""
        content = migration_file.read_text()
        
        # Check for index creation
        expected_indexes = [
            "ix_executed_paths_path_hash",
            "ix_executed_paths_profit_usd", 
            "ix_token_metadata_asset_id",
            "ix_token_metadata_symbol"
        ]
        
        for index in expected_indexes:
            assert index in content, f"Should create index {index}"
    
    def test_migration_creates_constraints(self, migration_file):
        """Test that migration creates necessary constraints."""
        content = migration_file.read_text()
        
        # Check for unique constraints
        assert "uq_executed_paths_path_hash_transaction_hash" in content
        assert "uq_token_metadata_asset_id" in content
        
        # Check for primary keys
        assert "PrimaryKeyConstraint('id')" in content or 'PrimaryKeyConstraint("id")' in content
    
    def test_migration_uses_correct_column_types(self, migration_file):
        """Test that migration uses correct PostgreSQL column types."""
        content = migration_file.read_text()
        
        # Check for PostgreSQL-specific types
        assert "postgresql.UUID" in content, "Should use PostgreSQL UUID type"
        assert "postgresql.ARRAY" in content, "Should use PostgreSQL ARRAY type"
        assert "postgresql.JSONB" in content, "Should use PostgreSQL JSONB type"
        
        # Check for numeric types with precision
        assert "sa.Numeric(precision=" in content, "Should use precise numeric types"
        assert "sa.DateTime(timezone=True)" in content, "Should use timezone-aware datetime"
    
    def test_migration_has_proper_downgrade(self, migration_file):
        """Test that migration has proper downgrade implementation."""
        content = migration_file.read_text()
        
        # Check that downgrade drops tables
        assert "op.drop_table('token_metadata')" in content
        assert "op.drop_table('executed_paths')" in content
        
        # Check that downgrade drops indexes
        assert "op.drop_index" in content
    
    def test_migration_sql_generation(self):
        """Test that migration can generate SQL without errors."""
        project_root = Path(__file__).parent.parent.parent
        
        try:
            result = subprocess.run(
                ['python', '-m', 'alembic', 'upgrade', 'head', '--sql'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0, f"SQL generation failed: {result.stderr}"
            
            sql_output = result.stdout.lower()
            
            # Check that SQL contains expected DDL
            assert "create table executed_paths" in sql_output
            assert "create table token_metadata" in sql_output
            assert "create index" in sql_output
            assert "insert into alembic_version" in sql_output
            
        except subprocess.TimeoutExpired:
            pytest.fail("SQL generation timed out")
    
    def test_migration_history_display(self):
        """Test that migration appears correctly in history."""
        project_root = Path(__file__).parent.parent.parent
        
        try:
            # Use --verbose to get detailed history
            result = subprocess.run(
                ['python', '-m', 'alembic', 'history', '--verbose'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            assert result.returncode == 0, f"History command failed: {result.stderr}"
            
            # Check that our migration appears
            assert "create_initial_tables" in result.stdout
            # With verbose, we should see the description
            assert "executed_paths" in result.stdout.lower() or "arbitrage" in result.stdout.lower()
            
        except subprocess.TimeoutExpired:
            pytest.fail("History command timed out")
    
    def test_migration_check_command(self):
        """Test alembic check command behavior."""
        project_root = Path(__file__).parent.parent.parent
        
        try:
            result = subprocess.run(
                ['python', '-m', 'alembic', 'check'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            # Check command may fail if database doesn't exist, but it should not crash
            # Return code 255 is expected when database is not up to date
            assert result.returncode in [0, 255], f"Check command crashed: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Check command timed out")