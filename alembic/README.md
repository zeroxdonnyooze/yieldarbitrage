# Alembic Database Migrations

This directory contains Alembic configuration and migration scripts for the Yield Arbitrage database schema.

## Configuration

- **alembic.ini**: Main Alembic configuration file
- **env.py**: Environment configuration for async SQLAlchemy
- **script.py.mako**: Template for migration scripts
- **versions/**: Directory containing migration files

## Features

- ✅ Async SQLAlchemy support
- ✅ Auto-imports database models
- ✅ Environment-based database URL configuration
- ✅ Date/time prefixed migration files
- ✅ Comprehensive type and default comparison
- ✅ Batch operations for better compatibility

## Usage

### Basic Commands

```bash
# Check current revision
alembic current

# Show migration history
alembic history

# Show head revisions
alembic heads

# Check for pending migrations
alembic check
```

### Creating Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "description_of_changes"

# Create empty migration
alembic revision -m "description_of_changes"
```

### Applying Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade to specific revision
alembic upgrade revision_id

# Downgrade one revision
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade revision_id
```

### Database URL Configuration

The database URL is automatically configured from your environment settings:

1. First checks `settings.database_url` from your application config
2. Falls back to the URL in `alembic.ini` if not set

### Migration File Naming

Migration files are automatically named with date/time prefix:
```
YYYY_MM_DD_HHMM-revision_id-description.py
```

Example: `2025_06_20_2212-a1b2c3d4e5f6-add_user_table.py`

## Model Integration

The Alembic configuration automatically imports all database models:

- `ExecutedPath` - Arbitrage execution tracking
- `TokenMetadata` - Token information and pricing

When you add new models to `src/yield_arbitrage/database/models.py`, they will be automatically detected for autogeneration.

## Best Practices

1. **Always review auto-generated migrations** before applying them
2. **Test migrations on a copy of production data** before deploying
3. **Use descriptive migration messages** that explain the business purpose
4. **Keep migrations atomic** - each migration should be a single logical change
5. **Don't edit existing migrations** once they've been committed to version control

## Troubleshooting

### Common Issues

**ImportError during migration**
- Ensure `PYTHONPATH` includes the `src` directory
- Check that all model imports in `env.py` are correct

**Database connection errors**
- Verify database URL in settings
- Ensure database server is running
- Check network connectivity

**Migration conflicts**
- Use `alembic merge` to resolve conflicts
- Check `alembic branches` for multiple heads

### Testing Migrations

Run the test script to verify configuration:
```bash
python scripts/test_alembic_runtime.py
```

Run unit tests:
```bash
pytest tests/unit/test_alembic_configuration.py -v
```

## Advanced Usage

### Custom Migration Templates

The `script.py.mako` template can be customized to include standard headers, imports, or other boilerplate code.

### Environment Variables

Set these environment variables to override defaults:
- `DATABASE_URL`: Complete database connection string
- `ALEMBIC_CONFIG`: Path to alternative alembic.ini file

### Offline Migrations

Generate SQL scripts without database connection:
```bash
alembic upgrade head --sql > migration.sql
```

## Security Notes

- Database credentials should never be committed to version control
- Use environment variables or secure configuration management
- Migration files may contain sensitive schema information
- Consider using database-specific migration tools for production environments