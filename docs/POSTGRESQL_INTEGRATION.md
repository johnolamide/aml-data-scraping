# PostgreSQL Integration Guide

This guide covers how to load OpenSanctions data into PostgreSQL using the `csv_to_postgresql.py` loader.

## Key Features

### Dynamic Table Naming

The loader automatically derives PostgreSQL table names from CSV filenames:

| CSV Filename                 | Generated Table Name         |
| ---------------------------- | ---------------------------- |
| `opensanctions_entities.csv` | `opensanctions_entities`     |
| `us-ofac-sanctions.csv`      | `us_ofac_sanctions`          |
| `EU_Sanctions_List.csv`      | `eu_sanctions_list`          |
| `sample@data#special.csv`    | `sample_data_special`        |
| `123_starts_with_number.csv` | `tbl_123_starts_with_number` |

#### Naming Rules

1. **Lowercase conversion**: All uppercase letters converted to lowercase
2. **Special character replacement**: Non-alphanumeric characters (except underscore and dollar sign) replaced with underscores
3. **Valid start character**: Names starting with numbers get `tbl_` prefix
4. **Length limit**: Names truncated to PostgreSQL's 63-character identifier limit
5. **Edge case handling**: Invalid or empty names fallback to `opensanctions_entities`

### Loading Modes

#### Full Load

-   Replaces all data in the table
-   Optional duplicate detection and skipping
-   Suitable for initial data loads or complete refreshes

#### Incremental Load

-   Only adds new entities not already in database
-   Compares `entity_id` values to avoid duplicates
-   Ideal for regular updates with new data
-   Maintains data integrity and performance

### Schema Management

The loader automatically creates tables with:

-   **Primary Key**: Auto-incrementing `id` column (SERIAL)
-   **Data Columns**: 23 standardized columns matching CSV output
-   **Metadata**: `created_at` and `updated_at` timestamps
-   **Performance Indexes**: Optimized indexes on frequently queried columns

## Setup Instructions

### Prerequisites

1. PostgreSQL server (version 9.5+ recommended)
2. Python packages: `psycopg2-binary`
3. Database with appropriate permissions

### Database Configuration

Create a configuration file `db_config.json`:

```json
{
	"host": "localhost",
	"database": "aml_database",
	"user": "postgres",
	"password": "your_password",
	"port": 5432
}
```

## Usage Examples

### Basic Loading

```bash
# Load CSV with automatic table naming
python csv_to_postgresql.py --csv-file us_ofac_sanctions.csv --db-config db_config.json

# This creates a table named 'us_ofac_sanctions'
```

### First-Time Setup

```bash
# Drop existing table and create fresh
python csv_to_postgresql.py \
  --csv-file sanctions_data.csv \
  --db-config db_config.json \
  --drop-table
```

### Regular Updates

```bash
# Incremental loading (recommended for updates)
python csv_to_postgresql.py \
  --csv-file updated_sanctions.csv \
  --db-config db_config.json \
  --incremental
```

### Advanced Options

```bash
# Custom batch size for large datasets
python csv_to_postgresql.py \
  --csv-file large_dataset.csv \
  --db-config db_config.json \
  --batch-size 5000

# Allow duplicate entity_ids
python csv_to_postgresql.py \
  --csv-file data.csv \
  --db-config db_config.json \
  --allow-duplicates

# Verbose logging for troubleshooting
python csv_to_postgresql.py \
  --csv-file data.csv \
  --db-config db_config.json \
  --verbose
```

### Command Line Parameters

Alternative to config file:

```bash
python csv_to_postgresql.py \
  --csv-file data.csv \
  --host localhost \
  --database aml_db \
  --user postgres \
  --password secret \
  --incremental
```

## Table Schema

### Generated Table Structure

```sql
CREATE TABLE us_ofac_sanctions (
    id SERIAL PRIMARY KEY,
    name TEXT,
    source TEXT,
    dataset TEXT,
    dataset_title TEXT,
    entity_type TEXT,
    entity_id TEXT,
    alias TEXT,
    country TEXT,
    birth_date TEXT,
    nationality TEXT,
    position TEXT,
    registration_number TEXT,
    topics TEXT,
    sanction_program TEXT,
    description TEXT,
    last_updated TEXT,
    source_url TEXT,
    legal_form TEXT,
    incorporation_date TEXT,
    address TEXT,
    phone TEXT,
    email TEXT,
    website TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes

The loader creates indexes for optimal query performance:

-   `idx_name` on `name` column
-   `idx_entity_id` on `entity_id` column
-   `idx_dataset` on `dataset` column
-   `idx_entity_type` on `entity_type` column
-   `idx_country` on `country` column
-   `idx_source` on `source` column
-   `idx_created_at` on `created_at` column

## Querying Data

### Sample Queries

```sql
-- Count entities by country
SELECT country, COUNT(*) as entity_count
FROM us_ofac_sanctions
WHERE country IS NOT NULL
GROUP BY country
ORDER BY entity_count DESC;

-- Find recent additions
SELECT name, entity_type, country, created_at
FROM us_ofac_sanctions
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY created_at DESC;

-- Search by name pattern
SELECT name, entity_type, sanction_program
FROM us_ofac_sanctions
WHERE name ILIKE '%company%'
LIMIT 10;

-- Entities by sanctions program
SELECT sanction_program, COUNT(*) as count
FROM us_ofac_sanctions
WHERE sanction_program IS NOT NULL
GROUP BY sanction_program
ORDER BY count DESC;
```

## Performance Considerations

### Large Datasets

-   Use `--batch-size` to tune memory usage (default: 1000)
-   Enable `--incremental` for regular updates
-   Monitor database connection limits during bulk loads

### Incremental Updates

-   Incremental mode compares all existing `entity_id` values
-   For very large databases, consider partitioning strategies
-   Regular `VACUUM ANALYZE` recommended after bulk updates

### Monitoring

```bash
# Check loading statistics
python csv_to_postgresql.py \
  --csv-file data.csv \
  --db-config config.json \
  --verbose
```

## Troubleshooting

### Common Issues

1. **Connection Errors**: Verify database credentials and network connectivity
2. **Permission Errors**: Ensure user has CREATE TABLE and INSERT privileges
3. **Memory Issues**: Reduce `--batch-size` for large datasets
4. **Character Encoding**: CSV files should be UTF-8 encoded

### Error Recovery

-   Use `--drop-table` to restart failed loads
-   Check PostgreSQL logs for detailed error messages
-   Validate CSV format matches expected schema

## Integration Examples

### Python Code

```python
from csv_to_postgresql import CSVToPostgreSQLLoader

# Initialize with config
db_config = {
    "host": "localhost",
    "database": "aml_db",
    "user": "postgres",
    "password": "secret"
}

loader = CSVToPostgreSQLLoader(db_config)

try:
    loader.connect()

    # Load CSV - table name automatically derived
    stats = loader.load_csv_file_incremental("sanctions_data.csv")
    print(f"Loaded {stats['inserted']} new records")

    # Get table statistics
    table_stats = loader.get_table_stats()
    print(f"Total records: {table_stats['total_records']}")

finally:
    loader.disconnect()
```

### Workflow Integration

```bash
#!/bin/bash
# Daily sanctions update workflow

# Extract latest data
python opensanctions_extractor.py --dataset us-ofac-sdn --output daily_update.csv

# Load incrementally into database
python csv_to_postgresql.py \
  --csv-file daily_update.csv \
  --db-config prod_config.json \
  --incremental \
  --verbose

# Cleanup
rm daily_update.csv
```

## Best Practices

1. **Use incremental loading** for regular updates
2. **Monitor database growth** and plan for scaling
3. **Regular backups** before major data loads
4. **Test with sample data** before production loads
5. **Use appropriate batch sizes** based on available memory
6. **Validate data quality** before and after loading
