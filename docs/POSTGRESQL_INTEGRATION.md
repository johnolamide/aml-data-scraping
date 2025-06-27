# PostgreSQL Integration Guide

This guide covers the PostgreSQL database integration features added to the OpenSanctions Data Extractor project.

## Overview

The `csv_to_postgresql.py` script provides seamless integration between the CSV output from the OpenSanctions extractor and PostgreSQL databases. This allows for:

-   Structured storage of entity data in a relational database
-   Efficient querying and filtering of large datasets
-   Integration with existing data pipelines and applications
-   Proper indexing for performance optimization

## Files Added

-   **`csv_to_postgresql.py`**: Main PostgreSQL loader script
-   **`demo_postgresql.py`**: Demonstration script showing complete workflow
-   **`db_config.json.example`**: Sample database configuration file
-   **`requirements.txt`**: Updated with PostgreSQL dependencies

## Database Schema

### Table: `sterlingai-aml-ctf-pep-tbl`

| Column              | Type      | Description                               |
| ------------------- | --------- | ----------------------------------------- |
| id                  | SERIAL    | Auto-generated primary key                |
| name                | TEXT      | Primary entity name                       |
| source              | TEXT      | Data source (always "opensanctions")      |
| dataset             | TEXT      | Source dataset name                       |
| dataset_title       | TEXT      | Human-readable dataset title              |
| entity_type         | TEXT      | Type of entity (Person, Organization)     |
| entity_id           | TEXT      | Unique entity identifier                  |
| alias               | TEXT      | Alternative names (semicolon-separated)   |
| country             | TEXT      | Associated countries                      |
| birth_date          | TEXT      | Birth date (for persons)                  |
| nationality         | TEXT      | Nationality information                   |
| position            | TEXT      | Job title or position                     |
| registration_number | TEXT      | Company/organization registration numbers |
| topics              | TEXT      | Topics/categories                         |
| sanction_program    | TEXT      | Specific sanction program names           |
| description         | TEXT      | Entity description                        |
| last_updated        | TEXT      | When the data was last updated            |
| source_url          | TEXT      | Link to original source                   |
| legal_form          | TEXT      | Legal form (for organizations)            |
| incorporation_date  | TEXT      | Date of incorporation                     |
| address             | TEXT      | Address information                       |
| phone               | TEXT      | Phone numbers                             |
| email               | TEXT      | Email addresses                           |
| website             | TEXT      | Website URLs                              |
| created_at          | TIMESTAMP | Record creation timestamp                 |
| updated_at          | TIMESTAMP | Record update timestamp                   |

### Indexes

The following indexes are automatically created for optimal query performance:

-   `idx_name` on `name` column
-   `idx_entity_id` on `entity_id` column
-   `idx_dataset` on `dataset` column
-   `idx_entity_type` on `entity_type` column
-   `idx_country` on `country` column
-   `idx_source` on `source` column
-   `idx_created_at` on `created_at` column

## Usage Examples

### Basic Loading

```bash
# Extract data and load to PostgreSQL
python opensanctions_extractor.py --mode specific --datasets 'us_ofac_sdn' --output ofac_data.csv
python csv_to_postgresql.py --csv-file ofac_data.csv --db-config db_config.json
```

### Advanced Options

```bash
# Drop existing table and reload
python csv_to_postgresql.py --csv-file data.csv --db-config config.json --drop-table

# Skip duplicate checking (faster for large datasets)
python csv_to_postgresql.py --csv-file data.csv --db-config config.json --allow-duplicates

# Custom batch size for memory optimization
python csv_to_postgresql.py --csv-file data.csv --db-config config.json --batch-size 5000
```

### Sample Queries

Once data is loaded, you can query it using standard SQL:

```sql
-- Count entities by type
SELECT entity_type, COUNT(*) as count
FROM "sterlingai-aml-ctf-pep-tbl"
GROUP BY entity_type
ORDER BY count DESC;

-- Find entities from specific countries
SELECT name, entity_type, dataset
FROM "sterlingai-aml-ctf-pep-tbl"
WHERE country LIKE '%US%' OR country LIKE '%United States%'
LIMIT 10;

-- Search by name pattern
SELECT name, alias, country, dataset
FROM "sterlingai-aml-ctf-pep-tbl"
WHERE name ILIKE '%john%' OR alias ILIKE '%john%';

-- Recent sanctions (if timestamp data available)
SELECT name, sanction_program, dataset, last_updated
FROM "sterlingai-aml-ctf-pep-tbl"
WHERE sanction_program IS NOT NULL
ORDER BY created_at DESC
LIMIT 20;
```

## Configuration

### Database Configuration File

Create `db_config.json` with your PostgreSQL credentials:

```json
{
	"host": "localhost",
	"database": "your_database_name",
	"user": "your_username",
	"password": "your_password",
	"port": 5432
}
```

### Environment Setup

Install PostgreSQL dependencies:

```bash
# Install from requirements
pip install -r requirements.txt

# Or install directly
pip install psycopg2-binary
```

## Performance Considerations

-   **Batch Size**: Default 1000 records per batch. Increase for faster loading of large datasets
-   **Indexing**: Indexes are created automatically but may slow initial loading
-   **Memory Usage**: Script uses efficient batching to handle large CSV files
-   **Duplicate Handling**: By default, skips entities with duplicate `entity_id` values

## Error Handling

The loader includes comprehensive error handling:

-   **Connection failures**: Clear error messages for database connectivity issues
-   **Data validation**: Handles missing columns and malformed data gracefully
-   **Transaction safety**: Uses database transactions to ensure data consistency
-   **Encoding issues**: Handles various character encodings in CSV files

## Integration with Existing Workflows

The PostgreSQL loader is designed to integrate seamlessly with existing data workflows:

1. **Extract**: Use `opensanctions_extractor.py` to extract data to CSV
2. **Transform**: CSV format is standardized and ready for loading
3. **Load**: Use `csv_to_postgresql.py` to load into database
4. **Query**: Use standard SQL tools to query and analyze the data

This ETL (Extract, Transform, Load) pipeline provides a robust foundation for compliance, AML, and sanctions screening applications.
