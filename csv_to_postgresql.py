#!/usr/bin/env python3
"""
CSV to PostgreSQL Database Loader

This script loads the standardized CSV output from the OpenSanctions extractor
into a PostgreSQL database table with proper schema and indexing.

Usage:
    python csv_to_postgresql.py --csv-file opensanctions_entities.csv --db-config config.json
    python csv_to_postgresql.py --csv-file output.csv --host localhost --database aml_db --user postgres --password secret
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, Optional, List, Any

try:
    import psycopg2
    from psycopg2.extras import execute_values
    from psycopg2 import sql
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CSVToPostgreSQLLoader:
    """Load CSV data into PostgreSQL database"""

    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize the loader with database configuration

        Args:
            db_config: Dictionary with database connection parameters
                      (host, database, user, password, port)
        """
        self.db_config = db_config
        self.table_name = "sterlingai-aml-ctf-pep-tbl"
        self.connection = None
        self.cursor = None

        # Define the standardized CSV columns from the extractor
        self.csv_columns = [
            'name', 'source', 'dataset', 'dataset_title', 'entity_type', 'entity_id',
            'alias', 'country', 'birth_date', 'nationality', 'position', 'registration_number',
            'topics', 'sanction_program', 'description', 'last_updated', 'source_url',
            'legal_form', 'incorporation_date', 'address', 'phone', 'email', 'website'
        ]

    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            logger.info(
                f"Connecting to PostgreSQL database at {self.db_config['host']}...")
            self.connection = psycopg2.connect(  # type: ignore
                **self.db_config)  # type: ignore
            self.cursor = self.connection.cursor()

            # Test connection
            self.cursor.execute("SELECT version();")
            version = self.cursor.fetchone()
            logger.info(f"Connected to PostgreSQL: {version[0]}")

        except psycopg2.Error as e:  # type: ignore
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Disconnected from PostgreSQL")

    def create_table(self, drop_existing: bool = False):
        """
        Create the PostgreSQL table with proper schema

        Args:
            drop_existing: Whether to drop the table if it already exists
        """
        try:
            # Drop table if requested
            if drop_existing:
                drop_query = sql.SQL("DROP TABLE IF EXISTS {}").format(  # type: ignore
                    sql.Identifier(self.table_name)  # type: ignore
                )
                self.cursor.execute(drop_query)  # type: ignore
                logger.info(f"Dropped existing table {self.table_name}")

            # Create table with proper schema
            create_query = sql.SQL(  # type: ignore
                """ 
                CREATE TABLE IF NOT EXISTS {} (
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
                )
            """).format(sql.Identifier(self.table_name))  # type: ignore

            self.cursor.execute(create_query)  # type: ignore
            self.connection.commit()  # type: ignore
            logger.info(f"Created/verified table {self.table_name}")

            # Create indexes for better query performance
            self.create_indexes()

        except psycopg2.Error as e:  # type: ignore
            logger.error(f"Failed to create table: {e}")
            raise

    def create_indexes(self):
        """Create indexes on frequently queried columns"""
        indexes = [
            ("idx_name", "name"),
            ("idx_entity_id", "entity_id"),
            ("idx_dataset", "dataset"),
            ("idx_entity_type", "entity_type"),
            ("idx_country", "country"),
            ("idx_source", "source"),
            ("idx_created_at", "created_at")
        ]

        for index_name, column in indexes:
            try:
                index_query = sql.SQL(  # type: ignore
                    """
                    CREATE INDEX IF NOT EXISTS {} ON {} ({})
                """).format(
                    sql.Identifier( # type: ignore
                        f"{self.table_name}_{index_name}"),  # type: ignore
                    sql.Identifier(self.table_name),  # type: ignore
                    sql.Identifier(column)  # type: ignore
                )
                self.cursor.execute(index_query)  # type: ignore
                logger.debug(f"Created index {index_name} on column {column}")
            except psycopg2.Error as e:  # type: ignore
                logger.warning(f"Failed to create index {index_name}: {e}")

        self.connection.commit()  # type: ignore
        logger.info("Created database indexes")

    def load_csv_file(self, csv_file_path: str, batch_size: int = 1000,
                      skip_duplicates: bool = True) -> int:
        """
        Load CSV file into PostgreSQL table

        Args:
            csv_file_path: Path to the CSV file
            batch_size: Number of records to insert in each batch
            skip_duplicates: Whether to skip duplicate entity_ids

        Returns:
            Number of records inserted
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        logger.info(f"Loading CSV file: {csv_file_path}")

        # Get file size for progress reporting
        file_size = os.path.getsize(csv_file_path)
        logger.info(f"CSV file size: {file_size / (1024*1024):.2f} MB")

        inserted_count = 0
        skipped_count = 0
        error_count = 0
        batch_data = []

        # Track existing entity_ids if skipping duplicates
        existing_entities = set()
        if skip_duplicates:
            existing_entities = self.get_existing_entity_ids()
            logger.info(
                f"Found {len(existing_entities)} existing entities in database")

        try:
            with open(csv_file_path, 'r', encoding='utf-8', newline='') as csvfile:
                # Try to detect CSV dialect, but fallback to default comma-separated
                try:
                    sample = csvfile.read(1024)
                    csvfile.seek(0)
                    dialect = csv.Sniffer().sniff(sample)
                except csv.Error:
                    # Fallback to default CSV format if detection fails
                    logger.info(
                        "CSV dialect detection failed, using default comma-separated format")
                    dialect = csv.excel  # Default comma-separated format

                csv_reader = csv.DictReader(csvfile, dialect=dialect)

                # Verify CSV columns match expected schema
                # type: ignore
                # type: ignore
                if not self.validate_csv_columns(csv_reader.fieldnames): # type: ignore
                    raise ValueError("CSV columns don't match expected schema")

                for row_num, row in enumerate(csv_reader, 1):
                    try:
                        # Clean and validate row data
                        cleaned_row = self.clean_row_data(row)

                        # Skip duplicates if requested
                        if skip_duplicates and cleaned_row['entity_id'] in existing_entities:
                            skipped_count += 1
                            continue

                        batch_data.append(cleaned_row)

                        # Insert batch when it reaches the specified size
                        if len(batch_data) >= batch_size:
                            inserted_batch = self.insert_batch(batch_data)
                            inserted_count += inserted_batch
                            batch_data = []

                            # Log progress
                            if row_num % (batch_size * 10) == 0:
                                logger.info(
                                    f"Processed {row_num:,} rows, inserted {inserted_count:,} records")

                    except Exception as e:
                        error_count += 1
                        logger.warning(f"Error processing row {row_num}: {e}")
                        continue

                # Insert remaining batch
                if batch_data:
                    inserted_batch = self.insert_batch(batch_data)
                    inserted_count += inserted_batch

                self.connection.commit()  # type: ignore

        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            if self.connection:
                self.connection.rollback()
            raise

        logger.info(f"CSV loading completed:")
        logger.info(f"  - Inserted: {inserted_count:,} records")
        logger.info(f"  - Skipped: {skipped_count:,} duplicates")
        logger.info(f"  - Errors: {error_count:,} rows")

        return inserted_count

    def validate_csv_columns(self, csv_columns: List[str]) -> bool:
        """Validate that CSV has expected columns"""
        if not csv_columns:
            return False

        missing_columns = set(self.csv_columns) - set(csv_columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        return True

    def clean_row_data(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Clean and validate row data"""
        cleaned = {}

        for column in self.csv_columns:
            value = row.get(column, '').strip()

            # Handle empty values
            if not value or value.lower() in ['', 'null', 'none', 'n/a']:
                cleaned[column] = None
            else:
                # Truncate very long text fields to prevent database errors
                if len(value) > 65535:  # PostgreSQL TEXT limit
                    value = value[:65532] + "..."
                    logger.warning(f"Truncated long value in column {column}")

                cleaned[column] = value

        return cleaned

    def insert_batch(self, batch_data: List[Dict[str, Any]]) -> int:
        """Insert a batch of records"""
        if not batch_data:
            return 0

        try:
            # Prepare insert query
            insert_query = sql.SQL(  # type: ignore
                """
                INSERT INTO {} ({}) VALUES %s
            """).format(
                sql.Identifier(self.table_name),  # type: ignore
                sql.SQL(', ').join( # type: ignore
                    map(sql.Identifier, self.csv_columns))  # type: ignore
            )

            # Prepare values
            values = []
            for row in batch_data:
                values.append(tuple(row[col] for col in self.csv_columns))

            # Execute batch insert
            execute_values(  # type: ignore
                self.cursor,
                insert_query,
                values,
                template=None,
                page_size=len(values)
            )

            return len(values)

        except psycopg2.Error as e:  # type: ignore
            logger.error(f"Failed to insert batch: {e}")
            raise

    def get_existing_entity_ids(self) -> set:
        """Get set of existing entity_ids from database"""
        try:
            query = sql.SQL("SELECT entity_id FROM {} WHERE entity_id IS NOT NULL").format(  # type: ignore
                sql.Identifier(self.table_name)  # type: ignore
            )
            self.cursor.execute(query)  # type: ignore
            return {row[0] for row in self.cursor.fetchall()}  # type: ignore
        except psycopg2.Error as e:  # type: ignore
            logger.warning(f"Failed to get existing entity IDs: {e}")
            return set()

    def get_table_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded data"""
        try:
            stats_query = sql.SQL(  # type: ignore
                """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT entity_id) as unique_entities,
                    COUNT(DISTINCT dataset) as unique_datasets,
                    COUNT(DISTINCT country) as unique_countries,
                    MIN(created_at) as first_loaded,
                    MAX(created_at) as last_loaded
                FROM {}
            """).format(sql.Identifier(self.table_name))  # type: ignore

            self.cursor.execute(stats_query)  # type: ignore
            result = self.cursor.fetchone()  # type: ignore

            column_names = [desc[0]
                            for desc in self.cursor.description]  # type: ignore
            return dict(zip(column_names, result))

        except psycopg2.Error as e:  # type: ignore
            logger.error(f"Failed to get table statistics: {e}")
            return {}


def load_db_config_from_file(config_file: str) -> Dict[str, str]:
    """Load database configuration from JSON file"""
    with open(config_file, 'r') as f:
        config = json.load(f)

    required_keys = ['host', 'database', 'user', 'password']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Set default port if not specified
    if 'port' not in config:
        config['port'] = 5432

    return config


def main():
    """Main function"""
    # Check if psycopg2 is available
    if not POSTGRES_AVAILABLE:
        print("Error: psycopg2 is required for PostgreSQL integration.")
        print("Please install it with: pip install psycopg2-binary")
        print("Or install all requirements with: pip install -r requirements.txt")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Load OpenSanctions CSV data into PostgreSQL database"
    )

    # CSV file argument
    parser.add_argument(
        '--csv-file',
        required=True,
        help='Path to the CSV file to load'
    )

    # Database configuration options
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--db-config',
        help='Path to JSON file with database configuration'
    )
    config_group.add_argument(
        '--host',
        help='Database host (use with --database, --user, --password)'
    )

    # Individual database parameters
    parser.add_argument('--database', help='Database name')
    parser.add_argument('--user', help='Database user')
    parser.add_argument('--password', help='Database password')
    parser.add_argument('--port', type=int, default=5432,
                        help='Database port (default: 5432)')

    # Loading options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of records to insert in each batch (default: 1000)'
    )
    parser.add_argument(
        '--drop-table',
        action='store_true',
        help='Drop existing table before creating new one'
    )
    parser.add_argument(
        '--allow-duplicates',
        action='store_true',
        help='Allow duplicate entity_ids (default: skip duplicates)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Prepare database configuration
    if args.db_config:
        db_config = load_db_config_from_file(args.db_config)
    else:
        if not all([args.host, args.database, args.user, args.password]):
            parser.error(
                "When not using --db-config, all of --host, --database, --user, --password are required")

        db_config = {
            'host': args.host,
            'database': args.database,
            'user': args.user,
            'password': args.password,
            'port': args.port
        }

    # Initialize loader
    loader = CSVToPostgreSQLLoader(db_config)

    try:
        # Connect to database
        loader.connect()

        # Create table
        loader.create_table(drop_existing=args.drop_table)

        # Load CSV data
        inserted_count = loader.load_csv_file(
            args.csv_file,
            batch_size=args.batch_size,
            skip_duplicates=not args.allow_duplicates
        )

        # Show statistics
        stats = loader.get_table_stats()
        if stats:
            logger.info("Database table statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")

        logger.info(
            f"Successfully loaded {inserted_count:,} records into PostgreSQL")

    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

    finally:
        loader.disconnect()


if __name__ == "__main__":
    main()
