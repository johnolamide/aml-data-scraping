#!/usr/bin/env python3
"""
CSV to PostgreSQL Database Loader - FIXED VERSION

This script loads CSV files into PostgreSQL database tables with dynamic schema detection.
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, Optional, List, Any
from collections import defaultdict
import re

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
    """Load CSV data into PostgreSQL database with dynamic schema"""

    def __init__(self, db_config: Dict[str, str], table_name: Optional[str] = None):
        if not POSTGRES_AVAILABLE:
            raise ImportError(
                "psycopg2 is required. Install with: pip install psycopg2-binary")

        self.db_config = db_config
        self.table_name = table_name or "default_table"
        self.connection = None
        self.cursor = None
        self.csv_columns = []
        self.column_types = {}
        self.original_columns = []  # Keep track of original CSV column names

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

    def set_table_name_from_csv(self, csv_file_path: str):
        """Set table name based on CSV filename"""
        filename = os.path.basename(csv_file_path)
        table_name = os.path.splitext(filename)[0]

        # Clean the name for PostgreSQL
        table_name = table_name.lower()
        table_name = re.sub(r'[^a-z0-9_]', '_', table_name)
        table_name = re.sub(r'_+', '_', table_name)
        table_name = table_name.strip('_')

        if not table_name or not table_name[0].isalpha():
            table_name = 'tbl_' + table_name

        if len(table_name) > 63:
            table_name = table_name[:63].rstrip('_')

        self.table_name = table_name
        logger.info(f"Set table name to: {self.table_name}")

    def discover_csv_schema(self, csv_file_path: str) -> Dict[str, str]:
        """Analyze CSV file to discover schema and infer column types"""
        logger.info(f"Analyzing CSV file structure: {csv_file_path}")

        column_samples = defaultdict(list)

        try:
            with open(csv_file_path, 'r', encoding='utf-8', newline='') as csvfile:
                # Try to detect CSV dialect
                try:
                    # Read more for better detection
                    sample = csvfile.read(8192)
                    csvfile.seek(0)
                    dialect = csv.Sniffer().sniff(sample, delimiters=',;|\t')
                except csv.Error:
                    logger.warning(
                        "Could not detect CSV dialect, using default")
                    dialect = csv.excel

                csv_reader = csv.DictReader(csvfile, dialect=dialect)

                if not csv_reader.fieldnames:
                    raise ValueError("CSV file has no column headers")

                # Store original column names
                self.original_columns = list(csv_reader.fieldnames)

                # Clean column names for PostgreSQL
                self.csv_columns = []
                for col in csv_reader.fieldnames:
                    clean_col = self._clean_column_name(col)
                    self.csv_columns.append(clean_col)

                # Sample data to infer types
                sample_count = 0
                for row in csv_reader:
                    if sample_count >= 1000:
                        break

                    for original_col, clean_col in zip(self.original_columns, self.csv_columns):
                        value = row.get(original_col, '').strip()
                        if value and value.lower() not in ['', 'null', 'none', 'n/a', 'nan']:
                            column_samples[clean_col].append(value)

                    sample_count += 1

            # Infer column types
            schema = {}
            for col_name in self.csv_columns:
                samples = column_samples[col_name]
                inferred_type = self._infer_column_type(samples)
                schema[col_name] = inferred_type

            logger.info(f"Discovered {len(schema)} columns:")
            for col, col_type in list(schema.items())[:10]:
                sample_count = len(column_samples[col])
                logger.info(f"  - {col} ({col_type}): {sample_count} samples")

            if len(schema) > 10:
                logger.info(f"  ... and {len(schema) - 10} more columns")

            self.column_types = schema
            return schema

        except Exception as e:
            logger.error(f"Failed to analyze CSV schema: {e}")
            raise

    def _clean_column_name(self, column_name: str) -> str:
        """Clean column name for PostgreSQL compatibility"""
        if not column_name:
            return "unnamed_column"

        clean_name = str(column_name).strip().lower()
        clean_name = re.sub(r'[^a-z0-9_]', '_', clean_name)
        clean_name = re.sub(r'_+', '_', clean_name)
        clean_name = clean_name.strip('_')

        if not clean_name or not clean_name[0].isalpha():
            clean_name = 'col_' + clean_name

        if not clean_name:
            clean_name = 'unnamed_column'

        if len(clean_name) > 63:
            clean_name = clean_name[:63].rstrip('_')

        return clean_name

    def _infer_column_type(self, samples: List[str]) -> str:
        """Infer PostgreSQL column type from sample values with BIGINT support"""
        if not samples:
            return "TEXT"

        # Check first 100 samples
        test_samples = samples[:100]
        total = len(test_samples)

        integers = 0
        large_integers = 0  # Track integers that exceed INTEGER range
        decimals = 0
        dates = 0
        booleans = 0

        for value in test_samples:
            value = str(value).strip()

            if self._is_integer(value):
                integers += 1
                # Check if integer exceeds PostgreSQL INTEGER range (-2^31 to 2^31-1)
                try:
                    int_val = int(value)
                    if abs(int_val) > 2147483647:
                        large_integers += 1
                except (ValueError, OverflowError):
                    # If we can't parse it or it's too large, treat as large integer
                    large_integers += 1
            elif self._is_decimal(value):
                decimals += 1
            elif self._is_date_like(value):
                dates += 1
            elif self._is_boolean(value):
                booleans += 1

        # Use 80% threshold for type decision
        threshold = total * 0.8

        if integers > threshold:
            # Use BIGINT if any samples exceed INTEGER range, otherwise INTEGER
            if large_integers > 0:
                return "BIGINT"
            else:
                return "INTEGER"
        elif decimals > threshold or (integers + decimals) > threshold:
            return "NUMERIC"
        elif dates > threshold:
            return "TEXT"  # Use TEXT for dates to avoid parsing issues
        elif booleans > threshold:
            return "BOOLEAN"
        else:
            return "TEXT"

    def _is_integer(self, value: str) -> bool:
        """Check if value is an integer"""
        try:
            int(value)
            return '.' not in value and 'e' not in value.lower()
        except (ValueError, TypeError):
            return False

    def _is_decimal(self, value: str) -> bool:
        """Check if value is a decimal number"""
        try:
            float(value)
            return '.' in value or 'e' in value.lower()
        except (ValueError, TypeError):
            return False

    def _is_date_like(self, value: str) -> bool:
        """Check if value looks like a date"""
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',
            r'^\d{2}/\d{2}/\d{4}$',
            r'^\d{2}-\d{2}-\d{4}$',
            r'^\d{4}/\d{2}/\d{2}$',
        ]

        for pattern in date_patterns:
            if re.match(pattern, value):
                return True
        return False

    def _is_boolean(self, value: str) -> bool:
        """Check if value is boolean-like"""
        return value.lower() in ['true', 'false', 'yes', 'no', '1', '0', 't', 'f']

    def create_table(self, drop_existing: bool = False):
        """Create PostgreSQL table with dynamic schema"""
        try:
            if drop_existing:
                drop_query = sql.SQL("DROP TABLE IF EXISTS {}").format(  # type: ignore
                    sql.Identifier(self.table_name)  # type: ignore
                )
                self.cursor.execute(drop_query)  # type: ignore
                logger.info(f"Dropped existing table {self.table_name}")

            # Build CREATE TABLE query
            column_definitions = ["object_key SERIAL PRIMARY KEY"]

            for col_name in self.csv_columns:
                col_type = self.column_types.get(col_name, "TEXT")
                column_definitions.append(f"{col_name} {col_type}")

            # Add metadata columns
            column_definitions.extend([
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ])

            create_query = sql.SQL(  # type: ignore
                "CREATE TABLE IF NOT EXISTS {} ({})"
            ).format(
                sql.Identifier(self.table_name),  # type: ignore
                sql.SQL(", ".join(column_definitions))  # type: ignore
            )

            self.cursor.execute(create_query)  # type: ignore
            self.connection.commit()  # type: ignore

            logger.info(
                f"Created table {self.table_name} with {len(self.csv_columns)} columns")

            # Create basic indexes (only on columns that likely exist)
            self._create_safe_indexes()

        except psycopg2.Error as e:  # type: ignore
            logger.error(f"Failed to create table: {e}")
            if self.connection:
                self.connection.rollback()
            raise

    def _create_safe_indexes(self):
        """Create indexes only on columns that actually exist"""
        common_index_columns = ['name', 'entity_id',
                                'dataset', 'entity_type', 'source']

        for col in common_index_columns:
            if col in self.csv_columns:
                try:
                    index_name = f"{self.table_name}_{col}_idx"
                    index_query = sql.SQL(  # type: ignore
                        "CREATE INDEX IF NOT EXISTS {} ON {} ({})"
                    ).format(
                        sql.Identifier(index_name),  # type: ignore
                        sql.Identifier(self.table_name),  # type: ignore
                        sql.Identifier(col)  # type: ignore
                    )
                    self.cursor.execute(index_query)  # type: ignore
                    logger.debug(f"Created index on column {col}")
                except psycopg2.Error as e:  # type: ignore
                    logger.warning(f"Failed to create index on {col}: {e}")

        try:
            self.connection.commit()  # type: ignore
            logger.info("Created database indexes")
        except psycopg2.Error as e:  # type: ignore
            logger.warning(f"Failed to commit indexes: {e}")

    def load_csv_file(self, csv_file_path: str, batch_size: int = 1000) -> int:
        """Load CSV file into PostgreSQL table"""
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        logger.info(f"Loading CSV file: {csv_file_path}")
        logger.info(f"Target table: {self.table_name}")

        file_size = os.path.getsize(csv_file_path)
        logger.info(f"File size: {file_size / (1024*1024):.2f} MB")

        inserted_count = 0
        error_count = 0
        batch_data = []

        try:
            with open(csv_file_path, 'r', encoding='utf-8', newline='') as csvfile:
                # Use same dialect detection as schema discovery
                try:
                    sample = csvfile.read(8192)
                    csvfile.seek(0)
                    dialect = csv.Sniffer().sniff(sample, delimiters=',;|\t')
                except csv.Error:
                    dialect = csv.excel

                csv_reader = csv.DictReader(csvfile, dialect=dialect)

                for row_num, row in enumerate(csv_reader, 1):
                    try:
                        cleaned_row = self._clean_row_data(row)
                        batch_data.append(cleaned_row)

                        if len(batch_data) >= batch_size:
                            inserted_batch = self._insert_batch(batch_data)
                            inserted_count += inserted_batch
                            batch_data = []

                            if row_num % (batch_size * 10) == 0:
                                logger.info(
                                    f"Processed {row_num:,} rows, inserted {inserted_count:,}")

                    except Exception as e:
                        error_count += 1
                        logger.warning(f"Error processing row {row_num}: {e}")
                        continue

                # Insert remaining batch
                if batch_data:
                    inserted_batch = self._insert_batch(batch_data)
                    inserted_count += inserted_batch

                self.connection.commit()  # type: ignore

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            if self.connection:
                self.connection.rollback()
            raise

        logger.info(
            f"Loading completed: {inserted_count:,} inserted, {error_count:,} errors")
        return inserted_count

    def _clean_row_data(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Clean and validate row data"""
        cleaned = {}

        for orig_col, clean_col in zip(self.original_columns, self.csv_columns):
            value = row.get(orig_col, '').strip()

            if not value or value.lower() in ['', 'null', 'none', 'n/a', 'nan']:
                cleaned[clean_col] = None
            else:
                col_type = self.column_types.get(clean_col, 'TEXT')
                cleaned[clean_col] = self._convert_value(value, col_type)

        return cleaned

    def _convert_value(self, value: str, col_type: str) -> Any:
        """Convert string value to appropriate type"""
        if not value or value.lower() in ['null', 'none', 'n/a', 'nan']:
            return None

        try:
            if col_type in ['INTEGER', 'BIGINT']:
                return int(float(value))  # Handle "123.0" -> 123
            elif col_type == 'NUMERIC':
                return float(value)
            elif col_type == 'BOOLEAN':
                return value.lower() in ['true', 'yes', '1', 't']
            else:  # TEXT
                if len(value) > 65535:
                    return value[:65532] + "..."
                return value
        except (ValueError, TypeError):
            return str(value)  # Fallback to string

    def _insert_batch(self, batch_data: List[Dict[str, Any]]) -> int:
        """Insert batch of records"""
        if not batch_data:
            return 0

        try:
            insert_query = sql.SQL(  # type: ignore
                "INSERT INTO {} ({}) VALUES %s"
            ).format(
                sql.Identifier(self.table_name),  # type: ignore
                sql.SQL(', ').join(  # type: ignore
                    map(sql.Identifier, self.csv_columns))  # type: ignore
            )

            values = []
            for row in batch_data:
                values.append(tuple(row.get(col) for col in self.csv_columns))

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
            if self.connection:
                self.connection.rollback()
            raise


def main():
    """Main function"""
    if not POSTGRES_AVAILABLE:
        print("Error: psycopg2 is required for PostgreSQL integration.")
        print("Please install it with: pip install psycopg2-binary")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Load CSV data into PostgreSQL")
    parser.add_argument('--csv-file', required=True, help="Path to CSV file")
    parser.add_argument('--db-config', required=True,
                        help="Path to DB config JSON")
    parser.add_argument('--drop-table', action='store_true',
                        help="Drop existing table")
    parser.add_argument('--batch-size', type=int,
                        default=1000, help="Batch size")

    args = parser.parse_args()

    # Load database config
    try:
        with open(args.db_config, 'r') as f:
            db_config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load database configuration: {e}")
        sys.exit(1)

    # Initialize and run loader
    loader = CSVToPostgreSQLLoader(db_config)

    try:
        loader.connect()
        loader.set_table_name_from_csv(args.csv_file)

        # Discover schema first
        schema = loader.discover_csv_schema(args.csv_file)
        logger.info(f"Schema discovery complete: {len(schema)} columns")

        # Create table
        loader.create_table(drop_existing=args.drop_table)

        # Load data
        inserted_count = loader.load_csv_file(args.csv_file, args.batch_size)
        logger.info(f"Successfully loaded {inserted_count:,} records")

    except Exception as e:
        logger.error(f"Operation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

    finally:
        loader.disconnect()


if __name__ == "__main__":
    main()
