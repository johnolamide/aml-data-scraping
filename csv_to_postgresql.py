#!/usr/bin/env python3
"""
CSV to PostgreSQL Database Loader - FIXED VERSION WITH INCREMENTAL SUPPORT

This script loads CSV files into PostgreSQL database tables with dynamic schema detection.
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, Optional, List, Any, Set
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
        self.unique_column = None  # Track which column to use for incremental loading
        self.db_table_columns = []  # Store existing table columns for incremental mode

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

    def get_existing_table_schema(self) -> Dict[str, str]:
        """Get the existing table schema from the database"""
        if not self.table_exists():
            return {}

        try:
            # Get column information from the existing table
            query = """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND column_name NOT IN ('object_key', 'created_at', 'updated_at')
                ORDER BY ordinal_position;
            """
            self.cursor.execute(query, (self.table_name,))
            columns = self.cursor.fetchall()

            schema = {}
            self.db_table_columns = []

            for col_name, data_type in columns:
                self.db_table_columns.append(col_name)
                # Map PostgreSQL types back to our internal types
                if data_type.upper() in ['INTEGER', 'BIGINT']:
                    schema[col_name] = data_type.upper()
                elif data_type.upper() in ['NUMERIC', 'DECIMAL', 'REAL', 'DOUBLE PRECISION']:
                    schema[col_name] = 'NUMERIC'
                elif data_type.upper() == 'BOOLEAN':
                    schema[col_name] = 'BOOLEAN'
                else:
                    schema[col_name] = 'TEXT'

            logger.info(
                f"Found existing table schema with {len(schema)} columns")
            return schema

        except psycopg2.Error as e:
            logger.warning(f"Failed to get existing table schema: {e}")
            return {}

    def reconcile_schemas_for_incremental(self, csv_schema: Dict[str, str]) -> Dict[str, str]:
        """Reconcile CSV schema with existing table schema for incremental loading"""
        if not self.db_table_columns:
            # No existing table, use CSV schema as-is
            return csv_schema

        # Get intersection of CSV columns and DB table columns
        csv_cols_set = set(self.csv_columns)
        db_cols_set = set(self.db_table_columns)

        # Columns that exist in both
        common_columns = csv_cols_set.intersection(db_cols_set)

        # Columns in CSV but not in DB (will be ignored)
        extra_csv_columns = csv_cols_set - db_cols_set

        # Columns in DB but not in CSV (will be filled with NULL)
        missing_csv_columns = db_cols_set - csv_cols_set

        if extra_csv_columns:
            logger.info(
                f"CSV has {len(extra_csv_columns)} extra columns that will be ignored: {sorted(extra_csv_columns)}")

        if missing_csv_columns:
            logger.info(
                f"CSV is missing {len(missing_csv_columns)} columns that exist in DB (will be NULL): {sorted(missing_csv_columns)}")

        # Update csv_columns to only include columns that exist in the DB table
        # This ensures we only insert into existing columns
        self.csv_columns = [
            col for col in self.db_table_columns if col in csv_cols_set or col in missing_csv_columns]

        # Update original_columns mapping to match
        new_original_columns = []
        for db_col in self.csv_columns:
            # Find the original column name that maps to this cleaned column
            found = False
            for orig_col, clean_col in zip(self.original_columns, self.csv_columns if hasattr(self, 'csv_columns') else []):
                if clean_col == db_col:
                    new_original_columns.append(orig_col)
                    found = True
                    break
            if not found:
                # This is a missing column, add a placeholder
                new_original_columns.append(f"__missing__{db_col}")

        self.original_columns = new_original_columns

        # Create reconciled schema using existing table schema for existing columns
        existing_schema = self.get_existing_table_schema()
        reconciled_schema = {}

        for col in self.csv_columns:
            if col in existing_schema:
                reconciled_schema[col] = existing_schema[col]
            elif col in csv_schema:
                reconciled_schema[col] = csv_schema[col]
            else:
                reconciled_schema[col] = 'TEXT'  # Default fallback

        logger.info(
            f"Reconciled schema: {len(reconciled_schema)} columns will be used for incremental load")
        return reconciled_schema

    def discover_csv_schema(self, csv_file_path: str, incremental: bool = False) -> Dict[str, str]:
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

                # Identify unique column for incremental loading
                self._identify_unique_column()

                # Sample more data to infer types more accurately
                sample_count = 0
                for row in csv_reader:
                    if sample_count >= 2000:  # Increased sample size
                        break

                    for original_col, clean_col in zip(self.original_columns, self.csv_columns):
                        value = row.get(original_col, '')
                        # Handle None values and convert to string
                        if value is None:
                            value = ''
                        else:
                            value = str(value).strip()

                        if value and value.lower() not in ['', 'null', 'none', 'n/a', 'nan']:
                            column_samples[clean_col].append(value)

                    sample_count += 1

            # Infer column types with more conservative approach
            schema = {}
            for col_name in self.csv_columns:
                samples = column_samples[col_name]
                inferred_type = self._infer_column_type(samples)
                schema[col_name] = inferred_type

            # If incremental mode, reconcile with existing table schema
            if incremental:
                schema = self.reconcile_schemas_for_incremental(schema)

            logger.info(f"Discovered {len(schema)} columns:")
            for col, col_type in list(schema.items())[:10]:
                sample_count = len(column_samples[col])
                logger.info(f"  - {col} ({col_type}): {sample_count} samples")

            if len(schema) > 10:
                logger.info(f"  ... and {len(schema) - 10} more columns")

            if self.unique_column:
                logger.info(
                    f"Using '{self.unique_column}' for incremental loading")

            self.column_types = schema
            return schema

        except Exception as e:
            logger.error(f"Failed to analyze CSV schema: {e}")
            raise

    def _identify_unique_column(self):
        """Identify which column to use for incremental loading"""
        # Check for common unique identifier columns
        unique_candidates = ['id', 'entity_id', 'record_id', 'uid', 'uuid']

        for candidate in unique_candidates:
            # Check both original and cleaned column names
            for orig_col, clean_col in zip(self.original_columns, self.csv_columns):
                if orig_col.lower() == candidate or clean_col.lower() == candidate:
                    self.unique_column = clean_col
                    return

        # If no standard unique column found, use the first column
        if self.csv_columns:
            self.unique_column = self.csv_columns[0]
            logger.warning(
                f"No standard unique column found, using '{self.unique_column}' for incremental loading")

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
        """Infer PostgreSQL column type from sample values with BIGINT support and conservative approach"""
        if not samples:
            return "TEXT"

        # Use more samples for better accuracy
        test_samples = samples[:500] if len(samples) > 500 else samples
        total = len(test_samples)

        if total == 0:
            return "TEXT"

        integers = 0
        large_integers = 0  # Track integers that exceed INTEGER range
        decimals = 0
        dates = 0
        booleans = 0
        non_numeric = 0  # Track clearly non-numeric values

        for value in test_samples:
            value = str(value).strip()

            # Skip empty values
            if not value or value.lower() in ['null', 'none', 'n/a', 'nan', '']:
                continue

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
            else:
                # Check if it's clearly non-numeric (contains letters, special chars, etc.)
                if self._is_clearly_text(value):
                    non_numeric += 1

        # Use more conservative thresholds (95% instead of 80%)
        strict_threshold = total * 0.95
        loose_threshold = total * 0.80

        # If we have any clearly non-numeric values, default to TEXT
        if non_numeric > 0:
            return "TEXT"

        # Very strict for integers - must be 95% or more
        if integers >= strict_threshold:
            # Use BIGINT if any samples exceed INTEGER range, otherwise INTEGER
            if large_integers > 0:
                return "BIGINT"
            else:
                return "INTEGER"
        # Allow decimals with slightly looser threshold
        elif decimals >= loose_threshold or (integers + decimals) >= strict_threshold:
            return "NUMERIC"
        elif dates >= loose_threshold:
            return "TEXT"  # Use TEXT for dates to avoid parsing issues
        elif booleans >= strict_threshold:
            return "BOOLEAN"
        else:
            # Default to TEXT for mixed or uncertain data
            return "TEXT"

    def _is_integer(self, value: str) -> bool:
        """Check if value is an integer"""
        try:
            # Must be purely numeric
            val = value.strip()
            if not val:
                return False

            # Handle negative numbers
            if val.startswith('-'):
                val = val[1:]

            # Must be all digits
            if not val.isdigit():
                return False

            int(value)
            return '.' not in value and 'e' not in value.lower()
        except (ValueError, TypeError):
            return False

    def _is_decimal(self, value: str) -> bool:
        """Check if value is a decimal number"""
        try:
            val = value.strip()
            if not val:
                return False

            float(value)
            # Must contain decimal point or scientific notation
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

    def _is_clearly_text(self, value: str) -> bool:
        """Check if value is clearly textual (contains letters, spaces, special chars)"""
        # If it contains letters, it's clearly text
        if any(c.isalpha() for c in value):
            return True

        # If it contains spaces (except leading/trailing), it's likely text
        if ' ' in value.strip():
            return True

        # If it contains common text punctuation
        text_chars = ['"', "'", '(', ')', '[', ']', '{', '}', '&', '#', '@']
        if any(char in value for char in text_chars):
            return True

        return False

    def table_exists(self) -> bool:
        """Check if table exists"""
        try:
            self.cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                (self.table_name,)
            )
            return self.cursor.fetchone()[0]
        except psycopg2.Error:
            return False

    def get_existing_records(self) -> Set[str]:
        """Get set of existing unique identifiers from the database"""
        if not self.table_exists() or not self.unique_column:
            return set()

        try:
            query = sql.SQL("SELECT {} FROM {}").format(
                sql.Identifier(self.unique_column),
                sql.Identifier(self.table_name)
            )
            self.cursor.execute(query)
            existing_ids = {
                str(row[0]) for row in self.cursor.fetchall() if row[0] is not None}
            logger.info(
                f"Found {len(existing_ids)} existing records in database")
            return existing_ids
        except psycopg2.Error as e:
            logger.warning(f"Failed to get existing records: {e}")
            return set()

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

    def load_csv_file(self, csv_file_path: str, batch_size: int = 1000, incremental: bool = False) -> int:
        """Load CSV file into PostgreSQL table"""
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        logger.info(f"Loading CSV file: {csv_file_path}")
        logger.info(f"Target table: {self.table_name}")
        logger.info(f"Incremental mode: {incremental}")

        file_size = os.path.getsize(csv_file_path)
        logger.info(f"File size: {file_size / (1024*1024):.2f} MB")

        # Get existing records if incremental mode
        existing_records = set()
        if incremental:
            existing_records = self.get_existing_records()

        inserted_count = 0
        skipped_count = 0
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
                        cleaned_row = self._clean_row_data(row, incremental)

                        # Check if record should be skipped in incremental mode
                        if incremental and self._should_skip_record(cleaned_row, existing_records):
                            skipped_count += 1
                            continue

                        batch_data.append(cleaned_row)

                        if len(batch_data) >= batch_size:
                            inserted_batch = self._insert_batch(batch_data)
                            inserted_count += inserted_batch
                            batch_data = []

                            if row_num % (batch_size * 10) == 0:
                                logger.info(
                                    f"Processed {row_num:,} rows, inserted {inserted_count:,}, skipped {skipped_count:,}")

                    except Exception as e:
                        error_count += 1
                        logger.warning(f"Error processing row {row_num}: {e}")

                        # If we have too many errors, there might be a systematic issue
                        if error_count > 100:
                            logger.error(
                                "Too many errors encountered. There might be a schema mismatch.")
                            logger.error(
                                "Consider checking the CSV file format or adjusting the schema inference.")
                            break
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
            f"Loading completed: {inserted_count:,} inserted, {skipped_count:,} skipped, {error_count:,} errors")
        return inserted_count

    def _should_skip_record(self, row: Dict[str, Any], existing_records: Set[str]) -> bool:
        """Check if record should be skipped in incremental mode"""
        if not self.unique_column or not existing_records:
            return False

        unique_value = row.get(self.unique_column)
        if unique_value is None:
            return False

        return str(unique_value) in existing_records

    def _clean_row_data(self, row: Dict[str, str], incremental: bool = False) -> Dict[str, Any]:
        """Clean and validate row data with improved None handling and schema reconciliation"""
        cleaned = {}

        # In incremental mode, we need to handle schema mismatches
        if incremental and self.db_table_columns:
            # Process only columns that exist in the database table
            for db_col in self.csv_columns:
                # Find the corresponding original column name
                orig_col = None
                for i, clean_col in enumerate(self.csv_columns):
                    if clean_col == db_col and i < len(self.original_columns):
                        orig_col_candidate = self.original_columns[i]
                        if not orig_col_candidate.startswith("__missing__"):
                            orig_col = orig_col_candidate
                        break

                # Get the value
                if orig_col and orig_col in row:
                    value = row.get(orig_col)
                else:
                    # Column doesn't exist in CSV, set to None
                    value = None

                # Process the value
                if value is None:
                    cleaned[db_col] = None
                else:
                    value = str(value).strip()
                    if not value or value.lower() in ['', 'null', 'none', 'n/a', 'nan']:
                        cleaned[db_col] = None
                    else:
                        col_type = self.column_types.get(db_col, 'TEXT')
                        cleaned[db_col] = self._convert_value(value, col_type)
        else:
            # Normal processing for non-incremental mode
            for orig_col, clean_col in zip(self.original_columns, self.csv_columns):
                # Get the value, handling None cases
                value = row.get(orig_col)

                # Convert None or empty values to empty string, then strip
                if value is None:
                    value = ''
                else:
                    value = str(value).strip()

                # Check for empty or null values
                if not value or value.lower() in ['', 'null', 'none', 'n/a', 'nan']:
                    cleaned[clean_col] = None
                else:
                    col_type = self.column_types.get(clean_col, 'TEXT')
                    cleaned[clean_col] = self._convert_value(value, col_type)

        return cleaned

    def _convert_value(self, value: str, col_type: str) -> Any:
        """Convert string value to appropriate type with better error handling"""
        if not value or value.lower() in ['null', 'none', 'n/a', 'nan']:
            return None

        try:
            if col_type in ['INTEGER', 'BIGINT']:
                # Be more careful with integer conversion
                clean_val = value.strip()
                if not clean_val:
                    return None

                # Check if it's actually numeric before conversion
                if not self._is_integer(clean_val):
                    # If it's not actually an integer, return as string
                    # This shouldn't happen with proper schema inference, but just in case
                    logger.warning(
                        f"Expected integer but got '{value}' - converting to string")
                    return str(value)

                return int(float(clean_val))  # Handle "123.0" -> 123

            elif col_type == 'NUMERIC':
                clean_val = value.strip()
                if not clean_val:
                    return None

                if not self._is_decimal(clean_val) and not self._is_integer(clean_val):
                    logger.warning(
                        f"Expected numeric but got '{value}' - converting to string")
                    return str(value)

                return float(clean_val)

            elif col_type == 'BOOLEAN':
                return value.lower() in ['true', 'yes', '1', 't']
            else:  # TEXT
                if len(value) > 65535:
                    return value[:65532] + "..."
                return value

        except (ValueError, TypeError) as e:
            logger.warning(
                f"Failed to convert '{value}' to {col_type}: {e} - using string instead")
            return str(value)  # Fallback to string

    def _insert_batch(self, batch_data: List[Dict[str, Any]]) -> int:
        """Insert batch of records with better error handling"""
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

            # Try to insert rows one by one to identify problematic rows
            if len(batch_data) > 1:
                logger.info(
                    "Attempting individual row insertion to identify problem rows...")
                success_count = 0
                for i, row in enumerate(batch_data):
                    try:
                        single_values = [tuple(row.get(col)
                                               for col in self.csv_columns)]
                        execute_values(  # type: ignore
                            self.cursor,
                            insert_query,  # type: ignore
                            single_values,
                            template=None,
                            page_size=1
                        )
                        success_count += 1
                    except psycopg2.Error as single_e:  # type: ignore
                        logger.warning(f"Row {i+1} failed: {single_e}")
                        logger.debug(f"Problematic row data: {row}")
                        continue

                if success_count > 0:
                    logger.info(
                        f"Successfully inserted {success_count}/{len(batch_data)} rows from batch")
                    return success_count

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
    parser.add_argument('--incremental', action='store_true',
                        help="Incremental loading - only insert new records")

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
        schema = loader.discover_csv_schema(
            args.csv_file, incremental=args.incremental)
        logger.info(f"Schema discovery complete: {len(schema)} columns")

        # Create table (don't drop if incremental mode)
        create_table = True
        if args.incremental and loader.table_exists():
            logger.info("Table exists, skipping creation for incremental load")
            create_table = False

        if create_table:
            loader.create_table(drop_existing=args.drop_table)

        # Load data
        inserted_count = loader.load_csv_file(
            args.csv_file,
            args.batch_size,
            incremental=args.incremental
        )

        if args.incremental:
            logger.info(
                f"Incremental load completed: {inserted_count:,} new records added")
        else:
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
