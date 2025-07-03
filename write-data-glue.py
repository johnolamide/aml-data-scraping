import sys
import json
import csv
import logging
import os
import tempfile
from typing import Dict, Optional, List, Any, Set
from collections import defaultdict
import re
from io import StringIO
import boto3
from datetime import datetime

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext

try:
    import psycopg2
    from psycopg2.extras import execute_values
    from psycopg2 import sql
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class S3ToRDSGlueLoader:
    """AWS Glue job to load CSV files from S3 to RDS PostgreSQL with IAM Authentication"""

    def __init__(self, s3_bucket: str, db_config: Dict[str, str], use_iam_auth: bool = True):
        if not POSTGRES_AVAILABLE:
            raise ImportError(
                "psycopg2 is required for PostgreSQL integration")

        self.s3_bucket = s3_bucket
        self.db_config = db_config
        self.use_iam_auth = use_iam_auth
        self.s3_client = boto3.client('s3')
        self.rds_client = boto3.client('rds')

        # Database connection
        self.connection = None
        self.cursor = None

        # CSV to table mapping with Sterling AI naming convention
        self.csv_to_table_mapping = {
            'sanctions_entities.csv': 'sterlingai-aml-ctf-sanctions-entities-tbl',
            'pep_entities.csv': 'sterlingai-aml-ctf-pep-entities-tbl',
            'debarment_exclusions_entities.csv': 'sterlingai-aml-ctf-debarment-exclusions-entities-tbl',
            'wanted_lists_entities.csv': 'sterlingai-aml-ctf-wanted-lists-entities-tbl',
            'regulatory_entities.csv': 'sterlingai-aml-ctf-regulatory-entities-tbl',
            'wikidata_entities.csv': 'sterlingai-aml-ctf-wikidata-entities-tbl'
        }

        # Track processing state
        self.csv_columns = []
        self.column_types = {}
        self.original_columns = []
        self.unique_column = None
        self.db_table_columns = []

    def connect_to_database(self):
        """Establish connection to PostgreSQL database using IAM authentication"""
        try:
            if self.use_iam_auth:
                logger.info(
                    f"Connecting to PostgreSQL database using IAM authentication at {self.db_config['host']}...")

                # Generate RDS auth token
                token = self.rds_client.generate_db_auth_token(
                    DBHostname=self.db_config['host'],
                    Port=self.db_config['port'],
                    DBUsername=self.db_config['user'],
                    Region=self.db_config.get('region', 'eu-west-1')
                )

                # Download SSL certificate
                ssl_cert_path = '/tmp/global-bundle.pem'
                self.s3_client.download_file(
                    'aml-external-csv-dataset-simulate',
                    'rds-certs/global-bundle.pem',
                    ssl_cert_path
                )

                # Use psycopg2 for IAM authentication
                self.connection = psycopg2.connect(
                    host=self.db_config['host'],
                    port=self.db_config['port'],
                    database=self.db_config['database'],
                    user=self.db_config['user'],
                    password=token,
                    sslmode='verify-full',
                    sslrootcert=ssl_cert_path
                )

                self.cursor = self.connection.cursor()

            else:
                logger.info(
                    f"Connecting to PostgreSQL database using password authentication at {self.db_config['host']}...")
                self.connection = psycopg2.connect(**self.db_config)
                self.cursor = self.connection.cursor()

            # Test connection
            self.cursor.execute("SELECT version();")
            version = self.cursor.fetchone()
            logger.info(f"Connected to PostgreSQL: {version[0]}")

            # DON'T set autocommit - psycopg2 defaults to autocommit=False
            # which is what we want for transactions

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def disconnect_from_database(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Disconnected from PostgreSQL")

    def get_latest_date_folder(self) -> str:
        """Find the latest date folder in the S3 bucket"""
        try:
            logger.info(
                f"Looking for latest date folder in s3://{self.s3_bucket}/")

            # List all objects in the bucket
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Delimiter='/')

            date_folders = []
            for page in pages:
                for prefix_info in page.get('CommonPrefixes', []):
                    prefix = prefix_info['Prefix'].rstrip('/')
                    # Check if it looks like a date folder (YYYYMMDD format)
                    if re.match(r'^\d{8}$', prefix):
                        date_folders.append(prefix)

            if not date_folders:
                raise ValueError("No date folders found in the S3 bucket")

            # Sort and get the latest
            latest_folder = max(date_folders)
            logger.info(f"Found latest date folder: {latest_folder}")

            return latest_folder

        except Exception as e:
            logger.error(f"Failed to find latest date folder: {e}")
            raise

    def list_csv_files_in_folder(self, date_folder: str) -> List[str]:
        """List all CSV files in the specified date folder"""
        try:
            logger.info(f"Listing CSV files in folder: {date_folder}")

            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=f"{date_folder}/"
            )

            csv_files = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                filename = os.path.basename(key)
                if filename.endswith('.csv') and filename in self.csv_to_table_mapping:
                    csv_files.append(key)

            logger.info(
                f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
            return csv_files

        except Exception as e:
            logger.error(f"Failed to list CSV files: {e}")
            raise

    def download_csv_from_s3(self, s3_key: str) -> str:
        """Download CSV file from S3 to temporary location"""
        try:
            filename = os.path.basename(s3_key)
            temp_file = tempfile.mktemp(suffix='.csv')

            logger.info(
                f"Downloading s3://{self.s3_bucket}/{s3_key} to {temp_file}")

            self.s3_client.download_file(self.s3_bucket, s3_key, temp_file)

            # Verify file was downloaded
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                file_size = os.path.getsize(temp_file)
                logger.info(
                    f"Successfully downloaded {filename} ({file_size:,} bytes)")
                return temp_file
            else:
                raise ValueError(
                    f"Downloaded file {temp_file} is empty or doesn't exist")

        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            raise

    def get_table_name_for_csv(self, csv_filename: str) -> str:
        """Get the corresponding table name for a CSV file"""
        return self.csv_to_table_mapping.get(csv_filename, f"tbl_{csv_filename.replace('.csv', '').replace('-', '_')}")

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        try:
            self.cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                (table_name,)
            )
            return self.cursor.fetchone()[0]
        except Exception:
            return False

    def discover_csv_schema(self, csv_file_path: str, table_name: str) -> Dict[str, str]:
        """Analyze CSV file to discover schema and infer column types"""
        logger.info(f"Analyzing CSV file structure: {csv_file_path}")

        column_samples = defaultdict(list)

        try:
            with open(csv_file_path, 'r', encoding='utf-8', newline='') as csvfile:
                # Try to detect CSV dialect
                try:
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

                # Identify unique column
                self._identify_unique_column()

                # Sample data to infer types
                sample_count = 0
                for row in csv_reader:
                    if sample_count >= 1000:  # Sample size for type inference
                        break

                    for original_col, clean_col in zip(self.original_columns, self.csv_columns):
                        value = row.get(original_col, '')
                        if value is None:
                            value = ''
                        else:
                            value = str(value).strip()

                        if value and value.lower() not in ['', 'null', 'none', 'n/a', 'nan']:
                            column_samples[clean_col].append(value)

                    sample_count += 1

            # Infer column types
            schema = {}
            for col_name in self.csv_columns:
                samples = column_samples[col_name]
                inferred_type = self._infer_column_type(samples)
                schema[col_name] = inferred_type

            # Check if table exists and reconcile schema
            if self.table_exists(table_name):
                logger.info(
                    f"Table {table_name} exists, checking for schema compatibility...")
                schema = self.reconcile_schemas_for_incremental(
                    schema, table_name)

            logger.info(
                f"Discovered {len(schema)} columns for table {table_name}")

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

    def _identify_unique_column(self):
        """Identify which column to use for incremental loading"""
        unique_candidates = ['id', 'entity_id', 'record_id', 'uid', 'uuid']

        for candidate in unique_candidates:
            for orig_col, clean_col in zip(self.original_columns, self.csv_columns):
                if orig_col.lower() == candidate or clean_col.lower() == candidate:
                    self.unique_column = clean_col
                    return

        # If no standard unique column found, use the first column
        if self.csv_columns:
            self.unique_column = self.csv_columns[0]
            logger.warning(
                f"No standard unique column found, using '{self.unique_column}' for incremental loading")

    def _infer_column_type(self, samples: List[str]) -> str:
        """Infer PostgreSQL column type from sample values"""
        if not samples:
            return "TEXT"

        test_samples = samples[:500] if len(samples) > 500 else samples
        total = len(test_samples)

        if total == 0:
            return "TEXT"

        integers = 0
        large_integers = 0
        decimals = 0
        booleans = 0
        non_numeric = 0

        for value in test_samples:
            value = str(value).strip()

            if not value or value.lower() in ['null', 'none', 'n/a', 'nan', '']:
                continue

            if self._is_integer(value):
                integers += 1
                try:
                    int_val = int(value)
                    if abs(int_val) > 2147483647:
                        large_integers += 1
                except (ValueError, OverflowError):
                    large_integers += 1
            elif self._is_decimal(value):
                decimals += 1
            elif self._is_boolean(value):
                booleans += 1
            else:
                if self._is_clearly_text(value):
                    non_numeric += 1

        # Conservative thresholds
        strict_threshold = total * 0.95
        loose_threshold = total * 0.80

        if non_numeric > 0:
            return "TEXT"

        if integers >= strict_threshold:
            if large_integers > 0:
                return "BIGINT"
            else:
                return "INTEGER"
        elif decimals >= loose_threshold or (integers + decimals) >= strict_threshold:
            return "NUMERIC"
        elif booleans >= strict_threshold:
            return "BOOLEAN"
        else:
            return "TEXT"

    def _is_integer(self, value: str) -> bool:
        """Check if value is an integer"""
        try:
            val = value.strip()
            if not val:
                return False
            if val.startswith('-'):
                val = val[1:]
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
            return '.' in value or 'e' in value.lower()
        except (ValueError, TypeError):
            return False

    def _is_boolean(self, value: str) -> bool:
        """Check if value is boolean-like"""
        return value.lower() in ['true', 'false', 'yes', 'no', '1', '0', 't', 'f']

    def _is_clearly_text(self, value: str) -> bool:
        """Check if value is clearly textual"""
        if any(c.isalpha() for c in value):
            return True
        if ' ' in value.strip():
            return True
        text_chars = ['"', "'", '(', ')', '[', ']', '{', '}', '&', '#', '@']
        if any(char in value for char in text_chars):
            return True
        return False

    def get_existing_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get the existing table schema from the database"""
        if not self.table_exists(table_name):
            return {}

        try:
            query = """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND column_name NOT IN ('object_key', 'created_at', 'updated_at')
                ORDER BY ordinal_position;
            """
            self.cursor.execute(query, (table_name,))
            columns = self.cursor.fetchall()

            schema = {}
            self.db_table_columns = []

            for col_name, data_type in columns:
                self.db_table_columns.append(col_name)
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

        except Exception as e:
            logger.warning(f"Failed to get existing table schema: {e}")
            return {}

    def reconcile_schemas_for_incremental(self, csv_schema: Dict[str, str], table_name: str) -> Dict[str, str]:
        """Reconcile CSV schema with existing table schema for incremental loading"""
        existing_schema = self.get_existing_table_schema(table_name)

        if not self.db_table_columns:
            return csv_schema

        csv_cols_set = set(self.csv_columns)
        db_cols_set = set(self.db_table_columns)

        extra_csv_columns = csv_cols_set - db_cols_set
        missing_csv_columns = db_cols_set - csv_cols_set

        if extra_csv_columns:
            logger.info(
                f"CSV has {len(extra_csv_columns)} extra columns that will be ignored: {sorted(extra_csv_columns)}")

        if missing_csv_columns:
            logger.info(
                f"CSV is missing {len(missing_csv_columns)} columns that exist in DB (will be NULL): {sorted(missing_csv_columns)}")

        # Update csv_columns to only include columns that exist in the DB table
        self.csv_columns = [
            col for col in self.db_table_columns if col in csv_cols_set or col in missing_csv_columns]

        # Update original_columns mapping
        new_original_columns = []
        for db_col in self.csv_columns:
            found = False
            for orig_col, clean_col in zip(self.original_columns, self.csv_columns if hasattr(self, 'csv_columns') else []):
                if clean_col == db_col:
                    new_original_columns.append(orig_col)
                    found = True
                    break
            if not found:
                new_original_columns.append(f"__missing__{db_col}")

        self.original_columns = new_original_columns

        # Create reconciled schema
        reconciled_schema = {}
        for col in self.csv_columns:
            if col in existing_schema:
                reconciled_schema[col] = existing_schema[col]
            elif col in csv_schema:
                reconciled_schema[col] = csv_schema[col]
            else:
                reconciled_schema[col] = 'TEXT'

        logger.info(
            f"Reconciled schema: {len(reconciled_schema)} columns will be used for incremental load")
        return reconciled_schema

    def create_table(self, table_name: str):
        """Create PostgreSQL table with dynamic schema"""
        try:
            # Build CREATE TABLE query
            column_definitions = ["object_key SERIAL PRIMARY KEY"]

            for col_name in self.csv_columns:
                col_type = self.column_types.get(col_name, "TEXT")
                column_definitions.append(f'"{col_name}" {col_type}')

            # Add metadata columns
            column_definitions.extend([
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ])

            create_query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(column_definitions)})'

            self.cursor.execute(create_query)
            self.connection.commit()

            logger.info(
                f"Created table {table_name} with {len(self.csv_columns)} columns")

            # Create basic indexes
            self._create_indexes(table_name)

        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            if self.connection:
                self.connection.rollback()
            raise

    def _create_indexes(self, table_name: str):
        """Create indexes on important columns"""
        index_columns = ['name', 'entity_id',
                         'dataset', 'entity_type', 'source']

        for col in index_columns:
            if col in self.csv_columns:
                try:
                    index_name = f"{table_name.replace('-', '_')}_{col}_idx"
                    index_query = f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table_name}" ("{col}")'
                    self.cursor.execute(index_query)
                    logger.debug(f"Created index on column {col}")
                except Exception as e:
                    logger.warning(f"Failed to create index on {col}: {e}")

        try:
            self.connection.commit()
            logger.info("Created database indexes")
        except Exception as e:
            logger.warning(f"Failed to commit indexes: {e}")

    def get_existing_records(self, table_name: str) -> Set[str]:
        """Get set of existing unique identifiers from the database"""
        if not self.table_exists(table_name) or not self.unique_column:
            return set()

        try:
            query = f'SELECT "{self.unique_column}" FROM "{table_name}"'
            self.cursor.execute(query)
            existing_ids = {
                str(row[0]) for row in self.cursor.fetchall() if row[0] is not None}
            logger.info(
                f"Found {len(existing_ids)} existing records in {table_name}")
            return existing_ids
        except Exception as e:
            logger.warning(f"Failed to get existing records: {e}")
            return set()

    def load_csv_to_table(self, csv_file_path: str, table_name: str, batch_size: int = 1000) -> int:
        """Load CSV file into PostgreSQL table with incremental support"""
        logger.info(f"Loading {csv_file_path} into table {table_name}")

        # Get existing records for incremental loading
        existing_records = self.get_existing_records(table_name)

        inserted_count = 0
        skipped_count = 0
        error_count = 0
        batch_data = []

        try:
            with open(csv_file_path, 'r', encoding='utf-8', newline='') as csvfile:
                try:
                    sample = csvfile.read(8192)
                    csvfile.seek(0)
                    dialect = csv.Sniffer().sniff(sample, delimiters=',;|\t')
                except csv.Error:
                    dialect = csv.excel

                csv_reader = csv.DictReader(csvfile, dialect=dialect)

                for row_num, row in enumerate(csv_reader, 1):
                    try:
                        cleaned_row = self._clean_row_data(
                            row, True)  # Always use incremental mode

                        # Check if record should be skipped
                        if self._should_skip_record(cleaned_row, existing_records):
                            skipped_count += 1
                            continue

                        batch_data.append(cleaned_row)

                        if len(batch_data) >= batch_size:
                            inserted_batch = self._insert_batch(
                                batch_data, table_name)
                            inserted_count += inserted_batch
                            batch_data = []

                            if row_num % (batch_size * 10) == 0:
                                logger.info(
                                    f"Processed {row_num:,} rows, inserted {inserted_count:,}, skipped {skipped_count:,}")

                    except Exception as e:
                        error_count += 1
                        logger.warning(f"Error processing row {row_num}: {e}")
                        if error_count > 100:
                            logger.error(
                                "Too many errors encountered, stopping")
                            break
                        continue

                # Insert remaining batch
                if batch_data:
                    inserted_batch = self._insert_batch(batch_data, table_name)
                    inserted_count += inserted_batch

                self.connection.commit()

            logger.info(
                f"Loading completed: {inserted_count:,} inserted, {skipped_count:,} skipped, {error_count:,} errors")
            return inserted_count

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            if self.connection:
                self.connection.rollback()
            raise

    def _should_skip_record(self, row: Dict[str, Any], existing_records: Set[str]) -> bool:
        """Check if record should be skipped in incremental mode"""
        if not self.unique_column or not existing_records:
            return False

        unique_value = row.get(self.unique_column)
        if unique_value is None:
            return False

        return str(unique_value) in existing_records

    def _clean_row_data(self, row: Dict[str, str], incremental: bool = True) -> Dict[str, Any]:
        """Clean and validate row data"""
        cleaned = {}

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

        return cleaned

    def _convert_value(self, value: str, col_type: str) -> Any:
        """Convert string value to appropriate type"""
        if not value or value.lower() in ['null', 'none', 'n/a', 'nan']:
            return None

        try:
            if col_type in ['INTEGER', 'BIGINT']:
                clean_val = value.strip()
                if not clean_val:
                    return None
                if not self._is_integer(clean_val):
                    return str(value)
                return int(float(clean_val))

            elif col_type == 'NUMERIC':
                clean_val = value.strip()
                if not clean_val:
                    return None
                if not self._is_decimal(clean_val) and not self._is_integer(clean_val):
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
            return str(value)

    def _insert_batch(self, batch_data: List[Dict[str, Any]], table_name: str) -> int:
        """Insert batch of records using executemany"""
        if not batch_data:
            return 0

        try:
            # Escape table name and column names properly
            escaped_table_name = f'"{table_name}"'
            escaped_columns = [f'"{col}"' for col in self.csv_columns]

            # Create placeholders for values
            placeholders = ', '.join(['%s'] * len(self.csv_columns))

            # Build the insert query
            insert_query = f'INSERT INTO {escaped_table_name} ({", ".join(escaped_columns)}) VALUES ({placeholders})'

            # Prepare values as tuples
            values = []
            for row in batch_data:
                values.append(tuple(row.get(col) for col in self.csv_columns))

            # Use executemany for batch inserts
            self.cursor.executemany(insert_query, values)

            return len(values)

        except Exception as e:
            logger.error(f"Failed to insert batch: {e}")
            logger.error(f"Query was: {insert_query}")
            if self.connection:
                self.connection.rollback()
            raise

    def process_csv_file(self, s3_key: str) -> Dict[str, Any]:
        """Process a single CSV file from S3 to PostgreSQL"""
        filename = os.path.basename(s3_key)
        table_name = self.get_table_name_for_csv(filename)

        logger.info(f"Processing {filename} -> {table_name}")

        temp_file = None
        try:
            # Download CSV from S3
            temp_file = self.download_csv_from_s3(s3_key)

            # Discover schema
            schema = self.discover_csv_schema(temp_file, table_name)

            # Create table if it doesn't exist
            if not self.table_exists(table_name):
                logger.info(f"Creating table {table_name}")
                self.create_table(table_name)
            else:
                logger.info(
                    f"Table {table_name} already exists, using incremental loading")

            # Load data
            inserted_count = self.load_csv_to_table(temp_file, table_name)

            return {
                'filename': filename,
                'table_name': table_name,
                'status': 'success',
                'inserted_count': inserted_count,
                'schema_columns': len(schema)
            }

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            return {
                'filename': filename,
                'table_name': table_name,
                'status': 'failed',
                'error': str(e),
                'inserted_count': 0
            }
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

    def run_etl_process(self) -> Dict[str, Any]:
        """Main ETL process to load all CSV files from latest S3 folder to PostgreSQL"""
        start_time = datetime.now()

        try:
            # Connect to database
            self.connect_to_database()

            # Find latest date folder
            latest_folder = self.get_latest_date_folder()

            # List CSV files in the folder
            csv_files = self.list_csv_files_in_folder(latest_folder)

            if not csv_files:
                raise ValueError(
                    f"No CSV files found in folder {latest_folder}")

            logger.info(
                f"Starting ETL process for {len(csv_files)} files from folder {latest_folder}")

            # Process each CSV file
            results = []
            total_inserted = 0

            for csv_file in csv_files:
                try:
                    result = self.process_csv_file(csv_file)
                    results.append(result)
                    total_inserted += result.get('inserted_count', 0)

                    logger.info(
                        f"✅ {result['filename']}: {result['inserted_count']:,} records")

                except Exception as e:
                    logger.error(f"❌ Failed to process {csv_file}: {e}")
                    results.append({
                        'filename': os.path.basename(csv_file),
                        'status': 'failed',
                        'error': str(e),
                        'inserted_count': 0
                    })
                    continue

            # Summary
            successful_files = [r for r in results if r['status'] == 'success']
            failed_files = [r for r in results if r['status'] == 'failed']

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            summary = {
                'status': 'completed',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'latest_folder': latest_folder,
                'total_files': len(csv_files),
                'successful_files': len(successful_files),
                'failed_files': len(failed_files),
                'total_records_inserted': total_inserted,
                'file_results': results
            }

            logger.info(f"\n{'='*80}")
            logger.info(f"🎉 ETL PROCESS COMPLETE")
            logger.info(f"{'='*80}")
            logger.info(f"Folder: {latest_folder}")
            logger.info(
                f"Files processed: {len(successful_files)}/{len(csv_files)} successful")
            logger.info(f"Total records inserted: {total_inserted:,}")
            logger.info(f"Duration: {duration:.1f} seconds")

            if failed_files:
                logger.warning(
                    f"❌ Failed files: {[f['filename'] for f in failed_files]}")

            return summary

        except Exception as e:
            logger.error(f"ETL process failed: {e}")
            raise
        finally:
            self.disconnect_from_database()


def main():
    """Main function for AWS Glue job"""
    sc = SparkContext.getOrCreate()
    glue_context = GlueContext(sc)
    job = Job(glue_context)

    args = getResolvedOptions(sys.argv, ['JOB_NAME'])
    job.init(args['JOB_NAME'], args)

    s3_bucket = "aml-external-csv-dataset-simulate"

    db_config = {
        'host': 'aml.c5vd2gfncckf.eu-west-1.rds.amazonaws.com',
        'port': 5432,
        'database': 'sterlingai',
        'user': 'aml_glue_user',
        'region': 'eu-west-1'
    }

    logger.info(f"Starting S3 to RDS ETL job with IAM authentication")
    logger.info(f"S3 bucket: {s3_bucket}")
    logger.info(
        f"Database: {db_config['host']}:{db_config['port']}/{db_config['database']}")
    logger.info(f"Database user: {db_config['user']}")

    try:
        etl_processor = S3ToRDSGlueLoader(
            s3_bucket=s3_bucket,
            db_config=db_config,
            use_iam_auth=True
        )

        results = etl_processor.run_etl_process()

        successful_count = sum(
            1 for result in results['file_results'] if result['status'] == 'success')
        total_files = len(results['file_results'])

        logger.info(
            f"Job completed: {successful_count}/{total_files} files processed successfully")

        job.commit()

        return {
            'statusCode': 200,
            'body': {
                'message': f'ETL process completed: {successful_count}/{total_files} files successful',
                's3_bucket': s3_bucket,
                'latest_folder': results['latest_folder'],
                'total_records_inserted': results['total_records_inserted'],
                'duration_seconds': results['duration_seconds'],
                'results': results
            }
        }

    except Exception as e:
        logger.error(f"Job failed: {e}")
        raise


if __name__ == "__main__":
    main()
