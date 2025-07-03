import sys
import json
import csv
import logging
import requests
import tempfile
import os
import subprocess
import zipfile
import gzip
import bz2
import re
import shutil
from typing import Dict, List, Any, Optional
from io import StringIO
import boto3
from datetime import datetime
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenSanctionsGlueExtractor:
    """AWS Glue version of OpenSanctions extractor that uploads to S3"""

    def __init__(self, s3_bucket: str, date_folder: str):
        self.base_url = "https://data.opensanctions.org/"
        self.index_url = "https://data.opensanctions.org/datasets/latest/index.json"
        self.s3_bucket = s3_bucket
        self.date_folder = date_folder  # Format: YYYYMMDD

        # Initialize AWS clients
        self.s3_client = boto3.client('s3')

        # Dataset category mappings - ALL categories will be processed
        self.dataset_categories = {
            'sanctions': ['us_ofac_sdn', 'eu_sanctions', 'un_sc_sanctions', 'maritime', 'securities'],
            'pep': ['ng_chipper_peps', 'peps', 'br_pep', 'wd_peps'],
            'debarment_exclusions': ['debarment', 'us_sam_exclusions', 'us_hhs_exclusions', 'us_ca_med_exclusions'],
            'wanted_lists': ['interpol_red_notices', 'crime', 'pl_wanted', 'us_dea_fugitives', 'tr_wanted'],
            'regulatory': ['regulatory'],
            'wikidata': ['wikidata']
        }

        # Configure session for downloads
        self.session = requests.Session()
        self._configure_session()

        # Column mapping for ID standardization
        self.id_column_mappings = {
            'id': 'entity_id',
            'entityid': 'entity_id',
            'entity_id': 'entity_id',
            'record_id': 'entity_id',
            'recordid': 'entity_id',
            'uid': 'entity_id',
            'uuid': 'entity_id',
            'identifier': 'entity_id',
            'person_id': 'entity_id',
            'personid': 'entity_id',
            'individual_id': 'entity_id',
            'individualid': 'entity_id',
            'subject_id': 'entity_id',
            'subjectid': 'entity_id',
            'target_id': 'entity_id',
            'targetid': 'entity_id'
        }

    def _normalize_column_name(self, column_name: str) -> str:
        """Normalize column names and map ID columns to entity_id"""
        if not column_name:
            return column_name

        # Clean the column name first
        clean_name = column_name.strip().lower().replace(' ', '_').replace('-', '_')
        clean_name = re.sub(r'[^a-z0-9_]', '_', clean_name)
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')

        # Check if this is an ID column that should be mapped to entity_id
        if clean_name in self.id_column_mappings:
            mapped_name = self.id_column_mappings[clean_name]
            if clean_name != mapped_name:
                logger.debug(
                    f"Mapping column '{column_name}' -> '{mapped_name}'")
            return mapped_name

        return clean_name

    def _configure_session(self):
        """Configure the requests session for better downloads"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        self.session.verify = False  # For SSL issues with OpenSanctions

    def get_datasets_for_category(self, category: str) -> List[str]:
        """Get list of dataset names for a given category"""
        if category in self.dataset_categories:
            return self.dataset_categories[category]
        else:
            # If it's not a predefined category, treat it as a single dataset name
            return [category]

    def fetch_datasets_index(self) -> Dict:
        """Fetch the main datasets index"""
        logger.info("Fetching OpenSanctions datasets index...")
        try:
            response = self.session.get(self.index_url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch datasets index: {e}")
            raise

    def get_target_datasets(self, category: str) -> List[Dict]:
        """Get list of datasets to process based on category"""
        dataset_names = self.get_datasets_for_category(category)
        index_data = self.fetch_datasets_index()
        all_datasets = index_data.get('datasets', [])

        target_datasets = []
        dataset_name_set = set(dataset_names)

        for dataset in all_datasets:
            if dataset.get('name') in dataset_name_set:
                target_datasets.append(dataset)

        logger.info(
            f"Found {len(target_datasets)} datasets for category '{category}': {[d.get('name') for d in target_datasets]}")
        return target_datasets

    def extract_from_csv_resource(self, dataset: Dict, csv_url: str) -> List[Dict]:
        """Extract entities from targets.simple.csv format"""
        logger.info(f"Downloading CSV from {csv_url}")
        entities = []

        try:
            response = self.session.get(csv_url, timeout=60)
            response.raise_for_status()

            # Parse CSV content
            csv_content = response.text
            csv_reader = csv.DictReader(csv_content.splitlines())

            for row in csv_reader:
                entity = self._create_entity_from_csv_row(row, dataset)
                if entity.get('name'):
                    entities.append(entity)

            logger.info(f"Extracted {len(entities)} entities from CSV")
            return entities

        except Exception as e:
            logger.error(f"Failed to process CSV {csv_url}: {e}")
            return []

    def extract_from_ftm_json(self, dataset: Dict, json_url: str) -> List[Dict]:
        """Extract entities from entities.ftm.json format"""
        logger.info(f"Downloading FTM JSON from {json_url}")
        entities = []

        try:
            response = self.session.get(json_url, timeout=60)
            response.raise_for_status()

            # Parse JSON Lines format
            for line in response.text.strip().split('\n'):
                if not line.strip():
                    continue

                try:
                    ftm_entity = json.loads(line)
                    entity = self._create_entity_from_ftm_json(
                        ftm_entity, dataset)
                    if entity.get('name'):
                        entities.append(entity)
                except json.JSONDecodeError:
                    continue

            logger.info(f"Extracted {len(entities)} entities from FTM JSON")
            return entities

        except Exception as e:
            logger.error(f"Failed to process FTM JSON {json_url}: {e}")
            return []

    def extract_dataset_entities(self, dataset: Dict) -> List[Dict]:
        """Extract entities from a single dataset"""
        dataset_name = dataset.get('name', 'unknown')
        logger.info(
            f"Processing dataset: {dataset_name} ({dataset.get('title', 'No title')})")

        # Get available resources
        resources = dataset.get('resources', [])

        # Prefer CSV format, fallback to FTM JSON
        csv_resource = None
        ftm_resource = None

        for resource in resources:
            resource_name = resource.get('name', '')
            if resource_name == 'targets.simple.csv':
                csv_resource = resource
            elif resource_name == 'entities.ftm.json':
                ftm_resource = resource

        # Try CSV first, then FTM JSON
        if csv_resource:
            return self.extract_from_csv_resource(dataset, csv_resource['url'])
        elif ftm_resource:
            return self.extract_from_ftm_json(dataset, ftm_resource['url'])
        else:
            logger.warning(
                f"No suitable resource found for dataset {dataset_name}")
            return []

    def _create_entity_from_ftm_json(self, ftm_entity: Dict, dataset: Dict) -> Dict:
        """Create entity dict from FTM JSON using actual available properties"""
        entity = {}

        # Always include core metadata
        entity['source'] = 'opensanctions'
        entity['dataset'] = dataset.get('name', '')
        entity['dataset_title'] = dataset.get('title', '')
        entity['last_updated'] = dataset.get('updated_at', '')
        entity['entity_id'] = ftm_entity.get('id', '')
        entity['entity_type'] = ftm_entity.get('schema', '')

        # Process all available properties
        properties = ftm_entity.get('properties', {})
        for prop_name, prop_values in properties.items():
            if prop_values:  # Only non-empty values
                # Join multiple values with semicolon
                if isinstance(prop_values, list):
                    clean_value = '; '.join(str(v).strip()
                                            for v in prop_values if str(v).strip())
                else:
                    clean_value = str(prop_values).strip()

                if clean_value:
                    # Normalize the property name (including ID mapping)
                    clean_key = self._normalize_column_name(prop_name)
                    entity[clean_key] = clean_value

        return entity

    def _create_entity_from_csv_row(self, row: Dict, dataset: Dict) -> Dict:
        """Create entity dict from CSV row using actual available columns"""
        entity = {}

        # Always include core metadata
        entity['source'] = 'opensanctions'
        entity['dataset'] = dataset.get('name', '')
        entity['dataset_title'] = dataset.get('title', '')
        entity['last_updated'] = dataset.get('updated_at', '')

        # Add all available columns from the CSV row with ID normalization
        for key, value in row.items():
            if key and value and str(value).strip():  # Only non-empty values
                # Normalize the column name (including ID mapping)
                clean_key = self._normalize_column_name(key)
                entity[clean_key] = str(value).strip()

        return entity

    def analyze_and_finalize_schema(self, all_entities: List[Dict]) -> List[str]:
        """Analyze collected data and finalize the CSV schema"""
        logger.info(f"Analyzing schema from {len(all_entities)} entities...")

        # Count column usage to prioritize common columns
        column_counts = {}
        for entity in all_entities:
            for column in entity.keys():
                column_counts[column] = column_counts.get(column, 0) + 1

        # Sort columns by usage frequency and importance
        core_columns = ['source', 'dataset', 'dataset_title',
                        'name', 'entity_id', 'entity_type']
        other_columns = [col for col in column_counts.keys()
                         if col not in core_columns]
        other_columns.sort(key=lambda x: column_counts[x], reverse=True)

        # Final column order: core columns first, then others by frequency
        csv_columns = []
        for col in core_columns:
            if col in column_counts:
                csv_columns.append(col)

        csv_columns.extend(other_columns)

        logger.info(f"Finalized schema with {len(csv_columns)} columns")
        return csv_columns

    def upload_csv_to_s3(self, all_entities: List[Dict], category: str) -> str:
        """Upload entities to S3 as CSV file in date-based folder"""
        # Analyze the data to determine the best schema
        csv_columns = self.analyze_and_finalize_schema(all_entities)

        # Create S3 key with date folder structure
        s3_key = f"{self.date_folder}/{category}_entities.csv"

        logger.info(
            f"Uploading {len(all_entities)} entities to s3://{self.s3_bucket}/{s3_key}")

        try:
            # Create CSV content in memory
            csv_buffer = StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=csv_columns)
            writer.writeheader()

            for entity in all_entities:
                # Create row with all columns, filling missing ones with empty string
                clean_entity = {}
                for col in csv_columns:
                    clean_entity[col] = entity.get(col, '')
                writer.writerow(clean_entity)

            # Upload to S3
            csv_content = csv_buffer.getvalue()
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=csv_content.encode('utf-8'),
                ContentType='text/csv'
            )

            logger.info(
                f"Successfully uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
            return s3_key

        except Exception as e:
            logger.error(f"Failed to upload CSV to S3: {e}")
            raise

    def extract_category(self, category: str) -> str:
        """Extract data for a specific category and upload to S3"""
        logger.info(f"Starting extraction for category: {category}")

        # Get target datasets for the category
        target_datasets = self.get_target_datasets(category)

        if not target_datasets:
            logger.warning(f"No datasets found for category: {category}")
            return None

        logger.info(
            f"Processing {len(target_datasets)} datasets for category '{category}'")

        all_entities = []

        for i, dataset in enumerate(target_datasets, 1):
            dataset_name = dataset.get('name', 'unknown')
            logger.info(
                f"[{i}/{len(target_datasets)}] Processing {dataset_name}")

            try:
                entities = self.extract_dataset_entities(dataset)
                all_entities.extend(entities)
                logger.info(
                    f"Total entities so far for {category}: {len(all_entities)}")

            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_name}: {e}")
                continue

        if not all_entities:
            logger.warning(f"No entities extracted for category: {category}")
            return None

        # Upload to S3
        s3_key = self.upload_csv_to_s3(all_entities, category)

        # Log summary for this category
        logger.info(f"=== {category.upper()} EXTRACTION COMPLETE ===")
        logger.info(f"Total entities: {len(all_entities):,}")
        logger.info(f"Datasets processed: {len(target_datasets)}")
        logger.info(f"S3 location: s3://{self.s3_bucket}/{s3_key}")

        # Entity type breakdown
        entity_types = {}
        for entity in all_entities:
            entity_type = entity.get('entity_type', 'Unknown')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        logger.info(f"Top entity types:")
        for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {entity_type}: {count:,}")

        return s3_key

    def run_all_extractions(self) -> Dict[str, str]:
        """Extract all dataset categories and upload to S3"""
        logger.info(f"Starting OpenSanctions extraction for ALL categories")
        logger.info(f"Target S3 bucket: {self.s3_bucket}")
        logger.info(f"Date folder: {self.date_folder}")

        results = {}
        total_entities = 0

        # Process each category
        for category in self.dataset_categories.keys():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"PROCESSING CATEGORY: {category.upper()}")
                logger.info(f"{'='*60}")

                s3_key = self.extract_category(category)
                if s3_key:
                    results[category] = s3_key
                    logger.info(f"✅ Successfully processed {category}")
                else:
                    logger.warning(f"❌ Failed to process {category}")
                    results[category] = None

            except Exception as e:
                logger.error(f"❌ Failed to process category {category}: {e}")
                results[category] = None
                continue

        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"🎉 ALL EXTRACTIONS COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Date folder: {self.date_folder}")
        logger.info(f"S3 bucket: s3://{self.s3_bucket}")

        successful_categories = [
            cat for cat, result in results.items() if result is not None]
        failed_categories = [cat for cat,
                             result in results.items() if result is None]

        logger.info(
            f"✅ Successful categories ({len(successful_categories)}): {', '.join(successful_categories)}")
        if failed_categories:
            logger.info(
                f"❌ Failed categories ({len(failed_categories)}): {', '.join(failed_categories)}")

        logger.info(f"\nFiles created in S3:")
        for category, s3_key in results.items():
            if s3_key:
                logger.info(f"  📄 s3://{self.s3_bucket}/{s3_key}")

        return results


def main():
    """Main function for AWS Glue job"""
    # Initialize Glue context
    sc = SparkContext.getOrCreate()
    glue_context = GlueContext(sc)
    job = Job(glue_context)

    # Get job parameters (only JOB_NAME is required now)
    args = getResolvedOptions(sys.argv, ['JOB_NAME'])
    job.init(args['JOB_NAME'], args)

    # Fixed S3 bucket (extract bucket name from ARN)
    s3_bucket = "aml-external-csv-dataset-simulate"

    # Generate date folder (YYYYMMDD format)
    date_folder = datetime.now().strftime("%Y%m%d")

    logger.info(f"Starting Glue job for ALL OpenSanctions categories")
    logger.info(f"  S3 bucket: {s3_bucket}")
    logger.info(f"  Date folder: {date_folder}")

    try:
        # Initialize extractor
        extractor = OpenSanctionsGlueExtractor(
            s3_bucket=s3_bucket,
            date_folder=date_folder
        )

        # Run extraction for all categories
        results = extractor.run_all_extractions()

        # Count successful extractions
        successful_count = sum(
            1 for result in results.values() if result is not None)
        total_categories = len(results)

        # Log final result
        logger.info(
            f"Job completed: {successful_count}/{total_categories} categories processed successfully")

        # Commit the job
        job.commit()

        # Return success summary
        return {
            'statusCode': 200,
            'body': {
                'message': f'Extraction completed: {successful_count}/{total_categories} categories successful',
                'date_folder': date_folder,
                's3_bucket': s3_bucket,
                'results': results,
                'successful_categories': [cat for cat, result in results.items() if result is not None],
                'failed_categories': [cat for cat, result in results.items() if result is None]
            }
        }

    except Exception as e:
        logger.error(f"Job failed: {e}")
        raise


if __name__ == "__main__":
    main()
