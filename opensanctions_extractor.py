#!/usr/bin/env python3
"""
OpenSanctions Data Extractor

Extracts entity data from OpenSanctions datasets and generates a unified CSV
with name as the main attribute, source as "opensanctions", and other relevant attributes.

Usage:
    python opensanctions_extractor.py --mode all --output opensanctions_entities.csv
    python opensanctions_extractor.py --mode specific --datasets wanted,un_sc_sanctions --output specific_entities.csv
    python opensanctions_extractor.py --mode consolidated --output consolidated_entities.csv
"""

import json
import csv
import logging
import requests
import argparse
import sys
import shutil
from typing import Dict, List, Any, Optional
import time
import ssl
import urllib3

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable SSL warnings for urllib3 (when using verify=False)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class OpenSanctionsExtractor:
    def __init__(self, output_file: str = 'opensanctions_entities.csv'):
        self.base_url = "https://data.opensanctions.org/"
        self.index_url = "https://data.opensanctions.org/datasets/latest/index.json"
        self.session = requests.Session()
        self.output_file = output_file

        # Configure session for better SSL handling
        self._configure_session()

        # Define CSV output columns
        self.csv_columns = [
            'name', 'source', 'dataset', 'dataset_title', 'entity_type', 'entity_id',
            'alias', 'country', 'birth_date', 'nationality', 'position', 'registration_number',
            'topics', 'sanction_program', 'description', 'last_updated', 'source_url',
            'legal_form', 'incorporation_date', 'address', 'phone', 'email', 'website'
        ]

    def _configure_session(self):
        """Configure the requests session for better SSL handling"""
        # Set headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        # For OpenSanctions, we'll use a more aggressive SSL configuration
        # due to known SSL/TLS compatibility issues with their server
        try:
            # Create multiple SSL context options to try
            ssl_contexts = []

            # Option 1: Very permissive context with legacy support
            try:
                ctx1 = ssl.create_default_context()
                ctx1.set_ciphers('DEFAULT@SECLEVEL=1')
                ctx1.check_hostname = False
                ctx1.verify_mode = ssl.CERT_NONE
                # Enable legacy protocols if available
                ctx1.options &= ~ssl.OP_NO_SSLv3
                ctx1.options &= ~ssl.OP_NO_TLSv1
                ctx1.options &= ~ssl.OP_NO_TLSv1_1
                ssl_contexts.append(("permissive_legacy", ctx1))
            except:
                pass

            # Option 2: High compatibility context
            try:
                ctx2 = ssl.create_default_context()
                ctx2.set_ciphers('HIGH:!DH:!aNULL')
                ctx2.check_hostname = False
                ctx2.verify_mode = ssl.CERT_NONE
                ssl_contexts.append(("high_compat", ctx2))
            except:
                pass

            # Option 3: Broad cipher support
            try:
                ctx3 = ssl.create_default_context()
                ctx3.set_ciphers(
                    'ALL:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA')
                ctx3.check_hostname = False
                ctx3.verify_mode = ssl.CERT_NONE
                ssl_contexts.append(("broad_cipher", ctx3))
            except:
                pass

            # Try to use the first working SSL context
            from requests.adapters import HTTPAdapter
            from urllib3.poolmanager import PoolManager

            class FlexibleSSLAdapter(HTTPAdapter):
                def __init__(self, ssl_contexts):
                    self.ssl_contexts = ssl_contexts
                    super().__init__()

                def init_poolmanager(self, *args, **kwargs):
                    # Try each SSL context until one works
                    for name, ctx in self.ssl_contexts:
                        try:
                            kwargs['ssl_context'] = ctx
                            return super().init_poolmanager(*args, **kwargs)
                        except Exception as e:
                            logger.debug(f"SSL context {name} failed: {e}")
                            continue

                    # If all fail, try without custom SSL context
                    kwargs.pop('ssl_context', None)
                    return super().init_poolmanager(*args, **kwargs)

            if ssl_contexts:
                self.session.mount(
                    'https://', FlexibleSSLAdapter(ssl_contexts))
                logger.info(
                    f"Configured flexible SSL context with {len(ssl_contexts)} options")
            else:
                # Ultimate fallback: disable SSL verification
                self.session.verify = False
                logger.warning(
                    "No SSL contexts available, disabled SSL verification")

        except Exception as e:
            logger.warning(f"SSL context configuration failed: {e}")
            # Fallback: disable SSL verification entirely
            self.session.verify = False
            logger.warning("Disabled SSL verification as fallback")

    def _make_request(self, url: str, timeout: int = 60, max_retries: int = 3, stream: bool = False):
        """Make HTTP request with retry logic and better error handling"""
        import time

        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"Attempting to fetch {url} (attempt {attempt + 1}/{max_retries})")
                response = self.session.get(
                    url, timeout=timeout, stream=stream)
                response.raise_for_status()
                return response

            except requests.exceptions.SSLError as e:
                last_exception = e
                logger.warning(f"SSL error on attempt {attempt + 1}: {e}")

                # For SSL errors, try different approaches
                if attempt == 0:
                    # First retry: disable SSL verification completely
                    try:
                        logger.warning(
                            "Retrying with SSL verification disabled")
                        response = requests.get(
                            url, timeout=timeout, verify=False, stream=stream,
                            headers=self.session.headers)
                        response.raise_for_status()
                        return response
                    except Exception as e2:
                        logger.warning(
                            f"Request with disabled SSL also failed: {e2}")

                elif attempt == 1:
                    # Second retry: try with different user agent and HTTP/1.1
                    try:
                        logger.warning("Retrying with different headers")
                        headers = dict(self.session.headers)
                        headers['User-Agent'] = 'curl/7.68.0'
                        headers['Connection'] = 'close'
                        response = requests.get(
                            url, timeout=timeout, verify=False, stream=stream,
                            headers=headers)
                        response.raise_for_status()
                        return response
                    except Exception as e3:
                        logger.warning(
                            f"Request with different headers failed: {e3}")

            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"Timeout on attempt {attempt + 1}: {e}")

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(
                    f"Connection error on attempt {attempt + 1}: {e}")

            except Exception as e:
                last_exception = e
                logger.warning(f"Request failed on attempt {attempt + 1}: {e}")

            # Wait before retrying (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

        # All retries failed
        if last_exception:
            raise last_exception
        else:
            raise requests.exceptions.RequestException(
                f"Failed to fetch {url} after {max_retries} attempts")

    def fetch_datasets_index(self) -> Dict:
        """Fetch the main datasets index"""
        logger.info("Fetching OpenSanctions datasets index...")
        try:
            response = self._make_request_with_fallback(
                self.index_url, timeout=30)
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch datasets index: {e}")
            raise

    # type: ignore
    # type: ignore
    # type: ignore
    # type: ignore
    # type: ignore
    def get_target_datasets(self, mode: str, specific_datasets: Optional[List[str]] = None) -> List[Dict]:
        """Get list of datasets to process based on mode"""
        index_data = self.fetch_datasets_index()
        all_datasets = index_data.get('datasets', [])

        if mode == 'consolidated':
            # Use the consolidated statements CSV
            return [{'name': 'default', 'use_consolidated': True}]

        elif mode == 'specific' and specific_datasets:
            # Filter for specific datasets
            target_datasets = []
            specific_set = set(specific_datasets)
            for dataset in all_datasets:
                if dataset.get('name') in specific_set:
                    target_datasets.append(dataset)
            return target_datasets

        elif mode == 'all':
            # Process all source datasets with reasonable entity counts
            target_datasets = []
            for dataset in all_datasets:
                entity_count = dataset.get('entity_count', 0)
                dataset_type = dataset.get('type', '')

                # Filter criteria:
                # 1. Source datasets (not collections or external)
                # 2. Has entities
                # 3. Has target entities (actual sanctioned/wanted people)
                # 4. Reasonable size (not too huge to avoid memory issues)
                if (dataset_type == 'source' and
                    entity_count > 0 and
                    dataset.get('target_count', 0) > 0 and
                        entity_count < 100000):  # Limit to manageable sizes
                    target_datasets.append(dataset)

            # Sort by entity count descending, take top 50
            target_datasets.sort(key=lambda x: x.get(
                'target_count', 0), reverse=True)
            return target_datasets[:50]

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def extract_from_csv_resource(self, dataset: Dict, csv_url: str) -> List[Dict]:
        """Extract entities from targets.simple.csv format"""
        logger.info(f"Downloading CSV from {csv_url}")
        entities = []

        try:
            # Check if this is a large dataset that needs special handling
            dataset_name = dataset.get('name', '')
            target_count = dataset.get('target_count', 0)

            # Use robust download for large datasets
            if target_count > 50000:
                logger.info(
                    f"Large dataset detected ({target_count:,} entities), using robust download...")

                # Try robust download method first
                downloaded_file = self._download_large_file(
                    csv_url, timeout=300)
                if downloaded_file:
                    return self._process_downloaded_csv_file(downloaded_file, dataset)
                else:
                    logger.error("All download methods failed for large file")
                    return []
            else:
                # For smaller datasets, use the regular streaming method
                timeout = 60
                response = self._make_request_with_fallback(
                    csv_url, timeout=timeout, stream=False)

                # Parse CSV content normally
                csv_content = response.text
                csv_reader = csv.DictReader(csv_content.splitlines())

                for row in csv_reader:
                    entity = self._create_entity_from_csv_row(row, dataset)
                    if entity['name']:
                        entities.append(entity)

            logger.info(f"Extracted {len(entities)} entities from CSV")
            return entities

        except Exception as e:
            logger.error(f"Failed to process CSV {csv_url}: {e}")
            return []

    def _download_large_file(self, url: str, timeout: int = 300) -> Optional[str]:
        """Download large files with robust error handling and SSL fallback"""
        import tempfile
        import shutil
        import os

        # Determine file extension from URL
        url_lower = url.lower()
        if '.zip' in url_lower:
            temp_file = tempfile.mktemp(suffix='.zip')
        elif '.gz' in url_lower:
            temp_file = tempfile.mktemp(suffix='.gz')
        elif '.bz2' in url_lower:
            temp_file = tempfile.mktemp(suffix='.bz2')
        else:
            temp_file = tempfile.mktemp(suffix='.csv')

        # Try multiple approaches for downloading large files
        methods = [
            ("requests_stream", self._download_with_requests_stream),
            ("requests_raw", self._download_with_requests_raw),
            ("curl_fallback", self._download_with_curl)
        ]

        for method_name, method_func in methods:
            try:
                logger.info(
                    f"Attempting large file download using {method_name}...")
                if method_func(url, temp_file, timeout):
                    logger.info(
                        f"Successfully downloaded large file using {method_name}")

                    # Check if file needs extraction
                    extracted_file = self._extract_if_compressed(temp_file)
                    if extracted_file:
                        # Clean up original compressed file
                        if os.path.exists(temp_file) and temp_file != extracted_file:
                            os.remove(temp_file)
                        return extracted_file
                    else:
                        return temp_file
                else:
                    logger.warning(f"Method {method_name} failed")
            except Exception as e:
                logger.warning(f"Method {method_name} failed with error: {e}")
                continue

        # All methods failed
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return None

    def _download_with_requests_stream(self, url: str, destination: str, timeout: int) -> bool:
        """Download using requests with stream=True and shutil.copyfileobj"""
        try:
            response = self.session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            with open(destination, 'wb') as out_file:
                # Use shutil.copyfileobj for efficient streaming
                shutil.copyfileobj(response.raw, out_file)
            return True
        except Exception as e:
            logger.debug(f"Requests stream method failed: {e}")
            return False

    def _download_with_requests_raw(self, url: str, destination: str, timeout: int) -> bool:
        """Download using requests with manual chunking and SSL disabled"""
        try:
            # Use a fresh session with SSL disabled for problematic endpoints
            session = requests.Session()
            session.verify = False
            session.headers.update(self.session.headers)

            response = session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            with open(destination, 'wb') as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        out_file.write(chunk)
            return True
        except Exception as e:
            logger.debug(f"Requests raw method failed: {e}")
            return False

    def _download_with_curl(self, url: str, destination: str, timeout: int) -> bool:
        """Download using curl as ultimate fallback"""
        return self._fallback_curl_request(url, destination) is not None

    def _process_streaming_csv(self, response, dataset: Dict) -> List[Dict]:
        """Process large CSV files using streaming to avoid memory issues"""
        entities = []
        header_read = False
        headers = []

        logger.info("Processing large CSV file with streaming...")

        # Read the response line by line
        for line_num, line in enumerate(response.iter_lines(decode_unicode=True)):
            if line_num % 50000 == 0 and line_num > 0:
                logger.info(
                    f"Processed {line_num:,} lines, extracted {len(entities):,} entities...")

            if not line.strip():
                continue

            if not header_read:
                # First line contains headers
                headers = [h.strip('"') for h in line.split(',')]
                header_read = True
                continue

            try:
                # Parse CSV line manually
                values = [v.strip('"') for v in line.split(',')]
                if len(values) != len(headers):
                    continue  # Skip malformed lines

                row = dict(zip(headers, values))
                entity = self._create_entity_from_csv_row(row, dataset)

                if entity['name']:
                    entities.append(entity)

            except Exception as e:
                # Skip problematic lines
                logger.debug(f"Skipping line {line_num}: {e}")
                continue

        return entities

    def _process_downloaded_csv_file(self, file_path: str, dataset: Dict) -> List[Dict]:
        """Process a downloaded CSV file"""
        entities = []

        logger.info(f"Processing downloaded CSV file: {file_path}")

        try:
            # File should already be extracted as plain CSV at this point
            with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
                csv_reader = csv.DictReader(csvfile)

                for line_num, row in enumerate(csv_reader):
                    if line_num % 50000 == 0 and line_num > 0:
                        logger.info(
                            f"Processed {line_num:,} lines, extracted {len(entities):,} entities...")

                    entity = self._create_entity_from_csv_row(row, dataset)
                    if entity['name']:
                        entities.append(entity)

            logger.info(
                f"Successfully processed {len(entities)} entities from downloaded file")
            return entities

        except UnicodeDecodeError:
            # Try different encodings if UTF-8 fails
            logger.warning("UTF-8 decoding failed, trying latin-1...")
            try:
                with open(file_path, 'r', encoding='latin-1', newline='') as csvfile:
                    csv_reader = csv.DictReader(csvfile)
                    for line_num, row in enumerate(csv_reader):
                        if line_num % 50000 == 0 and line_num > 0:
                            logger.info(
                                f"Processed {line_num:,} lines, extracted {len(entities):,} entities...")
                        entity = self._create_entity_from_csv_row(row, dataset)
                        if entity['name']:
                            entities.append(entity)
                logger.info(
                    f"Successfully processed {len(entities)} entities with latin-1 encoding")
                return entities
            except Exception as e2:
                logger.error(f"Failed with latin-1 encoding too: {e2}")
                return []
        except Exception as e:
            logger.error(f"Failed to process downloaded CSV file: {e}")
            return []
        finally:
            # Clean up the temporary file
            try:
                import os
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass

    def _create_entity_from_csv_row(self, row: Dict, dataset: Dict) -> Dict:
        """Create standardized entity dict from CSV row"""
        return {
            'name': row.get('name', '').strip(),
            'source': 'opensanctions',
            'dataset': dataset.get('name', ''),
            'dataset_title': dataset.get('title', ''),
            'entity_type': row.get('schema', ''),
            'entity_id': row.get('id', ''),
            'alias': self._extract_aliases_from_csv(row),
            'country': row.get('country', ''),
            'birth_date': row.get('birth_date', ''),
            'nationality': row.get('nationality', ''),
            'position': row.get('position', ''),
            'registration_number': row.get('registration_number', ''),
            'topics': row.get('topics', ''),
            'sanction_program': row.get('program', ''),
            'description': row.get('description', '').strip(),
            'last_updated': dataset.get('updated_at', ''),
            'source_url': row.get('source_url', ''),
            'legal_form': row.get('legal_form', ''),
            'incorporation_date': row.get('incorporation_date', ''),
            'address': row.get('address', ''),
            'phone': row.get('phone', ''),
            'email': row.get('email', ''),
            'website': row.get('website', '')
        }

    def _extract_aliases_from_csv(self, row: Dict) -> str:
        """Extract aliases from CSV row (may have multiple alias columns)"""
        aliases = []

        # Check for main alias field
        if row.get('alias'):
            aliases.append(row['alias'])

        # Check for numbered alias fields (alias_1, alias_2, etc.)
        for i in range(1, 10):  # Check up to alias_9
            alias_field = f'alias_{i}'
            if row.get(alias_field):
                aliases.append(row[alias_field])

        return '; '.join(filter(None, aliases))

    def extract_from_ftm_json(self, dataset: Dict, json_url: str) -> List[Dict]:
        """Extract entities from entities.ftm.json format"""
        logger.info(f"Downloading FTM JSON from {json_url}")
        entities = []

        try:
            response = self._make_request_with_fallback(json_url, timeout=60)

            # Parse JSON Lines format
            for line in response.text.strip().split('\n'):
                if not line.strip():
                    continue

                try:
                    ftm_entity = json.loads(line)
                    properties = ftm_entity.get('properties', {})

                    # Extract name (can be multiple)
                    names = properties.get('name', [])
                    if not names:
                        continue

                    primary_name = names[0] if names else ''
                    aliases = '; '.join(names[1:]) if len(names) > 1 else ''

                    # Add other aliases
                    other_aliases = []
                    for alias_field in ['alias', 'previousName', 'weakAlias']:
                        if alias_field in properties:
                            other_aliases.extend(properties[alias_field])

                    if other_aliases:
                        aliases = '; '.join(
                            [aliases] + other_aliases) if aliases else '; '.join(other_aliases)

                    entity = {
                        'name': primary_name,
                        'source': 'opensanctions',
                        'dataset': dataset.get('name', ''),
                        'dataset_title': dataset.get('title', ''),
                        'entity_type': ftm_entity.get('schema', ''),
                        'entity_id': ftm_entity.get('id', ''),
                        'alias': aliases,
                        'country': '; '.join(properties.get('country', [])),
                        'birth_date': '; '.join(properties.get('birthDate', [])),
                        'nationality': '; '.join(properties.get('nationality', [])),
                        'position': '; '.join(properties.get('position', [])),
                        'registration_number': '; '.join(properties.get('registrationNumber', [])),
                        'topics': '; '.join(properties.get('topics', [])),
                        'sanction_program': '; '.join(properties.get('program', [])),
                        'description': '; '.join(properties.get('description', [])),
                        'last_updated': dataset.get('updated_at', ''),
                        'source_url': '; '.join(properties.get('sourceUrl', [])),
                        'legal_form': '; '.join(properties.get('legalForm', [])),
                        'incorporation_date': '; '.join(properties.get('incorporationDate', [])),
                        'address': '; '.join(properties.get('address', [])),
                        'phone': '; '.join(properties.get('phone', [])),
                        'email': '; '.join(properties.get('email', [])),
                        'website': '; '.join(properties.get('website', []))
                    }

                    if entity['name']:
                        entities.append(entity)

                except json.JSONDecodeError:
                    continue

            logger.info(f"Extracted {len(entities)} entities from FTM JSON")
            return entities

        except Exception as e:
            logger.error(f"Failed to process FTM JSON {json_url}: {e}")
            return []

    def extract_from_consolidated_csv(self) -> List[Dict]:
        """Extract from the consolidated statements.csv file"""
        statements_url = "https://data.opensanctions.org/datasets/latest/default/statements.csv"
        logger.info(
            f"Downloading consolidated statements from {statements_url}")
        logger.warning(
            "This is a very large file (9+ GB). Processing may take significant time.")

        entities = []
        entity_dict = {}  # To group statements by entity

        try:
            response = self.session.get(
                statements_url, timeout=300, stream=True)
            response.raise_for_status()

            # Process CSV in chunks to handle large file
            csv_reader = csv.DictReader(
                response.iter_lines(decode_unicode=True))

            row_count = 0
            for row in csv_reader:
                row_count += 1
                if row_count % 100000 == 0:
                    logger.info(
                        f"Processed {row_count} statements, found {len(entity_dict)} unique entities")

                entity_id = row.get('entity_id', '')
                if not entity_id:
                    continue

                # Initialize entity if not seen before
                if entity_id not in entity_dict:
                    entity_dict[entity_id] = {
                        'name': '',
                        'source': 'opensanctions',
                        'dataset': 'default',
                        'dataset_title': 'OpenSanctions Default',
                        'entity_type': row.get('schema', ''),
                        'entity_id': entity_id,
                        'alias': set(),
                        'country': set(),
                        'birth_date': set(),
                        'nationality': set(),
                        'position': set(),
                        'registration_number': set(),
                        'topics': set(),
                        'sanction_program': set(),
                        'description': set(),
                        'last_updated': '',
                        'source_url': set(),
                        'legal_form': set(),
                        'incorporation_date': set(),
                        'address': set(),
                        'phone': set(),
                        'email': set(),
                        'website': set()
                    }

                entity = entity_dict[entity_id]
                prop_name = row.get('prop', '')
                prop_value = row.get('value', '').strip()

                if not prop_value:
                    continue

                # Map properties to our schema
                if prop_name == 'name' and not entity['name']:
                    entity['name'] = prop_value
                elif prop_name in ['alias', 'previousName', 'weakAlias']:
                    entity['alias'].add(prop_value)
                elif prop_name == 'country':
                    entity['country'].add(prop_value)
                elif prop_name == 'birthDate':
                    entity['birth_date'].add(prop_value)
                elif prop_name == 'nationality':
                    entity['nationality'].add(prop_value)
                elif prop_name == 'position':
                    entity['position'].add(prop_value)
                elif prop_name == 'registrationNumber':
                    entity['registration_number'].add(prop_value)
                elif prop_name == 'topics':
                    entity['topics'].add(prop_value)
                elif prop_name == 'program':
                    entity['sanction_program'].add(prop_value)
                elif prop_name == 'description':
                    entity['description'].add(prop_value)
                elif prop_name == 'sourceUrl':
                    entity['source_url'].add(prop_value)
                elif prop_name == 'legalForm':
                    entity['legal_form'].add(prop_value)
                elif prop_name == 'incorporationDate':
                    entity['incorporation_date'].add(prop_value)
                elif prop_name == 'address':
                    entity['address'].add(prop_value)
                elif prop_name == 'phone':
                    entity['phone'].add(prop_value)
                elif prop_name == 'email':
                    entity['email'].add(prop_value)
                elif prop_name == 'website':
                    entity['website'].add(prop_value)

            # Convert sets to strings and clean up
            for entity_id, entity in entity_dict.items():
                if entity['name']:  # Only include entities with names
                    # Convert sets to semicolon-separated strings
                    for field in ['alias', 'country', 'birth_date', 'nationality', 'position',
                                  'registration_number', 'topics', 'sanction_program', 'description',
                                  'source_url', 'legal_form', 'incorporation_date', 'address',
                                  'phone', 'email', 'website']:
                        if isinstance(entity[field], set):
                            entity[field] = '; '.join(
                                sorted(filter(None, entity[field])))

                    entities.append(entity)

            logger.info(
                f"Extracted {len(entities)} unique entities from consolidated CSV")
            return entities

        except Exception as e:
            logger.error(f"Failed to process consolidated CSV: {e}")
            return []

    def extract_dataset_entities(self, dataset: Dict) -> List[Dict]:
        """Extract entities from a single dataset"""
        dataset_name = dataset.get('name', 'unknown')
        logger.info(
            f"Processing dataset: {dataset_name} ({dataset.get('title', 'No title')})")

        # Check if this is consolidated mode
        if dataset.get('use_consolidated'):
            return self.extract_from_consolidated_csv()

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

    def save_to_csv(self, all_entities: List[Dict]):
        """Save all entities to CSV file"""
        logger.info(
            f"Saving {len(all_entities)} entities to {self.output_file}")

        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_columns)
                writer.writeheader()

                for entity in all_entities:
                    # Ensure all required columns exist
                    clean_entity = {}
                    for col in self.csv_columns:
                        clean_entity[col] = entity.get(col, '')
                    writer.writerow(clean_entity)

            logger.info(f"Successfully saved entities to {self.output_file}")

        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            raise

    # type: ignore
    # type: ignore
    # type: ignore
    # type: ignore
    # type: ignore
    def run_extraction(self, mode: str, specific_datasets: Optional[List[str]] = None, max_datasets: Optional[int] = None):
        """Main extraction process"""
        start_time = time.time()

        logger.info(f"Starting OpenSanctions extraction in '{mode}' mode")

        # Get target datasets
        target_datasets = self.get_target_datasets(mode, specific_datasets)

        if max_datasets:
            target_datasets = target_datasets[:max_datasets]

        logger.info(f"Processing {len(target_datasets)} datasets")

        all_entities = []

        for i, dataset in enumerate(target_datasets, 1):
            dataset_name = dataset.get('name', 'unknown')
            logger.info(
                f"[{i}/{len(target_datasets)}] Processing {dataset_name}")

            try:
                entities = self.extract_dataset_entities(dataset)
                all_entities.extend(entities)
                logger.info(f"Total entities so far: {len(all_entities)}")

                # Add small delay to be respectful to the server
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_name}: {e}")
                continue

        # Save results
        if all_entities:
            self.save_to_csv(all_entities)

            # Print summary
            elapsed = time.time() - start_time
            logger.info(f"\n=== EXTRACTION COMPLETE ===")
            logger.info(f"Total entities extracted: {len(all_entities):,}")
            logger.info(f"Datasets processed: {len(target_datasets)}")
            logger.info(f"Time taken: {elapsed:.1f} seconds")
            logger.info(f"Output file: {self.output_file}")

            # Entity type breakdown
            entity_types = {}
            for entity in all_entities:
                entity_type = entity.get('entity_type', 'Unknown')
                entity_types[entity_type] = entity_types.get(
                    entity_type, 0) + 1

            logger.info(f"\nEntity type breakdown:")
            for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  {entity_type}: {count:,}")
        else:
            logger.warning("No entities extracted!")

    def _fallback_curl_request(self, url: str, output_file: Optional[str] = None) -> Optional[str]:
        """Fallback method using curl when Python requests fail due to SSL issues"""
        try:
            import subprocess
            import tempfile
            import os

            logger.info(
                f"Attempting to download {url} using curl as fallback...")

            if output_file is None:
                # Determine appropriate file extension from URL
                url_lower = url.lower()
                if '.zip' in url_lower:
                    output_file = tempfile.mktemp(suffix='.zip')
                elif '.gz' in url_lower:
                    output_file = tempfile.mktemp(suffix='.gz')
                elif '.bz2' in url_lower:
                    output_file = tempfile.mktemp(suffix='.bz2')
                else:
                    output_file = tempfile.mktemp(suffix='.csv')

            # Check if curl is available
            if not shutil.which('curl'):
                logger.warning("Curl is not available on this system")
                return None

            # Use curl with SSL options that are more permissive
            curl_cmd = [
                'curl',
                '-L',  # Follow redirects
                '--retry', '3',  # Retry on failure
                '--retry-delay', '2',  # Delay between retries
                '--connect-timeout', '30',  # Connection timeout
                '--max-time', '300',  # Max total time
                '--insecure',  # Ignore SSL certificate errors
                '--tlsv1.2',  # Use TLS 1.2
                '--user-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                '--progress-bar',  # Show progress
                '--compressed',  # Handle compressed responses
                '-o', output_file,  # Output to file
                url
            ]

            result = subprocess.run(
                curl_cmd, capture_output=False, text=True, timeout=320)

            if result.returncode == 0 and os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                logger.info(
                    f"Successfully downloaded using curl ({file_size:,} bytes)")
                return output_file
            else:
                logger.error(
                    f"Curl failed with return code {result.returncode}")
                return None

        except subprocess.TimeoutExpired:  # type: ignore
            logger.error("Curl command timed out")
            return None
        except Exception as e:
            logger.error(f"Fallback curl method failed: {e}")
            return None

    def _make_request_with_fallback(self, url: str, timeout: int = 60, max_retries: int = 3, stream: bool = False):
        """Make HTTP request with curl fallback for SSL issues"""
        try:
            # First try the normal request method
            return self._make_request(url, timeout, max_retries, stream)
        except requests.exceptions.SSLError as e:
            logger.warning(
                f"All Python request methods failed with SSL error: {e}")
            logger.info("Attempting curl fallback...")

            # Try curl fallback
            temp_file = self._fallback_curl_request(url)
            if temp_file:
                # Create a mock response object
                class MockResponse:
                    def __init__(self, file_path):
                        self.file_path = file_path
                        with open(file_path, 'r', encoding='utf-8') as f:
                            self.text = f.read()
                        self.status_code = 200

                    def json(self):
                        import json
                        return json.loads(self.text)

                    def raise_for_status(self):
                        pass

                    def iter_lines(self, decode_unicode=True):
                        return self.text.splitlines()

                return MockResponse(temp_file)
            else:
                raise e
        except Exception as e:
            # For non-SSL errors, just re-raise
            raise e

    def _extract_if_compressed(self, file_path: str) -> Optional[str]:
        """Extract compressed files and return path to extracted CSV"""
        import os
        import tempfile
        import zipfile
        import gzip
        import bz2

        try:
            file_lower = file_path.lower()

            # Handle ZIP files
            if file_lower.endswith('.zip'):
                logger.info("Extracting ZIP archive...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Look for CSV files in the archive
                    csv_files = [f for f in zip_ref.namelist(
                    ) if f.lower().endswith('.csv')]
                    if not csv_files:
                        logger.warning("No CSV files found in ZIP archive")
                        return None

                    # Extract the first CSV file (or largest if multiple)
                    target_file = csv_files[0]
                    if len(csv_files) > 1:
                        # Choose the largest CSV file
                        file_sizes = [(f, zip_ref.getinfo(f).file_size)
                                      for f in csv_files]
                        target_file = max(file_sizes, key=lambda x: x[1])[0]
                        logger.info(
                            f"Multiple CSV files found, extracting largest: {target_file}")

                    # Extract to temporary location
                    extracted_path = tempfile.mktemp(suffix='.csv')
                    with zip_ref.open(target_file) as source, open(extracted_path, 'wb') as target:
                        shutil.copyfileobj(source, target)

                    logger.info(
                        f"Successfully extracted {target_file} from ZIP")
                    return extracted_path

            # Handle GZIP files
            elif file_lower.endswith('.gz'):
                logger.info("Extracting GZIP archive...")
                extracted_path = tempfile.mktemp(suffix='.csv')
                with gzip.open(file_path, 'rb') as gz_file:
                    with open(extracted_path, 'wb') as out_file:
                        shutil.copyfileobj(gz_file, out_file)
                logger.info("Successfully extracted GZIP file")
                return extracted_path

            # Handle BZIP2 files
            elif file_lower.endswith('.bz2'):
                logger.info("Extracting BZIP2 archive...")
                extracted_path = tempfile.mktemp(suffix='.csv')
                with bz2.open(file_path, 'rb') as bz2_file:
                    with open(extracted_path, 'wb') as out_file:
                        shutil.copyfileobj(bz2_file, out_file)
                logger.info("Successfully extracted BZIP2 file")
                return extracted_path

            # Not a compressed file, return as-is
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to extract compressed file {file_path}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract entities from OpenSanctions datasets')
    parser.add_argument('--mode', choices=['all', 'specific', 'consolidated'], required=True,
                        help='Extraction mode: all datasets, specific datasets, or consolidated data')
    parser.add_argument('--datasets', type=str,
                        help='Comma-separated list of dataset names (for specific mode)')
    parser.add_argument('--output', type=str, default='opensanctions_entities.csv',
                        help='Output CSV filename')
    parser.add_argument('--max-datasets', type=int,
                        help='Maximum number of datasets to process (for testing)')

    args = parser.parse_args()

    # Parse specific datasets if provided
    specific_datasets = None
    if args.datasets:
        specific_datasets = [name.strip() for name in args.datasets.split(',')]

    # Validate arguments
    if args.mode == 'specific' and not specific_datasets:
        parser.error('--datasets is required when using specific mode')

    # Create extractor and run
    extractor = OpenSanctionsExtractor(output_file=args.output)
    extractor.run_extraction(args.mode, specific_datasets,  # type: ignore
                             args.max_datasets)  # type: ignore


if __name__ == '__main__':
    main()
