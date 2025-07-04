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
import tempfile
import os
import subprocess
import zipfile
import gzip
import bz2
import re

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

        # Dynamic CSV columns - will be determined from actual data
        self.csv_columns = set()  # Use set to collect unique columns
        self.all_entities = []  # Store all entities to analyze schema later

        # Column mapping for ID standardization
        self.id_column_mappings = {
            'id': 'entity_id',
            'entityid': 'entity_id',
            'entity_id': 'entity_id',  # Keep as is
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

    def _fallback_curl_request(self, url: str, output_file: Optional[str] = None) -> Optional[str]:
        """Fallback method using curl when Python requests fail due to SSL issues"""
        try:
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
                '--user-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
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

        except subprocess.TimeoutExpired:
            logger.error("Curl command timed out")
            return None
        except Exception as e:
            logger.error(f"Fallback curl method failed: {e}")
            return None

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
            # Check dataset size
            target_count = dataset.get('target_count', 0)

            if target_count > 50000:
                logger.info(
                    f"Large dataset detected ({target_count:,} entities), using robust download...")

                # Download large file
                downloaded_file = self._download_large_file(
                    csv_url, timeout=300)
                if downloaded_file:
                    return self._process_downloaded_csv_file(downloaded_file, dataset)
                else:
                    logger.error("All download methods failed for large file")
                    return []
            else:
                # For smaller datasets, use regular method
                response = self._make_request_with_fallback(
                    csv_url, timeout=60, stream=False)

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

    def _download_large_file(self, url: str, timeout: int = 300) -> Optional[str]:
        """Download large files with robust error handling"""
        # Create temporary file
        temp_file = tempfile.mktemp(suffix='.csv')

        # Try multiple download methods
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
                        # Clean up original file
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
        """Download using requests with stream=True"""
        try:
            response = self.session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            with open(destination, 'wb') as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        out_file.write(chunk)

            # Verify file was downloaded
            if os.path.exists(destination) and os.path.getsize(destination) > 0:
                logger.info(
                    f"Downloaded {os.path.getsize(destination)} bytes to {destination}")
                return True
            else:
                logger.error("Downloaded file is empty or doesn't exist")
                return False

        except Exception as e:
            logger.debug(f"Requests stream method failed: {e}")
            return False

    def _download_with_requests_raw(self, url: str, destination: str, timeout: int) -> bool:
        """Download using requests raw response"""
        try:
            response = self.session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            with open(destination, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)

            # Verify file was downloaded
            if os.path.exists(destination) and os.path.getsize(destination) > 0:
                logger.info(
                    f"Downloaded {os.path.getsize(destination)} bytes to {destination}")
                return True
            else:
                logger.error("Downloaded file is empty or doesn't exist")
                return False

        except Exception as e:
            logger.debug(f"Requests raw method failed: {e}")
            return False

    def _download_with_curl(self, url: str, destination: str, timeout: int) -> bool:
        """Download using curl command"""
        try:
            if not shutil.which('curl'):
                logger.debug("Curl is not available")
                return False

            curl_cmd = [
                'curl',
                '-L',  # Follow redirects
                '--retry', '3',
                '--retry-delay', '2',
                '--connect-timeout', '30',
                '--max-time', str(timeout),
                '--insecure',
                '--user-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                '--compressed',
                '-o', destination,
                url
            ]

            result = subprocess.run(
                curl_cmd, capture_output=True, text=True, timeout=timeout + 20)

            if result.returncode == 0 and os.path.exists(destination) and os.path.getsize(destination) > 0:
                logger.info(
                    f"Downloaded {os.path.getsize(destination)} bytes using curl")
                return True
            else:
                logger.debug(
                    f"Curl failed with return code {result.returncode}")
                return False

        except Exception as e:
            logger.debug(f"Curl method failed: {e}")
            return False

    def _process_downloaded_csv_file(self, file_path: str, dataset: Dict) -> List[Dict]:
        """Process a downloaded CSV file"""
        entities = []
        logger.info(f"Processing downloaded CSV file: {file_path}")

        try:
            # Check if file is compressed or has issues
            with open(file_path, 'rb') as f:
                first_bytes = f.read(100)

            # Handle null bytes
            if b'\x00' in first_bytes:
                logger.warning("File contains null bytes, cleaning...")
                cleaned_file = self._clean_null_bytes_from_file(file_path)
                if cleaned_file:
                    return self._process_downloaded_csv_file(cleaned_file, dataset)
                else:
                    logger.error("Failed to clean file")
                    return []

            # Process with multiple encodings
            return self._process_csv_file_with_encoding(file_path, dataset)

        except Exception as e:
            logger.error(f"Failed to process downloaded CSV file: {e}")
            return []
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass

    def _clean_null_bytes_from_file(self, file_path: str) -> Optional[str]:
        """Clean null bytes from a file"""
        try:
            cleaned_file = tempfile.mktemp(suffix='.csv')

            with open(file_path, 'rb') as infile, open(cleaned_file, 'wb') as outfile:
                while True:
                    chunk = infile.read(8192)
                    if not chunk:
                        break
                    # Remove null bytes
                    cleaned_chunk = chunk.replace(b'\x00', b'')
                    outfile.write(cleaned_chunk)

            # Check if cleaned file has content
            if os.path.getsize(cleaned_file) > 0:
                logger.info(
                    f"Successfully cleaned file, size: {os.path.getsize(cleaned_file)} bytes")
                return cleaned_file
            else:
                os.remove(cleaned_file)
                return None

        except Exception as e:
            logger.error(f"Failed to clean null bytes: {e}")
            return None

    def _process_csv_file_with_encoding(self, file_path: str, dataset: Dict) -> List[Dict]:
        """Process CSV file trying multiple encodings"""
        entities = []
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                logger.info(f"Trying {encoding} encoding...")

                with open(file_path, 'r', encoding=encoding, newline='', errors='ignore') as csvfile:
                    csv_reader = csv.DictReader(csvfile)

                    # Check if we can read the header
                    if not csv_reader.fieldnames:
                        logger.warning(
                            f"No headers found with {encoding} encoding")
                        continue

                    entities = []
                    for line_num, row in enumerate(csv_reader):
                        if line_num % 50000 == 0 and line_num > 0:
                            logger.info(
                                f"Processed {line_num:,} lines, extracted {len(entities):,} entities...")

                        entity = self._create_entity_from_csv_row(row, dataset)
                        if entity.get('name'):
                            entities.append(entity)

                logger.info(
                    f"Successfully processed {len(entities)} entities with {encoding} encoding")
                return entities

            except UnicodeDecodeError as e:
                logger.warning(f"Failed with {encoding} encoding: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error with {encoding} encoding: {e}")
                continue

        logger.error("Failed to process file with any encoding")
        return []

    def _extract_if_compressed(self, file_path: str) -> Optional[str]:
        """Extract compressed files and return path to extracted CSV"""
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

        # Track all columns we've seen
        self.csv_columns.update(entity.keys())  # type: ignore

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

        # Track all columns we've seen
        self.csv_columns.update(entity.keys())  # type: ignore

        return entity

    def analyze_and_finalize_schema(self):
        """Analyze collected data and finalize the CSV schema"""
        logger.info(
            f"Analyzing schema from {len(self.all_entities)} entities...")

        # Count column usage to prioritize common columns
        column_counts = {}
        for entity in self.all_entities:
            for column in entity.keys():
                column_counts[column] = column_counts.get(column, 0) + 1

        # Sort columns by usage frequency and importance
        core_columns = ['source', 'dataset', 'dataset_title',
                        'name', 'entity_id', 'entity_type']
        other_columns = [col for col in column_counts.keys()
                         if col not in core_columns]
        other_columns.sort(key=lambda x: column_counts[x], reverse=True)

        # Final column order: core columns first, then others by frequency
        self.csv_columns = []
        for col in core_columns:
            if col in column_counts:
                self.csv_columns.append(col)

        self.csv_columns.extend(other_columns)

        logger.info(f"Finalized schema with {len(self.csv_columns)} columns:")
        for i, col in enumerate(self.csv_columns[:10]):  # Show first 10
            count = column_counts.get(col, 0)
            logger.info(
                f"  {i+1}. {col}: {count:,} entities ({count/len(self.all_entities)*100:.1f}%)")

        if len(self.csv_columns) > 10:
            logger.info(f"  ... and {len(self.csv_columns) - 10} more columns")

    def save_to_csv(self, all_entities: List[Dict]):
        """Save all entities to CSV file with dynamic schema"""
        # First, analyze the data to determine the best schema
        self.all_entities = all_entities
        self.analyze_and_finalize_schema()

        logger.info(
            f"Saving {len(all_entities)} entities to {self.output_file}")
        logger.info(
            f"Using dynamic schema with {len(self.csv_columns)} columns")

        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_columns)
                writer.writeheader()

                for entity in all_entities:
                    # Create row with all columns, filling missing ones with empty string
                    clean_entity = {}
                    for col in self.csv_columns:
                        clean_entity[col] = entity.get(col, '')
                    writer.writerow(clean_entity)

            logger.info(f"Successfully saved entities to {self.output_file}")

        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            raise

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


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract entities from OpenSanctions datasets")
    parser.add_argument('--mode', choices=['all', 'specific', 'consolidated'], required=True,
                        help="Extraction mode: 'all' for all datasets, 'specific' for selected datasets, 'consolidated' for the unified dataset")
    parser.add_argument('--datasets', type=str,
                        help="Comma-separated list of dataset names (for 'specific' mode)")
    parser.add_argument('--output', type=str, default='opensanctions_entities.csv',
                        help="Output CSV file name")
    parser.add_argument('--max-datasets', type=int,
                        help="Maximum number of datasets to process (for testing)")

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'specific' and not args.datasets:
        logger.error(
            "--datasets argument is required when using 'specific' mode")
        sys.exit(1)

    # Parse dataset names
    specific_datasets = None
    if args.datasets:
        specific_datasets = [name.strip() for name in args.datasets.split(',')]

    # Create extractor and run
    extractor = OpenSanctionsExtractor(output_file=args.output)

    try:
        extractor.run_extraction(
            mode=args.mode,
            specific_datasets=specific_datasets,
            max_datasets=args.max_datasets
        )
    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
