"""
OpenSanctions Test Script

This script demonstrates the extraction process on a small sample of data
without downloading large datasets. Use this to test the approach.
"""

import requests
import json
import csv
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_small_dataset_extraction():
    """Test extraction on a small dataset to verify the approach works"""

    # First, let's explore the data structure of one small dataset
    index_url = "https://data.opensanctions.org/datasets/latest/index.json"

    logger.info("Fetching dataset index...")
    response = requests.get(index_url)
    data = response.json()

    # Find a small dataset for testing
    datasets = data.get('datasets', [])
    small_datasets = []

    for dataset in datasets:
        entity_count = dataset.get('entity_count', 0)
        target_count = dataset.get('target_count', 0)
        dataset_type = dataset.get('type', '')

        # Look for small source datasets with targets
        if (dataset_type == 'source' and
            target_count > 0 and
            target_count < 1000 and
                entity_count < 5000):
            small_datasets.append(dataset)

    # Sort by target count
    small_datasets.sort(key=lambda x: x.get('target_count', 0))

    logger.info(
        f"Found {len(small_datasets)} small datasets suitable for testing")

    # Show first few for inspection
    logger.info("Small datasets available for testing:")
    for i, dataset in enumerate(small_datasets[:10]):
        logger.info(
            f"{i+1:2d}. {dataset['name']}: {dataset.get('title', 'No title')}")
        logger.info(
            f"     Entities: {dataset.get('entity_count', 0):,}, Targets: {dataset.get('target_count', 0):,}")
        logger.info(f"     Tags: {', '.join(dataset.get('tags', []))}")

        # Show available resources
        resources = dataset.get('resources', [])
        logger.info(
            f"     Resources: {', '.join([r.get('name', 'Unknown') for r in resources])}")
        logger.info("")

    if not small_datasets:
        logger.warning("No suitable small datasets found for testing")
        return

    # Test with the first small dataset
    test_dataset = small_datasets[0]
    logger.info(f"Testing with dataset: {test_dataset['name']}")

    # Look for CSV resource
    resources = test_dataset.get('resources', [])
    csv_resource = None

    for resource in resources:
        if resource.get('name') == 'targets.simple.csv':
            csv_resource = resource
            break

    if csv_resource:
        logger.info(f"Found CSV resource: {csv_resource['url']}")

        # Download and parse a few rows
        try:
            response = requests.get(csv_resource['url'], timeout=30)
            response.raise_for_status()

            csv_content = response.text
            lines = csv_content.split('\n')

            logger.info(f"CSV has {len(lines)} lines")

            # Show headers
            if lines:
                headers = lines[0].split(',')
                logger.info(f"CSV Headers: {headers}")

            # Parse first few rows
            csv_reader = csv.DictReader(csv_content.splitlines())
            sample_entities = []

            for i, row in enumerate(csv_reader):
                if i >= 5:  # Just first 5 rows
                    break
                sample_entities.append(row)

            logger.info(f"\nSample entities from {test_dataset['name']}:")
            for i, entity in enumerate(sample_entities, 1):
                logger.info(f"{i}. Name: {entity.get('name', 'N/A')}")
                logger.info(f"   Type: {entity.get('schema', 'N/A')}")
                logger.info(f"   Country: {entity.get('country', 'N/A')}")
                logger.info(f"   Topics: {entity.get('topics', 'N/A')}")
                logger.info("")

        except Exception as e:
            logger.error(f"Failed to download/parse CSV: {e}")

    else:
        logger.info("No CSV resource found, checking FTM JSON...")

        # Look for FTM JSON resource
        ftm_resource = None
        for resource in resources:
            if resource.get('name') == 'entities.ftm.json':
                ftm_resource = resource
                break

        if ftm_resource:
            logger.info(f"Found FTM JSON resource: {ftm_resource['url']}")

            try:
                response = requests.get(ftm_resource['url'], timeout=30)
                response.raise_for_status()

                # Parse first few JSON lines
                lines = response.text.strip().split('\n')
                logger.info(f"FTM JSON has {len(lines)} entities")

                sample_entities = []
                for i, line in enumerate(lines[:5]):  # First 5 entities
                    if line.strip():
                        entity = json.loads(line)
                        sample_entities.append(entity)

                logger.info(
                    f"\nSample entities from {test_dataset['name']} (FTM format):")
                for i, entity in enumerate(sample_entities, 1):
                    props = entity.get('properties', {})
                    names = props.get('name', [])
                    name = names[0] if names else 'N/A'

                    logger.info(f"{i}. Name: {name}")
                    logger.info(f"   Type: {entity.get('schema', 'N/A')}")
                    logger.info(f"   ID: {entity.get('id', 'N/A')}")
                    logger.info(
                        f"   Country: {props.get('country', ['N/A'])[0] if props.get('country') else 'N/A'}")
                    logger.info(
                        f"   Topics: {', '.join(props.get('topics', []))}")
                    logger.info("")

            except Exception as e:
                logger.error(f"Failed to download/parse FTM JSON: {e}")


def show_dataset_categories():
    """Show different categories of datasets available"""

    index_url = "https://data.opensanctions.org/datasets/latest/index.json"
    response = requests.get(index_url)
    data = response.json()

    datasets = data.get('datasets', [])

    # Categorize datasets
    categories = {
        'sanctions': [],
        'pep': [],
        'crime': [],
        'debarment': [],
        'wanted': [],
        'other': []
    }

    for dataset in datasets:
        name = dataset.get('name', '').lower()
        title = dataset.get('title', '').lower()
        tags = [tag.lower() for tag in dataset.get('tags', [])]

        if any('sanction' in x for x in [name, title] + tags):
            categories['sanctions'].append(dataset)
        elif any('pep' in x for x in [name, title] + tags):
            categories['pep'].append(dataset)
        elif any(word in x for word in ['crime', 'criminal', 'terror'] for x in [name, title] + tags):
            categories['crime'].append(dataset)
        elif any('debarment' in x for x in [name, title] + tags):
            categories['debarment'].append(dataset)
        elif any('wanted' in x for x in [name, title] + tags):
            categories['wanted'].append(dataset)
        else:
            categories['other'].append(dataset)

    logger.info("Dataset categories:")
    for category, datasets_in_cat in categories.items():
        if datasets_in_cat:
            logger.info(
                f"\n{category.upper()} ({len(datasets_in_cat)} datasets):")
            for dataset in sorted(datasets_in_cat, key=lambda x: x.get('target_count', 0), reverse=True)[:5]:
                logger.info(
                    f"  - {dataset['name']}: {dataset.get('title', 'No title')}")
                logger.info(
                    f"    Targets: {dataset.get('target_count', 0):,}, Total: {dataset.get('entity_count', 0):,}")


def create_sample_csv():
    """Create a sample CSV showing the expected output format"""

    sample_data = [
        {
            'name': 'John Smith',
            'source': 'opensanctions',
            'dataset': 'us_sanctions',
            'dataset_title': 'US Sanctions List',
            'entity_type': 'Person',
            'entity_id': 'us-sanctions-john-smith-1',
            'alias': 'Johnny Smith; J. Smith',
            'country': 'US',
            'birth_date': '1980-01-15',
            'nationality': 'US',
            'position': 'CEO',
            'registration_number': '',
            'topics': 'sanction; crime.fin',
            'sanction_program': 'OFAC SDN',
            'description': 'Financial crimes related to money laundering',
            'last_updated': '2025-06-27',
            'source_url': 'https://example.com/source'
        },
        {
            'name': 'ABC Corporation Ltd',
            'source': 'opensanctions',
            'dataset': 'eu_sanctions',
            'dataset_title': 'EU Sanctions List',
            'entity_type': 'Company',
            'entity_id': 'eu-sanctions-abc-corp-1',
            'alias': 'ABC Corp; ABC Limited',
            'country': 'GB',
            'birth_date': '',
            'nationality': '',
            'position': '',
            'registration_number': '12345678',
            'topics': 'sanction; export.control',
            'sanction_program': 'EU Restrictive Measures',
            'description': 'Company involved in prohibited exports',
            'last_updated': '2025-06-26',
            'source_url': 'https://example.com/eu-source'
        }
    ]

    output_file = 'sample_opensanctions_output.csv'

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'name', 'source', 'dataset', 'dataset_title', 'entity_type', 'entity_id',
            'alias', 'country', 'birth_date', 'nationality', 'position',
            'registration_number', 'topics', 'sanction_program', 'description',
            'last_updated', 'source_url'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_data)

    logger.info(f"Created sample output file: {output_file}")
    logger.info("This shows the expected format of the extracted data.")


if __name__ == '__main__':
    logger.info("=== OpenSanctions Test Script ===")
    logger.info(
        "This script demonstrates the data extraction approach without downloading large files.\n")

    # Show dataset categories
    show_dataset_categories()

    logger.info("\n" + "="*50 + "\n")

    # Test on small dataset
    test_small_dataset_extraction()

    logger.info("\n" + "="*50 + "\n")

    # Create sample output
    create_sample_csv()

    logger.info("\n=== Test Complete ===")
    logger.info(
        "The full extractor script 'opensanctions_extractor.py' is ready to use.")
    logger.info("\nUsage examples:")
    logger.info("1. Test with a few small datasets:")
    logger.info(
        "   python opensanctions_extractor.py --mode all --max-datasets 5")
    logger.info("")
    logger.info("2. Extract specific datasets:")
    logger.info(
        "   python opensanctions_extractor.py --mode specific --datasets 'us_ofac_sdn,eu_sanctions'")
    logger.info("")
    logger.info("3. Use consolidated data (recommended for full dataset):")
    logger.info("   python opensanctions_extractor.py --mode consolidated")
