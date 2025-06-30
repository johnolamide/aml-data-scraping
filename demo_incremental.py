#!/usr/bin/env python3
"""
Demo script for incremental CSV loading functionality

This script demonstrates how to use the incremental loading feature
of csv_to_postgresql.py to handle updated CSV files.
"""

import os
import csv
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_csv(filename: str, num_records: int, start_id: int = 1) -> str:
    """Create a sample CSV file with test data"""
    
    # Standard columns from the extractor
    columns = [
        'name', 'source', 'dataset', 'dataset_title', 'entity_type', 'entity_id',
        'alias', 'country', 'birth_date', 'nationality', 'position', 'registration_number',
        'topics', 'sanction_program', 'description', 'last_updated', 'source_url',
        'legal_form', 'incorporation_date', 'address', 'phone', 'email', 'website'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        
        for i in range(start_id, start_id + num_records):
            row = {
                'name': f'Test Entity {i}',
                'source': 'opensanctions',
                'dataset': 'demo_dataset',
                'dataset_title': 'Demo Dataset for Testing',
                'entity_type': 'Person' if i % 2 == 0 else 'Organization',
                'entity_id': f'demo-entity-{i:05d}',
                'alias': f'Alias {i}',
                'country': 'US' if i % 3 == 0 else 'UK',
                'birth_date': '1970-01-01' if i % 2 == 0 else '',
                'nationality': 'American' if i % 3 == 0 else 'British',
                'position': f'Position {i}' if i % 2 == 0 else '',
                'registration_number': f'REG{i:05d}' if i % 2 == 1 else '',
                'topics': 'sanctions',
                'sanction_program': 'Demo Sanctions',
                'description': f'Demo entity {i} for testing incremental loading',
                'last_updated': '2025-06-30',
                'source_url': f'https://example.com/entity/{i}',
                'legal_form': 'Corporation' if i % 2 == 1 else '',
                'incorporation_date': '2020-01-01' if i % 2 == 1 else '',
                'address': f'{i} Demo Street, Test City',
                'phone': f'+1-555-{i:04d}',
                'email': f'entity{i}@example.com',
                'website': f'https://entity{i}.example.com'
            }
            writer.writerow(row)
    
    logger.info(f"Created {filename} with {num_records} records (IDs {start_id} to {start_id + num_records - 1})")
    return filename

def demo_incremental_loading():
    """Demonstrate incremental loading functionality"""
    
    logger.info("🔄 Incremental Loading Demo")
    logger.info("=" * 50)
    
    # Step 1: Create initial CSV with 100 records
    logger.info("Step 1: Creating initial CSV file with 100 records...")
    initial_csv = create_sample_csv("demo_initial.csv", 100, start_id=1)
    
    # Step 2: Create updated CSV with 102 records (2 new ones)
    logger.info("Step 2: Creating updated CSV file with 102 records (2 new entries)...")
    updated_csv = create_sample_csv("demo_updated.csv", 102, start_id=1)
    
    logger.info("\n📋 Usage Instructions:")
    logger.info("=" * 50)
    
    print("\n1. First, load the initial data (100 records):")
    print("   python csv_to_postgresql.py --csv-file demo_initial.csv --db-config db_config.json --drop-table")
    
    print("\n2. Then, load the updated data incrementally (only 2 new records should be added):")
    print("   python csv_to_postgresql.py --csv-file demo_updated.csv --db-config db_config.json --incremental")
    
    print("\n3. Verify the results:")
    print("   - First load should insert 100 records")
    print("   - Second load should insert only 2 new records (IDs demo-entity-00101 and demo-entity-00102)")
    print("   - Total records in database should be 102")
    
    logger.info("\n📊 Expected Results:")
    logger.info("Initial load: 100 records inserted")
    logger.info("Incremental load: 2 new records inserted, 100 existing records skipped")
    logger.info("Final total: 102 records in database")
    
    # Step 3: Show the difference between the files
    logger.info("\n🔍 File Comparison:")
    logger.info(f"Initial CSV: {os.path.getsize(initial_csv)} bytes")
    logger.info(f"Updated CSV: {os.path.getsize(updated_csv)} bytes")
    
    # Read last few lines of each file to show the difference
    with open(initial_csv, 'r') as f:
        initial_lines = f.readlines()
    
    with open(updated_csv, 'r') as f:
        updated_lines = f.readlines()
    
    logger.info(f"Initial CSV has {len(initial_lines)-1} data rows")  # -1 for header
    logger.info(f"Updated CSV has {len(updated_lines)-1} data rows")
    
    if len(updated_lines) > len(initial_lines):
        logger.info("\n🆕 New entries in updated CSV:")
        for line in updated_lines[len(initial_lines):]:
            parts = line.strip().split(',')
            if len(parts) > 5:  # Ensure we have enough fields
                logger.info(f"   - Entity ID: {parts[5]}, Name: {parts[0]}")
    
    logger.info("\n✅ Demo files created successfully!")
    logger.info("Follow the usage instructions above to test incremental loading.")

if __name__ == "__main__":
    demo_incremental_loading()
