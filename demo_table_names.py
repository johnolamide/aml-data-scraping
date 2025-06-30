#!/usr/bin/env python3
"""
Demo script showing dynamic table name generation from CSV filenames

This script demonstrates how the PostgreSQL loader automatically sets
the table name based on the CSV filename, making it easier to load
multiple different datasets into appropriately named tables.
"""

from csv_to_postgresql import CSVToPostgreSQLLoader
import os
import json
import logging
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules


def create_sample_csv_files():
    """Create sample CSV files with different names to demonstrate table naming"""

    # Sample CSV data
    csv_data = """name,source,dataset,dataset_title,entity_type,entity_id,alias,country,birth_date,nationality,position,registration_number,topics,sanction_program,description,last_updated,source_url,legal_form,incorporation_date,address,phone,email,website
John Doe,OpenSanctions,us-ofac-sdn,US OFAC SDN List,Person,P001,Johnny,USA,1970-01-01,American,CEO,,crime.terror,SDGT,Suspected terrorist,2024-01-01,https://example.com,,,,123 Main St,555-1234,john@example.com,
Evil Corp,OpenSanctions,eu-sanctions,EU Sanctions List,Organization,O001,Bad Company,Russia,,Russian,,,crime.sanctions,EU-SANCTIONS,Sanctioned entity,2024-01-01,https://example.com,LLC,2000-01-01,456 Bad St,555-5678,info@evilcorp.com,www.evilcorp.com"""

    # Create different CSV files to show table naming
    test_files = [
        "us_ofac_sanctions.csv",
        "eu-consolidated-sanctions.csv",
        "UK_HMT_Sanctions_List.csv",
        "sample@data#with$special.csv",
        "123_numeric_start.csv"
    ]

    created_files = []

    for filename in test_files:
        with open(filename, 'w', newline='') as f:
            f.write(csv_data)
        created_files.append(filename)
        logger.info(f"Created sample CSV file: {filename}")

    return created_files


def demo_table_name_generation():
    """Demonstrate automatic table name generation"""

    # Mock database configuration (won't actually connect)
    mock_config = {
        "host": "localhost",
        "database": "demo_db",
        "user": "demo_user",
        "password": "demo_pass",
        "port": 5432
    }

    print("\n" + "="*70)
    print("DEMO: Dynamic PostgreSQL Table Name Generation")
    print("="*70)

    # Create sample CSV files
    csv_files = create_sample_csv_files()

    try:
        print(
            f"\nCreated {len(csv_files)} sample CSV files for demonstration:")

        # Show how table names are generated for each file
        loader = CSVToPostgreSQLLoader(mock_config)

        print(f"\n{'CSV Filename':<35} {'Generated Table Name':<30}")
        print("-" * 65)

        for csv_file in csv_files:
            # Generate table name without connecting to database
            table_name = loader._generate_table_name_from_csv(csv_file)
            print(f"{csv_file:<35} {table_name:<30}")

        print(f"\n" + "="*70)
        print("Key features of table name generation:")
        print("- Converts filename to lowercase")
        print("- Replaces special characters with underscores")
        print("- Ensures names start with letter or underscore")
        print("- Limits length to PostgreSQL's 63-character limit")
        print("- Handles edge cases gracefully")

        print(f"\nIn actual usage:")
        print("1. When loading a CSV, the table name is automatically set")
        print("2. The table is created with this name if it doesn't exist")
        print("3. Data is loaded into the appropriately named table")

        # Show how this works with the actual loader methods
        print(f"\n" + "-"*50)
        print("Example usage in code:")
        print("-"*50)

        sample_file = csv_files[0]
        print(f"""
# Initialize loader
loader = CSVToPostgreSQLLoader(db_config)

# Load CSV - table name automatically set from filename
loader.connect()
loader.load_csv_file('{sample_file}')  # Creates table '{loader._generate_table_name_from_csv(sample_file)}'
loader.disconnect()

# For incremental updates to the same dataset
loader.connect() 
loader.load_csv_file_incremental('{sample_file}')  # Uses same table name
loader.disconnect()
""")

    finally:
        # Clean up sample files
        for csv_file in csv_files:
            try:
                os.remove(csv_file)
                logger.info(f"Cleaned up: {csv_file}")
            except OSError:
                pass

    print("="*70)
    print("Demo completed successfully!")
    print("="*70)


if __name__ == "__main__":
    demo_table_name_generation()
