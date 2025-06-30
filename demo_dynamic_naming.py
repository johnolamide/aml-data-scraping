#!/usr/bin/env python3
"""
Demo: Dynamic Table Naming with CSV to PostgreSQL Loader

This demo shows how the csv_to_postgresql.py loader automatically creates
PostgreSQL tables with names derived from CSV filenames.

Example:
- sterling_aml_sanction_list.csv → sterling_aml_sanction_list (table)
- us-ofac-sdn.csv → us_ofac_sdn (table)
- EU@Sanctions#List.csv → eu_sanctions_list (table)
"""

import json
import logging
import os
import tempfile

from csv_to_postgresql import CSVToPostgreSQLLoader

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_dynamic_table_naming():
    """Demonstrate dynamic table naming feature"""

    print("\n" + "="*70)
    print("DEMO: Dynamic PostgreSQL Table Naming from CSV Files")
    print("="*70)

    # Sample CSV content
    sample_data = """name,source,dataset,dataset_title,entity_type,entity_id,alias,country,birth_date,nationality,position,registration_number,topics,sanction_program,description,last_updated,source_url,legal_form,incorporation_date,address,phone,email,website
John Smith,OpenSanctions,sterling-aml,Sterling AML Data,Person,P001,J.Smith,USA,1980-01-01,American,Manager,,crime.sanctions,AML-SANCTIONS,High-risk individual,2024-01-01,https://example.com,,,,123 Main St,555-0123,john@example.com,
Sterling Corp,OpenSanctions,sterling-aml,Sterling AML Data,Organization,O001,Sterling Company,UK,,British,,,crime.financial,AML-ENTITY,Financial services firm,2024-01-01,https://example.com,Ltd,1995-01-01,456 London St,44-20-1234,info@sterling.com,www.sterling.com"""

    # Create sample CSV files with different naming patterns
    test_files = [
        "sterling_aml_sanction_list.csv",
        "us-ofac-sdn-list.csv",
        "EU@Sanctions#List.csv",
        "UK_HMT_Consolidated_List.csv",
        "123_numeric_dataset.csv",
        "complex-name_with.special@chars#.csv"
    ]

    # Mock database config (won't actually connect)
    mock_config = {
        "host": "localhost",
        "database": "demo_db",
        "user": "demo_user",
        "password": "demo_pass",
        "port": 5432
    }

    print("\nCreating sample CSV files and showing generated table names:\n")

    created_files = []
    try:
        # Create sample files and show table naming
        for filename in test_files:
            # Create the CSV file
            with open(filename, 'w') as f:
                f.write(sample_data)
            created_files.append(filename)

            # Show what table name would be generated
            loader = CSVToPostgreSQLLoader(mock_config)
            table_name = loader._generate_table_name_from_csv(filename)

            print(f"CSV File: {filename:<35}")
            print(f"Table:    {table_name:<35}")
            print("-" * 50)

        print(f"\n✅ Dynamic table naming ensures:")
        print("   • Each CSV file gets its own appropriately named table")
        print("   • Names are PostgreSQL-compliant (lowercase, underscores)")
        print("   • Special characters are handled gracefully")
        print("   • Length limits are respected (63 char max)")
        print("   • Invalid names get proper fallbacks")

        print(f"\n📝 Usage in practice:")
        print(
            "   1. Extract data: python opensanctions_extractor.py --output my_dataset.csv")
        print("   2. Load to PostgreSQL: python csv_to_postgresql.py --csv-file my_dataset.csv --db-config config.json")
        print("   3. Query table: SELECT * FROM my_dataset;")

        print(f"\n🔄 For incremental updates:")
        print("   1. Extract updated data: python opensanctions_extractor.py --output my_dataset_update.csv")
        print("   2. Load incrementally: python csv_to_postgresql.py --csv-file my_dataset_update.csv --db-config config.json --incremental")
        print("   3. Only new entities are added to the my_dataset_update table")

    finally:
        # Clean up created files
        for filename in created_files:
            try:
                os.remove(filename)
            except OSError:
                pass

    print("\n" + "="*70)
    print("Dynamic table naming feature is ready for use!")
    print("="*70)


if __name__ == "__main__":
    demo_dynamic_table_naming()
