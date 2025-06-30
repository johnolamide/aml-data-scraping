#!/usr/bin/env python3
"""
Test script to verify dynamic table name generation from CSV filenames
"""

from csv_to_postgresql import CSVToPostgreSQLLoader
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_table_name_generation():
    """Test the table name generation from various CSV filenames"""

    # Create a mock database config
    mock_config = {
        "host": "localhost",
        "database": "test_db",
        "user": "test_user",
        "password": "test_pass",
        "port": 5432
    }

    # Initialize loader
    loader = CSVToPostgreSQLLoader(mock_config)

    # Test cases: CSV filename -> expected table name
    test_cases = [
        ("opensanctions_entities.csv", "opensanctions_entities"),
        ("us-ofac-sanctions.csv", "us_ofac_sanctions"),
        ("EU-Sanctions_List.csv", "eu_sanctions_list"),
        ("sample_output.csv", "sample_output"),
        ("complex-name_with.special@chars#.csv",
         "complex_name_with_special_chars_"),
        ("123_starts_with_number.csv", "tbl_123_starts_with_number"),
        ("_.csv", "opensanctions_entities"),  # fallback case
        ("very_long_filename_that_exceeds_postgresql_identifier_length_limit_of_63.csv",
         "very_long_filename_that_exceeds_postgresql_identifier_length_li")
    ]

    print("Testing table name generation from CSV filenames:")
    print("=" * 60)

    for csv_file, expected in test_cases:
        actual = loader._generate_table_name_from_csv(csv_file)
        status = "✓" if actual == expected else "✗"
        print(f"{status} {csv_file:40} -> {actual:30} (expected: {expected})")

        if actual != expected:
            print(f"  Expected: {expected}")
            print(f"  Got:      {actual}")

    print("\nTesting set_table_name_from_csv method:")
    print("-" * 40)

    # Test the method that actually sets the table name
    test_filename = "test_sanctions_data.csv"
    loader.set_table_name_from_csv(test_filename)
    print(f"CSV file: {test_filename}")
    print(f"Set table name: {loader.table_name}")
    print(f"Expected: test_sanctions_data")

    if loader.table_name == "test_sanctions_data":
        print("✓ Table name set correctly!")
    else:
        print("✗ Table name not set as expected")


if __name__ == "__main__":
    test_table_name_generation()
