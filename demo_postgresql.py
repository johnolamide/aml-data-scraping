#!/usr/bin/env python3
"""
Demo script showing how to use the CSV to PostgreSQL loader

This script demonstrates:
1. Extracting sample data using the OpenSanctions extractor
2. Loading the data into PostgreSQL using the CSV loader

Prerequisites:
- PostgreSQL server running
- Database created
- psycopg2-binary installed: pip install psycopg2-binary
"""

import os
import json
import logging
import subprocess
import sys

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_config():
    """Create a sample database configuration file"""
    config = {
        "host": "localhost",
        "database": "aml_test_db",
        "user": "postgres",
        "password": "postgres",
        "port": 5432
    }

    config_file = "demo_db_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Created sample config file: {config_file}")
    logger.info(
        "Please update the database credentials in this file before running the demo")
    return config_file


def extract_sample_data():
    """Extract sample data using the OpenSanctions extractor"""
    output_file = "demo_sample_entities.csv"

    logger.info("Extracting sample data from OpenSanctions...")

    try:
        # Run the extractor to get small sample datasets
        cmd = [
            sys.executable, "opensanctions_extractor.py",
            "--mode", "all",
            "--max-datasets", "3",
            "--output", output_file
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logger.info(
                    f"Successfully extracted sample data to {output_file}")
                return output_file
            else:
                logger.warning(
                    "No data was extracted, creating minimal test data...")
                return create_minimal_test_data()
        else:
            logger.error(f"Extractor failed: {result.stderr}")
            logger.info("Creating minimal test data instead...")
            return create_minimal_test_data()

    except subprocess.TimeoutExpired:
        logger.warning("Extraction timed out, creating minimal test data...")
        return create_minimal_test_data()
    except Exception as e:
        logger.error(f"Failed to run extractor: {e}")
        logger.info("Creating minimal test data instead...")
        return create_minimal_test_data()


def create_minimal_test_data():
    """Create minimal test CSV data for demo purposes"""
    output_file = "demo_minimal_entities.csv"

    # CSV header matching the expected schema
    csv_data = """name,source,dataset,dataset_title,entity_type,entity_id,alias,country,birth_date,nationality,position,registration_number,topics,sanction_program,description,last_updated,source_url,legal_form,incorporation_date,address,phone,email,website
"John Doe",opensanctions,demo_dataset,"Demo Dataset",Person,demo-001,"Johnny Doe; J. Doe",US,1970-01-15,American,"CEO","","sanctions","Demo Sanctions Program","Demo person for testing","2025-06-27","https://example.org","","","123 Main St","555-1234","john@example.com","https://johndoe.com"
"ACME Corporation",opensanctions,demo_dataset,"Demo Dataset",Organization,demo-002,"ACME Corp; ACME Ltd","","","","","REG123456","sanctions","Demo Sanctions Program","Demo organization for testing","2025-06-27","https://example.org","Corporation","2000-01-01","456 Business Ave","555-5678","info@acme.com","https://acme.com"
"Jane Smith",opensanctions,demo_dataset,"Demo Dataset",Person,demo-003,"J. Smith","UK,US","1985-03-22","British,American","Director","","crime,sanctions","Demo Program","Demo person with multiple attributes","2025-06-27","https://example.org","","","789 Another St","555-9999","jane@example.com",""
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(csv_data)

    logger.info(f"Created minimal test data: {output_file}")
    return output_file


def load_to_postgresql(csv_file, config_file):
    """Load CSV data to PostgreSQL using the loader script"""
    logger.info(f"Loading {csv_file} to PostgreSQL...")

    try:
        cmd = [
            sys.executable, "csv_to_postgresql.py",
            "--csv-file", csv_file,
            "--db-config", config_file,
            "--drop-table",  # Start fresh for demo
            "--verbose"
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info("Successfully loaded data to PostgreSQL!")
            logger.info("Output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        else:
            logger.error("Failed to load data to PostgreSQL:")
            logger.error(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        logger.error("Database loading timed out")
        return False
    except Exception as e:
        logger.error(f"Failed to run database loader: {e}")
        return False

    return True


def demo_database_queries(config_file):
    """Demonstrate some queries on the loaded data"""
    logger.info("Demonstrating database queries...")

    try:
        # Try to import psycopg2 and run sample queries
        import psycopg2
        import json

        with open(config_file, 'r') as f:
            config = json.load(f)

        conn = psycopg2.connect(**config)
        cursor = conn.cursor()

        # Query 1: Count total records
        cursor.execute('SELECT COUNT(*) FROM "sterlingai-aml-ctf-pep-tbl"')
        total_count = cursor.fetchone()[0]
        logger.info(f"Total records in database: {total_count}")

        # Query 2: Count by entity type
        cursor.execute('''
            SELECT entity_type, COUNT(*) 
            FROM "sterlingai-aml-ctf-pep-tbl" 
            GROUP BY entity_type 
            ORDER BY COUNT(*) DESC
        ''')
        logger.info("Records by entity type:")
        for entity_type, count in cursor.fetchall():
            logger.info(f"  {entity_type}: {count}")

        # Query 3: Sample records
        cursor.execute('''
            SELECT name, entity_type, country, dataset 
            FROM "sterlingai-aml-ctf-pep-tbl" 
            LIMIT 5
        ''')
        logger.info("Sample records:")
        for name, etype, country, dataset in cursor.fetchall():
            logger.info(f"  {name} ({etype}) - {country} [{dataset}]")

        cursor.close()
        conn.close()

    except ImportError:
        logger.warning(
            "psycopg2 not available, skipping database queries demo")
    except Exception as e:
        logger.error(f"Failed to run demo queries: {e}")


def main():
    """Main demo function"""
    logger.info("OpenSanctions CSV to PostgreSQL Demo")
    logger.info("=" * 50)

    # Step 1: Create sample database configuration
    config_file = create_sample_config()

    logger.info("\n" + "=" * 50)
    logger.info("IMPORTANT: Please update the database configuration")
    logger.info(f"Edit {config_file} with your PostgreSQL credentials")
    logger.info("Then run this demo again")
    logger.info("=" * 50)

    # Ask user if they want to continue (in a real scenario)
    print(f"\nPlease edit {config_file} with your database credentials.")
    user_input = input(
        "Press Enter to continue with demo (or Ctrl+C to exit): ")

    try:
        # Step 2: Extract sample data
        csv_file = extract_sample_data()

        if not csv_file or not os.path.exists(csv_file):
            logger.error("Failed to create sample data")
            return

        # Step 3: Load to PostgreSQL
        success = load_to_postgresql(csv_file, config_file)

        if success:
            # Step 4: Demonstrate queries
            demo_database_queries(config_file)

            logger.info("\n" + "=" * 50)
            logger.info("Demo completed successfully!")
            logger.info(
                f"Data is now available in the 'sterlingai-aml-ctf-pep-tbl' table")
            logger.info("You can query it using any PostgreSQL client")

        # Cleanup demo files
        cleanup_files = [csv_file, config_file]
        for file in cleanup_files:
            if os.path.exists(file) and file.startswith("demo_"):
                try:
                    os.remove(file)
                    logger.debug(f"Cleaned up {file}")
                except:
                    pass

    except KeyboardInterrupt:
        logger.info("\nDemo cancelled by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    main()
