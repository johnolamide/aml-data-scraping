#!/usr/bin/env python3
"""
OpenSanctions Demo Script

This script demonstrates different ways to use the OpenSanctions extractor.
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and show the output"""
    print(f"\n{'='*60}")
    print(f"DEMO: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Command timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    print("OpenSanctions Extractor Demo")
    print("This script demonstrates different extraction modes")

    # Demo 1: Test with small datasets
    print("\n1. Testing with small datasets (first 2 available):")
    run_command(
        "python opensanctions_extractor.py --mode all --max-datasets 2 --output demo_small.csv",
        "Extract from 2 small datasets"
    )

    # Demo 2: Specific datasets
    print("\n2. Testing with specific datasets:")
    run_command(
        "python opensanctions_extractor.py --mode specific --datasets 'eu_edes,validation' --output demo_specific.csv",
        "Extract from specific small datasets"
    )

    # Demo 3: Show what consolidated mode would do (without actually running it)
    print("\n3. Consolidated mode (DEMO ONLY - not actually running):")
    print("The consolidated mode extracts from the full 9GB+ statements.csv file.")
    print("This contains ALL entities from ALL datasets in one file.")
    print("To run this (WARNING: takes hours and lots of disk space):")
    print("python opensanctions_extractor.py --mode consolidated --output full_entities.csv")

    # Check output files
    print("\n4. Checking output files:")
    for filename in ['demo_small.csv', 'demo_specific.csv']:
        if os.path.exists(filename):
            lines = sum(1 for line in open(filename, 'r', encoding='utf-8'))
            print(f"  {filename}: {lines-1} entities (excluding header)")
        else:
            print(f"  {filename}: Not found")

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("The opensanctions_extractor.py script is ready for production use.")
    print("Choose your extraction mode based on your needs:")
    print("- 'all': Extract from all datasets (can be many GB)")
    print("- 'specific': Extract from chosen datasets")
    print("- 'consolidated': Extract from the full consolidated file (9+ GB)")


if __name__ == "__main__":
    main()
