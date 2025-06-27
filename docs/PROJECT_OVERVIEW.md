# OpenSanctions Data Extractor

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A robust Python tool for extracting entity names and key attributes from [OpenSanctions](https://opensanctions.org) datasets, generating unified CSV files with standardized columns.

## 🚀 Quick Start

```bash
# Install dependencies
pip install requests

# Test with small datasets
python opensanctions_extractor.py --mode all --max-datasets 5

# Extract specific datasets
python opensanctions_extractor.py --mode specific --datasets 'us_ofac_sdn,interpol_red_notices'

# Run test/demo
python test_opensanctions.py
python demo_opensanctions.py
```

## 📋 Project Structure

```bash
data-scraping/
├── opensanctions_extractor.py    # Main extraction script
├── test_opensanctions.py         # Test and demonstration script
├── demo_opensanctions.py         # Usage demonstration
├── README.md                     # Comprehensive documentation
├── DATASET_REFERENCE.md          # Dataset names and categories
├── PROJECT_SUMMARY.md            # Project completion summary
└── sample_opensanctions_output.csv # Sample output format
```

## 📊 Features

- **Multiple extraction modes**: All datasets, specific datasets, or consolidated
- **Format support**: CSV and FTM JSON from OpenSanctions
- **Standardized output**: 23 unified columns across all datasets
- **Memory efficient**: Streaming support for large datasets (9+ GB)
- **Error resilient**: Continues processing if individual datasets fail
- **300+ datasets**: Supports all available OpenSanctions datasets

## 📈 Dataset Coverage

| Category         | Datasets | Example Sources                  |
| ---------------- | -------- | -------------------------------- |
| **Sanctions**    | 106      | US OFAC, EU, UN Security Council |
| **PEP**          | 61       | Politicians, senior officials    |
| **Crime/Wanted** | 18       | INTERPOL, national wanted lists  |
| **Debarment**    | 36       | Procurement exclusions           |
| **Other**        | 85+      | Regulatory, enrichment data      |

## 🎯 Output Schema

The extractor generates CSV files with standardized columns:

| Column             | Description            | Example                  |
| ------------------ | ---------------------- | ------------------------ |
| `name`             | Primary entity name    | "John Doe"               |
| `source`           | Always "opensanctions" | "opensanctions"          |
| `dataset`          | Dataset identifier     | "us_ofac_sdn"            |
| `entity_type`      | Type of entity         | "Person", "Organization" |
| `country`          | Associated countries   | "US; UK"                 |
| `birth_date`       | Birth date (persons)   | "1970-01-15"             |
| `sanction_program` | Sanction program       | "UKRAINE-EO13661"        |
| ...                | 16+ additional columns | ...                      |

## 📖 Documentation

- [README.md](README.md) - Complete usage guide and examples
- [DATASET_REFERENCE.md](DATASET_REFERENCE.md) - Available datasets and categories
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Technical implementation details

## ✅ Tested & Verified

- ✅ Successfully extracts from 300+ datasets
- ✅ Handles datasets from 10 to 1M+ entities
- ✅ Memory-efficient streaming for large files
- ✅ Robust error handling and logging
- ✅ Production-ready implementation

## 🛠️ Usage Examples

```bash
# Extract top sanctions lists
python opensanctions_extractor.py --mode specific --datasets 'us_ofac_sdn,eu_sanctions,un_sc_sanctions'

# Extract crime and wanted data
python opensanctions_extractor.py --mode specific --datasets 'interpol_red_notices,crime'

# Full comprehensive extraction (large)
python opensanctions_extractor.py --mode consolidated --output full_data.csv
```

## 📞 Support

For questions about available datasets, run:

```bash
python test_opensanctions.py  # Shows all datasets with categories and counts
```

---

**Data Source**: [OpenSanctions.org](https://opensanctions.org) - The world's largest open sanctions and watchlist database
