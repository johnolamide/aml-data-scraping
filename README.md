# OpenSanctions Data Extractor

A robust Python script to extract entity names and key attributes from OpenSanctions datasets, generating a unified CSV with standardized columns.

## Features

- **Multiple extraction modes**: Extract from all datasets, specific datasets, or the consolidated data
- **Flexible data formats**: Supports both CSV and FTM JSON formats from OpenSanctions
- **Compressed file support**: Automatically detects and extracts ZIP, GZIP, and BZIP2 compressed files
- **Robust downloads**: Features retry logic, SSL fallback, and curl fallback for difficult downloads
- **Streaming architecture**: Memory-efficient processing with chunked downloads for large files
- **Standardized output**: Generates a unified CSV with consistent columns across all datasets
- **Error resilient**: Continues processing even if individual datasets fail with comprehensive error handling
- **Progress reporting**: Real-time progress updates for large dataset downloads and processing
- **Comprehensive attributes**: Extracts names, aliases, countries, birth dates, and many other relevant fields

## Quick Start

### Installation

No external dependencies beyond Python standard library and `requests`:

```bash
pip install requests
```

**Note**: The extractor automatically handles compressed files (ZIP, GZIP, BZIP2) and includes robust error handling for network issues.

### Basic Usage

1. **Test with small datasets** (recommended first run):

    ```bash
    python opensanctions_extractor.py --mode all --max-datasets 5 --output test_entities.csv
    ```

2. **Extract specific datasets**:

    ```bash
    python opensanctions_extractor.py --mode specific --datasets 'us_ofac_sdn,interpol_red_notices' --output specific_entities.csv
    ```

3. **Extract all available datasets** (large download):

    ```bash
    python opensanctions_extractor.py --mode all --output all_entities.csv
    ```

4. **Use consolidated data** (9+ GB file - most comprehensive):

    ```bash
    python opensanctions_extractor.py --mode consolidated --output consolidated_entities.csv
    ```

## Output Format

The script generates a CSV file with the following standardized columns:

| Column              | Description                                          |
| ------------------- | ---------------------------------------------------- |
| name                | Primary entity name                                  |
| source              | Always "opensanctions"                               |
| dataset             | Source dataset name                                  |
| dataset_title       | Human-readable dataset title                         |
| entity_type         | Type of entity (Person, Organization, Company, etc.) |
| entity_id           | Unique entity identifier                             |
| alias               | Alternative names (semicolon-separated)              |
| country             | Associated countries                                 |
| birth_date          | Birth date (for persons)                             |
| nationality         | Nationality information                              |
| position            | Job title or position                                |
| registration_number | Company/organization registration numbers            |
| topics              | Topics/categories (crime, sanctions, etc.)           |
| sanction_program    | Specific sanction program names                      |
| description         | Entity description                                   |
| last_updated        | When the data was last updated                       |
| source_url          | Link to original source                              |
| legal_form          | Legal form (for organizations)                       |
| incorporation_date  | Date of incorporation                                |
| address             | Address information                                  |
| phone               | Phone numbers                                        |
| email               | Email addresses                                      |
| website             | Website URLs                                         |

## Usage Examples

### Command Line Arguments

- `--mode`: Choose extraction mode
  - `all`: Extract from all available datasets
  - `specific`: Extract from specified datasets only
  - `consolidated`: Extract from the consolidated statements file
- `--datasets`: Comma-separated list of dataset names (required for `specific` mode)
- `--output`: Output CSV filename (default: `opensanctions_entities.csv`)
- `--max-datasets`: Maximum number of datasets to process (useful for testing)

### Example Commands

```bash
# Extract from top 10 datasets by size
python opensanctions_extractor.py --mode all --max-datasets 10

# Extract specific sanctions list
python opensanctions_extractor.py --mode specific --datasets 'us_ofac_sdn,eu_sanctions,un_sc_sanctions'

# Extract crime/wanted lists
python opensanctions_extractor.py --mode specific --datasets 'interpol_red_notices,crime,pl_wanted'

# Extract PEP data
python opensanctions_extractor.py --mode specific --datasets 'peps,br_pep'

# Extract debarment lists
python opensanctions_extractor.py --mode specific --datasets 'debarment,us_sam_exclusions'

# Full extraction (consolidated - most comprehensive)
python opensanctions_extractor.py --mode consolidated --output full_opensanctions.csv
```

## Testing

The repository includes test scripts to demonstrate functionality:

### Test Script

```bash
python test_opensanctions.py
```

Shows available datasets, categories, and extracts samples from small datasets.

### Demo Script

```bash
python demo_opensanctions.py
```

Demonstrates different extraction modes with actual data.

## Dataset Categories

OpenSanctions includes several types of datasets:

- **Sanctions**: Official government and international sanctions lists
- **PEP (Politically Exposed Persons)**: Politicians and senior officials
- **Crime**: Wanted persons and criminal entities
- **Debarment**: Companies and individuals barred from government contracts
- **Regulatory**: Various regulatory watchlists

### Popular Dataset Names for `--datasets` Option

#### 🎯 High-Value Sanctions Lists

- `us_ofac_sdn` - US OFAC Specially Designated Nationals (~18K entities)
- `eu_sanctions` - EU Consolidated Sanctions
- `un_sc_sanctions` - UN Security Council Sanctions
- `sanctions` - Consolidated Sanctions (~62K entities)
- `maritime` - Maritime-related sanctions (~69K entities)
- `securities` - Sanctioned Securities (~45K entities)

#### 🚨 Crime & Wanted Lists

- `interpol_red_notices` - INTERPOL Red Notices (~6.5K entities)
- `crime` - Warrants and Criminal Entities (~243K entities)
- `pl_wanted` - Poland Wanted Persons (~51K entities)
- `us_dea_fugitives` - US DEA Fugitives (~591 entities)
- `tr_wanted` - Türkiye Terrorist Wanted List (~2.6K entities)

#### 👤 PEP (Politically Exposed Persons)

- `ng_chipper_peps` - Nigerian Politically Exposed Persons (~25K entities)
- `peps` - Politically Exposed Persons Core Data (~679K entities)
- `br_pep` - Brazil Politically Exposed Persons (~110K entities)
- `wd_peps` - Wikidata Politically Exposed Persons (~228K entities)

#### 🏢 Debarment & Exclusions

- `debarment` - Debarred Companies and Individuals (~200K entities)
- `us_sam_exclusions` - US SAM Procurement Exclusions (~105K entities)
- `us_hhs_exclusions` - US Health and Human Sciences Exclusions (~80K entities)
- `us_ca_med_exclusions` - US California Medicaid Exclusions (~22K entities)

#### 📊 Other Datasets

- `regulatory` - Regulatory Watchlists (~155K entities)
- `enrichers` - Enrichment from External Databases (~343K entities)
- `wikidata` - Wikidata (~343K entities)

**💡 Tip**: Run `python test_opensanctions.py` to see the complete list of 300+ available datasets with their current entity counts.

## Performance Notes

### Dataset Sizes

- **Small datasets**: 2-1,000 entities (seconds to extract)
- **Medium datasets**: 1,000-50,000 entities (minutes to extract)
- **Large datasets**: 50,000+ entities (may take longer)
- **Consolidated file**: 1.1M+ entities from all datasets (9+ GB, requires streaming)

### Robustness Features

The extractor includes several reliability enhancements:

- **Automatic compression handling**: Detects and extracts ZIP, GZIP, and BZIP2 files transparently
- **Robust SSL handling**: Permissive SSL context with fallback options for certificate issues
- **Download resilience**: Retry logic with exponential backoff, streaming downloads for large files
- **Fallback mechanisms**: Automatic fallback to curl command-line tool if requests library fails
- **Progress tracking**: Real-time progress bars for large downloads and processing operations
- **Error recovery**: Continues processing other datasets even if individual ones fail

### Recommendations

1. **Start small**: Use `--max-datasets 5` for initial testing
2. **Specific extraction**: Target specific datasets for focused use cases
3. **Consolidated mode**: Use for comprehensive coverage, but expect long processing time
4. **Network issues**: The script automatically handles most network and SSL issues with fallbacks
5. **Compressed files**: No special handling needed - compressed datasets are processed automatically
6. **Memory usage**: The script manages memory efficiently with streaming, but consolidated mode will use significant disk space

## Example Output

```csv
name,source,dataset,dataset_title,entity_type,entity_id,alias,country,birth_date,...
"JOHN DOE",opensanctions,us_ofac_sdn,"US OFAC Specially Designated Nationals",Person,us-ofac-12345,"JOHN SMITH; J. DOE",US,1970-01-15,...
"ACME CORPORATION",opensanctions,eu_sanctions,"EU Consolidated Sanctions",Organization,eu-sanc-67890,"ACME CORP; ACME LTD",,,,,...
```

## Data Sources

All data comes from [OpenSanctions](https://opensanctions.org), which aggregates sanctions, watchlists, and persons of interest data from governments and international organizations worldwide.

## Additional Documentation

This project includes comprehensive documentation in the `docs/` folder:

- **`docs/DATASET_REFERENCE.md`**: Complete reference of all available datasets with descriptions and entity counts
- **`docs/TROUBLESHOOTING.md`**: Solutions for common issues including SSL errors, download failures, and encoding problems
- **`docs/PROJECT_SUMMARY.md`**: Technical overview of the project structure and implementation details
- **`docs/PROJECT_OVERVIEW.md`**: High-level project overview and architecture description
- **`docs/POSTGRESQL_INTEGRATION.md`**: Detailed PostgreSQL integration guide

## Troubleshooting

For common issues and solutions, see `docs/TROUBLESHOOTING.md`. Quick fixes:

- **SSL/Certificate errors**: The script automatically handles these with fallback options
- **Download failures**: Automatic retry logic and curl fallback are built-in
- **Compressed files**: Automatically detected and extracted (ZIP, GZIP, BZIP2)
- **Memory issues**: Use specific dataset mode instead of consolidated for large extractions
- **Encoding errors**: The script handles various character encodings automatically

Run the test script to verify your setup:

```bash
python test_opensanctions.py
```

## Database Integration

### PostgreSQL Loading

The project includes a PostgreSQL loader (`csv_to_postgresql.py`) that can load the extracted CSV data into a database table for easier querying and integration.

#### Prerequisites

Install the PostgreSQL adapter:

```bash
pip install -r requirements.txt
```

#### Database Setup

1. **Create database configuration file**:

```bash
cp db_config.json.example db_config.json
# Edit db_config.json with your PostgreSQL credentials
```

2. **Load CSV data to PostgreSQL**:

```bash
# Using config file
python csv_to_postgresql.py --csv-file opensanctions_entities.csv --db-config db_config.json

# Using command line parameters
python csv_to_postgresql.py --csv-file opensanctions_entities.csv --host localhost --database aml_db --user postgres --password secret
```

#### Database Schema

The PostgreSQL table `sterlingai-aml-ctf-pep-tbl` includes:

- **`id`**: Auto-generated primary key (SERIAL)
- **Standard columns**: All 23 standardized columns from the CSV output
- **Timestamps**: `created_at` and `updated_at` for tracking
- **Indexes**: Optimized indexes on frequently queried columns (name, entity_id, dataset, etc.)

#### Usage Options

```bash
# Drop existing table and reload
python csv_to_postgresql.py --csv-file data.csv --db-config config.json --drop-table

# Allow duplicate entity_ids
python csv_to_postgresql.py --csv-file data.csv --db-config config.json --allow-duplicates

# Custom batch size for large datasets
python csv_to_postgresql.py --csv-file data.csv --db-config config.json --batch-size 5000

# Verbose logging
python csv_to_postgresql.py --csv-file data.csv --db-config config.json --verbose
```

#### Demo

Run the PostgreSQL integration demo:

```bash
python demo_postgresql.py
```

This demo will:

1. Extract sample OpenSanctions data
2. Load it into PostgreSQL
3. Demonstrate basic queries on the loaded data

## Project Structure

```bash
opensanctions-extractor/
├── opensanctions_extractor.py    # Main extraction script
├── csv_to_postgresql.py          # PostgreSQL database loader
├── demo_opensanctions.py         # Extraction demonstration
├── demo_postgresql.py            # Database integration demo
├── test_opensanctions.py         # Test and validation script
├── requirements.txt              # Python dependencies
├── db_config.json.example        # Database configuration template
├── sample_output.csv             # Example output format
├── README.md                     # This file
├── PROJECT_PRESENTATION.md       # Project overview for presentation
└── docs/                         # Detailed documentation
    ├── DATASET_REFERENCE.md      # Complete dataset reference
    ├── POSTGRESQL_INTEGRATION.md # Database integration guide
    ├── PROJECT_OVERVIEW.md       # Technical architecture
    ├── PROJECT_SUMMARY.md        # Implementation details
    └── TROUBLESHOOTING.md        # Issues and solutions
```

## Quick Start Guide

1. **Clone and setup**:

    ```bash
    git clone <repository>
    cd opensanctions-extractor
    pip install -r requirements.txt
    ```

2. **Test the extractor**:

    ```bash
    python test_opensanctions.py
    ```

3. **Extract sample data**:

    ```bash
    python opensanctions_extractor.py --mode all --max-datasets 3 --output sample.csv
    ```

4. **Optional - Database setup**:

    ```bash
    cp db_config.json.example db_config.json
    # Edit with your PostgreSQL credentials
    python csv_to_postgresql.py --csv-file sample.csv --db-config db_config.json
    ```

## License

This extractor script is provided as-is. Please refer to OpenSanctions' terms of service for data usage rights.

---

**Note**: The consolidated mode downloads a very large file (9+ GB). The extractor includes robust download handling with automatic retries, progress reporting, and compression support, but ensure you have sufficient bandwidth, storage space, and time before running this mode.
