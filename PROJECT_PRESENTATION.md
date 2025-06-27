# OpenSanctions Data Extractor - Project Overview

## 🎯 Project Purpose

This project provides a robust, production-ready Python solution for extracting entity data from OpenSanctions datasets and loading it into PostgreSQL databases. It's designed for compliance, AML (Anti-Money Laundering), and sanctions screening applications.

## 📁 Project Structure

```
opensanctions-extractor/
├── opensanctions_extractor.py    # Main extraction script
├── csv_to_postgresql.py          # PostgreSQL database loader
├── demo_opensanctions.py         # Extraction demonstration
├── demo_postgresql.py            # Database loading demonstration
├── test_opensanctions.py         # Test and validation script
├── requirements.txt              # Python dependencies
├── db_config.json.example        # Database configuration template
├── sample_output.csv             # Example output format
├── README.md                     # Main documentation
└── docs/                         # Detailed documentation
    ├── DATASET_REFERENCE.md      # Available datasets reference
    ├── POSTGRESQL_INTEGRATION.md # Database integration guide
    ├── PROJECT_OVERVIEW.md       # Technical overview
    ├── PROJECT_SUMMARY.md        # Implementation details
    └── TROUBLESHOOTING.md        # Common issues & solutions
```

## 🚀 Key Features

### Data Extraction

-   **300+ datasets** from OpenSanctions (sanctions, PEPs, watchlists, crime data)
-   **Multiple formats** support (CSV, FTM JSON)
-   **Compression handling** (ZIP, GZIP, BZIP2)
-   **Streaming architecture** for large datasets
-   **Robust error handling** with retry logic and SSL fallbacks

### Database Integration

-   **PostgreSQL support** with optimized schema
-   **Batch processing** for efficient loading
-   **Duplicate detection** and handling
-   **Comprehensive indexing** for query performance
-   **Transaction safety** with rollback support

### Robustness

-   **SSL/TLS compatibility** with multiple fallback options
-   **Network resilience** with exponential backoff
-   **Memory efficiency** with streaming for large files
-   **Progress reporting** for long-running operations
-   **Comprehensive logging** for debugging and monitoring

## 📊 Data Schema

### Standardized CSV Output (23 columns)

| Column           | Description          | Example          |
| ---------------- | -------------------- | ---------------- |
| name             | Entity name          | "Vladimir Putin" |
| entity_type      | Person/Organization  | "Person"         |
| country          | Associated countries | "RU"             |
| sanction_program | Sanction details     | "EU Sanctions"   |
| dataset          | Source dataset       | "eu_sanctions"   |
| ...              | 18 more columns      | ...              |

### PostgreSQL Table Schema

-   **Primary Key**: Auto-incrementing `id` (SERIAL)
-   **Data Columns**: All 23 standardized columns
-   **Metadata**: `created_at`, `updated_at` timestamps
-   **Indexes**: 7 optimized indexes for query performance

## 🎮 Usage Examples

### Basic Extraction

```bash
# Extract specific high-value datasets
python opensanctions_extractor.py --mode specific --datasets 'us_ofac_sdn,eu_sanctions' --output sanctions.csv

# Extract all PEP data
python opensanctions_extractor.py --mode specific --datasets 'peps,br_pep' --output pep_data.csv
```

### Database Loading

```bash
# Load extracted data to PostgreSQL
python csv_to_postgresql.py --csv-file sanctions.csv --db-config db_config.json --drop-table
```

### Complete Workflow

```bash
# 1. Extract sanctions data
python opensanctions_extractor.py --mode specific --datasets 'us_ofac_sdn,eu_sanctions,un_sc_sanctions' --output sanctions.csv

# 2. Load to database
python csv_to_postgresql.py --csv-file sanctions.csv --db-config db_config.json

# 3. Query the data
psql -d your_database -c "SELECT name, country, sanction_program FROM \"sterlingai-aml-ctf-pep-tbl\" LIMIT 10;"
```

## 🔧 Technical Specifications

### Performance

-   **Small datasets**: 1-1K entities (seconds)
-   **Medium datasets**: 1K-50K entities (minutes)
-   **Large datasets**: 50K+ entities (streaming with progress)
-   **Consolidated data**: 1.1M+ entities (9+ GB, full streaming)

### Requirements

-   **Python**: 3.7+
-   **Dependencies**: requests, psycopg2-binary
-   **Database**: PostgreSQL 9.5+
-   **Memory**: Streaming architecture, minimal RAM usage
-   **Storage**: Variable based on dataset selection

### Error Handling

-   **Network issues**: Automatic retry with exponential backoff
-   **SSL problems**: Multiple fallback contexts + curl fallback
-   **File format issues**: Graceful handling of encoding/compression
-   **Database errors**: Transaction rollback and detailed logging

## 🎯 Use Cases

### Compliance & AML

-   **Sanctions screening**: Real-time entity checking
-   **PEP identification**: Politically exposed persons
-   **Watchlist management**: Comprehensive screening lists
-   **Regulatory compliance**: Meet AML/KYC requirements

### Data Analytics

-   **Risk assessment**: Entity risk profiling
-   **Network analysis**: Relationship mapping
-   **Trend analysis**: Sanctions and enforcement patterns
-   **Reporting**: Compliance and audit reports

### Integration Scenarios

-   **Existing systems**: ETL pipeline integration
-   **Real-time screening**: API-based entity checking
-   **Batch processing**: Nightly data updates
-   **Data warehousing**: Enterprise data lakes

## 📈 Dataset Coverage

### High-Value Datasets

-   **US OFAC SDN**: ~18K sanctioned entities
-   **EU Sanctions**: Consolidated EU sanctions
-   **UN Sanctions**: Security Council sanctions
-   **INTERPOL**: Red notices and wanted persons
-   **PEP Data**: 679K+ politically exposed persons

### Geographic Coverage

-   **Global**: 200+ countries and territories
-   **Regional**: EU, US, APAC, MENA, Latin America
-   **National**: Country-specific sanctions and watchlists

## 🛡️ Security & Compliance

### Data Handling

-   **Source verification**: Direct from OpenSanctions
-   **Data integrity**: Checksum validation
-   **Audit trail**: Complete processing logs
-   **Access control**: Database-level permissions

### Privacy Considerations

-   **Public data only**: No private information
-   **Lawful basis**: Compliance and public interest
-   **Data retention**: Configurable retention policies

## 🚀 Getting Started

1. **Installation**:

    ```bash
    git clone <repository>
    cd opensanctions-extractor
    pip install -r requirements.txt
    ```

2. **Basic Test**:

    ```bash
    python test_opensanctions.py
    ```

3. **Extract Sample Data**:

    ```bash
    python opensanctions_extractor.py --mode all --max-datasets 3 --output test.csv
    ```

4. **Set Up Database** (optional):
    ```bash
    cp db_config.json.example db_config.json
    # Edit db_config.json with your credentials
    python csv_to_postgresql.py --csv-file test.csv --db-config db_config.json
    ```

## 📞 Support

-   **Documentation**: Comprehensive docs in `docs/` folder
-   **Troubleshooting**: `docs/TROUBLESHOOTING.md`
-   **Examples**: Demo scripts included
-   **Testing**: Validation scripts provided

---

**Ready for production use with enterprise-grade reliability and performance.**
