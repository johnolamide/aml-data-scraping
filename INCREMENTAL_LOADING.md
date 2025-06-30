# Incremental Loading Feature Summary

## 🎯 Problem Solved

**Scenario**: You have a CSV file with 100 entities that you've already loaded into PostgreSQL. Later, your CSV file is updated to have 102 entities (2 new entries added). You want to add only the 2 new entries to the database without re-processing the existing 100 entries.

## ✅ Solution: `--incremental` Mode

The `csv_to_postgresql.py` script now supports incremental loading that:

1. **Identifies existing entities** by checking `entity_id` values already in the database
2. **Skips existing records** to avoid duplicates
3. **Adds only new entries** that don't exist in the database
4. **Provides detailed reporting** on what was added vs skipped

## 🚀 Usage

### Initial Load (First Time)

```bash
# Load 100 entities for the first time
python csv_to_postgresql.py --csv-file entities_v1.csv --db-config config.json --drop-table
# Result: 100 entities inserted
```

### Incremental Update (When CSV is Updated)

```bash
# Load updated CSV with 102 entities (2 new ones)
python csv_to_postgresql.py --csv-file entities_v2.csv --db-config config.json --incremental
# Result: 2 new entities inserted, 100 existing entities skipped
```

## 📊 Example Output

```bash
2025-06-30 08:15:23,456 - INFO - Loading CSV file incrementally: entities_v2.csv
2025-06-30 08:15:23,456 - INFO - CSV file size: 0.34 MB
2025-06-30 08:15:23,456 - INFO - Fetching existing entity IDs from database...
2025-06-30 08:15:23,567 - INFO - Found 100 existing entities in database
2025-06-30 08:15:23,567 - INFO - Processing CSV rows for incremental update...
2025-06-30 08:15:23,678 - INFO - Processed 102 rows, inserted 2 new records, skipped 100 existing
2025-06-30 08:15:23,789 - INFO - Incremental CSV loading completed:
2025-06-30 08:15:23,789 - INFO -   - Total rows processed: 102
2025-06-30 08:15:23,789 - INFO -   - New records inserted: 2
2025-06-30 08:15:23,789 - INFO -   - Existing records skipped: 100
2025-06-30 08:15:23,789 - INFO -   - Errors encountered: 0
2025-06-30 08:15:23,789 - INFO - ✅ Successfully added 2 new entries to the database
```

## 🔧 Technical Implementation

### Key Features

1. **Efficient entity lookup**: Fetches all existing `entity_id` values into memory for fast comparison
2. **Batch processing**: Still uses efficient batch inserts for new records
3. **Memory optimization**: Only stores entity IDs in memory, not full records
4. **Error handling**: Continues processing even if individual records fail
5. **Detailed logging**: Provides clear feedback on what was processed

### Performance Characteristics

- **Memory usage**: O(n) where n = number of existing entities in database
- **Time complexity**: O(m + n) where m = CSV rows, n = existing entities
- **Database queries**: 1 initial query to fetch existing IDs + batch inserts for new records

## ⚙️ Command Line Options

```bash
--incremental                    # Enable incremental mode
--batch-size 1000               # Batch size for insertions (default: 1000)
--verbose                       # Enable detailed logging
```

## 🎭 Demo

Run the included demo to see incremental loading in action:

```bash
python demo_incremental.py
```

This creates two CSV files:

- `demo_initial.csv`: 100 entities (demo-entity-00001 to demo-entity-00100)
- `demo_updated.csv`: 102 entities (adds demo-entity-00101 and demo-entity-00102)

## 🏆 Benefits

1. **Performance**: Only processes new data, not existing data
2. **Efficiency**: Minimal database operations (1 SELECT + batch INSERTs)
3. **Safety**: No risk of duplicating existing data
4. **Flexibility**: Works with any size increment (1 new record or thousands)
5. **Monitoring**: Clear reporting on what was added vs what was skipped

## 🔄 Typical Workflow

1. **Initial setup**: Load complete dataset with `--drop-table`
2. **Regular updates**: Use `--incremental` for updated CSV files
3. **Monitoring**: Check logs to see how many new entities were added
4. **Validation**: Query database to verify total count matches expectations

This feature is perfect for production environments where you receive regular updates to your entity data and need to efficiently maintain an up-to-date database.
