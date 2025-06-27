# OpenSanctions Dataset Reference

## Quick Reference for --datasets Parameter

### Most Popular Datasets

| Dataset Name           | Description                            | Approx. Entities | Category     |
| ---------------------- | -------------------------------------- | ---------------- | ------------ |
| `us_ofac_sdn`          | US OFAC Specially Designated Nationals | ~18,000          | Sanctions    |
| `interpol_red_notices` | INTERPOL Red Notices                   | ~6,500           | Crime/Wanted |
| `eu_sanctions`         | EU Consolidated Sanctions              | ~15,000          | Sanctions    |
| `crime`                | Warrants and Criminal Entities         | ~243,000         | Crime/Wanted |
| `sanctions`            | Consolidated Sanctions                 | ~62,000          | Sanctions    |
| `peps`                 | Politically Exposed Persons Core       | ~679,000         | PEP          |
| `debarment`            | Debarred Companies and Individuals     | ~200,000         | Debarment    |
| `us_sam_exclusions`    | US SAM Procurement Exclusions          | ~105,000         | Debarment    |

### Usage Examples

```bash
# High-value sanctions lists
--datasets 'us_ofac_sdn,eu_sanctions,un_sc_sanctions'

# Crime and wanted persons
--datasets 'interpol_red_notices,crime,pl_wanted'

# PEP data
--datasets 'peps,br_pep,wd_peps'

# Debarment and exclusions
--datasets 'debarment,us_sam_exclusions,us_hhs_exclusions'

# Mixed selection
--datasets 'us_ofac_sdn,interpol_red_notices,peps,debarment'
```

### How to Find Dataset Names

1. **Run the test script** to see all available datasets:

    ```bash
    python test_opensanctions.py
    ```

2. **Check the OpenSanctions website**: <https://opensanctions.org/datasets/>

3. **Use the dataset explorer** (built into test script) to browse by category

### Dataset Categories

- **SANCTIONS** (106 datasets): Government and international sanctions
- **PEP** (61 datasets): Politically exposed persons
- **CRIME** (4 datasets): Wanted persons and criminal entities
- **DEBARMENT** (36 datasets): Procurement exclusions
- **WANTED** (14 datasets): Various wanted lists
- **OTHER** (85+ datasets): Regulatory, enrichment, and other data

### Size Guidelines

- **Small** (< 1,000 entities): Good for testing - `validation`, `eu_edes`
- **Medium** (1K-50K entities): Most sanctions lists - `us_ofac_sdn`, `interpol_red_notices`
- **Large** (50K+ entities): Comprehensive datasets - `crime`, `peps`, `debarment`
- **Consolidated**: All datasets combined (~1.1M+ entities) - use `--mode consolidated`

## Complete Dataset List

Run this command to get the current complete list:

```bash
python test_opensanctions.py | grep -E "^\s*-\s+\w+:" | head -50
```

This will show you dataset names, descriptions, and entity counts for planning your extractions.
