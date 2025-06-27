# OpenSanctions Extractor - Troubleshooting Guide

## Common Issues and Solutions

### 🔐 SSL/TLS Connection Errors

**Error symptoms:**

```bash
ERROR - Failed to process CSV: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad record mac
ERROR - Failed to process CSV: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

**Root causes:**

- Corporate firewalls blocking SSL connections
- Outdated SSL certificates or certificate validation issues
- TLS version incompatibility
- Corporate proxy interference
- Python SSL configuration issues

**Solution implemented:** ✅
The extractor now includes robust SSL handling:

- Custom SSL context with permissive settings
- Automatic fallback to disabled SSL verification
- Browser-like headers to avoid blocking
- Retry logic with exponential backoff
- Multiple SSL configuration attempts

**Manual workarounds if needed:**

1. **Update certificates:**

    ```bash
    pip install --upgrade certifi requests urllib3
    ```

2. **Use corporate proxy settings:**

    ```bash
    export https_proxy=http://your-proxy:port
    export http_proxy=http://your-proxy:port
    ```

3. **Disable SSL system-wide (less secure):**

    ```bash
    export PYTHONHTTPSVERIFY=0
    ```

### 🌐 Network Connection Issues

**Error symptoms:**

```bash
ERROR - Connection error: HTTPSConnectionPool(...): Max retries exceeded
ERROR - Timeout error: Read timed out
```

**Solutions:**

- Check internet connectivity
- Increase timeout values
- Use retry logic (built into extractor)
- Try different network/VPN

### 🚫 Access Denied / Rate Limiting

**Error symptoms:**

```bash
ERROR - HTTP 403: Forbidden
ERROR - HTTP 429: Too Many Requests
```

**Solutions:**

- Wait and retry (built-in delays between requests)
- Use VPN if IP is blocked
- Reduce concurrent requests

### � Large Dataset Issues

**Error symptoms:**

```bash
ERROR - Failed to process CSV: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] (on large datasets)
WARNING - Request failed: Response ended prematurely
```

**Root causes:**

- Very large datasets (>50K entities) causing SSL timeouts
- Network instability during large file transfers
- Memory constraints with massive CSV files

**Solution implemented:** ✅

- **Automatic streaming**: Large datasets automatically use streaming downloads
- **Extended timeouts**: 5-minute timeouts for large files
- **Progress monitoring**: Shows progress every 50,000 lines
- **Memory efficiency**: Line-by-line processing instead of loading entire file

**Large datasets that benefit from streaming:**

- `peps` - ~679K entities
- `crime` - ~243K entities
- `debarment` - ~200K entities
- `enrichers` - ~343K entities

### 📊 Data Format Issues

**Error symptoms:**

```bash
ERROR - JSON decode error
ERROR - CSV parsing error
```

**Solutions:**

- Retry the extraction (temporary server issues)
- Check dataset availability at opensanctions.org
- Report to OpenSanctions if dataset is corrupted

## Testing Your Setup

Run these commands to test your installation:

```bash
# Test basic functionality
python test_opensanctions.py

# Test with small dataset
python opensanctions_extractor.py --mode specific --datasets 'validation' --output test.csv

# Test SSL handling with larger dataset
python opensanctions_extractor.py --mode specific --datasets 'us_ofac_sdn' --max-datasets 1 --output ssl_test.csv
```

## Performance Tuning

### For Slow Connections

- Use `--max-datasets 5` to limit data
- Process high-value datasets only: `--datasets 'us_ofac_sdn,interpol_red_notices'`

### For Corporate Networks

- The extractor automatically handles most corporate network issues
- Configure proxy settings if needed
- Contact IT department about OpenSanctions.org access

### For Large Extractions

- Use `--mode consolidated` for comprehensive data
- Run during off-peak hours
- Ensure sufficient disk space (9+ GB for full dataset)

## Getting Help

1. **Check logs:** Look for specific error messages in the output
2. **Test connectivity:** Try accessing <https://opensanctions.org> in browser
3. **Run diagnostics:** Use the test scripts to isolate issues
4. **Check documentation:** Review README.md and DATASET_REFERENCE.md

## Error Log Analysis

Common error patterns and their meanings:

| Error Type    | Meaning                 | Solution                       |
| ------------- | ----------------------- | ------------------------------ |
| `SSL:`        | SSL/TLS issue           | Built-in SSL fixes handle this |
| `Timeout`     | Network slow/unreliable | Retry logic handles this       |
| `403/429`     | Access restricted       | Wait and retry                 |
| `JSON decode` | Data format issue       | Retry or check dataset         |
| `Memory`      | Insufficient RAM        | Use smaller datasets           |

The extractor includes robust error handling and will automatically attempt to resolve most issues.
