# Trade Limit Configuration

## Overview

The application now includes a configurable trade limit warning system to alert users about potential performance issues with very large datasets. By default, files with more than 3,000 trades will show a warning message but still allow the file to be processed.

## Configuration Options

### Environment Variables

#### `MAX_TRADES_LIMIT`
- **Default**: `3000`
- **Description**: Maximum number of trades allowed in a single file
- **Example**: `MAX_TRADES_LIMIT=5000`

#### `DISABLE_TRADE_LIMIT`
- **Default**: `false`
- **Description**: Completely disable the trade limit check
- **Values**: `true` or `false`
- **Example**: `DISABLE_TRADE_LIMIT=true`

## Usage Examples

### Set Custom Limit
```bash
export MAX_TRADES_LIMIT=5000
python app.py
```

### Disable Limit Entirely
```bash
export DISABLE_TRADE_LIMIT=true
python app.py
```

### Azure App Service Configuration
Add these as Application Settings in the Azure portal:
- `MAX_TRADES_LIMIT`: `5000`
- `DISABLE_TRADE_LIMIT`: `false`

## Warning Messages

When a file exceeds the recommended limit, users will see a warning alert:
```
⚠️ WARNING: This file contains 5,247 trades. The recommended limit is 3,000 trades. 
Files larger than this may produce unexpected problems and slow performance. 
Consider splitting your data into smaller files.
```

The file will still be processed successfully, but the warning alerts users to potential performance issues.

## Why This Limit Exists

Large files with many trades can cause:
- **Memory Issues**: High RAM usage during processing
- **Performance Problems**: Slow chart rendering and calculations
- **Timeout Errors**: File processing may exceed timeout limits
- **Browser Issues**: Large data payloads can cause frontend problems

## Recommendations

1. **Split Large Files**: Break large datasets into smaller, manageable chunks
2. **Use Date Ranges**: Consider splitting by time periods (monthly/quarterly)
3. **Filter Data**: Remove unnecessary trades before upload
4. **Monitor Performance**: Watch for slow response times with large datasets

## Technical Details

- The limit is checked **before** portfolio loading begins
- Files exceeding the limit show a **warning** but are still processed
- The check applies to both **new uploads** and **saved file loading**
- Trade count is determined by the number of rows in the CSV file
- Warning messages are displayed as prominent alerts that auto-dismiss after 5 seconds
