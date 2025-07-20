# Historical Simulation Guide

This guide explains how to use the new historical simulation functionality that can dynamically fetch minute-level data for any past date.

## Overview

The historical simulation system consists of:

1. **`HistoricalSimulationFetcher`** - New library for fetching historical minute data from IBKR API
2. **`historical_simulation_intraday_data`** - New database table specifically for simulation data
3. **Updated `trading_simulation.py`** - Now automatically fetches missing historical data

## Key Features

✅ **Dynamic Data Fetching** - Automatically fetches minute-level data for any simulation date
✅ **Separate Storage** - Uses dedicated table, doesn't interfere with real-time data
✅ **Smart Caching** - Only fetches data once per symbol/date combination
✅ **IBKR Integration** - Uses same IBKR API as the main data collector
✅ **Error Handling** - Robust retry logic and rate limiting

## Usage

### Basic Simulation with Time Offset
```bash
# Simulate 1200 minutes (20 hours) ago
python src/real_time/trading_simulation.py --time_offset 1200
```

### Simulation with Exact Date/Time
```bash
# Simulate January 15, 2025 at 2:30 PM
python src/real_time/trading_simulation.py --simulation_date 202501151430
```

### Format for --simulation_date
- **YYYYMMDDHH24MI** (12 digits)
- **YYYY**: 4-digit year (e.g., 2025)
- **MM**: 2-digit month (01-12)
- **DD**: 2-digit day (01-31)
- **HH**: 2-digit hour in 24-hour format (00-23)
- **MI**: 2-digit minute (00-59)

## What Happens Automatically

When you run a simulation:

1. **Data Check** - System checks if minute data exists for the simulation date
2. **Auto-Fetch** - If missing, automatically fetches from IBKR API
3. **Smart Storage** - Stores in separate `historical_simulation_intraday_data` table
4. **Simulation** - Runs simulation using the fetched historical data

## Prerequisites

### IBKR Setup Required
- **IBKR Client Portal** must be running at `https://localhost:15000`
- **Authentication** - Must be logged in to IBKR Client Portal
- **Market Data Subscriptions** - Need appropriate data subscriptions

### Check IBKR Status
```bash
# Test the historical fetcher
python test_historical_fetcher.py
```

## Database Structure

### New Table: `historical_simulation_intraday_data`
```sql
CREATE TABLE historical_simulation_intraday_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    contract_id INTEGER,
    bar_time DATETIME NOT NULL,  -- UTC time
    open REAL,
    high REAL, 
    low REAL,
    close REAL,
    volume INTEGER,
    simulation_date DATE NOT NULL,  -- Date this data belongs to
    data_source TEXT DEFAULT 'ibkr_historical',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, bar_time, simulation_date)
);
```

## API Details

### HistoricalSimulationFetcher Class

```python
from real_time.historical_simulation_fetcher import HistoricalSimulationFetcher

# Initialize
fetcher = HistoricalSimulationFetcher()

# Ensure data for simulation
simulation_datetime = datetime(2025, 1, 15, 14, 30, tzinfo=ny_tz)
success = fetcher.ensure_data_for_simulation(simulation_datetime, ["NVDA", "AAPL"])

# Check if data exists
has_data = fetcher.has_data_for_date("NVDA", date(2025, 1, 15))

# Get minute data
data = fetcher.get_minute_data_for_date("NVDA", date(2025, 1, 15))
```

## Troubleshooting

### "Not authenticated to IBKR"
- Ensure IBKR Client Portal is running
- Login at https://localhost:15000
- Check network connectivity

### "No data returned"
- Check if the date is a trading day (not weekend/holiday)
- Verify market data subscriptions
- Try a more recent date

### Rate Limiting
- The system includes automatic rate limiting
- If you see 429 errors, wait and try again
- Consider fetching data for fewer symbols at once

### Missing Contract IDs
- Ensure symbol format is correct (e.g., "NVDA", not "NVIDIA")
- Check if symbol is available in IBKR
- Verify market data subscriptions

## Examples

### Example 1: Simulate Friday Close
```bash
# January 17, 2025 at 3:59 PM (market close)
python src/real_time/trading_simulation.py --simulation_date 202501171559
```

### Example 2: Simulate Market Open
```bash
# January 15, 2025 at 9:30 AM (market open)
python src/real_time/trading_simulation.py --simulation_date 202501150930
```

### Example 3: Test Historical Fetcher
```bash
# Run the test script to verify setup
python test_historical_fetcher.py
```

## Integration with Trading Simulation

The `trading_simulation.py` script now automatically:

1. **Detects Simulation Date** - From either `--time_offset` or `--simulation_date`
2. **Checks Data Availability** - Verifies minute data exists for that date
3. **Fetches Missing Data** - Automatically downloads from IBKR if needed
4. **Updates Database Queries** - Uses `historical_simulation_intraday_data` table
5. **Runs Simulation** - Proceeds with historical simulation using real minute data

## Benefits

- **No Manual Data Management** - System handles data fetching automatically
- **Historical Accuracy** - Uses real minute-level IBKR data
- **Performance** - Caches data to avoid re-fetching
- **Isolation** - Separate from real-time data collection
- **Flexibility** - Can simulate any historical trading day 