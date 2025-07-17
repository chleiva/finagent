# Time Offset Feature for Trading Simulation

## Overview

The `time_offset` feature allows you to simulate trading at any point in the past using historical intraday data. This is particularly useful for testing your trading strategy outside market hours or backtesting specific time periods.

## How It Works

When you specify a `time_offset` in minutes, the entire trading simulation logic "pretends" that the current time is `NOW - time_offset` minutes. This affects:

1. **Real-time Data**: Uses the closest intraday data point to the simulated time
2. **Intraday Data**: Fetches data from market open until simulated time minus 1 minute
3. **Daily Data**: Remains unchanged (uses historical daily data as normal)

## Usage

### Basic Syntax
```bash
cd src/real_time
python trading_simulation.py --time_offset <minutes>
```

### Examples

#### Real-time Mode (Default)
```bash
python trading_simulation.py
# or
python trading_simulation.py --time_offset 0
```

#### Simulate Trading 30 Minutes Ago
```bash
python trading_simulation.py --time_offset 30
```

#### Simulate Trading 2 Hours Ago
```bash
python trading_simulation.py --time_offset 120
```

#### Test During Market Hours (1 Hour Ago)
```bash
python trading_simulation.py --time_offset 60
```

## What Changes in Simulation Mode

### Data Sources
- **Real-time data**: Instead of live market data, uses the closest intraday bar to the simulated time
- **Bid/Ask prices**: Synthetically generated with a 0.1% spread around the last price
- **Intraday data**: Fetches from market open (9:30 AM ET) until simulated time minus 1 minute
- **Daily data**: Uses historical daily data (unchanged)

### Display
- Shows both actual time and simulated time
- Indicates "EXPERIMENTATION MODE" in the header
- Updates data freshness calculations relative to simulated time

## Benefits

✅ **Test Outside Market Hours**: Run your strategy when markets are closed  
✅ **Backtesting**: Test specific time periods with historical data  
✅ **Model Validation**: Validate model performance on past data  
✅ **Safe Experimentation**: No risk of real trading during testing  
✅ **Debugging**: Debug issues using specific historical scenarios  

## Limitations

⚠️ **Experimentation Only**: This feature is designed for testing, not real trading  
⚠️ **Data Dependency**: Requires sufficient intraday data in the database  
⚠️ **Synthetic Spreads**: Bid/ask spreads are simulated (0.1% around last price)  
⚠️ **Market Hours**: Best results when simulating during actual market hours  

## Technical Details

### Time Calculations
```python
simulated_time = actual_time - timedelta(minutes=time_offset)
```

### Data Queries
- **Real-time**: `SELECT * FROM intraday_minute_data WHERE bar_time <= simulated_time ORDER BY bar_time DESC LIMIT 1`
- **Intraday**: `SELECT * FROM intraday_minute_data WHERE bar_time >= market_open AND bar_time <= (simulated_time - 1 minute)`
- **Daily**: Normal query (unchanged)

### Market Open Calculation
The system calculates market open as 9:30 AM ET for the simulated date.

## Example Output

When running with `--time_offset 60`, you'll see:

```
🕐 Trading Simulation (EXPERIMENTATION MODE - 60 minutes offset)
================================================================================
Actual time: 2025-07-16 21:58:16 EST
Simulated time: 2025-07-16 20:58:16 EST
Time offset: 60 minutes
```

## Quick Start

1. **First time**: Run the example to understand the feature
   ```bash
   python examples/time_offset_simulation_example.py
   ```

2. **Test with 1 hour offset**:
   ```bash
   cd src/real_time
   python trading_simulation.py --time_offset 60
   ```

3. **Stop simulation**: Press `Ctrl+C` at any time

## Integration with Existing Features

The time offset feature is fully integrated with all existing trading simulation features:

- ✅ Technical assessments use simulated time
- ✅ Model inference works with historical data
- ✅ Feature calculations use appropriate time windows
- ✅ Data freshness checks relative to simulated time
- ✅ All debugging and logging functions work correctly

## Help

For detailed help and all available options:
```bash
cd src/real_time
python trading_simulation.py --help
``` 