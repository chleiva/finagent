# Stock Minute-by-Minute Feature Processor

This application processes stock data minute by minute for a given symbol and date, creating comprehensive trading features using the FeatureCalculator from `src/features.py`.

## Features Generated

The application generates 31 different trading features for each minute:

### Microstructure Features
- **Bid_Ask_Spread**: Current bid-ask spread
- **Order_Book_Imbalance**: Imbalance at NBBO
- **Price_Position_In_Spread**: Position of last trade within spread
- **Mid_Price_Momentum**: Rate of change in bid-ask midpoint
- **Spread_Percentile_20**: Current spread percentile vs 20-period distribution
- **Volume_Weighted_Order_Book_Imbalance**: Volume-weighted order book imbalance
- **Trade_Count_Per_Minute**: Number of trades per minute
- **Volume_Rate_of_Change**: Volume rate of change
- **Price_Impact**: Price impact per unit volume

### Volume Features
- **Volume_vs_SMA10**: Volume vs 10-period SMA
- **VWAP**: Volume Weighted Average Price
- **Normalized_Price_VWAP**: Price normalized by VWAP

### Technical Indicators
- **MACD_12_26_9**: MACD histogram
- **RSI_14**: 14-period RSI
- **Bollinger_Bands_Upper/Lower**: Bollinger bands levels
- **Bollinger_Band_Width**: Band width
- **Price_vs_SMA50**: Price vs 50-period SMA

### Return Features
- **Return_5min**: 5-minute return
- **Return_Since_Open**: Return since market open

### Volatility Features
- **ATR_14**: 14-period Average True Range
- **Volatility_Regime_Shift**: Volatility regime shift detection
- **Volatility_Percentile_100**: ATR percentile over 100 periods
- **ATR_Stop_Risk_Pct**: Volatility-normalized stop distance

### Price Pattern Features
- **Position_In_Day_Range**: Position within day's range
- **Support_Resistance_Proximity**: Distance to nearest S/R level
- **Breakout_Confirmation**: Volume-confirmed breakout detection
- **Price_Acceleration**: Second derivative of price movement
- **Max_Drawdown_30**: Maximum drawdown over 30 bars

### Time Features
- **Time_Of_Day_Cyclical_Sin/Cos**: Cyclical time encoding

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Prepare data files**:
```bash
python prepare_data.py
```

This will create two files:
- `data/intraday.csv` - Minute-level intraday data
- `data/daily.csv` - Daily OHLCV data

## Usage

### Basic Usage

Process a specific symbol and date:
```bash
python stock_minute_processor.py --symbol AAPL --date 2024-07-15
```

### Command Line Options

```bash
python stock_minute_processor.py [OPTIONS]

Options:
  -s, --symbol TEXT        Stock symbol to process
  -d, --date TEXT          Date to process (YYYY-MM-DD format)
  -i, --intraday-file TEXT Path to intraday CSV file [default: data/intraday.csv]
  -f, --daily-file TEXT    Path to daily CSV file [default: data/daily.csv]
  -o, --output TEXT        Output CSV file path [default: features_SYMBOL_DATE.csv]
  --list-symbols           List available symbols
  --list-dates TEXT        List available dates for a symbol
  --help                   Show this message and exit
```

### Examples

1. **List available symbols**:
```bash
python stock_minute_processor.py --list-symbols
```

2. **List available dates for a symbol**:
```bash
python stock_minute_processor.py --list-dates AAPL
```

3. **Process with custom output file**:
```bash
python stock_minute_processor.py --symbol AAPL --date 2024-07-15 --output my_features.csv
```

4. **Process with custom data files**:
```bash
python stock_minute_processor.py --symbol AAPL --date 2024-07-15 \
  --intraday-file /path/to/intraday.csv \
  --daily-file /path/to/daily.csv
```

## Data Processing Logic

For each minute of the trading day, the application:

1. **Creates three data sources**:
   - `real_time_df`: Current minute's data (simulating real-time feed)
   - `intra_day_df`: All data up to (but not including) current minute
   - `daily_df`: Previous days' daily summaries

2. **Calculates features**: Uses the FeatureCalculator to compute all 31 features

3. **Saves results**: Appends features with metadata (symbol, timestamp, date) to CSV

## Output Format

The output CSV contains:
- 31 feature columns (as listed above)
- 3 metadata columns:
  - `symbol`: Stock symbol
  - `timestamp`: Minute timestamp
  - `date`: Trading date

## Sample Output

```csv
Bid_Ask_Spread,Order_Book_Imbalance,Price_Position_In_Spread,...,symbol,timestamp,date
0.200,0.915,5.600,...,AAPL,2024-07-15 08:00:00+00:00,2024-07-15
0.260,0.905,1.923,...,AAPL,2024-07-15 08:01:00+00:00,2024-07-15
...
```

## Performance

- **Processing speed**: ~0.005 seconds per minute
- **Memory usage**: Loads full dataset into memory for efficient processing
- **Progress tracking**: Real-time progress bar with ETA

## Error Handling

- Graceful handling of missing data
- Continues processing on individual minute errors
- Comprehensive error reporting in summary
- Detailed success/failure statistics

## Data Requirements

### Intraday Data Format
Required columns:
- `ts_event_clean`: Timestamp
- `symbol_price` or `symbol`: Stock symbol
- `open_1min`, `high_1min`, `low_1min`, `close_1min`: OHLC data
- `volume_1min`: Volume data
- `bid_px_00`, `ask_px_00`: Bid/ask prices
- `bid_sz_00`, `ask_sz_00`: Bid/ask sizes

### Daily Data Format
Required columns:
- `symbol`: Stock symbol
- `date`: Date
- `open`, `high`, `low`, `close`: OHLC data
- `volume`: Volume data

## Troubleshooting

1. **No data found**: Check symbol and date availability using `--list-symbols` and `--list-dates`
2. **Feature calculation errors**: Verify data quality and column names
3. **Memory issues**: Process smaller date ranges or use data chunking
4. **Performance issues**: Consider processing data in parallel for multiple symbols

## Dependencies

- `pandas>=1.5.0`: Data manipulation
- `numpy>=1.20.0`: Numerical computations
- `tqdm>=4.60.0`: Progress bars 