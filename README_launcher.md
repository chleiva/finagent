# 🚀 Trading Model Launcher

A comprehensive, user-friendly launcher that orchestrates the entire ML pipeline from dataset preparation to model training and evaluation.

## Features

- **Smart Dataset Concatenation**: Automatically combines monthly CSV files based on stock symbols and time periods
- **Interactive Stock Selection**: Choose stocks by category, search, or manual selection
- **Flexible Time Filtering**: Select specific years/months or use all available data
- **Model Training Integration**: Seamlessly triggers the optimized model training pipeline
- **Comprehensive Logging**: Tracks all runs with detailed metrics and performance data
- **Error Handling**: Robust error handling with clear feedback
- **Multiple Modes**: Interactive, command-line, and quick modes for different use cases

## Quick Start

### Interactive Mode (Recommended for new users)
```bash
python trading_model_launcher.py
```

### Command-Line Mode
```bash
# Train on specific stocks and time period
python trading_model_launcher.py --stocks AAPL,MSFT,GOOGL --year 2025 --month 01 --description "Tech stocks Q1 2025"

# Quick mode - all stocks, all time, fast training
python trading_model_launcher.py --quick --description "Full dataset training"

# Fast mode for testing
python trading_model_launcher.py --stocks AAPL --year 2025 --fast --description "Quick test"
```

## Available Stocks

The launcher automatically detects available stocks from your monthly CSV files. Currently supports:

### Tech Giants
- AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA

### Semiconductors
- AMD, INTC, NVDA, ASML, KLAC, MRVL, QCOM

### Software
- ADBE, CRM, SNOW, NET, PLTR, ZS, WDAY

### Fintech
- PYPL, COIN, SOFI, PUBM

### Entertainment
- NFLX, ROKU, MELI

### Transportation
- UBER, TSLA

### Biotech
- BIIB, VRTX, ANGO

### And many more...
Total: 48+ stocks available

## Usage Modes

### 1. Interactive Mode
The most user-friendly option that guides you through each step:

1. **Stock Selection**: Choose from categories, search, or manual selection
2. **Time Period**: Select specific years/months or use all data
3. **Model Description**: Provide a meaningful description for tracking
4. **Training Options**: Choose fast mode and optional test CSV
5. **Confirmation**: Review settings before execution

### 2. Command-Line Mode
For automation and scripting:

```bash
# Basic usage
python trading_model_launcher.py --stocks AAPL,MSFT --year 2025 --description "Test run"

# With all options
python trading_model_launcher.py \
  --stocks AAPL,MSFT,GOOGL,NVDA \
  --year 2025 \
  --month 01 \
  --description "Tech sector Q1 2025" \
  --fast \
  --test-csv test_data.csv
```

### 3. Quick Mode
For rapid testing with all available data:

```bash
python trading_model_launcher.py --quick --description "Quick test run"
```

## Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--stocks` | Comma-separated stock symbols | `AAPL,MSFT,GOOGL` |
| `--year` | Year filter | `2025` |
| `--month` | Month filter | `01` |
| `--description` | Model description | `"Tech stocks Q1 2025"` |
| `--fast` | Use fast training mode | (flag) |
| `--test-csv` | Test CSV file path | `test_data.csv` |
| `--quick` | Quick mode (all data, fast training) | (flag) |

## Pipeline Overview

The launcher executes a complete ML pipeline:

1. **Dataset Preparation**
   - Concatenates monthly CSV files based on filters
   - Validates file existence and parameters
   - Creates timestamped output file

2. **Model Training**
   - Runs the optimized training pipeline
   - Uses high-impact features only
   - Performs comprehensive evaluation

3. **Logging & Tracking**
   - Logs all runs to `model_training_log.csv`
   - Tracks performance metrics
   - Enables model comparison

## Output Files

- **Training Dataset**: `training_dataset_YYYYMMDD_HHMMSS.csv`
- **Model Log**: `model_training_log.csv`
- **Evaluation Results**: `evaluation_results/` directory
- **Trained Model**: `trained_model/` directory

## Examples

### Example 1: Tech Sector Analysis
```bash
python trading_model_launcher.py \
  --stocks AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA \
  --year 2025 \
  --description "Tech giants 2025 analysis"
```

### Example 2: Semiconductor Focus
```bash
python trading_model_launcher.py \
  --stocks AMD,INTC,NVDA,ASML,KLAC,MRVL,QCOM \
  --year 2024 \
  --month 12 \
  --description "Semiconductor sector Dec 2024"
```

### Example 3: Single Stock Deep Dive
```bash
python trading_model_launcher.py \
  --stocks AAPL \
  --year 2025 \
  --month 01 \
  --fast \
  --description "AAPL January 2025 quick test"
```

### Example 4: Comprehensive Training
```bash
python trading_model_launcher.py \
  --quick \
  --description "Full dataset comprehensive training"
```

## Error Handling

The launcher includes robust error handling:

- **File Validation**: Checks for required files and data
- **Parameter Validation**: Validates stock symbols and time periods
- **Process Monitoring**: Tracks subprocess execution
- **Error Logging**: Records errors in the training log
- **Graceful Degradation**: Continues with available data when possible

## Integration with Existing Tools

The launcher integrates seamlessly with your existing pipeline:

- **concatenate_all.py**: Enhanced for multiple stock support
- **model_training_15Jul_optimized1M.py**: Full training pipeline
- **model_comparison_tool.py**: Analyze and compare runs
- **view_model_log.py**: View training history

## Tips for Best Results

1. **Start with Interactive Mode**: Use interactive mode to understand the options
2. **Use Meaningful Descriptions**: Good descriptions help track experiments
3. **Test with Fast Mode**: Use `--fast` for quick validation
4. **Monitor the Log**: Check `model_training_log.csv` for run history
5. **Compare Models**: Use the model comparison tool to analyze performance

## Troubleshooting

### Common Issues

1. **"No files found"**: Check that monthly CSV files exist in the current directory
2. **"Invalid stocks"**: Verify stock symbols are in the available list
3. **"Required files not found"**: Ensure `concatenate_all.py` and `model_training_15Jul_optimized1M.py` are present

### Getting Help

- Check the model training log for detailed error messages
- Use `--fast` mode for quicker debugging
- Start with a single stock to isolate issues

## Performance Notes

- **Dataset Size**: Large datasets may take significant time to process
- **Memory Usage**: Monitor memory usage with very large datasets
- **Fast Mode**: Use `--fast` for development and testing
- **Storage**: Ensure sufficient disk space for output files

---

**Happy Trading! 📈** 