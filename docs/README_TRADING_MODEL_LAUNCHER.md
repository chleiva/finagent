# 🚀 Trading Model Launcher - Complete Solution

A comprehensive, user-friendly launcher that orchestrates the entire ML pipeline from dataset preparation to model training and evaluation. This solution provides both interactive and command-line interfaces for easy experimentation with different stock combinations, time periods, and training configurations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Modes](#usage-modes)
- [Command-Line Options](#command-line-options)
- [Stock Categories](#stock-categories)
- [Pipeline Architecture](#pipeline-architecture)
- [File Structure](#file-structure)
- [Examples](#examples)
- [Integration](#integration)
- [Troubleshooting](#troubleshooting)
- [Performance Notes](#performance-notes)
- [Contributing](#contributing)

## 🎯 Overview

The Trading Model Launcher is designed to simplify the process of training machine learning models on financial data. It combines two main functionalities:

1. **Dataset Preparation**: Smart concatenation of monthly CSV files based on stock symbols and time periods
2. **Model Training**: Automated training pipeline with comprehensive evaluation and logging

The solution addresses the need for a flexible, user-friendly interface that can handle various use cases from quick testing to comprehensive model training.

## ✨ Features

### Core Features
- **Smart Dataset Concatenation**: Automatically combines monthly CSV files based on filters
- **Interactive Stock Selection**: Choose stocks by category, search, or manual selection
- **Flexible Time Filtering**: Select specific years/months or use all available data
- **Model Training Integration**: Seamlessly triggers the optimized training pipeline
- **Comprehensive Logging**: Tracks all runs with detailed metrics and performance data
- **Error Handling**: Robust error handling with clear feedback and validation

### User Experience Features
- **Multiple Modes**: Interactive, command-line, and quick modes for different use cases
- **Beautiful Interface**: ASCII art banners and clear progress indicators
- **Progress Tracking**: Real-time feedback on processing status
- **Validation**: Comprehensive parameter and file validation
- **Documentation**: Extensive help and examples

### Technical Features
- **Pandas Integration**: Efficient data processing and concatenation
- **Subprocess Management**: Robust execution of external scripts
- **File Management**: Automatic timestamped output files
- **Error Recovery**: Graceful handling of failures
- **Performance Optimization**: Fast mode for development and testing

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Required Python packages (see requirements.txt)
- Monthly CSV files in the format: `monthly_SYMBOL_YYYY-MM.csv`

### Setup
1. Ensure all required files are in the same directory:
   ```
   trading_model_launcher.py
   concatenate_all.py
   model_training_15Jul_optimized1M.py
   prepare.py
   train.py
   evaluate.py
   ```

2. Verify monthly CSV files are present:
   ```bash
   ls monthly_*.csv
   ```

3. Test the installation:
   ```bash
   python trading_model_launcher.py --help
   ```

## 🚀 Quick Start

### Interactive Mode (Recommended for new users)
```bash
python trading_model_launcher.py
```
This will guide you through each step with prompts and options.

### Command-Line Mode
```bash
# Train on specific stocks and time period
python trading_model_launcher.py --stocks AAPL,MSFT,GOOGL --year 2025 --month 01 --description "Tech stocks Q1 2025"

# Quick mode - all stocks, all time, fast training
python trading_model_launcher.py --quick --description "Full dataset training"

# Fast mode for testing
python trading_model_launcher.py --stocks AAPL --year 2025 --fast --description "Quick test"
```

### Demo Mode
```bash
python demo_launcher.py
```
Run through various examples to understand different use cases.

## 📊 Usage Modes

### 1. Interactive Mode
The most user-friendly option that guides you through each step:

#### Step 1: Stock Selection
Choose from multiple options:
- **Category Selection**: Pre-defined groups (Tech Giants, Semiconductors, etc.)
- **Search**: Find stocks by partial name
- **Manual Selection**: Direct input of stock symbols
- **All Stocks**: Select everything available

#### Step 2: Time Period Selection
- **Year Selection**: Choose specific year or all years
- **Month Selection**: Choose specific month or all months
- **Auto-validation**: Checks file existence for selected parameters

#### Step 3: Model Configuration
- **Description**: Provide meaningful description for tracking
- **Fast Mode**: Choose between fast and full training
- **Test CSV**: Optional test file for evaluation

#### Step 4: Confirmation
Review all settings before execution

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

## ⚙️ Command-Line Options

| Option | Type | Description | Example |
|--------|------|-------------|---------|
| `--stocks` | string | Comma-separated stock symbols | `AAPL,MSFT,GOOGL` |
| `--year` | string | Year filter | `2025` |
| `--month` | string | Month filter | `01` |
| `--description` | string | Model description | `"Tech stocks Q1 2025"` |
| `--fast` | flag | Use fast training mode | (no value) |
| `--test-csv` | string | Test CSV file path | `test_data.csv` |
| `--quick` | flag | Quick mode (all data, fast training) | (no value) |

### Option Details

#### `--stocks`
- Accepts comma-separated list of stock symbols
- Case-insensitive (automatically converted to uppercase)
- Validates against available stocks
- Examples: `AAPL`, `AAPL,MSFT`, `AAPL,MSFT,GOOGL,NVDA`

#### `--year` and `--month`
- Filters data by specific time periods
- If omitted, uses all available data
- Validates against existing files
- Examples: `2025`, `01`, `12`

#### `--description`
- Required for tracking and identification
- Used in logging and model comparison
- Should be descriptive and meaningful
- Examples: `"Tech giants 2025 analysis"`, `"Semiconductor sector Dec 2024"`

#### `--fast`
- Enables fast training mode
- Skips expensive optimizations
- Useful for development and testing
- Reduces training time significantly

#### `--test-csv`
- Optional test dataset for evaluation
- Should contain 'buy' column for full metrics
- Used for comprehensive model evaluation
- Examples: `test_data.csv`, `validation_set.csv`

#### `--quick`
- Convenience flag for comprehensive training
- Equivalent to: all stocks + all time + fast mode
- Good for initial testing and validation
- Automatically sets description if not provided

## 📈 Stock Categories

The launcher organizes stocks into logical categories for easier selection:

### Tech Giants (7 stocks)
- AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
- Large-cap technology companies

### Semiconductors (7 stocks)
- AMD, INTC, NVDA, ASML, KLAC, MRVL, QCOM
- Semiconductor manufacturers and suppliers

### Software (7 stocks)
- ADBE, CRM, SNOW, NET, PLTR, ZS, WDAY
- Software and cloud services companies

### Fintech (4 stocks)
- PYPL, COIN, SOFI, PUBM
- Financial technology companies

### Entertainment (3 stocks)
- NFLX, ROKU, MELI
- Entertainment and streaming services

### Transportation (2 stocks)
- UBER, TSLA
- Transportation and mobility companies

### Biotech (3 stocks)
- BIIB, VRTX, ANGO
- Biotechnology and pharmaceutical companies

### All Stocks (48+ stocks)
- Complete list of all available stocks
- Includes additional stocks not in specific categories

## 🔧 Pipeline Architecture

The launcher executes a complete ML pipeline with the following stages:

### Stage 1: Dataset Preparation
```
Input: Stock symbols, time periods
↓
File Discovery: Find matching monthly CSV files
↓
Validation: Check file existence and parameters
↓
Concatenation: Combine files using pandas
↓
Output: Timestamped training dataset
```

### Stage 2: Model Training
```
Input: Prepared dataset
↓
Data Preparation: Feature engineering and splitting
↓
Model Training: Optimized training pipeline
↓
Evaluation: Comprehensive model assessment
↓
Output: Trained model and evaluation results
```

### Stage 3: Logging and Tracking
```
Input: Training results
↓
Metrics Extraction: Performance metrics and statistics
↓
Log Entry: Create comprehensive log entry
↓
File Storage: Save to CSV log file
↓
Output: Updated training log
```

## 📁 File Structure

### Core Files
```
trading_model_launcher.py          # Main launcher application
concatenate_all.py                 # Enhanced dataset concatenation
model_training_15Jul_optimized1M.py # Optimized training pipeline
demo_launcher.py                   # Demonstration script
README_launcher.md                 # This documentation
```

### Supporting Files
```
prepare.py                         # Data preparation module
train.py                          # Model training module
evaluate.py                       # Model evaluation module
model_comparison_tool.py          # Model comparison utility
view_model_log.py                 # Log viewing utility
```

### Output Files
```
training_dataset_YYYYMMDD_HHMMSS.csv  # Generated training dataset
model_training_log.csv                # Training run log
evaluation_results/                   # Evaluation output directory
trained_model/                        # Trained model directory
```

## 💡 Examples

### Example 1: Tech Sector Analysis
```bash
python trading_model_launcher.py \
  --stocks AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA \
  --year 2025 \
  --description "Tech giants 2025 analysis"
```
**Use Case**: Analyze the performance of major technology companies in 2025.

### Example 2: Semiconductor Focus
```bash
python trading_model_launcher.py \
  --stocks AMD,INTC,NVDA,ASML,KLAC,MRVL,QCOM \
  --year 2024 \
  --month 12 \
  --description "Semiconductor sector Dec 2024"
```
**Use Case**: Focus on semiconductor industry performance in December 2024.

### Example 3: Single Stock Deep Dive
```bash
python trading_model_launcher.py \
  --stocks AAPL \
  --year 2025 \
  --month 01 \
  --fast \
  --description "AAPL January 2025 quick test"
```
**Use Case**: Quick analysis of Apple's performance in January 2025.

### Example 4: Comprehensive Training
```bash
python trading_model_launcher.py \
  --quick \
  --description "Full dataset comprehensive training"
```
**Use Case**: Comprehensive training on all available data for initial model development.

### Example 5: Sector Comparison
```bash
# Tech vs Semiconductor comparison
python trading_model_launcher.py \
  --stocks AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA \
  --year 2025 \
  --description "Tech giants 2025"

python trading_model_launcher.py \
  --stocks AMD,INTC,NVDA,ASML,KLAC,MRVL,QCOM \
  --year 2025 \
  --description "Semiconductors 2025"
```
**Use Case**: Compare performance between different sectors.

## 🔗 Integration

### Existing Tools Integration
The launcher integrates seamlessly with your existing pipeline:

- **concatenate_all.py**: Enhanced for multiple stock support
- **model_training_15Jul_optimized1M.py**: Full training pipeline
- **model_comparison_tool.py**: Analyze and compare runs
- **view_model_log.py**: View training history

### Workflow Integration
```bash
# 1. Prepare and train model
python trading_model_launcher.py --stocks AAPL,MSFT --year 2025 --description "Test run"

# 2. View training log
python view_model_log.py

# 3. Compare models
python model_comparison_tool.py

# 4. Analyze results
ls evaluation_results/
```

### Automation Integration
```bash
# Script for automated training
#!/bin/bash
python trading_model_launcher.py \
  --stocks AAPL,MSFT,GOOGL \
  --year 2025 \
  --fast \
  --description "Daily automated run $(date +%Y-%m-%d)"
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. "No files found" Error
**Problem**: No monthly CSV files match the specified criteria
**Solution**: 
- Check that monthly CSV files exist in the current directory
- Verify file naming format: `monthly_SYMBOL_YYYY-MM.csv`
- Use `ls monthly_*.csv` to see available files

#### 2. "Invalid stocks" Error
**Problem**: Specified stock symbols don't exist
**Solution**:
- Check available stocks: `python trading_model_launcher.py --help`
- Verify stock symbols are correct (case-insensitive)
- Use interactive mode to see available options

#### 3. "Required files not found" Error
**Problem**: Missing core pipeline files
**Solution**:
- Ensure all required files are in the same directory
- Check file permissions
- Verify Python environment

#### 4. Memory Issues
**Problem**: Out of memory during processing
**Solution**:
- Use `--fast` mode to reduce memory usage
- Process smaller subsets of data
- Increase system memory if possible

#### 5. Slow Performance
**Problem**: Processing takes too long
**Solution**:
- Use `--fast` mode for quicker results
- Reduce the number of stocks or time period
- Use `--quick` mode for initial testing

### Debugging Tips

1. **Start Small**: Begin with single stock and short time period
2. **Use Fast Mode**: Enable `--fast` for quicker debugging
3. **Check Logs**: Review `model_training_log.csv` for error details
4. **Validate Files**: Ensure all required files are present
5. **Monitor Resources**: Watch memory and disk usage

### Getting Help

1. **Check Documentation**: Review this README and help messages
2. **Use Interactive Mode**: Get guided through the process
3. **Run Demos**: Use `python demo_launcher.py` for examples
4. **Check Logs**: Review training logs for detailed information

## ⚡ Performance Notes

### Dataset Size Considerations
- **Small datasets** (< 1GB): Fast processing, minimal memory usage
- **Medium datasets** (1-5GB): Moderate processing time, reasonable memory
- **Large datasets** (> 5GB): Longer processing time, high memory usage

### Optimization Tips
1. **Use Fast Mode**: Significantly reduces training time
2. **Limit Time Periods**: Process shorter time ranges for quicker results
3. **Selective Stocks**: Choose specific stocks rather than all
4. **Monitor Resources**: Watch system resources during processing

### Memory Management
- **Available Memory**: Ensure sufficient RAM for dataset size
- **Disk Space**: Check available disk space for output files
- **Processing Time**: Large datasets may take significant time

### Scaling Considerations
- **Parallel Processing**: Consider splitting large datasets
- **Batch Processing**: Process multiple smaller runs
- **Resource Monitoring**: Monitor system resources during execution

## 🤝 Contributing

### Development Guidelines
1. **Code Style**: Follow PEP 8 guidelines
2. **Documentation**: Update documentation for new features
3. **Testing**: Test with various scenarios and edge cases
4. **Error Handling**: Implement robust error handling

### Feature Requests
1. **New Stock Categories**: Add relevant stock groupings
2. **Additional Filters**: Implement new filtering options
3. **Enhanced Logging**: Improve logging and tracking features
4. **Performance Optimization**: Optimize for speed and memory usage

### Bug Reports
1. **Reproduce**: Provide steps to reproduce the issue
2. **Environment**: Include system and Python version information
3. **Logs**: Attach relevant log files and error messages
4. **Data**: Provide sample data if relevant

## 📄 License

This project is part of the Proper Trading V1 system. Please refer to the main project license for usage terms.

## 🙏 Acknowledgments

- Built on top of the existing ML pipeline infrastructure
- Enhanced with user experience improvements
- Integrated with comprehensive logging and evaluation systems

---

**Happy Trading! 📈**

For questions, issues, or contributions, please refer to the main project documentation or create an issue in the project repository. 