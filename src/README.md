# Proper Trading V1 - Source Code Documentation

This document provides comprehensive documentation for all main Python files in the `src/` directory and its subfolders. The codebase implements a complete machine learning trading system with real-time data processing, feature engineering, model training, and trading simulation capabilities.

## 📁 Directory Structure

```
src/
├── data_processing/          # Data preparation and preprocessing
├── feature_engineering/      # Feature calculation and optimization
├── inference/               # Model inference and prediction
├── utils/                   # Utility functions and model management
├── scripts/                 # Batch processing and automation
├── evaluation/              # Model evaluation and analysis
├── model_training/          # Model training pipelines
├── launchers/              # Application launchers and orchestrators
├── real_time/              # Real-time trading and data collection
├── trading_simulation/     # Historical simulation and backtesting
├── models/                 # Model definitions (placeholder)
└── deprecated/             # Legacy code (not documented)
```

## 🔧 Core Components

### 1. Data Processing (`data_processing/`)

#### `prepare.py` - Main Data Preparation Pipeline
**Purpose**: Orchestrates the complete data preparation pipeline for model training.

**Key Features**:
- Loads and validates raw data with memory optimizations
- Applies data cleaning, outlier treatment, and normalization
- Implements intelligent sampling strategies
- Creates train/validation/test splits
- Saves prepared data artifacts

**Usage**:
```python
from data_processing.prepare import prepare_data
prepare_data("input_data.csv", output_dir="prepared_data")
```

**Arguments**:
- `df_path`: Path to input CSV file
- `output_dir`: Output directory for prepared data (default: "prepared_data")
- `top_n_features`: Number of top features to select (default: 20)
- `sample_size`: Number of samples to use (default: 200000)
- `sampling_strategy`: Sampling strategy ('balanced', 'stratified', etc.)
- `fast`: Fast mode for debugging (reduces sample size and features)

#### `concatenate_all.py` - Data Concatenation Utility
**Purpose**: Combines multiple data files into a single dataset for training.

**Key Features**:
- Merges monthly data files by stock symbol
- Filters by date ranges and stock symbols
- Handles large datasets efficiently
- Validates data integrity

### 2. Feature Engineering (`feature_engineering/`)

#### `features.py` - Core Feature Calculator
**Purpose**: Calculates comprehensive trading features from real-time, intraday, and daily data.

**Key Features**:
- **Microstructure Features**: Bid-ask spread, order book imbalance, price position
- **Volume Features**: Volume percentiles, VWAP, volume rate of change
- **Technical Indicators**: MACD, RSI, Bollinger Bands, SMA comparisons
- **Return Features**: 5-minute returns, returns since market open
- **Volatility Features**: ATR, volatility percentiles, regime shifts
- **Risk Management**: Stop loss calculations, drawdown analysis
- **Pattern Features**: Support/resistance proximity, breakout confirmation
- **Time Features**: Cyclical time encoding for market hours

**Usage**:
```python
from feature_engineering.features import calculate_features
features = calculate_features(real_time_df, intra_day_df, daily_df)
```

#### `buy_label.py` - Buy Signal Assessment
**Purpose**: Evaluates whether current market conditions meet basic buy criteria.

**Key Features**:
- Market hours validation (9:30 AM - 4:00 PM ET)
- Technical condition checks (price levels, volume, spread)
- Risk management validation (stop loss, targets)
- Future price action evaluation for signal labeling

**Usage**:
```python
from feature_engineering.buy_label import meets_basic_buy_conditions
assessment, reason = meets_basic_buy_conditions(features, current_time)
```

#### `feature_optimizer.py` - Feature Optimization
**Purpose**: Optimizes and enhances features for better model performance.

**Key Features**:
- Feature interaction calculations
- Advanced feature transformations
- Feature selection optimization
- Performance enhancement techniques

#### `extra_features.py` - Additional Feature Calculations
**Purpose**: Provides supplementary features beyond the core feature set.

**Key Features**:
- Extended technical indicators
- Market microstructure enhancements
- Advanced statistical features
- Custom trading signals

#### `stock_minute_processor_parallel.py` - Parallel Data Processing
**Purpose**: Processes large datasets in parallel for feature engineering.

**Key Features**:
- **Parallel Processing**: Multi-core processing for large datasets
- **Memory Optimization**: Efficient memory management for large files
- **Progress Tracking**: Real-time progress monitoring with tqdm
- **Performance Monitoring**: CPU and memory usage tracking
- **Batch Processing**: Processes multiple symbols and dates efficiently
- **Data Validation**: Comprehensive data quality checks

**Usage**:
```bash
# Process single symbol and date
python src/feature_engineering/stock_minute_processor_parallel.py --symbol AAPL --date 2025-01-15

# Process all symbols for a date range
python src/feature_engineering/stock_minute_processor_parallel.py --start-date 2025-01-01 --end-date 2025-01-31

# Merge monthly files
python src/feature_engineering/stock_minute_processor_parallel.py --merge
```

**Arguments**:
- `--symbol`: Stock symbol to process
- `--date`: Specific date to process (YYYY-MM-DD)
- `--start-date`: Start date for range processing
- `--end-date`: End date for range processing
- `--merge`: Merge monthly files into single dataset
- `--workers`: Number of parallel workers (default: auto-detect)

#### `features_cleaner.py` - Feature Cleaning and Validation
**Purpose**: Cleans and validates features for model training.

**Key Features**:
- **Data Cleaning**: Removes outliers and invalid values
- **Feature Validation**: Ensures feature consistency
- **Missing Value Handling**: Imputes missing values appropriately
- **Data Quality Checks**: Comprehensive validation pipeline

### 3. Inference (`inference/`)

#### `model_inference_adapter.py` - Model Inference Interface
**Purpose**: Provides a unified interface for real-time model inference.

**Key Features**:
- Loads trained models from artifacts
- Computes features for live data
- Runs predictions with configurable thresholds
- Handles missing features gracefully
- Returns comprehensive prediction results

**Usage**:
```python
from inference.model_inference_adapter import ModelInferenceAdapter
adapter = ModelInferenceAdapter("model_artifacts/model_index.csv")
result = adapter.infer("AAPL", real_time_df, intra_day_df, daily_df)
```

### 4. Utilities (`utils/`)

#### `model_wrapper.py` - Model Management
**Purpose**: Wraps trained models for easy access and prediction.

**Key Features**:
- Loads model artifacts from index
- Provides prediction interface
- Extracts model metadata and performance metrics
- Handles feature importance analysis
- Manages optimal thresholds

**Usage**:
```python
from utils.model_wrapper import ModelWrapper
model = ModelWrapper("AAPL")
prediction, probability = model.predict(data, threshold=0.5)
```

#### `model_manager.py` - Model Artifact Management
**Purpose**: Manages model artifacts, indexing, and versioning.

**Key Features**:
- Creates and maintains model index
- Handles model versioning
- Provides model discovery and selection
- Manages model metadata

#### `utils.py` - General Utilities
**Purpose**: Provides common utility functions for data processing.

**Key Features**:
- Optimized data loading
- Outlier treatment
- Missing value imputation
- Feature selection and normalization
- Intelligent sampling strategies

### 5. Model Training (`model_training/`)

#### `model_training_15Jul_optimized1M.py` - Main Training Pipeline
**Purpose**: Orchestrates the complete model training pipeline with optimizations.

**Key Features**:
- **Optuna Hyperparameter Optimization**: Automated hyperparameter tuning
- **Multiple Model Types**: XGBoost, LightGBM, Gradient Boosting
- **Cross-Validation**: Stratified k-fold validation
- **Feature Selection**: Automatic feature importance analysis
- **Model Artifacts**: Saves complete model packages
- **Performance Logging**: Comprehensive training metrics

**Usage**:
```python
from model_training.model_training_15Jul_optimized1M import train_model
train_model("prepared_data/data_splits.pkl", "trained_model")
```

**Arguments**:
- `prepared_data_path`: Path to prepared data splits (default: "prepared_data/data_splits.pkl")
- `output_dir`: Output directory for trained models (default: "trained_model")
- `n_trials`: Number of Optuna optimization trials (default: 50)
- `cv_folds`: Number of cross-validation folds (default: 5)
- `fast`: Fast mode for debugging (reduces trials and folds)

### 6. Evaluation (`evaluation/`)

#### `evaluate.py` - Model Evaluation
**Purpose**: Comprehensive model evaluation and performance analysis.

**Key Features**:
- **Performance Metrics**: AUC, F1, precision, recall, confusion matrix
- **ROC Analysis**: ROC curves and threshold optimization
- **Feature Importance**: SHAP analysis and feature rankings
- **Custom Test Sets**: Support for external test data
- **Visualization**: Performance plots and charts

**Usage**:
```python
from evaluation.evaluate import evaluate_model
evaluate_model("trained_model/model_package.pkl", "prepared_data/data_splits.pkl")
```

#### `model_comparison_tool.py` - Model Comparison
**Purpose**: Compares multiple models and their performance.

**Key Features**:
- Side-by-side model comparison
- Statistical significance testing
- Performance ranking
- Model selection recommendations

#### `streamlit_model_dashboard.py` - Interactive Dashboard
**Purpose**: Provides an interactive web interface for model analysis.

**Key Features**:
- Interactive model exploration
- Real-time performance visualization
- Feature importance analysis
- Prediction analysis tools

### 7. Launchers (`launchers/`)

#### `trading_model_launcher.py` - Main Application Launcher
**Purpose**: Orchestrates the complete ML pipeline from data preparation to model training.

**Key Features**:
- **Interactive Mode**: User-friendly interactive interface
- **Command-Line Mode**: Automated pipeline execution
- **Stock Selection**: Interactive stock selection with categories
- **Time Period Selection**: Flexible date range selection
- **Pipeline Orchestration**: Coordinates all pipeline stages
- **Progress Tracking**: Real-time progress monitoring

**Usage**:
```bash
# Interactive mode
python src/launchers/trading_model_launcher.py

# Command-line mode
python src/launchers/trading_model_launcher.py --stocks AAPL,MSFT --year 2025 --month 01
```

#### `demo_launcher.py` - Demo Application
**Purpose**: Provides a demonstration of the trading system capabilities.

**Key Features**:
- Quick demonstration setup
- Sample data processing
- Model training showcase
- Results visualization

### 8. Real-Time Trading (`real_time/`)

#### `enhanced_data_collector.py` - IBKR Data Collector
**Purpose**: Collects real-time market data from Interactive Brokers (IBKR) API.

**Key Features**:
- **IBKR Integration**: Connects to IBKR WebSocket and REST APIs
- **Real-Time Data**: Fetches live bid/ask, volume, and price data every 10 seconds
- **Intraday Data**: Collects 1-minute bars every 30 seconds
- **Historical Data**: Fetches 30-day historical data from Yahoo Finance
- **Database Storage**: Stores data in SQLite database
- **Authentication**: Handles IBKR session management
- **Multi-threading**: Parallel data collection for multiple symbols

**Prerequisites**:
- IBKR TWS or IB Gateway running on localhost:15000
- IBKR account with market data subscriptions
- Python packages: `websocket-client`, `requests`, `yfinance`

**Usage**:
```bash
# Start the data collector (must run before real-time trading)
python src/real_time/enhanced_data_collector.py
```

**Configuration**:
- **Symbols**: Top 10 NASDAQ stocks (configurable in SYMBOLS list)
- **Database**: `database/realtime_market_data.db`
- **IBKR URL**: `https://localhost:15000/v1/api`
- **WebSocket URL**: `wss://localhost:15000/v1/api/ws`

#### `real_time_trading.py` - Live Trading System
**Purpose**: Implements real-time trading with live market data and ML predictions.

**Key Features**:
- **Real-Time Data**: Fetches live market data from database
- **Technical Assessment**: Real-time buy condition evaluation using `buy_label.py`
- **Model Inference**: Live prediction generation using trained models
- **Feature Engineering**: Real-time feature calculation and optimization
- **Risk Management**: Live risk monitoring and position sizing
- **Data Freshness**: Monitors data latency and quality
- **Comprehensive Display**: Real-time table with predictions and metrics

**Prerequisites**:
- Enhanced data collector must be running
- Trained models available in `model_artifacts/`
- Database populated with real-time data

**Usage**:
```bash
# Start real-time trading (after data collector is running)
python src/real_time/real_time_trading.py
```

**Arguments**: None (configured via constants in the file)

**Configuration**:
- **Symbols**: Top 10 NASDAQ stocks
- **Fetch Interval**: 30 seconds
- **Database**: `database/realtime_market_data.db`
- **Model Path**: `model_artifacts/model_index.csv`

#### `data_collector_open_ai.py` - Alternative Data Source
**Purpose**: Collects data from alternative sources (OpenAI integration).

**Key Features**:
- Alternative data collection methods
- OpenAI API integration
- Backup data sources

### Real-Time Trading Setup Guide

#### Step 1: Start IBKR Data Collection
```bash
# 1. Ensure IBKR TWS or IB Gateway is running on localhost:15000
# 2. Login to IBKR and authenticate
# 3. Start the data collector
python src/real_time/enhanced_data_collector.py
```

**Expected Output**:
```
🚀 Starting IBKR Data Collector
📊 Symbols: NVDA, MSFT, AAPL, AMZN, GOOGL, META, AVGO, TSLA, NFLX, COST
🗄️ Database: database/realtime_market_data.db
✅ Authenticated with session: abc123def456...
✅ NVDA: Contract ID 76792991
✅ MSFT: Contract ID 272093
...
📈 NVDA: Stored 390 minute bars
📈 MSFT: Stored 390 minute bars
...
✅ Data collection started. Press Ctrl+C to stop.
```

#### Step 2: Start Real-Time Trading
```bash
# In a new terminal, start the trading system
python src/real_time/real_time_trading.py
```

**Expected Output**:
```
🚦 Trading System (REAL DATA MODE ONLY)
============================================================
Symbols: NVDA, MSFT, AAPL, AMZN, GOOGL, META, AVGO, TSLA, NFLX, COST
Fetching REAL data every 30 seconds. Press Ctrl+C to stop.

✅ Technical assessment passed for: NVDA, MSFT, AAPL

┌────────┬────────────┬────────────┬────────────────┬──────────┬────────────┬────────┬──────────────────┬───────────┐
│ Symbol │ Last Price │ Last Update│ Tech Assessment │ Prediction│ Probability│ Vol %  │ Missing Features │ Threshold │
├────────┼────────────┼────────────┼────────────────┼──────────┼────────────┼────────┼──────────────────┼───────────┤
│ NVDA   │ $485.09    │ 14:30:15   │ ✅ PASS        │ BUY      │ 0.8234     │ 0.756  │ 0                │ 0.500     │
│ MSFT   │ $378.85    │ 14:30:12   │ ✅ PASS        │ NO BUY   │ 0.2341     │ 0.623  │ 0                │ 0.500     │
...
```

#### Step 3: Monitor and Manage
- **Data Freshness**: Monitor latency and data quality
- **Buy Signals**: Watch for BUY predictions with high probability
- **Technical Assessment**: Ensure symbols pass basic conditions
- **Missing Features**: Address any feature calculation issues

#### Troubleshooting Real-Time Trading

**Common Issues**:
1. **"Not authenticated"**: Ensure IBKR TWS is running and logged in
2. **"No real-time data found"**: Check data collector is running
3. **"Model not found"**: Train models before running real-time trading
4. **"Missing features"**: Check feature engineering pipeline
5. **"Database connection error"**: Verify database path and permissions

**Debug Mode**:
```bash
# Check data freshness
python -c "from src.real_time.real_time_trading import check_data_freshness; print(check_data_freshness())"

# Test single symbol
python -c "from src.real_time.real_time_trading import fetch_real_time_data; print(fetch_real_time_data('AAPL'))"
```

### 9. Trading Simulation (`trading_simulation/`)

#### `trading_simulation.py` - Historical Simulation
**Purpose**: Runs historical backtesting and simulation.

**Key Features**:
- **Historical Mode**: Exact date/time simulation
- **Position Management**: Buy/sell execution simulation
- **Risk Management**: Stop loss and take profit simulation
- **Performance Tracking**: P&L and performance metrics
- **Market Hours**: Realistic market timing

**Usage**:
```bash
python src/trading_simulation/trading_simulation.py --simulation_date 202501161030
```

#### `buy_sell.py` - Position Management
**Purpose**: Manages trading positions and execution logic.

**Key Features**:
- **Position Tracking**: Database-backed position management
- **Buy Logic**: Position sizing and entry execution
- **Sell Logic**: Stop loss, take profit, and time-based exits
- **Risk Management**: Position sizing and risk controls
- **Performance Metrics**: P&L calculation and reporting

**Usage**:
```python
from trading_simulation.buy_sell import PositionManager
pm = PositionManager("database/realtime_market_data.db")
cash = pm.buy(simulation_id, symbol, price, features, time, cash)
```

#### `historical_simulation_fetcher.py` - Historical Data Fetcher
**Purpose**: Fetches historical data for simulation purposes.

**Key Features**:
- Historical data retrieval
- Data validation and cleaning
- Time period filtering
- Data format standardization

### 10. Scripts (`scripts/`)

#### `batch_train_top10.py` - Batch Training
**Purpose**: Trains models for multiple stocks in batch mode.

**Key Features**:
- **Top 10 NASDAQ**: Automated training for major stocks
- **Batch Processing**: Parallel or sequential training
- **Progress Tracking**: Batch progress monitoring
- **Error Handling**: Robust error recovery
- **Results Summary**: Batch training results

**Usage**:
```bash
python src/scripts/batch_train_top10.py
```

## 🔄 Data Flow

### Training Pipeline
1. **Data Preparation** (`data_processing/prepare.py`)
   - Load raw data → Clean → Normalize → Split
2. **Feature Engineering** (`feature_engineering/features.py`)
   - Calculate features → Optimize → Select
3. **Model Training** (`model_training/model_training_15Jul_optimized1M.py`)
   - Train models → Optimize → Evaluate
4. **Model Evaluation** (`evaluation/evaluate.py`)
   - Test performance → Analyze → Visualize

### Real-Time Pipeline
1. **Data Collection** (`real_time/enhanced_data_collector.py`)
   - Fetch live data → Validate → Store
2. **Feature Calculation** (`feature_engineering/features.py`)
   - Compute real-time features
3. **Model Inference** (`inference/model_inference_adapter.py`)
   - Load model → Predict → Return results
4. **Trading Logic** (`real_time/real_time_trading.py`)
   - Assess conditions → Execute trades → Manage positions

### Simulation Pipeline
1. **Historical Data** (`trading_simulation/historical_simulation_fetcher.py`)
   - Load historical data → Validate → Prepare
2. **Simulation Engine** (`trading_simulation/trading_simulation.py`)
   - Process data → Calculate features → Run inference
3. **Position Management** (`trading_simulation/buy_sell.py`)
   - Execute trades → Track positions → Calculate P&L

## 🚀 Quick Start

### 1. Train a Model
```bash
# Interactive mode
python src/launchers/trading_model_launcher.py

# Command-line mode
python src/launchers/trading_model_launcher.py --stocks AAPL,MSFT --year 2025 --month 01
```

### 2. Run Real-Time Trading
```bash
# Step 1: Start IBKR data collection (requires IBKR TWS running)
python src/real_time/enhanced_data_collector.py

# Step 2: In a new terminal, start real-time trading
python src/real_time/real_time_trading.py
```

### 3. Run Historical Simulation
```bash
python src/trading_simulation/trading_simulation.py --simulation_date 202501161030
```

### 4. Evaluate Models
```bash
python src/evaluation/evaluate.py
```

## 🔄 Complete Workflow Guide

### Phase 1: Data Preparation
```bash
# 1. Concatenate monthly data files
python src/data_processing/concatenate_all.py --stocks AAPL,MSFT,GOOGL --year 2025 --month 01

# 2. Process features in parallel (for large datasets)
python src/feature_engineering/stock_minute_processor_parallel.py --start-date 2025-01-01 --end-date 2025-01-31

# 3. Prepare data for training
python -c "from src.data_processing.prepare import prepare_data; prepare_data('merged_features.csv', fast=True)"
```

### Phase 2: Model Training
```bash
# 1. Train individual models
python src/model_training/model_training_15Jul_optimized1M.py merged_features.csv --fast

# 2. Or use the launcher for multiple stocks
python src/launchers/trading_model_launcher.py --stocks AAPL,MSFT,GOOGL --year 2025 --month 01 --description "Tech stocks Q1 2025"

# 3. Batch train top 10 NASDAQ stocks
python src/scripts/batch_train_top10.py
```

### Phase 3: Model Evaluation
```bash
# 1. Evaluate trained models
python src/evaluation/evaluate.py

# 2. Compare multiple models
python src/evaluation/model_comparison_tool.py

# 3. Launch interactive dashboard
streamlit run src/evaluation/streamlit_model_dashboard.py
```

### Phase 4: Real-Time Trading Setup
```bash
# 1. Ensure IBKR TWS is running on localhost:15000
# 2. Start data collection
python src/real_time/enhanced_data_collector.py

# 3. In new terminal, start trading
python src/real_time/real_time_trading.py
```

### Phase 5: Historical Simulation
```bash
# Run simulation for specific date/time
python src/trading_simulation/trading_simulation.py --simulation_date 202501161030
```

## 📊 Key Features

### Machine Learning
- **Multiple Algorithms**: XGBoost, LightGBM, Gradient Boosting
- **Hyperparameter Optimization**: Optuna-based automated tuning
- **Feature Engineering**: 50+ technical and microstructure features
- **Cross-Validation**: Robust model validation
- **Performance Metrics**: Comprehensive evaluation suite

### Trading System
- **Real-Time Processing**: Live market data processing
- **Risk Management**: Position sizing and stop losses
- **Technical Analysis**: Advanced technical indicators
- **Market Microstructure**: Order book and spread analysis
- **Backtesting**: Historical simulation capabilities

### Data Management
- **Multiple Sources**: Real-time, intraday, and daily data
- **Data Validation**: Comprehensive data quality checks
- **Optimized Storage**: Memory-efficient data handling
- **Batch Processing**: Automated data preparation

## 🔧 Configuration

### Database Configuration
- **Path**: `database/realtime_market_data.db`
- **Tables**: `realtime_summary`, `historical_simulation_intraday_data`, `daily_summary_data`, `positions`

### Model Configuration
- **Artifacts**: `model_artifacts/model_index.csv`
- **Features**: 20+ high-impact features
- **Thresholds**: Configurable prediction thresholds

### Trading Configuration
- **Symbols**: Top 10 NASDAQ stocks
- **Budget**: 5% per trade
- **Slippage**: 0.1% per trade
- **Commission**: $1 per trade

## 📈 Performance

### Model Performance
- **AUC**: Typically 0.65-0.75
- **F1 Score**: Typically 0.60-0.70
- **Training Time**: 5-15 minutes per model
- **Inference Time**: <100ms per prediction

### System Performance
- **Data Processing**: Handles 1M+ samples efficiently
- **Real-Time Latency**: <2 seconds end-to-end
- **Memory Usage**: Optimized for large datasets
- **Scalability**: Supports multiple symbols simultaneously

## 🛠️ Dependencies

### Core Dependencies
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning
- `xgboost`: Gradient boosting
- `lightgbm`: Light gradient boosting
- `optuna`: Hyperparameter optimization

### Trading Dependencies
- `sqlite3`: Database management
- `pytz`: Timezone handling
- `tabulate`: Table formatting

### Visualization Dependencies
- `matplotlib`: Plotting
- `seaborn`: Statistical visualization
- `streamlit`: Interactive dashboards

## 📝 Notes

- All paths are relative to the project root
- Database must be initialized before running real-time or simulation modes
- Model artifacts must be trained before running inference
- Timezone handling uses New York timezone for market hours
- All monetary values are in USD
- All timestamps are in UTC for consistency

## 🔍 Troubleshooting

### Common Issues
1. **Missing Data**: Ensure database is populated with required data
2. **Model Not Found**: Train models before running inference
3. **Feature Mismatch**: Ensure feature engineering is consistent
4. **Timezone Issues**: Verify timezone configuration
5. **Memory Issues**: Use fast mode for large datasets

### Debug Mode
Most scripts support a `--fast` flag for debugging:
```bash
python src/launchers/trading_model_launcher.py --fast
```

This reduces sample sizes and optimization trials for faster execution during development. 