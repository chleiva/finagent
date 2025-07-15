# Trading Model Launcher - ML Pipeline

A comprehensive machine learning pipeline for trading model development, training, and deployment with automatic logging and performance tracking.

## 🚀 Quick Start

```bash
# Setup virtual environment and install dependencies
make setup

# Run the main training pipeline
make train

# Or use the convenience script
./scripts/run_training.sh

# Launch the interactive dashboard
streamlit run src/evaluation/streamlit_model_dashboard.py
```

## 📁 Project Structure

```
Proper_Trading_V1/
├── src/                          # Source code
│   ├── data_processing/          # Data preparation and cleaning
│   ├── feature_engineering/      # Feature creation and optimization
│   ├── model_training/           # Model training scripts
│   ├── evaluation/               # Model evaluation and comparison
│   ├── launchers/                # Main application launchers
│   └── utils/                    # Utility functions
├── data/                         # Data storage
│   ├── raw/                      # Raw data files
│   ├── processed/                # Processed datasets
│   └── external/                 # External data sources
├── models/                       # Model artifacts
│   ├── saved/                    # Saved model files
│   └── checkpoints/              # Training checkpoints
├── config/                       # Configuration files
├── docs/                         # Documentation
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks
├── scripts/                      # Shell scripts
├── logs/                         # Application logs
├── model_training_log.csv        # 📊 AUTOMATIC MODEL TRAINING LOG
├── evaluation_results/           # Model evaluation outputs
├── prepared_data/                # Processed training data
└── trained_model/                # Saved trained models
```

## 🛠️ Features

- **Interactive Model Launcher**: User-friendly interface for model training
- **Automated Data Processing**: Concatenation and preparation of training datasets
- **Feature Engineering**: Advanced feature creation and optimization
- **Model Training**: Multiple training algorithms with hyperparameter optimization
- **Evaluation Tools**: Comprehensive model evaluation and comparison
- **Real-time Processing**: Parallel processing for minute-by-minute data
- **📊 Automatic Logging**: Every training run is automatically logged with detailed metrics
- **�� Performance Tracking**: Track model performance over time with trends and comparisons
- **🎨 Interactive Dashboard**: Beautiful Streamlit dashboard for model comparison and analysis

## 📊 Model Training Log & Monitoring

Every time you run the training pipeline, it automatically logs detailed information to `model_training_log.csv` including:

### 📋 **Tracked Metrics:**
- **Basic Info**: Date, Description, Training File, Test File
- **Configuration**: Fast Mode, Number of Features, Number of Samples
- **Performance**: Duration, Status, Error Messages
- **Model Results**: Best Model, Test AUC, Accuracy, Precision, Recall, F1
- **Trading Metrics**: Best Threshold, Best Precision, Number of Signals, Expected Value
- **Feature Analysis**: Top 3 Features, Low Impact Features
- **Performance Notes**: Automated performance assessment

### 🔧 **How to View the Log:**

#### **1. 🎨 Interactive Streamlit Dashboard (Recommended)**
```bash
# Launch the beautiful interactive dashboard
streamlit run src/evaluation/streamlit_model_dashboard.py

# Or use the Makefile command
make dashboard
```

**Dashboard Features:**
- 📋 **All runs visible** in a sortable, filterable table
- 🔍 **Interactive filters** by status, model, and date range
- 📈 **Performance trends** with interactive charts
- 🤖 **Model comparison** side-by-side
- 🔍 **Feature analysis** and impact visualization
- 🏆 **Best performers** highlighted
- ❌ **Error analysis** for failed runs

#### **2. Simple Log Viewer**
```bash
# View all training runs in readable format
python src/utils/view_model_log.py
```

#### **3. Interactive Model Comparison Tool**
```bash
# Comprehensive analysis with trends and comparisons
python src/evaluation/model_comparison_tool.py

# View specific number of recent runs
python src/evaluation/model_comparison_tool.py --recent 5
```

#### **4. Direct CSV Access**
```bash
# View raw CSV file
cat model_training_log.csv

# Open in spreadsheet application
open model_training_log.csv
```

### 📈 **What You Can Analyze:**

- **Recent Runs**: See your latest training experiments
- **Performance Trends**: Track how your models improve over time
- **Model Comparisons**: Compare different algorithms side-by-side
- **Feature Analysis**: See which features are most important
- **Error Tracking**: Identify and debug failed runs
- **Best Performers**: Find your highest-performing models

### 🎯 **Example Log Entry:**
```
Date: 2025-07-15 22:39:53
Description: "AAPL January 2025"
Training File: data/raw/monthly_AAPL_2025-01.csv
Best Model: LightGBM
Test AUC: 0.9104 (Excellent!)
Best F1: 0.763
Best Precision: 93.5%
Signals: 146
Expected Value: 0.870
Duration: 17.7 seconds
Features: 21
Samples: 4,061
```

## 📚 Documentation

- [Quick Start Guide](docs/QUICK_START.md)
- [Trading Model Launcher](docs/README_TRADING_MODEL_LAUNCHER.md)
- [Model Tools](docs/README_model_tools.md)
- [Solution Summary](docs/SOLUTION_SUMMARY.md)

## 🔧 Configuration

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_model_training.py
```

## 📊 Usage Examples

### Interactive Mode
```bash
python src/launchers/trading_model_launcher.py
```

### Command Line Mode
```bash
python src/launchers/trading_model_launcher.py --stocks AAPL,MSFT --year 2024 --month 7
```

### Data Processing
```bash
python src/data_processing/concatenate_all.py --symbols AAPL,MSFT --output training_data.csv
```

### Model Training (Direct)
```bash
# Train with specific data file
python src/model_training/model_training_15Jul_optimized1M.py data/raw/monthly_AAPL_2025-01.csv --description "AAPL January 2025" --fast

# Train with test data
python src/model_training/model_training_15Jul_optimized1M.py data/raw/monthly_AAPL_2025-01.csv --test-csv data/raw/monthly_AAPL_2025-02.csv --description "AAPL with test data"
```

### Monitoring & Analysis
```bash
# 🎨 Launch interactive dashboard (Recommended)
streamlit run src/evaluation/streamlit_model_dashboard.py

# Or use Makefile
make dashboard

# View training log
python src/utils/view_model_log.py

# Interactive model comparison
python src/evaluation/model_comparison_tool.py

# View recent runs
python src/evaluation/model_comparison_tool.py --recent 10
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions, please open an issue in the GitHub repository.
