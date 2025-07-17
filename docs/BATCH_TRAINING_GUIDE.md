# Batch Training Guide - Top 10 NASDAQ Shares

This guide explains how to use the batch training system to train individual models for the top 10 NASDAQ shares and how to access them using the ModelWrapper.

## 🎯 Overview

The batch training system allows you to:
- Train individual models for each of the top 10 NASDAQ shares
- Automatically organize model artifacts
- Easily access trained models using a simple Python wrapper
- Make predictions with trained models

## 📊 Target Shares

The system trains models for these 10 shares:

| Symbol | Company | Description |
|--------|---------|-------------|
| NVDA   | Nvidia | Graphics & AI chips |
| MSFT   | Microsoft | Software & cloud |
| AAPL   | Apple | Consumer electronics |
| AMZN   | Amazon | E-commerce & cloud |
| GOOGL  | Alphabet | Search & advertising |
| META   | Meta Platforms | Social media |
| AVGO   | Broadcom | Semiconductor |
| TSLA   | Tesla | Electric vehicles |
| NFLX   | Netflix | Streaming |
| COST   | Costco | Retail |

## 🚀 Quick Start

### Option 1: Using the Shell Script (Recommended)

```bash
# Run the batch training launcher
./scripts/run_batch_training.sh
```

This will:
- Activate the virtual environment
- Check dependencies
- Start interactive batch training
- Allow you to skip individual models
- Show progress and results

### Option 2: Direct Python Script

```bash
# Activate virtual environment
source venv/bin/activate

# Run batch training
python src/scripts/batch_train_top10.py
```

## 📋 Training Process

### What Happens During Training

1. **Data Validation**: Checks if data exists for each symbol
2. **Model Training**: Runs the full ML pipeline for each share
3. **Artifact Organization**: Saves models in organized directories
4. **Progress Tracking**: Shows real-time progress and results
5. **Error Handling**: Continues with other models if one fails

### Training Time Estimates

- **Per Model**: 5-15 minutes (depending on data size)
- **Total Time**: 2-3 hours for all 10 models
- **Memory Usage**: ~2-4 GB RAM per model

### Interactive Features

- **Skip Models**: Choose to skip individual models during training
- **Delay Control**: Add delays between models to manage system load
- **Progress Monitoring**: Real-time progress updates
- **Error Recovery**: Continue training even if some models fail

## 🔧 Using Trained Models

### ModelWrapper Class

The `ModelWrapper` class provides easy access to trained models:

```python
from src.utils.model_wrapper import ModelWrapper

# Load a model for AAPL
model = ModelWrapper("AAPL")

# Get model information
info = model.get_info()
print(f"AAPL Model AUC: {info['test_auc']:.4f}")

# Make predictions
predictions, probabilities = model.predict(your_data)

# Get feature importance
importance = model.get_feature_importance(top_n=10)

# Get trading metrics
trading_metrics = model.get_trading_metrics()
```

### Key Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_info()` | Get comprehensive model information | Dict |
| `predict(data)` | Make predictions on new data | (predictions, probabilities) |
| `predict_proba(data)` | Get prediction probabilities only | probabilities |
| `get_feature_importance(top_n)` | Get top N important features | DataFrame |
| `get_trading_metrics()` | Get trading-specific metrics | Dict |
| `save_predictions(data, path)` | Save predictions to CSV | str |

### Example Usage

```python
# Load multiple models
aapl_model = ModelWrapper("AAPL")
msft_model = ModelWrapper("MSFT")
nvda_model = ModelWrapper("NVDA")

# Compare model performance
models = [aapl_model, msft_model, nvda_model]
for model in models:
    info = model.get_info()
    print(f"{info['symbol']}: AUC={info['test_auc']:.4f}, F1={info['test_f1']:.4f}")

# Make predictions on new data
new_data = pd.read_csv("new_market_data.csv")
aapl_predictions, aapl_probs = aapl_model.predict(new_data)
msft_predictions, msft_probs = msft_model.predict(new_data)

# Save predictions
aapl_model.save_predictions(new_data, "aapl_predictions.csv")
```

## 📁 Artifact Organization

### Directory Structure

```
model_artifacts/
├── model_index.csv                    # Index of all models
├── 20250716_143022_AAPL-24-25/       # AAPL model artifacts
│   ├── models/
│   │   ├── model_package.pkl         # Trained models
│   │   └── optuna_study.pkl          # Optimization results
│   ├── evaluation/
│   │   ├── test_auc_comparison.csv   # Model comparison
│   │   ├── trading_strategies.csv    # Trading metrics
│   │   └── feature_importance_analysis.csv
│   ├── data/
│   │   └── data_summary.csv          # Training data info
│   ├── config/
│   │   └── model_config.csv          # Model configuration
│   └── plots/
│       └── roc_curves.png            # Performance plots
├── 20250716_143156_MSFT-24-25/       # MSFT model artifacts
└── ...
```

### Model Index

The `model_index.csv` file contains metadata for all trained models:

| Column | Description |
|--------|-------------|
| model_id | Unique model identifier |
| description | Model description (e.g., "AAPL-24-25") |
| date_created | Training date and time |
| best_model | Best performing model type |
| test_auc | Test AUC score |
| test_f1 | Test F1 score |
| best_precision | Best precision with reasonable signals |
| best_signals | Number of signals at best precision |

## 🔍 Utility Functions

### List All Models

```python
from src.utils.model_wrapper import list_available_models

# Get all available models
models_df = list_available_models()
print(f"Found {len(models_df)} trained models")

# Show latest models
latest_models = models_df.head(5)
for _, model in latest_models.iterrows():
    print(f"• {model['description']} (AUC: {model['test_auc']:.4f})")
```

### Get Latest Model for Symbol

```python
from src.utils.model_wrapper import get_latest_model_for_symbol

# Get latest AAPL model info
aapl_info = get_latest_model_for_symbol("AAPL")
if aapl_info:
    print(f"AAPL Model: {aapl_info['description']}")
    print(f"Test AUC: {aapl_info['test_auc']:.4f}")
```

## 🛠️ Troubleshooting

### Common Issues

1. **No Data Found**
   ```
   ❌ No data found for SYMBOL
   ```
   **Solution**: Ensure you have monthly CSV files in `data/raw/` for the symbol

2. **Model Not Found**
   ```
   ValueError: No models found for symbol: SYMBOL
   ```
   **Solution**: Train a model for that symbol first

3. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution**: Use `--fast` flag or reduce sample size in prepare.py

4. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'pandas'
   ```
   **Solution**: Activate virtual environment: `source venv/bin/activate`

### Performance Tips

- **Parallel Training**: Train models in parallel if you have sufficient RAM
- **Fast Mode**: Use `--fast` flag for quicker testing
- **Data Filtering**: Use year/month filters to reduce training time
- **Cleanup**: Remove old temporary files with `--cleanup-temp`

## 📈 Monitoring Training

### Real-time Monitoring

1. **Streamlit Dashboard**: Monitor training progress
   ```bash
   streamlit run src/evaluation/streamlit_model_dashboard.py
   ```

2. **Log Files**: Check training logs
   ```bash
   tail -f model_training_log.csv
   ```

3. **Model Manager**: List and manage models
   ```bash
   python src/utils/model_manager.py list
   ```

### Batch Results

After batch training, results are saved in:
- `batch_runs/top10_nasdaq_TIMESTAMP/batch_results.json`
- `model_artifacts/model_index.csv`

## 🎯 Next Steps

After training your models:

1. **Test Models**: Use the example script to test model loading
   ```bash
   python examples/model_usage_example.py
   ```

2. **Make Predictions**: Use ModelWrapper in your trading scripts

3. **Monitor Performance**: Use the Streamlit dashboard to track model performance

4. **Retrain Models**: Set up automated retraining schedules

## 📚 Additional Resources

- [Model Training Guide](MODEL_TRAINING_GUIDE.md)
- [Streamlit Dashboard Guide](DASHBOARD_GUIDE.md)
- [Feature Engineering Guide](FEATURE_ENGINEERING_GUIDE.md)
- [API Documentation](API_DOCUMENTATION.md) 