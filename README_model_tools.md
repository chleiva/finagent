# Model Training Tools

This directory contains tools for tracking and analyzing model training runs.

## 📊 Model Training Log

Every time you run the training pipeline, it automatically logs detailed information to `model_training_log.csv` including:

- **Date/Time**: When training started
- **Files**: Training and test dataset names
- **Configuration**: Fast mode, number of features, samples
- **Performance**: Duration, AUC, F1, precision, signals
- **Features**: Top 3 most important features
- **Analysis**: Low-impact features that could be removed

## 🔧 Tools Available

### 1. Model Training Pipeline
```bash
# Train a model (automatically logs to CSV)
python model_training_15Jul_optimized1M.py monthly_AAPL_2024-12.csv --test-csv AAPL_eval.csv --fast
```

### 2. View Model Log
```bash
# Simple log viewer
python view_model_log.py
```

### 3. Model Comparison Tool
```bash
# Interactive mode (recommended)
python model_comparison_tool.py --interactive

# Show recent runs
python model_comparison_tool.py --recent 5

# Compare models side by side
python model_comparison_tool.py --compare

# Show performance trends
python model_comparison_tool.py --trends

# Analyze feature usage
python model_comparison_tool.py --features

# Model performance breakdown
python model_comparison_tool.py --models

# Generate performance plots
python model_comparison_tool.py --plots

# Run all analyses
python model_comparison_tool.py --recent 3 --compare --trends --features --models --plots
```

## 🎯 Interactive Menu Options

When running in interactive mode, you can:

1. **Show recent runs** - Display the most recent model training runs
2. **Compare models** - Side-by-side comparison of top performing models
3. **Performance trends** - Analyze how performance changes over time
4. **Feature analysis** - Understand feature usage patterns
5. **Model performance breakdown** - Compare different model types
6. **Generate plots** - Create visualization charts
7. **Reload data** - Refresh the log data
8. **Exit** - Quit the tool

## 📈 Key Metrics Tracked

- **Test AUC**: Area Under Curve (0.5 = random, 1.0 = perfect)
- **Best F1**: F1-score at optimal threshold
- **Best Precision**: Precision with >20 signals
- **Signals**: Number of trading signals generated
- **Expected Value**: Expected return per trade
- **Duration**: Total training time
- **Features**: Number of features used

## 🏆 Performance Categories

- **Excellent**: AUC > 0.7
- **Good**: AUC > 0.6
- **Moderate**: AUC > 0.55
- **Poor**: AUC ≤ 0.55 (needs improvement)

## 📊 Generated Files

- `model_training_log.csv` - Main log file
- `model_performance_analysis.png` - Performance visualization plots
- `evaluation_results/` - Detailed evaluation outputs

## 💡 Tips for Model Analysis

1. **Track Progress**: Use the log to see if your models are improving over time
2. **Feature Optimization**: Identify which features are most important and which can be removed
3. **Model Selection**: Compare different model types to find the best performer
4. **Performance Trends**: Look for patterns in what makes models perform better
5. **Resource Management**: Monitor training time and resource usage

## 🔍 Example Workflow

```bash
# 1. Train a model
python model_training_15Jul_optimized1M.py monthly_AAPL_2024-12.csv --test-csv AAPL_eval.csv

# 2. Analyze the results
python model_comparison_tool.py --interactive

# 3. In interactive mode, explore:
#    - Recent runs (option 1)
#    - Model comparison (option 2)
#    - Performance trends (option 3)
#    - Feature analysis (option 4)
#    - Generate plots (option 6)
```

This comprehensive tracking system helps you understand what works best for your trading models and continuously improve their performance! 🚀 