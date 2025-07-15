# File Organization Guide

This document describes the organization of files in the Trading Model Launcher project.

## 📁 Directory Structure

```
src/
├── data_processing/          # Data preparation and cleaning
├── feature_engineering/      # Feature creation and optimization
├── model_training/           # Model training scripts
├── evaluation/               # Model evaluation and comparison
├── launchers/                # Main application launchers
├── utils/                    # Utility functions
└── deprecated/               # Old/unused files (kept for reference)
```

## 🔧 Active Files by Category

### **Data Processing** (`src/data_processing/`)
- **`concatenate_all.py`** - Main data concatenation tool for multiple stocks
- **`prepare_data.py`** - Data preparation and cleaning utilities
- **`prepare.py`** - Additional data preparation scripts

### **Feature Engineering** (`src/feature_engineering/`)
- **`features.py`** - Main feature engineering module
- **`features_cleaner.py`** - Feature cleaning and preprocessing
- **`features_old_13Jul.py`** - Legacy feature engineering (kept for reference)
- **`buy_label.py`** - Buy signal labeling logic
- **`buy_label_before_optimization.py`** - Pre-optimization labeling
- **`extra_features.py`** - Additional feature creation utilities
- **`stock_minute_processor_parallel.py`** - Parallel processing for minute-by-minute data
- **`feature_optimizer.py`** - Feature optimization and selection

### **Model Training** (`src/model_training/`)
- **`model_training_15Jul_optimized1M.py`** - **MAIN TRAINING SCRIPT** (optimized for 1M data)

### **Deprecated Model Training** (`src/deprecated/model_training/`)
- **`model_training_14Jul.py`** - Standard training script (archived)
- **`model_training_14Jul_removing_dominant_class.py`** - Training with class balancing (archived)
- **`model_training_good_14Jul.py`** - Alternative training approach (archived)
- **`train.py`** - Generic training script (archived)

### **Evaluation** (`src/evaluation/`)
- **`evaluate.py`** - Model evaluation and metrics calculation
- **`model_comparison_tool.py`** - Model comparison and analysis

### **Launchers** (`src/launchers/`)
- **`trading_model_launcher.py`** - **MAIN APPLICATION** - Interactive model launcher
- **`demo_launcher.py`** - Demo launcher for testing

### **Utils** (`src/utils/`)
- **`utils.py`** - General utility functions
- **`view_model_log.py`** - Model logging utilities

### **Deprecated** (`src/deprecated/`)
- **`_deprecated_stock_minute_processor_stable.py`** - Old stable processor (kept for reference)

## 🎯 Key Files to Use

### **For Daily Development:**
1. **`src/launchers/trading_model_launcher.py`** - Main application
2. **`src/model_training/model_training_15Jul_optimized1M.py`** - Best training script
3. **`src/feature_engineering/features.py`** - Feature engineering
4. **`src/data_processing/concatenate_all.py`** - Data preparation

### **For Feature Engineering:**
- `src/feature_engineering/features.py` - Main features
- `src/feature_engineering/stock_minute_processor_parallel.py` - Real-time processing
- `src/feature_engineering/feature_optimizer.py` - Feature optimization

### **For Model Training:**
- `src/model_training/model_training_15Jul_optimized1M.py` - **Main training script**

### **For Evaluation:**
- `src/evaluation/evaluate.py` - Model evaluation
- `src/evaluation/model_comparison_tool.py` - Model comparison

## 📋 File Status

### **✅ Active & Recommended:**
- `trading_model_launcher.py` - Main application
- `model_training_15Jul_optimized1M.py` - Best training script
- `features.py` - Main feature engineering
- `concatenate_all.py` - Data processing
- `stock_minute_processor_parallel.py` - Real-time processing

### **🔄 Alternative Versions:**
- `src/deprecated/model_training/model_training_14Jul.py` - Standard training (archived)
- `src/deprecated/model_training/model_training_14Jul_removing_dominant_class.py` - Class balancing (archived)
- `src/deprecated/model_training/model_training_good_14Jul.py` - Alternative approach (archived)

### **📚 Reference/Historical:**
- `features_old_13Jul.py` - Legacy features
- `_deprecated_stock_minute_processor_stable.py` - Old processor

## 🚀 Quick Commands

```bash
# Run main application
make run-launcher

# Run specific training script
python src/model_training/model_training_15Jul_optimized1M.py

# Process data
python src/data_processing/concatenate_all.py

# Feature engineering
python src/feature_engineering/features.py
```

## 🔄 Migration Notes

- All feature engineering files are preserved in `src/feature_engineering/`
- All model training files are preserved in `src/model_training/`
- Deprecated files are kept in `src/deprecated/` for reference
- Empty directories have been removed
- File organization follows ML best practices 