# Archived Model Training Scripts

This directory contains archived model training scripts that are no longer in active use.

## Archived Files

- **`model_training_14Jul.py`** - Standard training script (superseded by optimized version)
- **`model_training_14Jul_removing_dominant_class.py`** - Training with class balancing (superseded by optimized version)
- **`model_training_good_14Jul.py`** - Alternative training approach (superseded by optimized version)
- **`train.py`** - Generic training script (superseded by optimized version)

## Current Active Training Script

The main training script is now:
- **`src/model_training/model_training_15Jul_optimized1M.py`** - Optimized for 1-minute data

## Why Archived?

These scripts were archived to:
- Reduce confusion about which training script to use
- Focus on the most optimized and tested version
- Maintain a clean, organized codebase
- Keep historical versions for reference if needed

## Accessing Archived Scripts

If you need to use any of these archived scripts, you can copy them back to the main model_training directory:

```bash
# Example: Copy back the class balancing version
cp src/deprecated/model_training/model_training_14Jul_removing_dominant_class.py src/model_training/
``` 