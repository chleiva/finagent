# Trading Model Launcher - ML Pipeline

A comprehensive machine learning pipeline for trading model development, training, and deployment.

## 🚀 Quick Start

```bash
# Setup virtual environment and install dependencies
make setup

# Run the main training pipeline
make train

# Or use the convenience script
./scripts/run_training.sh
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
└── logs/                         # Application logs
```

## 🛠️ Features

- **Interactive Model Launcher**: User-friendly interface for model training
- **Automated Data Processing**: Concatenation and preparation of training datasets
- **Feature Engineering**: Advanced feature creation and optimization
- **Model Training**: Multiple training algorithms with hyperparameter optimization
- **Evaluation Tools**: Comprehensive model evaluation and comparison
- **Real-time Processing**: Parallel processing for minute-by-minute data

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
python src/launchers/trading_model_launcher.py --symbols AAPL,MSFT --year 2024 --month 7
```

### Data Processing
```bash
python src/data_processing/concatenate_all.py --symbols AAPL,MSFT --output training_data.csv
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
