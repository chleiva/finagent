# 🚀 Proper Trading V1 - Real-Time Trading Simulation & ML Pipeline

Proper Trading V1 is a comprehensive, modular platform designed to enable **real-time trading simulation, live data collection, and robust model validation** for financial markets. The core of the application is the `src/real_time/` module, which provides advanced tools for simulating trading strategies on live or historical data, collecting real-time market data, and validating trading databases.

---

## 🎯 Main Purpose

**The primary goal of this application is to empower users to:**
- **Simulate trading strategies in real time** using live or historical market data.
- **Collect and process real-time financial data** for use in machine learning models.
- **Validate and analyze trading results** to refine strategies and improve model performance.

All other components—data processing, feature engineering, model training, and evaluation—are designed to support and enhance the real-time trading simulation workflow.

---

## 📁 Component Tree

```plaintext
Proper_Trading_V1/
│
├── src/
│   ├── data_processing/
│   │   ├── concatenate_all.py         # Main data concatenation tool
│   │   ├── prepare.py                 # Data preparation logic
│   │   └── prepare_data.py            # Data cleaning utilities
│   ├── feature_engineering/
│   │   ├── features.py                # Main feature engineering module (latest)
│   │   ├── extra_features.py          # Additional feature creation
│   │   ├── feature_optimizer.py       # Feature selection/optimization
│   │   ├── features_cleaner.py        # Feature cleaning/preprocessing
│   │   ├── stock_minute_processor_parallel.py # Parallel minute-level processing
│   │   ├── buy_label.py               # Buy signal labeling
│   │   └── buy_label_before_optimization.py # Pre-optimization labeling
│   ├── model_training/
│   │   └── model_training_15Jul_optimized1M.py # Main, most advanced training script
│   ├── evaluation/
│   │   ├── evaluate.py                # Model evaluation and metrics
│   │   ├── model_comparison_tool.py   # Model comparison and analysis
│   │   └── streamlit_model_dashboard.py # Streamlit dashboard for model results
│   ├── inference/
│   │   └── model_inference_adapter.py # Model inference utilities
│   ├── real_time/
│   │   ├── trading_simulation.py      # Real-time trading simulation (latest)
│   │   ├── trading_simulation_working.py # Alternative simulation version
│   │   ├── enhanced_data_collector.py # Real-time data collection
│   │   └── validate_database.py       # Database validation
│   ├── launchers/
│   │   ├── trading_model_launcher.py  # Main entry point: orchestrates the pipeline
│   │   └── demo_launcher.py           # Demo launcher for usage examples
│   ├── scripts/
│   │   └── batch_train_top10.py       # Batch training for top 10 stocks
│   ├── utils/
│   │   ├── model_wrapper.py           # Model wrapper utilities
│   │   ├── model_manager.py           # Model management
│   │   ├── utils.py                   # General utilities
│   │   └── view_model_log.py          # Model log viewer
│   └── deprecated/
│       ├── _deprecated_stock_minute_processor_stable.py # Old processor
│       └── model_training/
│           ├── model_training_14Jul.py
│           ├── model_training_14Jul_removing_dominant_class.py
│           ├── model_training_good_14Jul.py
│           └── train.py
│
├── examples/
│   ├── model_usage_example.py         # Example: model usage
│   └── time_offset_simulation_example.py # Example: time offset simulation
│
├── scripts/
│   ├── run_batch_training.sh          # Shell script for batch training
│   ├── run_training.sh                # Shell script for training
│   ├── launch_dashboard.sh            # Launch Streamlit dashboard
│   └── monitor.sh                     # Monitoring script
│
├── tests/
│   ├── unit/
│   │   └── test_data_processing.py    # Unit tests for data processing
│   └── integration/
│
├── docs/
│   ├── README_TRADING_MODEL_LAUNCHER.md # Full documentation
│   ├── FILE_ORGANIZATION.md           # File/folder organization
│   ├── SOLUTION_SUMMARY.md            # Solution summary
│   ├── QUICK_START.md                 # Quick start guide
│   ├── BATCH_TRAINING_GUIDE.md        # Batch training documentation
│   └── TIME_OFFSET_FEATURE.md         # Time offset feature documentation
│
├── model_artifacts/                   # Saved models and artifacts
├── trained_model/                     # Trained model outputs
├── evaluation_results/                # Evaluation results (e.g., ROC curves)
├── batch_runs/                        # Batch training results
├── prepared_data/                     # Prepared datasets
├── data/                              # Raw, processed, and external data
├── config/                            # Configuration files
├── logs/                              # Log files
├── Makefile
├── launcher.sh
├── setup.py
└── README.md                          # (You are here)
```

---

## 🧩 How the Pipeline Supports Real-Time Trading

1. **Data Preparation (`src/data_processing/`)**
   - Cleans and aggregates raw market data, ensuring high-quality inputs for both model training and real-time simulation.

2. **Feature Engineering (`src/feature_engineering/`)**
   - Extracts and optimizes features from market data, which are used by both offline models and real-time trading logic.

3. **Model Training (`src/model_training/`)**
   - Trains machine learning models on historical data, producing models that can be deployed in real-time trading simulations.

4. **Evaluation (`src/evaluation/`)**
   - Assesses model performance, ensuring only robust models are used in live or simulated trading.

5. **Real-Time Trading & Simulation (`src/real_time/`)**
   - **`trading_simulation.py`**: The main engine for simulating trading strategies in real time, using trained models and live/streamed data.
   - **`enhanced_data_collector.py`**: Collects and processes real-time market data, feeding it into the simulation and model inference pipeline.
   - **`validate_database.py`**: Ensures the integrity and quality of the trading database, critical for reliable simulation and analysis.

---

## 🏆 Key Real-Time Components

- **`src/real_time/trading_simulation.py`**  
  The heart of the application. Simulates trading strategies using live or historical data, applying trained ML models to make buy/sell decisions, and logs all trades and performance metrics.

- **`src/real_time/enhanced_data_collector.py`**  
  Continuously collects real-time market data, processes it, and makes it available for both simulation and model inference.

- **`src/real_time/validate_database.py`**  
  Validates the structure and content of the trading database, ensuring data quality for both backtesting and live simulation.

---

## 🚦 End-to-End Workflow

1. **Prepare and clean data** → 2. **Engineer features** → 3. **Train and evaluate models** →  
4. **Deploy models in real-time simulation** → 5. **Collect and validate live data** → 6. **Analyze and refine strategies**

---

## 📝 Example Usage

**Run a real-time trading simulation:**
```bash
python src/real_time/trading_simulation.py --config config/config.yaml
```

**Collect live market data:**
```bash
python src/real_time/enhanced_data_collector.py --symbols AAPL,MSFT,NVDA
```

**Validate your trading database:**
```bash
python src/real_time/validate_database.py --db data/processed/market_data.db
```

---

## 📚 Documentation

- See `docs/README_TRADING_MODEL_LAUNCHER.md` for full details.
- `docs/FILE_ORGANIZATION.md` for file/folder explanations.
- `docs/SOLUTION_SUMMARY.md` for a high-level overview.
- `docs/BATCH_TRAINING_GUIDE.md` for batch training instructions.
- `docs/TIME_OFFSET_FEATURE.md` for time offset simulation details.

---

## 🏅 Best Practices

- Use only the latest scripts for new experiments:
  - `trading_model_launcher.py` (main entry)
  - `model_training_15Jul_optimized1M.py` (training)
  - `features.py` (feature engineering)
- Reference deprecated scripts only for legacy support or comparison.
- Log all experiments with meaningful descriptions for reproducibility.

---

## 🧪 Testing

- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`

Run all tests:
```bash
pytest tests/
```

---

## 🤝 Contributing

1. Fork the repo and create a new branch.
2. Add your feature or fix.
3. Ensure all tests pass.
4. Submit a pull request with a clear description.

---

## 💬 Support

For questions, open an issue or consult the documentation in the `docs/` folder.
