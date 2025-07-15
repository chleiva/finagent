# 🚀 Quick Start Guide - Trading Model Launcher

## ⚡ Get Started in 30 Seconds

### 1. Interactive Mode (Easiest)
```bash
python trading_model_launcher.py
```
Follow the prompts to select stocks, time periods, and training options.

### 2. Command-Line Mode (Fast)
```bash
# Quick test with one stock
python trading_model_launcher.py --stocks AAPL --year 2025 --fast --description "Quick test"

# Tech sector analysis
python trading_model_launcher.py --stocks AAPL,MSFT,GOOGL --year 2025 --description "Tech 2025"

# Comprehensive training
python trading_model_launcher.py --quick --description "Full training"
```

### 3. Demo Mode (Learn by Example)
```bash
python demo_launcher.py
```

## 📊 Available Stock Categories

- **Tech Giants**: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
- **Semiconductors**: AMD, INTC, NVDA, ASML, KLAC, MRVL, QCOM
- **Software**: ADBE, CRM, SNOW, NET, PLTR, ZS, WDAY
- **Fintech**: PYPL, COIN, SOFI, PUBM
- **Entertainment**: NFLX, ROKU, MELI
- **Transportation**: UBER, TSLA
- **Biotech**: BIIB, VRTX, ANGO

## ⚙️ Key Options

| Option | Description | Example |
|--------|-------------|---------|
| `--stocks` | Stock symbols | `AAPL,MSFT,GOOGL` |
| `--year` | Year filter | `2025` |
| `--month` | Month filter | `01` |
| `--fast` | Quick training | (flag) |
| `--quick` | All data, fast mode | (flag) |
| `--description` | Model name | `"My experiment"` |

## 📁 Output Files

- **Training Dataset**: `training_dataset_YYYYMMDD_HHMMSS.csv`
- **Model Log**: `model_training_log.csv`
- **Results**: `evaluation_results/` directory

## 🔧 Next Steps

1. **View Results**: `python view_model_log.py`
2. **Compare Models**: `python model_comparison_tool.py`
3. **Read Full Docs**: `README_TRADING_MODEL_LAUNCHER.md`

## 🆘 Need Help?

- **Interactive Mode**: Best for beginners
- **Fast Mode**: Use `--fast` for quick testing
- **Demo Script**: `python demo_launcher.py`
- **Full Documentation**: See `README_TRADING_MODEL_LAUNCHER.md`

---

**Happy Trading! 📈** 