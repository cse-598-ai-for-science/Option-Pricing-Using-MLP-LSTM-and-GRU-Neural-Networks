# Option Pricing Using Neural Networks

This repository implements neural network models for option pricing, with support for multiple architectures and alternative data sources. The project builds upon the original MLP-based approach with two significant improvements: a gated residual architecture and physics-informed training.

## Features

- Multiple model architectures: baseline MLP and gated residual network
- Support for alternative data sources: Economic Policy Uncertainty (EPU) index and news sentiment
- Physics-informed training with Black-Scholes PDE constraints
- Real-time option data fetching from Yahoo Finance
- Comprehensive evaluation framework with statistical validation

## Quick Start

### 1. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train a Model

**Basic MLP (baseline):**
```bash
python3 train_from_yf.py --ticker SPY --expirations 2 --epochs 100 --batch_size 32
```

**Gated Residual Architecture (recommended):**
```bash
python3 train_from_yf.py --ticker SPY --expirations 2 --epochs 100 --batch_size 32 \
  --model gated --hidden_width 64 --n_layers 4
```

## Model Architectures

### Baseline MLP

The original architecture uses a sequential dense network:
```
Input(5) -> Dense(30) -> LeakyReLU -> Dense(60) -> ELU -> Dense(90) -> LeakyReLU -> Dense(1)
```

### Gated Residual Architecture

The improved architecture incorporates residual connections and learnable gating:
```
Input(N) -> InputProjection(hidden_width) -> [ResidualBlock x N_LAYERS] -> GatedOutput
```

Key features:
- **Residual Blocks**: Skip connections for improved gradient flow
- **Gating Mechanism**: Adaptive feature weighting via sigmoid gates
- **Layer Normalization**: Stabilized training dynamics
- **L2 Regularization**: Prevents overfitting

## Training Scripts

### train_from_yf.py

Standard training script with Yahoo Finance data:

```bash
# With baseline MLP
python3 train_from_yf.py --ticker SPY --expirations 2 --epochs 100 --batch_size 32

# With gated architecture
python3 train_from_yf.py --ticker SPY --expirations 2 --epochs 100 --batch_size 32 \
  --model gated --hidden_width 64 --n_layers 4

# With custom hyperparameters
python3 train_from_yf.py --ticker SPY --expirations 2 --epochs 100 --batch_size 32 \
  --model gated --hidden_width 128 --n_layers 6
```

### train_alt.py

Training with alternative data sources:

```bash
# With EPU index
python3 train_alt.py --ticker SPY --expirations 2 --epochs 100 --batch_size 32 \
  --model gated --hidden_width 64 --n_layers 4 \
  --epu_csv All_Daily_Policy_Data.csv

# With sentiment data
python3 train_alt.py --ticker SPY --expirations 2 --epochs 100 --batch_size 32 \
  --model gated --hidden_width 64 --n_layers 4 \
  --sentiment_csv sp500_news_290k_articles_cleaned.csv

# With both EPU and sentiment
python3 train_alt.py --ticker SPY --expirations 2 --epochs 100 --batch_size 32 \
  --model gated --hidden_width 64 --n_layers 4 \
  --epu_csv All_Daily_Policy_Data.csv \
  --sentiment_csv sp500_news_290k_articles_cleaned.csv
```

### train_pinn.py

Physics-informed training with Black-Scholes PDE constraints:

```bash
# Basic physics-informed training
python3 train_pinn.py --ticker SPY --expirations 2 --epochs 100 --batch_size 32 \
  --model gated --hidden_width 64 --n_layers 4 \
  --physics_weight 1.0 --n_physics_samples 1000

# With higher physics weight
python3 train_pinn.py --ticker SPY --expirations 2 --epochs 100 --batch_size 32 \
  --model gated --hidden_width 64 --n_layers 4 \
  --physics_weight 5.0 --n_physics_samples 2000

# With alternative data sources
python3 train_pinn.py --ticker SPY --expirations 2 --epochs 100 --batch_size 32 \
  --model gated --hidden_width 64 --n_layers 4 \
  --physics_weight 5.0 --n_physics_samples 1000 \
  --epu_csv All_Daily_Policy_Data.csv \
  --sentiment_csv sp500_news_290k_articles_cleaned.csv
```

## Evaluation

### evaluate_data_sources.py

Compare model performance across different data configurations:

```bash
# Compare MLP vs Gated architecture across all data sources
python3 evaluate_data_sources.py \
  --ticker SPY \
  --expirations 3 \
  --epochs 100 \
  --batch_size 32 \
  --model both \
  --hidden_width 64 \
  --n_layers 4 \
  --repeats 3 \
  --epu_csv All_Daily_Policy_Data.csv \
  --sentiment_csv sp500_news_290k_articles_cleaned.csv \
  --out_dir evaluation_results/gated_architecture

# Test single architecture
python3 evaluate_data_sources.py \
  --ticker SPY \
  --expirations 3 \
  --epochs 100 \
  --batch_size 32 \
  --model gated \
  --hidden_width 64 \
  --n_layers 4 \
  --repeats 3 \
  --epu_csv All_Daily_Policy_Data.csv \
  --sentiment_csv sp500_news_290k_articles_cleaned.csv \
  --out_dir evaluation_results/comparison
```

This generates:
- `aggregated_metrics.csv`: Mean metrics across runs
- `per_run_metrics.csv`: Individual run results
- `rmse_by_data_source.png`: RMSE comparison visualization
- `mae_by_data_source.png`: MAE comparison visualization
- Additional charts when comparing both architectures

## Project Structure

```
.
├── models/
│   ├── __init__.py           # Package exports
│   ├── base_mlp.py           # Baseline MLP architecture
│   ├── gated_mlp.py          # Gated residual architecture
│   └── physics_loss.py       # Black-Scholes PDE loss components
│
├── train_from_yf.py          # Standard training script
├── train_alt.py              # Training with alternative data
├── train_pinn.py             # Physics-informed training
├── evaluate_data_sources.py  # Evaluation framework
├── data_loader.py            # Yahoo Finance data fetching
├── data_loader_alt.py        # Alternative data processing
│
├── evaluation_results/       # Experiment outputs
│   ├── gated_architecture/   # Architecture comparison results
│   └── physics-informed-training/  # PINN results
│
├── All_Daily_Policy_Data.csv           # EPU index data
├── sp500_news_290k_articles_cleaned.csv # Sentiment data
│
└── requirements.txt          # Python dependencies
```

## Data Sources

### Base Features (Yahoo Finance)
- `r`: Risk-free interest rate
- `K_over_S`: Strike-to-spot ratio
- `Maturity`: Time to expiration (years)
- `IV`: Implied volatility
- `cond_vol`: Conditional volatility (GARCH-based)

### Alternative Data
- **EPU Index**: Economic Policy Uncertainty from policyuncertainty.com
- **Sentiment**: News sentiment scores (positive, neutral, negative) from S&P 500 articles

## Experimental Results

The gated residual architecture shows consistent improvement when integrating external data sources:

| Configuration | MLP RMSE | Gated RMSE | Improvement |
|--------------|----------|------------|-------------|
| yf_only      | 0.01020  | 0.00714    | 30.0%       |
| epu          | 0.01165  | 0.00801    | 31.3%       |
| sentiment    | 0.01348  | 0.00725    | 46.2%       |
| both         | 0.01220  | 0.01143    | 6.3%        |

Note: Results vary between runs due to stochastic training (random weight initialization, mini-batch sampling, data shuffling). Values shown are means across 3 runs.

Physics-informed training shows minimal benefit for this well-sampled dataset but may help in low-data scenarios.

See `experiment_report.html` for the full analysis and `results.md` for formal thesis documentation.

## Command Line Arguments

### Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--ticker` | Stock ticker symbol | SPY |
| `--expirations` | Number of expiration dates | 2 |
| `--epochs` | Training epochs | 100 |
| `--batch_size` | Batch size | 32 |
| `--model` | Architecture (mlp, gated, both) | mlp |
| `--hidden_width` | Hidden layer width (gated) | 64 |
| `--n_layers` | Number of residual layers (gated) | 4 |

### Physics-Informed Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--physics_weight` | Weight for PDE loss | 1.0 |
| `--n_physics_samples` | Number of physics collocation points | 1000 |

### Alternative Data Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--epu_csv` | Path to EPU index CSV | None |
| `--sentiment_csv` | Path to sentiment CSV | None |

## Notes

- `data_loader.py` contains `yfinance_to_model_df` which maps Yahoo Finance option chain fields to model features
- The script trains on log(C/S) by default; modify `prepare_xy` for raw prices
- Use `--use_garch` to compute GARCH-based conditional volatility (increases training time)
- Use `python3` instead of `python` for consistent execution across environments

## Full Experimental Run

To run a complete evaluation comparing both architectures across all data configurations:

```bash
# Clear previous results (optional)
rm -rf evaluation_results/gated_architecture
rm -rf evaluation_results/physics-informed-training
mkdir -p evaluation_results/gated_architecture
mkdir -p evaluation_results/physics-informed-training

# Experiment 1: Gated Architecture vs MLP
python3 evaluate_data_sources.py \
  --ticker SPY \
  --expirations 3 \
  --epochs 100 \
  --batch_size 32 \
  --model both \
  --hidden_width 64 \
  --n_layers 4 \
  --repeats 3 \
  --epu_csv All_Daily_Policy_Data.csv \
  --sentiment_csv sp500_news_290k_articles_cleaned.csv \
  --out_dir evaluation_results/gated_architecture

# Experiment 2: Physics-Informed Training (varying physics weights)
for weight in 0.0 0.1 1.0 5.0 10.0; do
  python3 train_pinn.py \
    --ticker SPY \
    --expirations 3 \
    --epochs 100 \
    --batch_size 32 \
    --model gated \
    --hidden_width 64 \
    --n_layers 4 \
    --physics_weight $weight \
    --n_physics_samples 1000 \
    --repeats 3 \
    --out_dir evaluation_results/physics-informed-training
done
```

## References

- Original dissertation: Option Pricing Using MLP, LSTM, and GRU Neural Networks
- PINN reference: Physics-Informed Neural Networks for Option Pricing

