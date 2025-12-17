# Stocks Predictions (LSTM)

This repo contains two training pipelines for forecasting Apple stock movements using LSTM models:

- Single-LSTM: minimal, unidirectional LSTM that predicts next-day scaled close-price change using only the `Close` feature.
- Multi-LSTM: stacked bidirectional LSTM that uses multiple features (OHLC; an advanced variant also adds technical indicators and separate target scaling).

## Environment Setup

Requirements: macOS, Python 3.11 recommended.

```bash
# Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Optional GPU (CUDA) is detected automatically by PyTorch if available.

## Data

Place CSVs under `data/`. The repo includes `data/aapl.us.csv` with at least these columns:

- `Date` (YYYY-MM-DD)
- `Open`, `High`, `Low`, `Close`

Scripts scale features to `[-1, 1]` via `MinMaxScaler`. The single-LSTM uses only the `Close` column as input; multi-LSTM uses OHLC (and optionally indicators).

## Scripts

- Single model: `single-ltsm.py`
- Multi model (basic): `multi-ltsm.py`

### Run

```bash
# Single-LSTM (unidirectional, input_dims=1)
python single-ltsm.py

# Multi-LSTM (2-layer bidirectional, input_dims=4)
python multi-ltsm.py

```

Outputs:
- Console logs show train/val loss and test accuracy within a tolerance (on scaled targets).
- Plots compare predicted vs actual prices after inverse-transform.
- Models saved under `results/` (e.g., `model.ckpt`, `improved_model.pt`).

## What Each Script Does

### single-ltsm.py
- Splits data 70/10/20 into train/val/test.
- Scales OHLC to `[-1, 1]`.
- Input features: only `Close` (`input_dims=1`).
- Target: next-day scaled close-price difference (small magnitude; loss values are tiny by design).
- Model: one unidirectional `nn.LSTM` + a single `nn.Linear` head.
- Training: Huber loss, Adam, gradient clipping, early stopping on validation loss.
- Evaluation: tolerance-based accuracy and inverse-transform to original price for plotting.

### multi-ltsm.py (basic)
- Uses OHLC as inputs (`input_dims=4`).
- Target: next-day scaled close-price difference.
- Model: 2-layer bidirectional `nn.LSTM` with dropout between stacked layers; MLP head.
- Training: Huber loss, Adam, gradient clipping, early stopping.
- Evaluation: tolerance accuracy; inverse-transform to price for plots.

## Key Hyperparameters (where to adjust)

- `SEQ_LEN` / `sequence_length`: number of past days the model sees.
- `hidden_size`: LSTM capacity; larger values increase model power and runtime.
- `num_layers`: stacked LSTM layers (dropout inside LSTM applies when `num_layers > 1`).
- `dropout`: regularization; in LSTM only active between layers, also used in the MLP head.
- `learning_rate`: optimizer step size (typical: `1e-3` to `5e-4`).
- `tolerance`: threshold for accuracy on scaled targets.

## Interpreting Results

- Very small losses are expected when predicting scaled differences (targets in the 0.0001–0.01 range).
- High tolerance accuracy (e.g., ≥95%) indicates predictions are within the chosen error band.
- Use MAE on original price via inverse-transform for more intuitive error values.

## Common Pitfalls

- Data leakage: do not fit scalers on combined train+val+test (fixed in the advanced script).
- Dropout with single LSTM layer: PyTorch applies dropout only between stacked layers; with `num_layers=1`, LSTM dropout is ignored.
- Constant predictions: using MSE on tiny targets can lead to mean-predicting; Huber loss helps.

## Upload to GitHub

From the project root:

```bash
# Initialize and commit
git init
git add .
git commit -m "Initialize LSTM stock prediction project"

# Create a new repo on GitHub first, then add your remote
# Replace <YOUR_USER> and <REPO_NAME>
git remote add origin https://github.com/<YOUR_USER>/<REPO_NAME>.git

git branch -M main
git push -u origin main
```

## Troubleshooting

- If batch size causes OOM on GPU, reduce `batch_size` (e.g., 256 → 128 → 64).
- If validation loss plateaus, try: increasing `sequence_length` (e.g., 60–90), lowering `learning_rate`, adding/adjusting indicators, and using `ReduceLROnPlateau` + early stopping (advanced script).
- If plots do not display in headless environments, save figures instead of `plt.show()`.

## License

See `LICENSE`.
