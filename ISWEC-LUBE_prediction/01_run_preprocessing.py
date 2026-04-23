import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

# ── SETTINGS ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH   = os.path.join(BASE_DIR, 'dataset',
                               'training_set_results_simulation_150days_window9000.csv')
INPUT_WINDOW_SIZE = 24   # past timesteps used as input  (24 steps = 6 h at 15-min sampling)
PREDICTED_STEP    = 16   # future timesteps to predict   (16 steps = 4 h at 15-min sampling)
OUTPUT_NAME       = 'simulation_150days_ISWEC_LUBE'
# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT_NAME is the dataset_name key used by engine_trainer.py / bayesian_engine_trainer.py.
# It determines:
#   • the supervised CSV  : dataset/{PREDICTED_STEP}_{OUTPUT_NAME}_supervised_wind_power.csv
#   • the saved scaler    : dataset/scaler_{OUTPUT_NAME}.pkl
#
# In 02_run_pipeline.py pass it as:
#   t.run(dataset_name=OUTPUT_NAME)
# ──────────────────────────────────────────────────────────────────────────────


def series_to_supervised(data, n_in=1, n_out=1):
    """
    Transforms a plain time series into a supervised learning dataset
    using a sliding window approach.

    Each output row contains:
        n_in  columns : past power values   power(t-n_in) … power(t-1)
        n_out columns : future power values power(t)      … power(t+n_out-1)
    """
    df = pd.DataFrame(data)
    cols, names = [], []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'power(t-{i})']
    for i in range(n_out):
        cols.append(df.shift(-i))
        names += ['power(t)' if i == 0 else f'power(t+{i})']
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Input file not found: {DATASET_PATH}\n"
        "Please make sure your dataset file is inside the 'dataset' directory.\n"
        "Expected file: dataset/training_set_results_simulation_150days_window9000.csv"
    )

df           = pd.read_csv(DATASET_PATH)
power_values = df['TrifPelect_downsampled'].values.reshape(-1, 1)

print(f"Dataset loaded : {len(power_values)} timesteps")
print(f"Power range    : {power_values.min():.3f} W -> {power_values.max():.3f} W")
print(f"Power mean     : {power_values.mean():.3f} W")


# ── NORMALIZE ─────────────────────────────────────────────────────────────────
# Fit the scaler ONLY on training data  (scale to [0, 1]).
# The fitted scaler is saved to disk so that engine_trainer.py can apply
# .transform() (NOT .fit_transform()) to the external test CSV, guaranteeing
# both sets share the identical normalisation range.
scaler       = MinMaxScaler(feature_range=(0, 1))
scaled_power = scaler.fit_transform(power_values)

print(f"\nNormalised range : [{scaled_power.min():.4f}, {scaled_power.max():.4f}]")


# ── BUILD SUPERVISED DATASET ──────────────────────────────────────────────────
# Each row: INPUT_WINDOW_SIZE past values (input) + PREDICTED_STEP future values (targets)
supervised_df = series_to_supervised(scaled_power, INPUT_WINDOW_SIZE, PREDICTED_STEP)

print(f"\nSliding window applied:")
print(f"  Input  columns : {INPUT_WINDOW_SIZE}  (power(t-{INPUT_WINDOW_SIZE}) … power(t-1))")
print(f"  Output columns : {PREDICTED_STEP}  (power(t) … power(t+{PREDICTED_STEP-1}))")
print(f"  Total  samples : {supervised_df.shape[0]}  (from {len(power_values)} raw rows)")


# ── SAVE ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(BASE_DIR, 'dataset')
os.makedirs(out_dir, exist_ok=True)

# 1) Supervised training CSV  (loaded by trainer.load_training_data)
out_path = os.path.join(out_dir, f'{PREDICTED_STEP}_{OUTPUT_NAME}_supervised.csv')
supervised_df.to_csv(out_path, index_label='index')
print(f"\nSupervised CSV saved -> {out_path}")

# 2) Fitted MinMaxScaler  (loaded by trainer.load_pi_sets to normalise the test CSV)
#    engine_trainer.py calls  scaler.transform()  on test_set_results_simulation_150days_window9000.csv
#    using this file — do NOT retrain or replace it between preprocessing and training runs.
scaler_path = os.path.join(out_dir, f'scaler_{OUTPUT_NAME}.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved       -> {scaler_path}")


# ── USAGE REMINDER ────────────────────────────────────────────────────────────
print("\nYou can now run the trainer with:")
print(f"  t = trainer(predicted_step={PREDICTED_STEP}, input_window_size={INPUT_WINDOW_SIZE}, ...)")
print(f"  t.run(dataset_name='{OUTPUT_NAME}')")
