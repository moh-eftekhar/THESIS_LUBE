"""
02_run_pipeline.py
══════════════════════════════════════════════════════════════════════════════
Master script — runs the complete MOGD-LUBE pipeline.

IMPLEMENTED IMPROVEMENTS:
  1. Early stopping     — halts training when PI-Val loss stops improving
                          (patience=20 epochs). Prevents overfitting.
  2. Multi-alpha        — trains and evaluates both 90% and 95% PI levels
                          so results are directly comparable with KDE output
  3. Per-horizon metrics— PICP, MPIW, ACE, Winkler computed for each of the
                          16 forecast steps (+15 min → +4 hours)
  4. Multi-horizon eval — available after MLP and LSTM are complete

WORKFLOW:
  Step 0 — Read parameters directly from KDE model file (.h5)
  Step 1 — (Optional) Tune MLP hyperparameters via grid search
  Step 2 — Train model for each confidence level (90%, 95%)
  Step 3 — Evaluate on held-out PI-Test set with per-horizon breakdown

HOW TO RUN:
  python 02_run_pipeline.py

SWITCHES (edit below):
  KDE_MODEL_PATH : path to your .h5 file — reads params automatically
  MODEL_TYPE     : 'MLP' or 'LSTM'
  RUN_TUNING     : True only for MLP (skip for LSTM — use KDE params)
  RUN_TRAINING   : True to train
  RUN_TESTING    : True to evaluate
  ALPHAS         : confidence levels to train and evaluate
══════════════════════════════════════════════════════════════════════════════
"""

# ── Configuration ─────────────────────────────────────────────────────────────
KDE_MODEL_PATH = 'lstm_vector_150days_w9000_22.h5'  # your KDE model file

MODEL_TYPE   = 'LSTM'    # 'MLP' or 'LSTM'
DATASET_NAME = 'simulation_150days'

# Confidence levels to train and evaluate
# Derived from list_confidence_interval = [0.975, 0.95, 0.90]
# alpha = 1 - confidence_level
ALPHAS = [0.025, 0.05, 0.10]   # 0.025 -> 97.5% PI | 0.05 -> 95% PI | 0.10 -> 90% PI

# ── Pipeline switches ─────────────────────────────────────────────────────────
RUN_TUNING   = False   # Skipped for LSTM -- parameters read from KDE .h5 file
RUN_TRAINING = True    # train model for each alpha in ALPHAS
RUN_TESTING  = True    # evaluate on held-out PI-Test set

# ── Tuning override (used only when RUN_TUNING=False for MLP) ─────────────────
# These are filled automatically from the .h5 file (see Step 0 below).
# You can also set them manually here if needed.
BEST_PARAMS = None     # set to None to read from .h5 file automatically

# ── Fixed training settings ───────────────────────────────────────────────────
NUM_EPOCH    = 100     # maximum epochs (early stopping may stop earlier)
PATIENCE     = 20      # early stopping patience — stops after 20 epochs with
                       # no improvement. Recommended: 20–30.
LAMBDA1      = 0.001   # QD loss penalty obj.1 (MOGD-specific)
LAMBDA2      = 0.0008  # QD loss penalty obj.2 (MOGD-specific)
BATCH_SIZE   = 128

# ── Test visualisation settings ────────────────────────────────────────────────
DISPLAY_SIZE = 200     # timesteps shown in zoomed plot (200×15min = 50h)
STEP_TO_SHOW = 0       # horizon to highlight: 0=+15min | 7=+2h | 15=+4h

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 0 — READ PARAMETERS FROM KDE MODEL FILE
# ══════════════════════════════════════════════════════════════════════════════
from module_kde_reader import read_kde_params, print_kde_params

print("\n" + "="*80)
print(f"  MOGD-LUBE Pipeline  —  Model: {MODEL_TYPE}")
print("="*80)
print(f"  Reading KDE parameters from: {KDE_MODEL_PATH}")

kde_params = read_kde_params(KDE_MODEL_PATH)
print_kde_params(kde_params)

# Use KDE params as defaults — tuning may override these for MLP
if BEST_PARAMS is None:
    BEST_PARAMS = {
        'input_window_size' : kde_params['input_window_size'],
        'num_neurons'       : kde_params['num_neurons'],
        'lambda1_'          : LAMBDA1,
        'lambda2_'          : LAMBDA2,
        'batch_size'        : BATCH_SIZE,
    }

PREDICTED_STEP = kde_params['predicted_step']

print(f"  Parameters to be used:")
print(f"    input_window_size : {BEST_PARAMS['input_window_size']}  "
      f"(from .h5)" if BEST_PARAMS['input_window_size'] == kde_params['input_window_size']
      else f"    input_window_size : {BEST_PARAMS['input_window_size']}  (tuned)")
print(f"    num_neurons       : {BEST_PARAMS['num_neurons']}")
print(f"    predicted_step    : {PREDICTED_STEP}")
print(f"    lambda1_          : {BEST_PARAMS['lambda1_']}")
print(f"    lambda2_          : {BEST_PARAMS['lambda2_']}")
print(f"    batch_size        : {BEST_PARAMS['batch_size']}")
print(f"    alphas            : {[int((1-a)*100) for a in ALPHAS]}% PI")
print(f"    early_stopping    : patience={PATIENCE} epochs")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — TUNING  (MLP only, optional)
# ══════════════════════════════════════════════════════════════════════════════
from engine_trainer   import trainer
from engine_tester import LUBETester

if RUN_TUNING:
    print("\n" + "="*80)
    print("  STEP 1 — Hyperparameter Tuning (MLP)")
    print("="*80)
    from engine_tuner import run_tuning
    results_df, best = run_tuning()
    BEST_PARAMS = {
        'input_window_size' : int(best['input_window_size']),
        'num_neurons'       : int(best['num_neurons']),
        'lambda1_'          : float(best['lambda1_']),
        'lambda2_'          : float(best['lambda2_']),
        'batch_size'        : int(best['batch_size']),
    }
    print(f"\n  Best params found: {BEST_PARAMS}")
else:
    print("\n  Tuning skipped.")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — TRAINING  (one model per confidence level)
# ══════════════════════════════════════════════════════════════════════════════
if RUN_TRAINING:
    print("\n" + "="*80)
    print(f"  STEP 2 — Training  [{MODEL_TYPE}]")
    print(f"  Training {len(ALPHAS)} model(s): "
          f"{[str(int((1-a)*100))+'%' for a in ALPHAS]} PI")
    print("="*80)

    for alpha in ALPHAS:
        conf = int((1 - alpha) * 100)
        print(f"\n  ── Training {conf}% PI model ────────────────────────────")

        t = trainer(
            modelType         = MODEL_TYPE,
            lossType          = 'qd',
            num_task          = 2,                              # MOGD
            input_window_size = BEST_PARAMS['input_window_size'],
            predicted_step    = PREDICTED_STEP,
            num_neurons       = BEST_PARAMS['num_neurons'],
            lambda1_          = BEST_PARAMS['lambda1_'],
            lambda2_          = BEST_PARAMS['lambda2_'],
            batch_size        = BEST_PARAMS['batch_size'],
            num_epoch         = NUM_EPOCH,
            alpha_            = alpha,
            early_stopping    = True,
            patience          = PATIENCE,
            draw_training     = True,
            dataset_name      = DATASET_NAME,
        )
        t.run()
        # Model saved to: saved_models/{DATASET_NAME}_{MODEL_TYPE}_alpha{conf}_model.pth

else:
    print("\n  Training skipped — using saved models.")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — TESTING  (per-horizon metrics + multi-alpha comparison)
# ══════════════════════════════════════════════════════════════════════════════
if RUN_TESTING:
    print("\n" + "="*80)
    print(f"  STEP 3 — Evaluation  [{MODEL_TYPE}]")
    print(f"  Evaluating {[int((1-a)*100) for a in ALPHAS]}% PI levels")
    print(f"  Per-horizon metrics: steps 1–{PREDICTED_STEP} "
          f"(+15 min → +{PREDICTED_STEP*15//60} h)")
    print("="*80)

    tester = LUBETester(
        dataset_name = DATASET_NAME,
        modelType    = MODEL_TYPE,
        display_size = DISPLAY_SIZE,
        step_to_show = STEP_TO_SHOW,
    )
    # Evaluates both 90% and 95% PI, then shows a side-by-side comparison
    all_results = tester.run(alphas=ALPHAS)

else:
    print("\n  Testing skipped.")

print("\n" + "="*80)
print(f"  Pipeline complete.  Model: {MODEL_TYPE}  |  Dataset: {DATASET_NAME}")
print("="*80 + "\n")
