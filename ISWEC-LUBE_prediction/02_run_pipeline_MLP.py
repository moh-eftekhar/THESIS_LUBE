"""
02_run_pipeline_MLP.py
================================================================================
Master pipeline script — MLP model variant.

KEY DIFFERENCES FROM 02_run_pipeline.py (LSTM):
  - MODEL_TYPE = 'MLP'
  - RUN_TUNING = True  ->  grid-search finds the best window size, neurons,
                           batch size automatically via engine_tuner.py
  - NO dependency on any .h5 file -- all parameters come from the tuning grid
    or from BEST_PARAMS set manually below
  - Hyperparameters come from the tuning grid (or BEST_PARAMS if tuning skipped)

WORKFLOW:
  Step 1 -- Grid-search tuning across window / neurons / batch combinations
  Step 2 -- Train MLP for each confidence level (97.5%, 95%, 90%)
  Step 3 -- Evaluate on held-out PI-Test set with per-horizon metrics

HOW TO RUN:
  python 02_run_pipeline_MLP.py

SWITCHES:
  RUN_TUNING   : set False + fill BEST_PARAMS manually to skip grid search
  RUN_TRAINING : set False if model already saved and you only want to re-test
  RUN_TESTING  : set False to skip evaluation
================================================================================
"""

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_TYPE   = 'MLP'               # fixed -- do not change in this file
DATASET_NAME = 'simulation_150days'

# Prediction horizon -- how many future steps to forecast
# 16 steps x 15 min = 4 hours ahead  (must match the value used in preprocessing)
PREDICTED_STEP = 16

# Confidence levels to train and evaluate
# Derived from list_confidence_interval = [0.975, 0.95, 0.90]
ALPHAS = [0.025, 0.05, 0.10]      # 0.025 -> 97.5% PI | 0.05 -> 95% PI | 0.10 -> 90% PI

# ── Pipeline switches ─────────────────────────────────────────────────────────
RUN_TUNING   = True    # run grid search to find best MLP hyperparameters
RUN_TRAINING = True    # train MLP for each alpha in ALPHAS
RUN_TESTING  = True    # evaluate on held-out PI-Test set

# ── Manual best params (used only when RUN_TUNING = False) ───────────────────
# Fill these in from a previous tuning run to skip the grid search entirely.
BEST_PARAMS = {
    'input_window_size' : 24,      # past timesteps as input (24 steps = 6 h)
    'num_neurons'       : 64,      # MLP hidden layer width
    'lambda1_'          : 0.001,   # QD loss penalty obj.1
    'lambda2_'          : 0.0008,  # QD loss penalty obj.2
    'batch_size'        : 128,
}

# ── Fixed training settings ───────────────────────────────────────────────────
NUM_EPOCH    = 100     # maximum epochs (early stopping may halt sooner)
PATIENCE     = 20      # early stopping patience in epochs
LAMBDA1      = 0.001
LAMBDA2      = 0.0008
BATCH_SIZE   = 128

# ── Test visualisation settings ───────────────────────────────────────────────
DISPLAY_SIZE = 200     # timesteps shown in zoomed plot (200 x 15 min = 50 h)
STEP_TO_SHOW = 0       # horizon to highlight: 0=+15min | 7=+2h | 15=+4h

# ==============================================================================
#  STEP 1 -- HYPERPARAMETER TUNING  (grid search)
# ==============================================================================
from engine_trainer import trainer
from engine_tester  import LUBETester

print("\n" + "="*80)
print(f"  MOGD-LUBE Pipeline  --  Model: {MODEL_TYPE}")
print("="*80)
print(f"  predicted_step : {PREDICTED_STEP}  steps  =  {PREDICTED_STEP*15/60:.0f} h ahead")
print(f"  alphas         : {[int((1-a)*100) for a in ALPHAS]}% PI")
print(f"  early_stopping : patience={PATIENCE} epochs")

# ==============================================================================
#  STEP 1 -- HYPERPARAMETER TUNING  (grid search)
# ==============================================================================
from engine_trainer import trainer
from engine_tester  import LUBETester

if RUN_TUNING:
    print("\n" + "="*80)
    print("  STEP 1 -- Hyperparameter Tuning (MLP)")
    print("  Searching: input_window_size / num_neurons / batch_size")
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

    print("\n  Best parameters found by tuning:")
    for k, v in BEST_PARAMS.items():
        print(f"    {k:20s} = {v}")

else:
    print("\n  Tuning skipped -- using BEST_PARAMS defined at top of this file:")
    for k, v in BEST_PARAMS.items():
        print(f"    {k:20s} = {v}")

print(f"\n  Parameters to be used:")
print(f"    model_type        : {MODEL_TYPE}")
print(f"    input_window_size : {BEST_PARAMS['input_window_size']}")
print(f"    num_neurons       : {BEST_PARAMS['num_neurons']}")
print(f"    predicted_step    : {PREDICTED_STEP}")
print(f"    lambda1_          : {BEST_PARAMS['lambda1_']}")
print(f"    lambda2_          : {BEST_PARAMS['lambda2_']}")
print(f"    batch_size        : {BEST_PARAMS['batch_size']}")
print(f"    alphas            : {[int((1-a)*100) for a in ALPHAS]}% PI")
print(f"    early_stopping    : patience={PATIENCE} epochs")

# ==============================================================================
#  STEP 2 -- TRAINING  (one model per confidence level)
# ==============================================================================
if RUN_TRAINING:
    print("\n" + "="*80)
    print(f"  STEP 2 -- Training  [{MODEL_TYPE}]")
    print(f"  Training {len(ALPHAS)} model(s): "
          f"{[str(int((1-a)*100))+'%' for a in ALPHAS]} PI")
    print("="*80)

    for alpha in ALPHAS:
        conf = int((1 - alpha) * 100)
        print(f"\n  -- Training {conf}% PI model ----------------------------------")

        t = trainer(
            modelType         = MODEL_TYPE,
            lossType          = 'qd',
            num_task          = 2,                               # MOGD
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
        # Saved to: saved_models/{DATASET_NAME}_MLP_alpha{conf}_model.pth

else:
    print("\n  Training skipped -- using saved MLP models.")

# ==============================================================================
#  STEP 3 -- TESTING  (per-horizon metrics + multi-alpha comparison)
# ==============================================================================
if RUN_TESTING:
    print("\n" + "="*80)
    print(f"  STEP 3 -- Evaluation  [{MODEL_TYPE}]")
    print(f"  Evaluating {[int((1-a)*100) for a in ALPHAS]}% PI levels")
    print(f"  Per-horizon metrics: steps 1-{PREDICTED_STEP} "
          f"(+15 min -> +{PREDICTED_STEP*15//60} h)")
    print("="*80)

    tester = LUBETester(
        dataset_name = DATASET_NAME,
        modelType    = MODEL_TYPE,
        display_size = DISPLAY_SIZE,
        step_to_show = STEP_TO_SHOW,
    )
    all_results = tester.run(alphas=ALPHAS)

else:
    print("\n  Testing skipped.")

print("\n" + "="*80)
print(f"  Pipeline complete.  Model: {MODEL_TYPE}  |  Dataset: {DATASET_NAME}")
print("="*80 + "\n")
