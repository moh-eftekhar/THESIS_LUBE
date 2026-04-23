"""
engine_tuner.py
══════════════════════════════════════════════════════════════════════════════
Grid-search hyperparameter tuning for the MOGD-LUBE MLP model.

Evaluates every combination of parameters on the PI-Val set.
Uses early stopping during tuning to keep each trial fast.
Saves full results and best config to tuning_results/.

HOW TO RUN:
  python engine_tuner.py

OUTPUT:
  tuning_results/tuning_results.csv   — all combinations ranked by Winkler
  tuning_results/best_config.txt      — ready-to-paste trainer() call
  tuning_results/tuning_plots.png     — visualisation of all results
══════════════════════════════════════════════════════════════════════════════
"""

import os
import math
import copy
import itertools
import time

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

from module_mogd_solver import MOSolver
from module_models import MLP, qd_objective

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_FILE  = 'dataset/training_set_results_simulation_150days_window9000.csv'
TEST_FILE   = 'dataset/test_set_results_simulation_150days_window9000.csv'
POWER_COL   = 'TrifPelect_downsampled'
RESULTS_DIR = 'tuning_results'

# ══════════════════════════════════════════════════════════════════════════════
#  TUNING GRID — edit these lists to control what gets tested
# ══════════════════════════════════════════════════════════════════════════════
TUNING_GRID = {
    'input_window_size': [20, 24, 48],
    'num_neurons'      : [32, 64, 128],
    'lambda1_'         : [0.0005, 0.001, 0.005],
    'lambda2_'         : [0.0004, 0.0008, 0.004],
    'batch_size'       : [64, 128],
}

# ── Fixed settings for tuning ─────────────────────────────────────────────────
FIXED = {
    'predicted_step' : 16,
    'alpha_'         : 0.05,
    'num_task'       : 2,
    'soften_'        : 160.,
    'num_epoch'      : 30,      # reduced for speed; use 100 for final training
    'patience'       : 10,      # early stopping patience during tuning
    'min_delta'      : 1e-5,
}


def load_raw_data():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, TRAIN_FILE)
    test_path  = os.path.join(base_dir, TEST_FILE)
    for p in [train_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")
    train_raw = pd.read_csv(train_path)[POWER_COL].values.reshape(-1, 1)
    test_raw  = pd.read_csv(test_path)[POWER_COL].values.reshape(-1, 1)
    scaler       = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw).flatten()
    test_scaled  = scaler.transform(test_raw).flatten()
    print(f"  Train: {len(train_raw)} rows | Test: {len(test_raw)} rows")
    return train_scaled, test_scaled


def make_windows(data, n_in, n_out):
    X, y = [], []
    for i in range(n_in, len(data) - n_out + 1):
        X.append(data[i - n_in : i])
        y.append(data[i        : i + n_out])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_one_config(config, train_scaled, test_scaled):
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_in       = config['input_window_size']
    n_out      = FIXED['predicted_step']
    batch_size = config['batch_size']
    alpha_     = FIXED['alpha_']

    X_train, y_train       = make_windows(train_scaled, n_in, n_out)
    X_test_all, y_test_all = make_windows(test_scaled,  n_in, n_out)
    split    = len(X_test_all) // 2
    X_pi_val = X_test_all[split:];  y_pi_val = y_test_all[split:]   # second 50% = PI-Val

    model = MLP(num_neurons=config['num_neurons'], input_window_size=n_in,
                predicted_step=n_out).to(device)

    criterion = {
        0: qd_objective(lambda_=config['lambda1_'], alpha_=alpha_,
                        soften_=FIXED['soften_'], device=device,
                        batch_size=batch_size).to(device),
        1: qd_objective(lambda_=config['lambda2_'], alpha_=alpha_,
                        soften_=FIXED['soften_'], device=device,
                        batch_size=batch_size).to(device),
    }
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val  = float('inf')
    best_wts  = None
    patience_ctr = 0
    t_start   = time.time()

    for epoch in range(FIXED['num_epoch']):
        model.train()
        x_tr, y_tr = shuffle(X_train, y_train)
        for batch in range(math.ceil(len(x_tr) / batch_size)):
            s, e    = batch * batch_size, (batch + 1) * batch_size
            inputs  = torch.tensor(x_tr[s:e], dtype=torch.float).to(device)
            targets = torch.tensor(y_tr[s:e], dtype=torch.float).to(device)
            grads, scale = {}, {}
            for i in range(2):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss    = criterion[i](outputs, targets)
                loss.backward()
                grads[i] = [p.grad.data.clone() for p in model.parameters()
                            if p.grad is not None]
            sol = MOSolver.find_min_norm_element([grads[0], grads[1]])
            scale[0], scale[1] = float(sol[0]), float(sol[1])
            outputs    = model(inputs)
            total_loss = scale[0]*criterion[0](outputs,targets) + \
                         scale[1]*criterion[1](outputs,targets)
            optimizer.zero_grad(); total_loss.backward(); optimizer.step()

        # PI-Val
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in range(math.ceil(len(X_pi_val) / batch_size)):
                s, e    = batch * batch_size, (batch + 1) * batch_size
                inputs  = torch.tensor(X_pi_val[s:e], dtype=torch.float).to(device)
                targets = torch.tensor(y_pi_val[s:e], dtype=torch.float).to(device)
                outputs = model(inputs)
                val_loss += criterion[0](outputs, targets).item() * inputs.size(0)
        val_loss /= len(X_pi_val)

        if val_loss < best_val - FIXED['min_delta']:
            best_val = val_loss;  best_wts = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= FIXED['patience']:
                break

    if best_wts: model.load_state_dict(best_wts)

    # Evaluate on PI-Val
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_pi_val, dtype=torch.float).to(device))
        y_u  = pred[:, ::2].cpu().numpy()
        y_l  = pred[:, 1::2].cpu().numpy()

    K_u  = np.maximum(0.0, np.sign(y_u - y_pi_val))
    K_l  = np.maximum(0.0, np.sign(y_pi_val - y_l))
    picp = float(np.mean(K_u * K_l))
    mpiw = float(np.mean(np.abs(y_u - y_l)))
    ace  = picp - (1 - alpha_)
    S_t  = (np.abs(y_u-y_pi_val)
            + (2/alpha_)*np.multiply(y_l-y_pi_val, np.maximum(0.0, np.sign(y_l-y_pi_val)))
            + (2/alpha_)*np.multiply(y_pi_val-y_u, np.maximum(0.0, np.sign(y_pi_val-y_u))))
    winkler = float(np.mean(S_t))

    return picp, mpiw, ace, winkler, round(time.time()-t_start, 1)


def plot_tuning_results(df):
    df = df.sort_values('winkler').reset_index(drop=True)
    x  = range(len(df))
    target = 1 - FIXED['alpha_']

    fig, axes = plt.subplots(2, 2, figsize=(36, 24))

    ax = axes[0, 0]
    colors = ['gold' if i == 0 else '#33FFE3' for i in x]
    ax.bar(x, df['winkler'], color=colors, edgecolor='black', lw=0.8)
    ax.set_title('Winkler Score (gold = best)', fontsize=24, fontweight='bold')
    ax.set_xlabel('Configuration rank', fontsize=20)
    ax.set_ylabel('Winkler Score', fontsize=20)
    ax.grid(True, alpha=0.3, axis='y'); ax.tick_params(labelsize=14)

    ax = axes[0, 1]
    colors2 = ['green' if v >= target else 'salmon' for v in df['picp']]
    ax.bar(x, df['picp'], color=colors2, edgecolor='black', lw=0.8)
    ax.axhline(target, color='red', ls='--', lw=2.5, label=f'Target ≥ {target:.2f}')
    ax.set_title('PICP (green = meets target)', fontsize=24, fontweight='bold')
    ax.set_xlabel('Configuration rank', fontsize=20)
    ax.set_ylabel('PICP', fontsize=20)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3, axis='y'); ax.tick_params(labelsize=14)

    ax = axes[1, 0]
    ax.bar(x, df['mpiw'], color='#FFA500', edgecolor='black', lw=0.8)
    ax.set_title('MPIW (lower = tighter)', fontsize=24, fontweight='bold')
    ax.set_xlabel('Configuration rank', fontsize=20)
    ax.set_ylabel('MPIW', fontsize=20)
    ax.grid(True, alpha=0.3, axis='y'); ax.tick_params(labelsize=14)

    ax = axes[1, 1]
    sc = ax.scatter(df['mpiw'], df['picp'], c=df['winkler'],
                    cmap='RdYlGn_r', s=100, edgecolors='black', lw=0.8, zorder=3)
    plt.colorbar(sc, ax=ax, label='Winkler Score')
    ax.axhline(target, color='red', ls='--', lw=2, label=f'Target PICP={target:.2f}')
    best = df.iloc[0]
    ax.scatter(best['mpiw'], best['picp'], color='gold', s=300, zorder=5,
               edgecolors='black', lw=2, label='Best config')
    ax.set_title('PICP vs MPIW  (top-left = ideal)', fontsize=24, fontweight='bold')
    ax.set_xlabel('MPIW', fontsize=20); ax.set_ylabel('PICP', fontsize=20)
    ax.legend(fontsize=16); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=14)

    plt.suptitle(f'Tuning Results — MOGD-LUBE MLP  |  {len(df)} configurations',
                 fontsize=28, fontweight='bold')
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, 'tuning_plots.png'), dpi=150,
                bbox_inches='tight')
    plt.show()


def run_tuning():
    keys   = list(TUNING_GRID.keys())
    combos = list(itertools.product(*[TUNING_GRID[k] for k in keys]))
    total  = len(combos)

    print(f"\n{'='*80}")
    print(f"  Hyperparameter Tuning — MOGD-LUBE MLP")
    print(f"  Total combinations : {total}")
    print(f"  Epochs per trial   : max {FIXED['num_epoch']} "
          f"(early stop patience={FIXED['patience']})")
    print(f"{'='*80}\n")

    train_scaled, test_scaled = load_raw_data()
    results = []
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for idx, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        print(f"[{idx+1:3d}/{total}]  "
              f"win={config['input_window_size']}  "
              f"neu={config['num_neurons']}  "
              f"λ1={config['lambda1_']}  "
              f"λ2={config['lambda2_']}  "
              f"bs={config['batch_size']}", end='  →  ')
        try:
            picp, mpiw, ace, winkler, elapsed = train_one_config(
                config, train_scaled, test_scaled)
            ok = '✓' if picp >= (1 - FIXED['alpha_']) else '✗'
            print(f"PICP={picp:.4f}{ok}  MPIW={mpiw:.4f}  "
                  f"Winkler={winkler:.4f}  ({elapsed:.0f}s)")
            results.append({**config, 'picp': round(picp,4), 'mpiw': round(mpiw,4),
                             'ace': round(ace,4), 'winkler': round(winkler,4),
                             'time_s': elapsed})
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({**config, 'picp':None,'mpiw':None,'ace':None,
                             'winkler':None,'time_s':None})

        pd.DataFrame(results).to_csv(
            os.path.join(RESULTS_DIR, 'tuning_results.csv'), index=False)

    df   = pd.DataFrame(results).dropna()
    df   = df.sort_values('winkler').reset_index(drop=True)
    best = df.iloc[0]

    print(f"\n{'='*80}")
    print(f"  TOP 10 CONFIGURATIONS")
    print(f"{'='*80}")
    print(df.head(10).to_string(index=True))

    print(f"\n{'='*80}")
    print(f"  BEST CONFIG  →  use these in engine_trainer.py")
    print(f"{'='*80}")
    best_path = os.path.join(RESULTS_DIR, 'best_config.txt')
    lines = [
        "Best hyperparameter config from engine_tuner.py",
        "=" * 50,
        f"input_window_size : {int(best['input_window_size'])}",
        f"num_neurons       : {int(best['num_neurons'])}",
        f"lambda1_          : {best['lambda1_']}",
        f"lambda2_          : {best['lambda2_']}",
        f"batch_size        : {int(best['batch_size'])}",
        "-" * 50,
        f"PICP    : {best['picp']:.4f}",
        f"MPIW    : {best['mpiw']:.4f}",
        f"ACE     : {best['ace']:+.4f}",
        f"Winkler : {best['winkler']:.4f}",
        "=" * 50,
        "",
        "Paste into 02_run_pipeline.py → BEST_PARAMS:",
        "",
        "BEST_PARAMS = {",
        f"    'input_window_size' : {int(best['input_window_size'])},",
        f"    'num_neurons'       : {int(best['num_neurons'])},",
        f"    'lambda1_'          : {best['lambda1_']},",
        f"    'lambda2_'          : {best['lambda2_']},",
        f"    'batch_size'        : {int(best['batch_size'])},",
        "}",
    ]
    with open(best_path, 'w') as f:
        f.write('\n'.join(lines))
    for l in lines: print(' ', l)

    plot_tuning_results(df)
    return df, best


if __name__ == '__main__':
    run_tuning()
