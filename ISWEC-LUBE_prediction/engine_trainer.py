"""
engine_trainer.py
══════════════════════════════════════════════════════════════════════════════
Trains the MOGD-LUBE model for electric power prediction intervals.

NEW IN THIS VERSION:
  • Early stopping  — training halts when PI-Val loss stops improving,
                      saving time and preventing overfitting
  • Multi-alpha     — train one model per confidence level (90%, 95%)
                      so results can be compared directly with KDE

DATA FLOW:
  TRAINING SET  ←  training_set_results_simulation_150days_window9000.csv
                   All 8640 rows. Updates model weights every epoch.

  PI-VAL SET    <-  test_set_results_simulation_150days_window9000.csv
                   Second 50% (rows 1440+). Monitors validation loss.
                   Also drives early stopping decisions.
                   No weight updates here.

  PI-TEST SET   ←  NEVER touched here.
                   Reserved exclusively for engine_tester.py.

OUTPUT (per alpha level):
  saved_models/{dataset_name}_{modelType}_alpha{pct}_model.pth
  saved_models/{dataset_name}_scaler.pkl
══════════════════════════════════════════════════════════════════════════════
"""

import os
import math
import pickle
import copy

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

from module_mogd_solver import MOSolver
from module_models import (MLP, SNN, qd_objective, LSTM, GRU,
                   winkler_objective, DeepAR, gaussian_log_likelihood)

# ── File paths ────────────────────────────────────────────────────────────────
TRAIN_FILE     = 'dataset/training_set_results_simulation_150days_window9000.csv'
TEST_FILE      = 'dataset/test_set_results_simulation_150days_window9000.csv'
POWER_COL      = 'TrifPelect_downsampled'
MODEL_SAVE_DIR = 'saved_models'


class trainer():

    def __init__(
        self,
        modelType         = 'MLP',
        lossType          = 'qd',
        lambda1_          = 0.001,
        lambda2_          = 0.0008,
        soften_           = 160.,
        num_epoch         = 100,
        alpha_            = 0.05,
        batch_size        = 128,
        num_task          = 2,
        input_window_size = 24,
        predicted_step    = 16,
        num_neurons       = 64,
        threshold         = 0.5,
        draw_training     = True,
        dataset_name      = 'simulation_150days',
        # ── Early stopping ────────────────────────────────────────────────
        early_stopping    = True,
        patience          = 20,
        min_delta         = 1e-5,
    ):
        """
        Parameters
        ----------
        modelType         : 'MLP', 'LSTM', 'BiGRU', 'SNN'
        lossType          : 'qd', 'winkler', 'deepAR'
        num_task          : 2 = MOGD (recommended) | 1 = standard
        input_window_size : past timesteps as input  (20 steps = 5 h)
        predicted_step    : future timesteps to predict (16 steps = 4 h)
        alpha_            : 0.05 → 95% PI | 0.10 → 90% PI | 0.01 → 99% PI
        draw_training     : plot loss curves after training
        dataset_name      : label used in saved filenames and print output
        early_stopping    : stop training when PI-Val loss stops improving
        patience          : epochs to wait for improvement before stopping
                            recommended: 20–30
        min_delta         : minimum improvement in loss to count as progress
        """
        self.alpha_ = alpha_
        if   self.alpha_ == 0.05: self.n_std_devs = 1.96
        elif self.alpha_ == 0.10: self.n_std_devs = 1.645
        elif self.alpha_ == 0.01: self.n_std_devs = 2.575
        else:                     self.n_std_devs = 1.96

        assert num_task  in [1, 2]
        assert lossType  in ['qd', 'winkler', 'deepAR']
        assert modelType in ['MLP', 'LSTM', 'BiGRU', 'SNN']

        self.num_task          = num_task
        self.batch_size        = batch_size
        self.input_window_size = input_window_size
        self.predicted_step    = predicted_step
        self.num_neurons       = num_neurons
        self.threshold         = threshold
        self.modelType         = modelType
        self.lossType          = lossType
        self.rnn               = (modelType != 'MLP')
        self.device            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lambda1_          = lambda1_
        self.lambda2_          = lambda2_
        self.soften_           = soften_
        self.num_epoch         = num_epoch
        self.draw_training     = draw_training
        self.dataset_name      = dataset_name
        self.early_stopping    = early_stopping
        self.patience          = patience
        self.min_delta         = min_delta

    # ── Sliding-window helper ─────────────────────────────────────────────────
    def _to_supervised(self, data):
        X, y, n_in, n_out = [], [], self.input_window_size, self.predicted_step
        for i in range(n_in, len(data) - n_out + 1):
            X.append(data[i - n_in : i])
            y.append(data[i        : i + n_out])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ── Data loading ──────────────────────────────────────────────────────────
    def load_data(self):
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(base_dir, TRAIN_FILE)
        test_path  = os.path.join(base_dir, TEST_FILE)

        for p in [train_path, test_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"File not found: {p}\n"
                    "Place both CSV files in the 'dataset/' folder.")

        train_raw = pd.read_csv(train_path)[POWER_COL].values.reshape(-1, 1)
        test_raw  = pd.read_csv(test_path)[POWER_COL].values.reshape(-1, 1)

        print(f"  Train : {len(train_raw)} rows | "
              f"{float(train_raw.min()):.1f}–{float(train_raw.max()):.1f} W")
        print(f"  Test  : {len(test_raw)} rows | "
              f"{float(test_raw.min()):.1f}–{float(test_raw.max()):.1f} W")

        # Normalize — scaler fitted on training data ONLY (no leakage)
        self.scaler  = MinMaxScaler(feature_range=(0, 1))
        train_scaled = self.scaler.fit_transform(train_raw).flatten()
        test_scaled  = self.scaler.transform(test_raw).flatten()

        X_train, y_train       = self._to_supervised(train_scaled)
        X_test_all, y_test_all = self._to_supervised(test_scaled)

        # First 50% of test = PI-Test (reserved for engine_tester.py)
        # Second 50% of test = PI-Val  (monitors early stopping here)
        split    = len(X_test_all) // 2
        X_pi_val = X_test_all[split:]
        y_pi_val = y_test_all[split:]

        if self.rnn:
            X_train  = X_train.reshape( -1, self.input_window_size, 1)
            X_pi_val = X_pi_val.reshape(-1, self.input_window_size, 1)

        print(f"\n  Split -> Train: {len(X_train)} | "
              f"PI-Val (2nd half): {len(X_pi_val)} | "
              f"PI-Test (1st half): {split} (reserved)\n")

        return X_train, y_train, X_pi_val, y_pi_val

    # ── Build model ───────────────────────────────────────────────────────────
    def _build_model(self):
        if self.modelType == 'MLP':
            return MLP(num_neurons=self.num_neurons,
                       input_window_size=self.input_window_size,
                       predicted_step=self.predicted_step)
        elif self.modelType == 'LSTM':
            return LSTM(num_neurons=self.num_neurons,
                        input_window_size=self.input_window_size,
                        predicted_step=self.predicted_step, device=self.device)
        elif self.modelType == 'SNN':
            return SNN(num_neurons=self.num_neurons, threshold=self.threshold,
                       input_window_size=self.input_window_size,
                       predicted_step=self.predicted_step)
        elif self.lossType == 'deepAR':
            return DeepAR(num_neurons=self.num_neurons,
                          input_window_size=self.input_window_size,
                          predicted_step=self.predicted_step,
                          bidirectional=True, device=self.device)
        elif self.modelType == 'BiGRU':
            return GRU(num_neurons=self.num_neurons,
                       input_window_size=self.input_window_size,
                       predicted_step=self.predicted_step,
                       bidirectional=True, device=self.device)

    # ── Build loss functions ──────────────────────────────────────────────────
    def _build_criterion(self):
        criterion = {}
        if self.lossType == 'qd':
            criterion[0] = qd_objective(lambda_=self.lambda1_, alpha_=self.alpha_,
                soften_=self.soften_, device=self.device, batch_size=self.batch_size)
            if self.num_task == 2:
                criterion[1] = qd_objective(lambda_=self.lambda2_, alpha_=self.alpha_,
                    soften_=self.soften_, device=self.device, batch_size=self.batch_size)
        elif self.lossType == 'winkler':
            criterion[0] = winkler_objective(lambda_=self.lambda1_, alpha_=self.alpha_,
                soften_=self.soften_, device=self.device, batch_size=self.batch_size)
            if self.num_task == 2:
                criterion[1] = winkler_objective(lambda_=self.lambda2_, alpha_=self.alpha_,
                    soften_=self.soften_, device=self.device, batch_size=self.batch_size)
        elif self.lossType == 'deepAR':
            criterion[0] = gaussian_log_likelihood()
        return criterion

    # ── Training loop with early stopping ────────────────────────────────────
    def training_loop(self, X_train, y_train, X_pi_val, y_pi_val,
                      model, optimizer, scheduler, criterion):
        """
        Trains the model epoch by epoch with early stopping.

        EARLY STOPPING LOGIC:
          After each epoch, the combined PI-Val loss (sum across objectives)
          is compared against the best seen so far.
          If it does not improve by at least min_delta for 'patience' consecutive
          epochs, training stops and the best model weights are restored.

          This prevents wasting time on epochs that do not help, and avoids
          overfitting on the training data.
        """
        train_history = {i: [] for i in range(self.num_task)}
        val_history   = {i: [] for i in range(self.num_task)}

        # Early stopping state
        best_val_loss   = float('inf')
        best_weights    = None
        patience_counter= 0
        stopped_epoch   = self.num_epoch

        for epoch in range(self.num_epoch):

            # ── Training phase ────────────────────────────────────────────
            train_loss = {i: 0.0 for i in range(self.num_task)}
            model.train()
            x_tr, y_tr = shuffle(X_train, y_train)

            for batch in range(math.ceil(len(x_tr) / self.batch_size)):
                s, e    = batch * self.batch_size, (batch + 1) * self.batch_size
                inputs  = torch.tensor(x_tr[s:e], dtype=torch.float).to(self.device)
                targets = torch.tensor(y_tr[s:e], dtype=torch.float).to(self.device)
                loss_data = {}

                if self.num_task == 2:
                    grads, scale = {}, {}
                    for i in range(2):
                        optimizer.zero_grad()
                        outputs      = model(inputs)
                        loss         = criterion[i](outputs, targets)
                        loss_data[i] = loss.item()
                        loss.backward()
                        grads[i] = [p.grad.data.clone() for p in model.parameters()
                                    if p.grad is not None]
                    sol = MOSolver.find_min_norm_element([grads[i] for i in range(2)])
                    for i in range(2): scale[i] = float(sol[i])
                    outputs    = model(inputs)
                    total_loss = sum(scale[i] * criterion[i](outputs, targets) for i in range(2))
                    for i in range(2): loss_data[i] = criterion[i](outputs, targets).item()
                    total_loss.backward(); optimizer.step()

                elif self.lossType == 'deepAR':
                    optimizer.zero_grad()
                    mu, sigma = model(inputs)
                    outputs   = torch.cat([mu + self.n_std_devs * sigma,
                                           mu - self.n_std_devs * sigma], 1)
                    loss = criterion[0](mu, sigma, targets)
                    loss_data[0] = loss.item()
                    loss.backward(); optimizer.step()

                else:
                    optimizer.zero_grad()
                    outputs      = model(inputs)
                    loss         = criterion[0](outputs, targets)
                    loss_data[0] = loss.item()
                    loss.backward(); optimizer.step()

                for i in range(self.num_task):
                    train_loss[i] += loss_data[i] * inputs.size(0)

            for i in range(self.num_task):
                train_history[i].append(train_loss[i] / len(X_train))

            # ── PI-Val phase ──────────────────────────────────────────────
            val_loss = {i: 0.0 for i in range(self.num_task)}
            model.eval()
            with torch.no_grad():
                for batch in range(math.ceil(len(X_pi_val) / self.batch_size)):
                    s, e    = batch * self.batch_size, (batch + 1) * self.batch_size
                    inputs  = torch.tensor(X_pi_val[s:e], dtype=torch.float).to(self.device)
                    targets = torch.tensor(y_pi_val[s:e], dtype=torch.float).to(self.device)
                    if self.lossType == 'deepAR':
                        mu, sigma = model(inputs)
                        outputs = torch.cat([mu + self.n_std_devs * sigma,
                                             mu - self.n_std_devs * sigma], 1)
                    else:
                        outputs = model(inputs)
                    for i in range(self.num_task):
                        lt = (criterion[i](mu, sigma, targets) if self.lossType == 'deepAR'
                              else criterion[i](outputs, targets))
                        val_loss[i] += lt.item() * inputs.size(0)

            for i in range(self.num_task):
                val_history[i].append(val_loss[i] / len(X_pi_val))

            scheduler.step()

            # ── Epoch log ────────────────────────────────────────────────
            combined_val = sum(val_loss[i] / len(X_pi_val) for i in range(self.num_task))
            print('─' * 90)
            if self.num_task == 2:
                print(f'Epoch {epoch+1:3d}  │  MOGD: {scale[0]:.3f}/{scale[1]:.3f}  │  '
                      f'Train: {train_loss[0]/len(X_train):.5f}  │  '
                      f'PI-Val: {val_loss[0]/len(X_pi_val):.5f}', end='')
            else:
                print(f'Epoch {epoch+1:3d}  │  '
                      f'Train: {train_loss[0]/len(X_train):.5f}  │  '
                      f'PI-Val: {val_loss[0]/len(X_pi_val):.5f}', end='')

            # ── Early stopping check ──────────────────────────────────────
            if self.early_stopping:
                if combined_val < best_val_loss - self.min_delta:
                    best_val_loss    = combined_val
                    best_weights     = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    print('  ✓ improved')
                else:
                    patience_counter += 1
                    print(f'  — no improvement ({patience_counter}/{self.patience})')
                    if patience_counter >= self.patience:
                        stopped_epoch = epoch + 1
                        print(f'\n  ⏹  Early stopping at epoch {stopped_epoch}. '
                              f'Best PI-Val loss: {best_val_loss:.6f}')
                        model.load_state_dict(best_weights)
                        break
            else:
                print()

        # If training completed without early stopping, save best weights anyway
        if self.early_stopping and best_weights is not None:
            model.load_state_dict(best_weights)

        actual_epochs = list(range(1, stopped_epoch + 1))
        return train_history, val_history, stopped_epoch

    # ── Save model ────────────────────────────────────────────────────────────
    def _save(self, model, alpha_label):
        """
        Saves model under a filename that includes the confidence level.
        e.g.  simulation_150days_MLP_alpha95_model.pth
              simulation_150days_MLP_alpha90_model.pth
        This allows multiple confidence levels to coexist in saved_models/.
        """
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        pct        = int((1 - self.alpha_) * 100)
        model_path = os.path.join(MODEL_SAVE_DIR,
            f'{self.dataset_name}_{self.modelType}_alpha{pct}_model.pth')
        scaler_path= os.path.join(MODEL_SAVE_DIR,
            f'{self.dataset_name}_scaler.pkl')

        torch.save({
            'model_state_dict'  : model.state_dict(),
            'modelType'         : self.modelType,
            'lossType'          : self.lossType,
            'input_window_size' : self.input_window_size,
            'predicted_step'    : self.predicted_step,
            'num_neurons'       : self.num_neurons,
            'threshold'         : self.threshold,
            'alpha_'            : self.alpha_,
            'n_std_devs'        : self.n_std_devs,
            'dataset_name'      : self.dataset_name,
        }, model_path)

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"  ✓ Model  → {model_path}")
        print(f"  ✓ Scaler → {scaler_path}")
        return model_path

    # ── Plot training curves ──────────────────────────────────────────────────
    def _plot_training_curves(self, train_history, val_history, stopped_epoch):
        actual = min(stopped_epoch, len(train_history[0]))
        epochs = range(1, actual + 1)
        fig, axes = plt.subplots(1, self.num_task, figsize=(16 * self.num_task, 10))
        if self.num_task == 1: axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(epochs, train_history[i][:actual], color='red',
                    linewidth=2.5, label='Training Loss')
            ax.plot(epochs, val_history[i][:actual],   color='blue',
                    linewidth=2.5, label='PI-Val Loss')
            if self.early_stopping and stopped_epoch < self.num_epoch:
                ax.axvline(stopped_epoch, color='green', linestyle='--',
                           linewidth=2, label=f'Early stop (epoch {stopped_epoch})')
            ax.set_xlabel('Epoch', fontsize=20)
            ax.set_ylabel('Loss',  fontsize=20)
            ax.set_title(f'Objective {i+1} — Loss Curve', fontsize=22)
            ax.legend(fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=14)
        conf = int((1 - self.alpha_) * 100)
        plt.suptitle(
            f'{self.dataset_name} — {self.modelType} — {conf}% PI — '
            f'Stopped at epoch {stopped_epoch}',
            fontsize=24, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self):
        """
        Trains the model and saves it to disk.
        Supports both single and multi-alpha training.
        After this, run engine_tester.py for evaluation.
        """
        conf = int((1 - self.alpha_) * 100)
        print(f"\n{'='*80}")
        print(f"  MOGD-LUBE — Training  [{conf}% PI]")
        print(f"{'='*80}")
        print(f"  Model    : {self.modelType} | Loss: {self.lossType} | Tasks: {self.num_task}")
        print(f"  Window   : {self.input_window_size} steps → Predict: {self.predicted_step} steps")
        print(f"  Epochs   : max {self.num_epoch} | "
              f"Early stop: {self.early_stopping} (patience={self.patience})")
        print(f"  Device   : {self.device}")
        print(f"{'='*80}\n")

        X_train, y_train, X_pi_val, y_pi_val = self.load_data()

        model     = self._build_model().to(self.device)
        criterion = self._build_criterion()
        for i in range(self.num_task):
            criterion[i] = criterion[i].to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[], gamma=1)

        train_history, val_history, stopped_epoch = self.training_loop(
            X_train, y_train, X_pi_val, y_pi_val,
            model, optimizer, scheduler, criterion)

        model_path = self._save(model, alpha_label=conf)

        if self.draw_training:
            self._plot_training_curves(train_history, val_history, stopped_epoch)

        print(f"\n  Training complete at epoch {stopped_epoch}/{self.num_epoch}.")
        print(f"  Run engine_tester.py to evaluate on the PI-Test set.\n")

        return model_path
