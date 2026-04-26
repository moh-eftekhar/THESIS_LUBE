"""
engine_tester.py
================================================================================
Evaluates the trained MOGD-LUBE model on the held-out PI-Test set.

FEATURES:
  - Multi-alpha evaluation  : evaluates both 90% and 95% PI in one call
  - Per-horizon metrics     : PICP, MPIW, ACE, Winkler per forecast step
                              (step 1 = +15 min through step 16 = +4 hours)
  - Overall summary metrics : averaged across all steps and all samples
  - Publication-quality plots (300 DPI) auto-saved to figures/

DATA USED:
  PI-TEST SET  <-  test_set_results_simulation_150days_window9000.csv
                   Second 50% only. NEVER seen during training.

HOW TO RUN:
  python engine_tester.py

  Or from a notebook / 02_run_pipeline.py:
    from engine_tester import LUBETester
    tester = LUBETester(dataset_name='simulation_150days', modelType='MLP')
    tester.run()                   # evaluates both 90% and 95% PI
    tester.run(alphas=[0.05])      # evaluates 95% PI only
================================================================================
"""

import os
import json
import pickle

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend so saving always works
import matplotlib.pyplot as plt

from module_models import MLP, LSTM, GRU, SNN, DeepAR

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_FILE      = 'dataset/test_set_results_simulation_150days_window9000.csv'
POWER_COL      = 'TrifPelect_downsampled'
MODEL_SAVE_DIR = 'saved_models'

# ── Global matplotlib style ───────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'         : 150,
    'savefig.dpi'        : 300,
    'font.family'        : 'DejaVu Sans',
    'font.size'          : 12,
    'axes.titlesize'     : 14,
    'axes.titlepad'      : 12,
    'axes.labelsize'     : 13,
    'xtick.labelsize'    : 11,
    'ytick.labelsize'    : 11,
    'legend.fontsize'    : 11,
    'legend.framealpha'  : 0.9,
    'axes.grid'          : True,
    'grid.alpha'         : 0.3,
    'axes.facecolor'     : 'white',
    'figure.facecolor'   : 'white',
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
})


class LUBETester():

    def __init__(
        self,
        dataset_name = 'simulation_150days',
        modelType    = 'MLP',
        display_size = 200,
        step_to_show = 0,
    ):
        """
        Parameters
        ----------
        dataset_name  : must match dataset_name used in engine_trainer.py
        modelType     : 'MLP', 'LSTM', 'BiGRU', 'SNN'
        display_size  : timesteps shown in the zoomed time-series plot
        step_to_show  : which horizon to highlight in the time-series plots
                        0  = +15 min  |  7 = +2 h  |  15 = +4 h
        """
        self.dataset_name = dataset_name
        self.modelType    = modelType
        self.display_size = display_size
        self.step_to_show = step_to_show
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # figures directory (resolved relative to this script file)
        self._fig_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'figures')
        os.makedirs(self._fig_dir, exist_ok=True)

    # ── Load model for a specific alpha ───────────────────────────────────────
    def _load_model(self, alpha):
        """
        Loads the model saved for a specific confidence level.
        File name pattern:
          saved_models/{dataset_name}_{modelType}_alpha{pct}_model.pth
          saved_models/{dataset_name}_scaler.pkl
        """
        pct         = int((1 - alpha) * 100)
        model_path  = os.path.join(MODEL_SAVE_DIR,
            f'{self.dataset_name}_{self.modelType}_alpha{pct}_model.pth')
        scaler_path = os.path.join(MODEL_SAVE_DIR,
            f'{self.dataset_name}_scaler.pkl')

        for p in [model_path, scaler_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"File not found: {p}\n"
                    "Run engine_trainer.py first for this alpha level.")

        ckpt = torch.load(model_path, map_location=self.device)
        self.input_window_size = ckpt['input_window_size']
        self.predicted_step    = ckpt['predicted_step']
        self.num_neurons       = ckpt['num_neurons']
        self.threshold         = ckpt.get('threshold', 0.5)
        self.lossType          = ckpt['lossType']
        n_std_devs             = ckpt['n_std_devs']

        mt = self.modelType
        if mt == 'MLP':
            model = MLP(num_neurons=self.num_neurons,
                        input_window_size=self.input_window_size,
                        predicted_step=self.predicted_step)
        elif mt == 'LSTM':
            model = LSTM(num_neurons=self.num_neurons,
                         input_window_size=self.input_window_size,
                         predicted_step=self.predicted_step, device=self.device)
        elif mt == 'BiGRU':
            model = GRU(num_neurons=self.num_neurons,
                        input_window_size=self.input_window_size,
                        predicted_step=self.predicted_step,
                        bidirectional=True, device=self.device)
        elif mt == 'SNN':
            model = SNN(num_neurons=self.num_neurons,
                        threshold=self.threshold,
                        input_window_size=self.input_window_size,
                        predicted_step=self.predicted_step)
        else:
            model = DeepAR(num_neurons=self.num_neurons,
                           input_window_size=self.input_window_size,
                           predicted_step=self.predicted_step,
                           bidirectional=True, device=self.device)

        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        print(f"  [OK] Loaded {pct}% PI model <- {model_path}")
        return model, scaler, n_std_devs

    # ── Load PI-Test data ─────────────────────────────────────────────────────
    def _load_pi_test(self, scaler):
        """
        Loads the FIRST 50% of the test file (PI-Test set).
        Applies the training scaler -- never refits it on test data.

        Data split convention:
          First  50%  -> PI-Test  (loaded here, final evaluation only)
          Second 50%  -> PI-Val   (used in engine_trainer.py for early stopping)
        """
        base_dir  = os.path.dirname(os.path.abspath(__file__))
        test_path = os.path.join(base_dir, TEST_FILE)
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")

        test_raw    = pd.read_csv(test_path)[POWER_COL].values.reshape(-1, 1)
        test_scaled = scaler.transform(test_raw).flatten()

        # Build sliding windows
        X_all, y_all = [], []
        n_in, n_out  = self.input_window_size, self.predicted_step
        for i in range(n_in, len(test_scaled) - n_out + 1):
            X_all.append(test_scaled[i - n_in : i])
            y_all.append(test_scaled[i         : i + n_out])
        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.float32)

        # First 50% only (PI-Test set)
        split     = len(X_all) // 2
        X_pi_test = X_all[:split]
        y_pi_test = y_all[:split]
        # Second 50% (PI-Val set) -- kept for JSON export
        X_pi_val  = X_all[split:]
        y_pi_val  = y_all[split:]

        if self.modelType != 'MLP':
            X_pi_test = X_pi_test.reshape(-1, self.input_window_size, 1)
            X_pi_val  = X_pi_val.reshape(-1, self.input_window_size, 1)

        print(f"  PI-Test: {len(X_pi_test)} samples "
              f"(first 50% of test CSV -- {len(X_pi_test)*15/60:.0f} h)\n")
        return X_pi_test, y_pi_test, X_pi_val, y_pi_val

    # ── Predict ───────────────────────────────────────────────────────────────
    def _predict(self, model, X_pi_test, n_std_devs):
        """Returns y_u, y_l -- both shape (n_samples, predicted_step)."""
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_pi_test, dtype=torch.float).to(self.device)
            if self.lossType == 'deepAR':
                mu, sigma = model(inputs)
                y_u = (mu + n_std_devs * sigma).cpu().numpy()
                y_l = (mu - n_std_devs * sigma).cpu().numpy()
            else:
                pred = model(inputs)
                y_u  = pred[:, ::2].cpu().numpy()
                y_l  = pred[:, 1::2].cpu().numpy()
        return y_u, y_l

    # ── Compute metrics ───────────────────────────────────────────────────────
    def _compute_metrics(self, y_u, y_l, y_true, alpha, scaler=None):
        """
        Computes PICP, MPIW, ACE, Winkler at two levels:
          OVERALL    -- averaged across all samples AND all 16 predicted steps.
          PER-HORIZON-- one value per forecast step (step 1 to step 16).

        If scaler is provided, also returns per-step mean error bounds in raw Watts:
          mean_lower_err_W[s] = mean( y_l_raw[:,s] - y_true_raw[:,s] )  (typically negative)
          mean_upper_err_W[s] = mean( y_u_raw[:,s] - y_true_raw[:,s] )  (typically positive)
        """
        n_steps = y_true.shape[1]

        # Overall metrics
        K_u  = np.maximum(0.0, np.sign(y_u    - y_true))
        K_l  = np.maximum(0.0, np.sign(y_true - y_l))
        picp = float(np.mean(K_u * K_l))
        mpiw = float(np.mean(np.abs(y_u - y_l)))
        ace  = picp - (1 - alpha)
        S_t  = (np.abs(y_u - y_l)
                + (2/alpha) * np.multiply(y_l - y_true,
                    np.maximum(0.0, np.sign(y_l - y_true)))
                + (2/alpha) * np.multiply(y_true - y_u,
                    np.maximum(0.0, np.sign(y_true - y_u))))
        winkler = float(np.mean(S_t))
        overall = dict(picp=picp, mpiw=mpiw, ace=ace, winkler=winkler)

        # Per-horizon metrics
        ph_picp, ph_mpiw, ph_ace, ph_winkler = [], [], [], []
        ph_lower_err_w, ph_upper_err_w = [], []
        for s in range(n_steps):
            ku = np.maximum(0.0, np.sign(y_u[:, s] - y_true[:, s]))
            kl = np.maximum(0.0, np.sign(y_true[:, s] - y_l[:, s]))
            p  = float(np.mean(ku * kl))
            m  = float(np.mean(np.abs(y_u[:, s] - y_l[:, s])))
            a  = p - (1 - alpha)
            st = (np.abs(y_u[:, s] - y_l[:, s])
                  + (2/alpha) * np.multiply(y_l[:, s] - y_true[:, s],
                      np.maximum(0.0, np.sign(y_l[:, s] - y_true[:, s])))
                  + (2/alpha) * np.multiply(y_true[:, s] - y_u[:, s],
                      np.maximum(0.0, np.sign(y_true[:, s] - y_u[:, s]))))
            ph_picp.append(p)
            ph_mpiw.append(m)
            ph_ace.append(a)
            ph_winkler.append(float(np.mean(st)))

            # Raw error bounds in Watts (vectorised inverse-scale)
            if scaler is not None:
                def _inv_col(col):
                    return scaler.inverse_transform(col.reshape(-1, 1)).flatten()
                mu_l_w    = float(np.mean(_inv_col(y_l[:, s])))
                mu_u_w    = float(np.mean(_inv_col(y_u[:, s])))
                mu_true_w = float(np.mean(_inv_col(y_true[:, s])))
                ph_lower_err_w.append(round(mu_l_w - mu_true_w, 1))
                ph_upper_err_w.append(round(mu_u_w - mu_true_w, 1))
            else:
                ph_lower_err_w.append(None)
                ph_upper_err_w.append(None)

        per_step = dict(
            picp=ph_picp, mpiw=ph_mpiw, ace=ph_ace, winkler=ph_winkler,
            lower_err_w=ph_lower_err_w, upper_err_w=ph_upper_err_w
        )
        return overall, per_step

    # ── Print metrics table ───────────────────────────────────────────────────
    def _print_metrics(self, overall, per_step, alpha):
        conf = int((1 - alpha) * 100)
        n    = len(per_step['picp'])
        print(f"\n  {'─'*70}")
        print(f"  {conf}% PREDICTION INTERVAL -- EVALUATION RESULTS")
        print(f"  {'─'*70}")
        print(f"  OVERALL (averaged across all {n} horizon steps)")
        print(f"    PICP  (target >= {1-alpha:.2f}) : {overall['picp']:.4f}")
        print(f"    MPIW                     : {overall['mpiw']:.4f}")
        print(f"    ACE   (target = 0.0000)  : {overall['ace']:+.4f}")
        print(f"    Winkler Score            : {overall['winkler']:.4f}")
        print(f"\n  PER-HORIZON BREAKDOWN")
        print(f"  {'Step':>5}  {'Min':>6}  {'PICP':>7}  {'MPIW':>7}  "
              f"{'ACE':>8}  {'Winkler':>9}  Status")
        print(f"  {'─'*70}")
        for s in range(n):
            mins   = (s + 1) * 15
            p      = per_step['picp'][s]
            m      = per_step['mpiw'][s]
            a      = per_step['ace'][s]
            w      = per_step['winkler'][s]
            status = 'OK' if p >= (1 - alpha) else 'UNDER-COVERAGE'
            print(f"  {s+1:>5}  {mins:>5}m  {p:>7.4f}  {m:>7.4f}  "
                  f"{a:>+8.4f}  {w:>9.4f}  {status}")
        print(f"  {'─'*70}\n")

    # ── Helper: save and show ─────────────────────────────────────────────────
    def _save_fig(self, fig, filename):
        """Saves figure to figures/ directory at 300 DPI and shows it."""
        fpath = os.path.join(self._fig_dir, filename)
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f"  Figure saved -> {fpath}")
        plt.show()
        plt.close(fig)

    # ── Visualisation ─────────────────────────────────────────────────────────
    def _plot(self, y_u, y_l, y_true, overall, per_step, alpha):
        """
        Produces 4 publication-quality figures for one confidence level
        and auto-saves each one to figures/.

        Plot 1 -- Full PI-Test time series with prediction interval band
        Plot 2 -- Zoomed view (display_size timesteps)
        Plot 3+4 -- PICP and MPIW per forecast horizon (side by side)
        """
        conf      = int((1 - alpha) * 100)
        clabel    = f'{conf}%'
        n_s       = self.step_to_show
        sl        = f'+{(n_s+1)*15} min'
        n_t       = len(y_true)
        t_f       = np.arange(n_t) * 15
        n_z       = min(self.display_size, n_t)
        t_z       = np.arange(n_z) * 15
        step_mins = np.arange(1, self.predicted_step + 1) * 15

        uu = y_u[:, n_s]
        ul = y_l[:, n_s]
        yt = y_true[:, n_s]

        # ── Plot 1: Full set ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(18, 5))
        ax.fill_between(t_f, uu, ul, color='#33FFE3', alpha=0.55,
                        label=f'MOGD-LUBE {clabel} PI', zorder=2)
        ax.plot(t_f, uu, lw=0.8, color='#00A8C8', alpha=0.75, zorder=3)
        ax.plot(t_f, ul, lw=0.8, color='#00A8C8', alpha=0.75, zorder=3)
        ax.plot(t_f, yt,  lw=1.5, color='black', label='Observed', zorder=4)
        ax.set_xlabel('Time (min into PI-Test set)')
        ax.set_ylabel('Normalized Electric Power')
        ax.set_title(
            f'MOGD-LUBE [{self.modelType}]  |  {clabel} PI  |  Full PI-Test  |  Horizon: {sl}\n'
            f'PICP={overall["picp"]:.4f}   MPIW={overall["mpiw"]:.4f}   '
            f'ACE={overall["ace"]:+.4f}   Winkler={overall["winkler"]:.4f}')
        ax.legend(loc='upper right')
        plt.tight_layout()
        self._save_fig(fig, f'{self.modelType}_alpha{conf}_full_PI_{(n_s+1)*15}min.png')

        # ── Plot 2: Zoomed view ───────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(18, 5))
        ax.fill_between(t_z, uu[:n_z], ul[:n_z], color='#33FFE3', alpha=0.55,
                        label=f'MOGD-LUBE {clabel} PI', zorder=2)
        ax.plot(t_z, uu[:n_z], lw=1.0, color='#00A8C8', alpha=0.85, zorder=3)
        ax.plot(t_z, ul[:n_z], lw=1.0, color='#00A8C8', alpha=0.85, zorder=3)
        ax.plot(t_z, yt[:n_z], lw=2.0, color='black', label='Observed', zorder=4)
        ax.set_xlabel('Time (min into PI-Test set)')
        ax.set_ylabel('Normalized Electric Power')
        ax.set_title(
            f'MOGD-LUBE [{self.modelType}]  |  {clabel} PI  |  '
            f'Zoomed ({n_z} steps = {n_z*15//60} h)  |  Horizon: {sl}')
        ax.legend(loc='upper right')
        plt.tight_layout()
        self._save_fig(fig, f'{self.modelType}_alpha{conf}_zoomed_PI_{(n_s+1)*15}min.png')

        # ── Plot 3 & 4: PICP and MPIW per horizon ────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        fig.subplots_adjust(top=0.84, wspace=0.30)

        # PICP
        ax1.plot(step_mins, per_step['picp'], marker='o', lw=2.5, ms=7,
                 color='#00CED1', label='PICP per step')
        ax1.axhline(1 - alpha, color='crimson', ls='--', lw=2.0,
                    label=f'Target >= {1-alpha:.2f}')
        ax1.fill_between(step_mins, per_step['picp'], 1 - alpha,
                         where=[p < 1 - alpha for p in per_step['picp']],
                         color='crimson', alpha=0.15, label='Under-coverage zone')
        ax1.set_xlabel('Forecast Horizon (minutes)')
        ax1.set_ylabel('PICP')
        ax1.set_title(f'Coverage per Horizon  |  {clabel} PI')
        ax1.set_ylim(0, 1.08)
        ax1.legend(loc='lower left')

        # MPIW
        ax2.plot(step_mins, per_step['mpiw'], marker='s', lw=2.5, ms=7,
                 color='#FFA500', label='MPIW per step')
        ax2.set_xlabel('Forecast Horizon (minutes)')
        ax2.set_ylabel('MPIW')
        ax2.set_title(f'Interval Width per Horizon  |  {clabel} PI')
        ax2.legend(loc='upper left')

        fig.suptitle(
            f'MOGD-LUBE [{self.modelType}]  |  Per-Horizon Quality  |  '
            f'{clabel} PI  |  {self.predicted_step} steps  |  PI-Test set',
            fontsize=14, fontweight='bold', y=0.99)
        self._save_fig(fig, f'{self.modelType}_alpha{conf}_per_horizon_metrics.png')

    # ── Multi-alpha comparison plot ───────────────────────────────────────────
    def _plot_alpha_comparison(self, results):
        """
        Side-by-side comparison of per-horizon PICP and MPIW for all alphas.
        Auto-saved to figures/ at 300 DPI.
        """
        if len(results) < 2:
            return

        step_mins = np.arange(1, self.predicted_step + 1) * 15
        colors    = ['#00CED1', '#FFA500', '#9B59B6', '#E74C3C']
        conf_labels = ' vs '.join([f"{int((1-a)*100)}%" for a, _, __ in results])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        fig.subplots_adjust(top=0.84, wspace=0.30)

        for idx, (alpha, overall, per_step) in enumerate(results):
            conf  = int((1 - alpha) * 100)
            color = colors[idx % len(colors)]
            ax1.plot(step_mins, per_step['picp'], marker='o', lw=2.5, ms=7,
                     color=color,
                     label=f'{conf}% PI  (overall PICP={overall["picp"]:.4f})')
            ax1.axhline(1 - alpha, color=color, ls='--', lw=1.5, alpha=0.45)
            ax2.plot(step_mins, per_step['mpiw'], marker='s', lw=2.5, ms=7,
                     color=color,
                     label=f'{conf}% PI  (overall MPIW={overall["mpiw"]:.4f})')

        ax1.set_xlabel('Forecast Horizon (minutes)')
        ax1.set_ylabel('PICP')
        ax1.set_title(f'Coverage Comparison  |  {conf_labels} PI')
        ax1.set_ylim(0, 1.08)
        ax1.legend(loc='lower left')

        ax2.set_xlabel('Forecast Horizon (minutes)')
        ax2.set_ylabel('MPIW')
        ax2.set_title(f'Interval Width Comparison  |  {conf_labels} PI')
        ax2.legend(loc='upper left')

        fig.suptitle(
            f'MOGD-LUBE [{self.modelType}]  |  Confidence Level Comparison\n'
            f'Dataset: {self.dataset_name}  |  PI-Test set (held-out)  |  {conf_labels}',
            fontsize=14, fontweight='bold', y=0.99)
        self._save_fig(fig, f'{self.modelType}_alpha_comparison_{conf_labels.replace(" vs ", "_")}.png')

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self, alphas=None):
        """
        Evaluates the model for one or more confidence levels.

        Parameters
        ----------
        alphas : list of float, default [0.10, 0.05]
                 Confidence levels to evaluate.
                 0.10 -> 90% PI  |  0.05 -> 95% PI
                 A separate saved model must exist for each alpha.

        Returns
        -------
        all_results : list of (alpha, y_u, y_l, y_true, overall, per_step)
        """
        if alphas is None:
            alphas = [0.10, 0.05]

        print(f"\n{'='*80}")
        print(f"  MOGD-LUBE -- Evaluation")
        print(f"  Model     : {self.modelType}  |  Dataset: {self.dataset_name}")
        print(f"  Alphas    : {[int((1-a)*100) for a in alphas]}% PI levels")
        print(f"  Test file : {TEST_FILE}")
        print(f"              -> second 50% = PI-Test (held-out)")
        print(f"  Figures   : {self._fig_dir}")
        print(f"{'='*80}\n")

        all_results   = []
        comparison    = []
        json_by_alpha = {}   # keyed by alpha, value = (overall, per_step_test, per_step_val)

        for alpha in alphas:
            conf = int((1 - alpha) * 100)
            print(f"\n  -- Evaluating {conf}% PI ----------------------------------------")

            model, scaler, n_std_devs           = self._load_model(alpha)
            X_pi_test, y_pi_test, X_pi_val, y_pi_val = self._load_pi_test(scaler)

            # ── PI-Test evaluation ────────────────────────────────────────────
            y_u, y_l          = self._predict(model, X_pi_test, n_std_devs)
            overall, per_step = self._compute_metrics(y_u, y_l, y_pi_test, alpha, scaler)

            # ── PI-Val evaluation (for JSON export only, not printed) ─────────
            y_u_val, y_l_val         = self._predict(model, X_pi_val, n_std_devs)
            _, per_step_val          = self._compute_metrics(
                y_u_val, y_l_val, y_pi_val, alpha, scaler)

            self._print_metrics(overall, per_step, alpha)
            self._plot(y_u, y_l, y_pi_test, overall, per_step, alpha)

            all_results.append((alpha, y_u, y_l, y_pi_test, overall, per_step))
            comparison.append((alpha, overall, per_step))
            json_by_alpha[alpha] = (overall, per_step, per_step_val)

        # Side-by-side comparison if multiple alphas were evaluated
        if len(alphas) > 1:
            self._plot_alpha_comparison(comparison)

        # Final summary table
        print(f"\n{'='*80}")
        print(f"  SUMMARY -- {self.modelType}  |  {self.dataset_name}")
        print(f"{'='*80}")
        print(f"  {'Conf':>6}  {'PICP':>8}  {'MPIW':>8}  {'ACE':>9}  {'Winkler':>10}")
        print(f"  {'─'*55}")
        for alpha, overall, per_step in comparison:
            conf = int((1 - alpha) * 100)
            print(f"  {conf:>5}%  {overall['picp']:>8.4f}  {overall['mpiw']:>8.4f}  "
                  f"{overall['ace']:>+9.4f}  {overall['winkler']:>10.4f}")
        print(f"{'='*80}\n")

        # Export JSON results
        self._save_json(json_by_alpha, alphas)

        return all_results

    # ── Save JSON results (legacy-compatible format) ───────────────────────────
    def _save_json(self, json_by_alpha, alphas):
        """
        Saves per-horizon, per-confidence metrics to a JSON file in the project
        root directory, using a structure compatible with the legacy KDE output.

        Fields per horizon step:
          pred_hor             : forecast step (1 = +15 min, 16 = +4 h)
          model_type           : LSTM / MLP
          input_window_size    : window used
          num_neurons          : hidden units
          predicted_step       : total horizon steps
          conf_interval[]:
            confidence         : 0.975 / 0.95 / 0.90
            lower_limit        : mean(y_l_Watts) - mean(y_true_Watts)  per step
            upper_limit        : mean(y_u_Watts) - mean(y_true_Watts)  per step
            PICP_testSet       : PICP on PI-Test set  (as %, 0-100)
            pinaw_testSet      : MPIW on PI-Test set  (normalized, 0-1)
            ACE_testSet        : ACE  on PI-Test set
            Winkler_testSet    : Winkler score on PI-Test set
            PICP_piEvalSet     : PICP on PI-Val set   (as %, 0-100)
            pinaw_piEvalSet    : MPIW on PI-Val set   (normalized, 0-1)
        """
        n_steps = self.predicted_step
        output  = []

        for step_idx in range(n_steps):
            conf_list = []
            for alpha in sorted(alphas):          # 0.025 -> 0.05 -> 0.10
                overall, ps_test, ps_val = json_by_alpha[alpha]
                conf_entry = {
                    'confidence'     : round(1 - alpha, 3),
                    'lower_limit'    : ps_test['lower_err_w'][step_idx],
                    'upper_limit'    : ps_test['upper_err_w'][step_idx],
                    'PICP_testSet'   : round(ps_test['picp'][step_idx] * 100, 4),
                    'pinaw_testSet'  : round(ps_test['mpiw'][step_idx], 10),
                    'ACE_testSet'    : round(ps_test['ace'][step_idx], 6),
                    'Winkler_testSet': round(ps_test['winkler'][step_idx], 6),
                    'PICP_piEvalSet' : round(ps_val['picp'][step_idx] * 100, 4),
                    'pinaw_piEvalSet': round(ps_val['mpiw'][step_idx], 10),
                }
                conf_list.append(conf_entry)

            horizon_entry = {
                'pred_hor'         : step_idx + 1,
                'model_type'       : self.modelType,
                'input_window_size': getattr(self, 'input_window_size', None),
                'num_neurons'      : getattr(self, 'num_neurons', None),
                'predicted_step'   : n_steps,
                'conf_interval'    : conf_list,
                'errors'           : [],
            }
            output.append(horizon_entry)

        # Save to project root
        base_dir  = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(
            base_dir, f'{self.dataset_name}_{self.modelType}_PI_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"  JSON results saved -> {json_path}")


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tester = LUBETester(
        dataset_name = 'simulation_150days',
        modelType    = 'MLP',
        display_size = 200,
        step_to_show = 0,
    )
    tester.run(alphas=[0.10, 0.05])
