import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
import seaborn as sns
from sklearn.utils import shuffle
import joblib
import math
import os
import time
from module_models import qd_objective, winkler_objective, VariationalLSTM

sns.set(rc={"figure.figsize": (32, 24)})
plt.rcParams['axes.facecolor'] = 'white'


# ─────────────────────────────────────────────────────────────────────────────
# HELPER  – mirrors series_to_supervised in 01_run_preprocessing.py exactly so the
#           test CSV receives the same windowing as the training data.
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
class bayesian_trainer():
    """
    Variational-dropout Bayesian trainer for ISWEC wave-energy prediction
    intervals (VariationalLSTM).

    Uncertainty decomposition
    ─────────────────────────
    • Aleatoric  – mean PI band  (average over MC forward passes)
    • Epistemic  – spread of PI band  (std over MC forward passes)

    Data flow (aligned with engine_trainer.py)
    ────────────────────────────────────
    Training data  : full supervised CSV from 01_run_preprocessing.py
                     (no portion held back from training)
    PI-validation  : first  50% of test_set_results_simulation_150days_window9000.csv
                     – used only to monitor val loss per epoch
    PI-test        : second 50% of the same file
                     – used for final metrics + all visualisation

    NOTE: display_size must equal batch_size because VariationalLSTM samples
    its dropout masks once per batch; mismatching them changes the effective
    mask during visualisation.
    """

    def __init__(self, modelType='VariationalLSTM', trainingType='CrossValidation',
                 lossType='winkler', layer_dropout=0.2, time_dropout=0.2,
                 num_forward_passes=100, lambda_=0.0005, soften_=160.,
                 num_epoch=100, alpha_=0.05, fold_size=8, train_prop=0.8,
                 batch_size=128, input_window_size=24, predicted_step=1,
                 num_neurons=64, draw=True, display_size=128):

        self.alpha_            = alpha_
        self.batch_size        = batch_size
        assert fold_size > 1
        self.input_window_size = input_window_size
        self.predicted_step    = predicted_step
        self.num_neurons       = num_neurons
        self.modelType         = modelType
        self.trainingType      = trainingType

        assert trainingType in ['CrossValidation', 'SinglePass']
        self.fold_size = fold_size if trainingType == 'CrossValidation' else 1

        self.train_prop        = train_prop
        self.device            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lambda_           = lambda_
        self.soften_           = soften_
        self.num_epoch         = num_epoch
        self.draw              = draw
        self.display_size      = display_size   # keep == batch_size for VariationalLSTM

        assert lossType in ['qd', 'winkler']
        self.lossType          = lossType
        self.layer_dropout     = layer_dropout
        self.time_dropout      = time_dropout
        self.num_forward_passes = num_forward_passes

        # n_std_devs used for optional Gaussian PI width reference only
        if   self.alpha_ == 0.05: self.n_std_devs = 1.96
        elif self.alpha_ == 0.10: self.n_std_devs = 1.645
        elif self.alpha_ == 0.01: self.n_std_devs = 2.575

    # ─────────────────────────────────────────────────────────────────────────
    # DATA LOADING
    # ─────────────────────────────────────────────────────────────────────────

    def load_training_data(self, dataset_name='simulation_150days_ISWEC_LUBE'):
        """
        Loads the pre-processed, already-normalised supervised training CSV
        produced by 01_run_preprocessing.py.

        Expected file:
            dataset/{predicted_step}_{dataset_name}_supervised.csv

        The entire file is used for training – no portion is withheld.
        CrossValidation : k-fold splits operate on this training file only.
        SinglePass      : entire file becomes X_train.

        Returns
        -------
        X_train_list : list of numpy arrays (N, input_window_size, 1), one per fold
        y_train_list : list of numpy arrays (N, predicted_step),       one per fold
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, 'dataset',
                            f'{self.predicted_step}_{dataset_name}_supervised.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Preprocessed training file not found:\n  {path}\n"
                "Run 01_run_preprocessing.py to generate it before training."
            )

        df = pd.read_csv(path).set_index('index')
        X_train_list, y_train_list = [], []

        if self.trainingType == 'CrossValidation':
            batch_num_1fold = df.shape[0] // (self.batch_size * self.fold_size)
            df = df.head(batch_num_1fold * self.batch_size * self.fold_size)
            for i in range(self.fold_size):
                train = np.delete(
                    df.values,
                    range(i * batch_num_1fold * self.batch_size,
                          (i + 1) * batch_num_1fold * self.batch_size),
                    axis=0
                )
                y_train = train[:, -self.predicted_step:].reshape(-1, self.predicted_step)
                X_train = train[:, :-self.predicted_step].reshape(-1, self.input_window_size, 1)
                X_train_list.append(X_train)
                y_train_list.append(y_train)

        elif self.trainingType == 'SinglePass':
            values  = df.values
            y_train = values[:, -self.predicted_step:].reshape(-1, self.predicted_step)
            X_train = values[:, :-self.predicted_step].reshape(-1, self.input_window_size, 1)
            X_train_list.append(X_train)
            y_train_list.append(y_train)

        return X_train_list, y_train_list

    def load_pi_sets(self, dataset_name='simulation_150days_ISWEC_LUBE'):
        """
        Loads, normalises, and 50/50-splits the external test CSV.

        Expected files:
            dataset/test_set_results_simulation_150days_window9000.csv
            dataset/scaler_{dataset_name}.pkl   (saved by 01_run_preprocessing.py)

        Normalisation uses scaler.transform() – NOT fit_transform() – so the
        test set shares the exact min/max range fitted on the training data.

        Returns
        -------
        X_pi_val  : (N//2, input_window_size, 1)  – first  50% → val-loss monitor
        y_pi_val  : (N//2, predicted_step)
        X_pi_test : (N//2, input_window_size, 1)  – second 50% → final evaluation
        y_pi_test : (N//2, predicted_step)
        """
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        test_csv   = os.path.join(base_dir, 'dataset',
                                  'test_set_results_simulation_150days_window9000.csv')
        scaler_pkl = os.path.join(base_dir, 'dataset', f'scaler_{dataset_name}.pkl')

        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"Test CSV not found:\n  {test_csv}")
        if not os.path.exists(scaler_pkl):
            raise FileNotFoundError(
                f"Scaler not found:\n  {scaler_pkl}\n"
                "Make sure 01_run_preprocessing.py has joblib.dump(scaler, scaler_path) "
                "in its SAVE section."
            )

        df_test      = pd.read_csv(test_csv)
        power_values = df_test['TrifPelect_downsampled'].values.reshape(-1, 1)

        scaler = joblib.load(scaler_pkl)
        scaled = scaler.transform(power_values)   # transform only, NOT fit_transform

        supervised_df = series_to_supervised(scaled, self.input_window_size, self.predicted_step)
        values = supervised_df.values

        half         = len(values) // 2
        pi_val_data  = values[:half]
        pi_test_data = values[half:]

        def split_xy(arr):
            y = arr[:, -self.predicted_step:].reshape(-1, self.predicted_step)
            X = arr[:, :-self.predicted_step].reshape(-1, self.input_window_size, 1)
            return X, y

        X_pi_val,  y_pi_val  = split_xy(pi_val_data)
        X_pi_test, y_pi_test = split_xy(pi_test_data)

        print(f"[load_pi_sets]  PI-validation : {X_pi_val.shape[0]:>6} samples  (first 50%)")
        print(f"[load_pi_sets]  PI-test       : {X_pi_test.shape[0]:>6} samples  (second 50%)")
        return X_pi_val, y_pi_val, X_pi_test, y_pi_test

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def training_loop(self, X_train, y_train, X_val, y_val,
                      model, optimizer, scheduler, criterion):
        """
        Inner epoch loop.
        X_val / y_val is the PI-validation set (first 50% of test CSV).
        It is used only to compute val loss per epoch – never for gradient updates.
        """
        lrs             = []
        train_loss_list = []
        valid_loss_list = []

        for epoch in range(self.num_epoch):
            # ── training ──────────────────────────────────────────────────────
            train_loss = 0.0
            model.train()
            x_train, Y_train = shuffle(X_train, y_train)

            for batch in range(math.ceil(X_train.shape[0] / self.batch_size)):
                start   = batch * self.batch_size
                end     = start + self.batch_size
                inputs  = Variable(torch.tensor(x_train[start:end], dtype=torch.float)).to(self.device)
                targets = Variable(torch.tensor(Y_train[start:end], dtype=torch.float)).to(self.device)

                optimizer.zero_grad()
                outputs    = model(inputs)
                loss       = criterion(outputs, targets)
                loss_data  = loss.item()
                loss.backward()
                optimizer.step()

                train_loss += loss_data * inputs.size(0)

            train_loss_list.append(train_loss / X_train.shape[0])

            # ── validation on PI-validation set ───────────────────────────────
            valid_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch in range(math.ceil(X_val.shape[0] / self.batch_size)):
                    start   = batch * self.batch_size
                    end     = start + self.batch_size
                    inputs  = Variable(torch.tensor(X_val[start:end], dtype=torch.float)).to(self.device)
                    targets = Variable(torch.tensor(y_val[start:end], dtype=torch.float)).to(self.device)
                    outputs   = model(inputs)
                    loss_data = criterion(outputs, targets).item()
                    valid_loss += loss_data * inputs.size(0)

            valid_loss_list.append(valid_loss / X_val.shape[0])
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

            print('-' * 110)
            print(f'Epoch {epoch+1}  '
                  f'Train Loss: {train_loss/X_train.shape[0]:.6f}  '
                  f'Val Loss (PI-val): {valid_loss/X_val.shape[0]:.6f}')

        return lrs, train_loss_list, valid_loss_list

    # ─────────────────────────────────────────────────────────────────────────
    # ONE-FOLD TRAINING + VISUALISATION
    # ─────────────────────────────────────────────────────────────────────────

    def one_fold_training(self, X_train, y_train,
                          X_pi_val, y_pi_val,
                          X_pi_test, y_pi_test,
                          dataset_name='simulation_150days_ISWEC_LUBE'):
        """
        Builds and trains VariationalLSTM for one fold, then evaluates on the
        PI-test set using MC dropout (num_forward_passes stochastic passes).

        Visualisation (when draw=True):
            1. Convergence plot   – train / val loss curves
            2. Uncertainty plot   – aleatoric + epistemic PI bands on PI-test set
               • Green fill  : aleatoric uncertainty (mean upper/lower band)
               • Blue fill   : epistemic spread of upper bound
               • Red fill    : epistemic spread of lower bound
        """
        assert self.modelType in ['VariationalLSTM']
        model = VariationalLSTM(
            num_neurons       = self.num_neurons,
            input_window_size = self.input_window_size,
            predicted_step    = self.predicted_step,
            layer_dropout     = self.layer_dropout,
            time_dropout      = self.time_dropout,
            batch_size        = self.batch_size,
            device            = self.device
        )

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1)

        assert self.lossType in ['qd', 'winkler']
        if self.lossType == 'qd':
            criterion = qd_objective(lambda_=self.lambda_, alpha_=self.alpha_,
                                     soften_=self.soften_, device=self.device,
                                     batch_size=self.batch_size)
        elif self.lossType == 'winkler':
            criterion = winkler_objective(lambda_=self.lambda_, alpha_=self.alpha_,
                                          soften_=self.soften_, device=self.device,
                                          batch_size=self.batch_size)

        model     = model.to(self.device)
        criterion = criterion.to(self.device)

        # ── training (val loss monitored on PI-val set) ───────────────────────
        lrs, train_loss_list, valid_loss_list = self.training_loop(
            X_train=X_train, y_train=y_train,
            X_val=X_pi_val,  y_val=y_pi_val,
            model=model, optimizer=optimizer,
            scheduler=scheduler, criterion=criterion
        )

        # ── convergence plot ──────────────────────────────────────────────────
        if self.draw:
            fig, ax = plt.subplots(figsize=(32, 12))
            ax.plot(train_loss_list, color='r', linewidth=3, label='Training loss')
            ax.plot(valid_loss_list, color='b', linewidth=3, label='PI-Validation loss')
            ax.set_xlabel("Epoch", fontsize=48)
            ax.set_ylabel("Loss", fontsize=48)
            ax.set_title(f'Convergence  [{dataset_name}  –  {self.modelType}]', fontsize=40)
            ax.tick_params(labelsize=32)
            ax.legend(loc="upper right", fontsize=36)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # ── MC forward passes over display_size PI-test samples (zoomed view) ─
        # NOTE: display_size must equal batch_size so VariationalLSTM draws
        #       exactly one set of dropout masks per call.
        disp      = min(self.display_size, X_pi_test.shape[0])
        x_disp    = X_pi_test[:disp]
        y_disp    = y_pi_test[:disp]

        if self.draw:
            predictions = []
            model.train()   # keep dropout active for MC sampling
            for _ in range(self.num_forward_passes):
                with torch.no_grad():
                    out = model(Variable(
                        torch.tensor(x_disp, dtype=torch.float)).to(self.device)
                    ).cpu().detach().numpy()
                predictions.append(out)

            pred_array = np.array(predictions)   # (num_forward_passes, disp, 2*predicted_step)
            y_u_pred   = pred_array[:, :, 0]     # upper bounds across passes
            y_l_pred   = pred_array[:, :, 1]     # lower bounds across passes
            y_u_mean   = y_u_pred.mean(axis=0)
            y_u_std    = y_u_pred.std(axis=0)
            y_l_mean   = y_l_pred.mean(axis=0)
            y_l_std    = y_l_pred.std(axis=0)

            # aleatoric band extremes; epistemic spread around upper/lower means
            y_u_u = y_u_mean + self.n_std_devs * y_u_std   # epistemic upper of upper bound
            y_u_l = y_u_mean - self.n_std_devs * y_u_std   # epistemic lower of upper bound
            y_l_u = y_l_mean + self.n_std_devs * y_l_std   # epistemic upper of lower bound
            y_l_l = y_l_mean - self.n_std_devs * y_l_std   # epistemic lower of lower bound

            x_axis = np.arange(disp)
            fig, ax = plt.subplots(figsize=(32, 18))

            ax.plot(x_axis, y_disp[:, 0], linewidth=3, color='black',
                    label='Observed TrifPelect (normalised)', zorder=5)

            # aleatoric uncertainty (mean PI band)
            ax.plot(x_axis, y_u_mean, color='g', linewidth=1.5, zorder=4)
            ax.plot(x_axis, y_l_mean, color='g', linewidth=1.5, zorder=4)
            ax.fill_between(x_axis, y_u_mean, y_l_mean,
                            color='g', alpha=0.50, zorder=3,
                            label='Aleatoric Uncertainty (mean PI band)')

            # epistemic spread of upper bound
            ax.plot(x_axis, y_u_u, color='b', linewidth=1.0, zorder=2)
            ax.plot(x_axis, y_u_l, color='b', linewidth=1.0, zorder=2)
            ax.fill_between(x_axis, y_u_u, y_u_l,
                            color='b', alpha=0.30, zorder=1,
                            label='Epistemic Uncertainty – Upper Bound')

            # epistemic spread of lower bound
            ax.plot(x_axis, y_l_u, color='r', linewidth=1.0, zorder=2)
            ax.plot(x_axis, y_l_l, color='r', linewidth=1.0, zorder=2)
            ax.fill_between(x_axis, y_l_u, y_l_l,
                            color='r', alpha=0.30, zorder=1,
                            label='Epistemic Uncertainty – Lower Bound')

            conf_pct = int((1 - self.alpha_) * 100)
            ax.set_title(
                f'Bayesian VariationalLSTM  –  {conf_pct}% PI  [{dataset_name}]\n'
                f'PI-test set  (first {disp} steps  |  {self.num_forward_passes} MC passes)',
                fontsize=36, pad=14
            )
            ax.set_xlabel("Time step (PI-test portion)", fontsize=32)
            ax.set_ylabel("Normalised Power  (TrifPelect_downsampled)", fontsize=32)
            ax.tick_params(labelsize=26)
            ax.legend(loc="upper left", fontsize=28)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # ── MC stats over the full PI-test set ────────────────────────────────
        predictions = []
        model.train()   # keep dropout active for MC sampling
        for _ in range(self.num_forward_passes):
            prediction = None
            for batch in range(math.ceil(X_pi_test.shape[0] / self.batch_size)):
                start = batch * self.batch_size
                end   = start + self.batch_size
                with torch.no_grad():
                    chunk = model(Variable(
                        torch.tensor(X_pi_test[start:end], dtype=torch.float)
                    ).to(self.device)).cpu().detach().numpy()
                prediction = chunk if prediction is None else np.concatenate((prediction, chunk), axis=0)
            predictions.append(prediction)

        pred_array = np.array(predictions)
        y_u_pred   = pred_array[:, :, 0]
        y_l_pred   = pred_array[:, :, 1]
        y_u_mean   = y_u_pred.mean(axis=0)
        y_u_std    = y_u_pred.std(axis=0)
        y_l_mean   = y_l_pred.mean(axis=0)
        y_l_std    = y_l_pred.std(axis=0)

        K_u        = np.maximum(0.0, np.sign(y_u_mean - y_pi_test[:, 0]))
        K_l        = np.maximum(0.0, np.sign(y_pi_test[:, 0] - y_l_mean))
        picp       = np.mean(K_u * K_l)
        mpiw       = np.mean(np.absolute(y_u_mean - y_l_mean))
        S_t        = (np.absolute(y_u_mean - y_l_mean)
                      + (2/self.alpha_) * np.multiply(
                          y_l_mean - y_pi_test[:, 0],
                          np.maximum(0.0, np.sign(y_l_mean - y_pi_test[:, 0])))
                      + (2/self.alpha_) * np.multiply(
                          y_pi_test[:, 0] - y_u_mean,
                          np.maximum(0.0, np.sign(y_pi_test[:, 0] - y_u_mean))))
        S_overline = np.mean(S_t)

        print('─' * 60)
        print('[Bayesian VariationalLSTM — PI Test Set results]')
        print(f'  PICP          : {picp:.4f}   (target ≥ {1-self.alpha_:.2f})')
        print(f'  MPIW          : {mpiw:.4f}')
        print(f'  Winkler Score : {S_overline:.6f}')
        print(f'  ACE           : {picp - (1-self.alpha_):+.4f}')
        print(f'  Upper Bound – Mean: {np.mean(y_u_mean):.4f}   Std: {np.mean(y_u_std):.4f}')
        print(f'  Lower Bound – Mean: {np.mean(y_l_mean):.4f}   Std: {np.mean(y_l_std):.4f}')
        print('─' * 60)

        return (picp, mpiw, S_overline,
                np.mean(y_u_mean), np.mean(y_u_std),
                np.mean(y_l_mean), np.mean(y_l_std))

    # ─────────────────────────────────────────────────────────────────────────
    # ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, dataset_name='simulation_150days_ISWEC_LUBE'):
        """
        Main entry point.

        Steps
        ─────
        1. load_training_data()  – full supervised training CSV (no split)
        2. load_pi_sets()        – test CSV → 50% PI-val / 50% PI-test
        3. one_fold_training()   – one iteration per fold

        Usage:
            bt = bayesian_trainer(
                    modelType='VariationalLSTM', trainingType='SinglePass',
                    lossType='winkler', input_window_size=24, predicted_step=16,
                    num_epoch=100, batch_size=128, display_size=128, alpha_=0.05)
            bt.run(dataset_name='simulation_150days_ISWEC_LUBE')
        """
        X_train_list, y_train_list = self.load_training_data(dataset_name=dataset_name)
        X_pi_val, y_pi_val, X_pi_test, y_pi_test = self.load_pi_sets(dataset_name=dataset_name)

        picp_list, mpiw_list, ace_list, score_list = [], [], [], []
        bayesian_dict = {key: [] for key in ['ubm', 'ubs', 'lbm', 'lbs']}

        for i in range(self.fold_size):
            print(f'{"─"*50} fold {i} {"─"*50}')
            picp, mpiw, score, ubm, ubs, lbm, lbs = self.one_fold_training(
                X_train_list[i], y_train_list[i],
                X_pi_val,  y_pi_val,
                X_pi_test, y_pi_test,
                dataset_name=dataset_name
            )
            picp_list.append(picp)
            mpiw_list.append(mpiw)
            ace_list.append(picp - (1 - self.alpha_))
            score_list.append(score)
            bayesian_dict['ubm'].append(ubm)
            bayesian_dict['ubs'].append(ubs)
            bayesian_dict['lbm'].append(lbm)
            bayesian_dict['lbs'].append(lbs)

        print('\n' + '═' * 60)
        print('  Bayesian VariationalLSTM — Aggregated Results')
        print('═' * 60)
        print(f'  PICP           mean {np.mean(picp_list):.4f}   std {np.std(picp_list):.4f}')
        print(f'  MPIW           mean {np.mean(mpiw_list):.4f}   std {np.std(mpiw_list):.4f}')
        print(f'  ACE            mean {np.mean(ace_list):+.4f}  std {np.std(ace_list):.4f}')
        print(f'  Winkler Score  mean {np.mean(score_list):.6f}   std {np.std(score_list):.6f}')
        print(f'  Upper Bound    mean of means {np.mean(bayesian_dict["ubm"]):.4f}   std of means {np.std(bayesian_dict["ubm"]):.4f}')
        print(f'                 mean of stds  {np.mean(bayesian_dict["ubs"]):.4f}   std of stds  {np.std(bayesian_dict["ubs"]):.4f}')
        print(f'  Lower Bound    mean of means {np.mean(bayesian_dict["lbm"]):.4f}   std of means {np.std(bayesian_dict["lbm"]):.4f}')
        print(f'                 mean of stds  {np.mean(bayesian_dict["lbs"]):.4f}   std of stds  {np.std(bayesian_dict["lbs"]):.4f}')
        print('═' * 60)
