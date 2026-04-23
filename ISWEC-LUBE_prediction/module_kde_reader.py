"""
read_kde_module_models.py
══════════════════════════════════════════════════════════════════════════════
Reads a Keras LSTM model (.h5) and extracts all architecture parameters.
No tensorflow or h5py required — uses only Python built-ins + json + re.

USAGE:
  from module_kde_reader import read_kde_params, print_kde_params
  params = read_kde_params('lstm_vector_150days_w9000_22.h5')
  print_kde_params(params)
══════════════════════════════════════════════════════════════════════════════
"""

import re
import json
import os


def read_kde_params(h5_path):
    """
    Extracts architecture and training parameters from a Keras .h5 file.

    Returns dict with:
      input_window_size, num_neurons, predicted_step,
      learning_rate, dropout, activation, model_name, keras_version, optimizer
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Model file not found: {h5_path}")

    with open(h5_path, 'rb') as f:
        content = f.read()

    # ── Extract model_config JSON ─────────────────────────────────────────
    model_cfg_raw = None
    for marker in [b'{"class_name": "Sequential"', b'{"class_name":"Sequential"']:
        start = content.find(marker)
        if start != -1:
            depth, in_str, escape = 0, False, False
            chunk = content[start:start+10000].decode('ascii', errors='replace')
            for i, ch in enumerate(chunk):
                if escape:          escape = False; continue
                if ch == '\\':      escape = True;  continue
                if ch == '"':       in_str = not in_str; continue
                if not in_str:
                    if ch == '{':   depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            model_cfg_raw = chunk[:i+1]; break
            if model_cfg_raw: break

    if not model_cfg_raw:
        raise ValueError("Could not parse model_config from .h5 file.")

    model_cfg = json.loads(model_cfg_raw)
    layers    = model_cfg['config']['layers']

    # ── Extract training_config JSON ──────────────────────────────────────
    train_cfg = {}
    for marker in [b'{"loss":', b'{"loss": "']:
        t_start = content.find(marker)
        if t_start != -1:
            chunk = content[t_start:t_start+1000].decode('ascii', errors='replace')
            depth, in_str, escape = 0, False, False
            for i, ch in enumerate(chunk):
                if escape:          escape = False; continue
                if ch == '\\':      escape = True;  continue
                if ch == '"':       in_str = not in_str; continue
                if not in_str:
                    if ch == '{':   depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            try:    train_cfg = json.loads(chunk[:i+1])
                            except: pass
                            break
            break

    # ── Parse layer configs ───────────────────────────────────────────────
    input_window_size = None
    for layer in layers:
        if layer['class_name'] == 'InputLayer':
            input_window_size = layer['config']['batch_input_shape'][1]
            break
        if layer['class_name'] == 'LSTM' and 'batch_input_shape' in layer['config']:
            input_window_size = layer['config']['batch_input_shape'][1]
            break

    num_neurons = None; activation = 'tanh'; dropout = 0.0
    for layer in layers:
        if layer['class_name'] == 'LSTM':
            num_neurons = layer['config']['units']
            activation  = layer['config'].get('activation', 'tanh')
            dropout     = layer['config'].get('dropout', 0.0)
            break

    predicted_step = None
    for layer in reversed(layers):
        if layer['class_name'] == 'Dense':
            predicted_step = layer['config']['units']
            break

    model_name = model_cfg['config'].get('name', 'sequential')

    # ── Keras version ─────────────────────────────────────────────────────
    keras_ver  = 'unknown'
    ver_area   = content[content.find(b'keras_version'):
                         content.find(b'keras_version')+200]
    ver_match  = re.search(rb'(\d+\.\d+\.\d+)', ver_area)
    if ver_match: keras_ver = ver_match.group(1).decode()

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer_name = 'Adam'; learning_rate = 0.001
    if train_cfg:
        opt_cfg        = train_cfg.get('optimizer_config', {})
        optimizer_name = opt_cfg.get('class_name', 'Adam')
        lr             = opt_cfg.get('config', {}).get('learning_rate', 0.001)
        learning_rate  = round(float(lr), 6)

    return {
        'input_window_size' : input_window_size,
        'num_neurons'       : num_neurons,
        'predicted_step'    : predicted_step,
        'learning_rate'     : learning_rate,
        'dropout'           : dropout,
        'activation'        : activation,
        'model_name'        : model_name,
        'keras_version'     : keras_ver,
        'optimizer'         : optimizer_name,
    }


def print_kde_params(params):
    print("\n" + "="*60)
    print("  Parameters from KDE model (.h5 file)")
    print("="*60)
    print(f"  model_name        : {params['model_name']}")
    print(f"  keras_version     : {params['keras_version']}")
    print(f"  optimizer         : {params['optimizer']}  "
          f"lr={params['learning_rate']}")
    print(f"  input_window_size : {params['input_window_size']}  steps  "
          f"= {params['input_window_size']*15/60:.2f} h")
    print(f"  num_neurons       : {params['num_neurons']}  LSTM units")
    print(f"  predicted_step    : {params['predicted_step']}  steps  "
          f"= {params['predicted_step']*15/60:.0f} h ahead")
    print(f"  activation        : {params['activation']}")
    print(f"  dropout           : {params['dropout']}")
    print("="*60 + "\n")


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'lstm_vector_150days_w9000_22.h5'
    print_kde_params(read_kde_params(path))
