#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ------------------------------------------------
#              Model_Training
# -------------------------------------------------

""" 
Final production-ready script for next-day Open/Close forecasting.

    Features:
     - Loads engineered features CSV (expects 'Open' and 'Close' columns)
     - Creates next-day targets: next_open, next_close
     - Adds configurable lag features
     - Standard scaling
     - Walk-forward (expanding-window) cross-validation
     - XGBoost training (recommended) with a robust fit call (works across xgboost versions)
     - Optional Optuna hyperparameter tuning for XGBoost
     - RandomForest baseline option
     - Saves model artifacts and training summary
     - Works inside Jupyter notebooks via parse_known_args()
"""


# In[10]:


import argparse
import json
import os
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Optional libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

SEED = 42


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    return df


def create_next_day_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Open' not in df.columns or 'Close' not in df.columns:
        raise ValueError("Input features must contain 'Open' and 'Close' columns.")
    df['next_open'] = df['Open'].shift(-1)
    df['next_close'] = df['Close'].shift(-1)
    return df


def add_lag_features(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        for l in lags:
            df[f"{c}_lag_{l}"] = df[c].shift(l)
    return df


def get_predictor_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    exclude_set = set(exclude)
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_set]


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    return {'mae': float(mae), 'rmse': rmse, 'r2': float(r2), 'mape': float(mape)}


def train_xgb(X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              params: Optional[Dict[str, Any]] = None) -> Any:
    if not XGB_AVAILABLE:
        raise RuntimeError('xgboost is not available. Install xgboost to use this trainer.')

    params = params or {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': SEED
    }
    model = xgb.XGBRegressor(**params)

    if X_val is not None and y_val is not None:
        try:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
        except TypeError:
            try:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            except TypeError:
                model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    return model


def walk_forward_split(n_samples: int, n_splits: int = 5, initial_train_size: Optional[int] = None) -> List[Tuple[slice, slice]]:
    if initial_train_size is None:
        initial_train_size = int(n_samples * 0.5)
    test_size = int((n_samples - initial_train_size) / n_splits)
    if test_size < 1:
        test_size = 1
    splits = []
    train_end = initial_train_size
    for i in range(n_splits):
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)
        if test_start >= n_samples:
            break
        splits.append((slice(0, train_end), slice(test_start, test_end)))
        train_end = test_end
        if test_end == n_samples:
            break
    return splits


def run_walk_forward_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, initial_train_size: Optional[int] = None,
                        params: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    splits = walk_forward_split(len(X), n_splits=n_splits, initial_train_size=initial_train_size)
    fold_results = []

    for i, (train_slice, test_slice) in enumerate(splits):
        X_tr, y_tr = X[train_slice], y[train_slice]
        X_te, y_te = X[test_slice], y[test_slice]
        model = train_xgb(X_tr, y_tr, X_val=X_te, y_val=y_te, params=params)
        preds = model.predict(X_te)
        metrics = eval_metrics(y_te, preds)
        fold_results.append({'fold': i, 'metrics': metrics})
    return fold_results, (params or {})


def optuna_objective(trial, X: np.ndarray, y: np.ndarray, n_splits: int = 3, initial_train_size: Optional[int] = None) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': SEED,
    }

    splits = walk_forward_split(len(X), n_splits=n_splits, initial_train_size=initial_train_size)
    val_scores = []
    for train_slice, test_slice in splits:
        X_tr, y_tr = X[train_slice], y[train_slice]
        X_te, y_te = X[test_slice], y[test_slice]
        try:
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], early_stopping_rounds=20, verbose=False)
        except Exception:
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        metrics = eval_metrics(y_te, preds)
        val_scores.append(metrics['rmse'])
    return float(np.mean(val_scores))


def save_artifacts(model: Any, scaler: Optional[StandardScaler], out_dir: str, name: str, predictors: Optional[List[str]] = None) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    meta: Dict[str, Any] = {}
    try:
        if hasattr(model, 'get_booster'):
            model.get_booster().save_model(os.path.join(out_dir, f"{name}_xgb.json"))
            meta['model_path'] = os.path.join(out_dir, f"{name}_xgb.json")
            meta['framework'] = 'xgboost'
        else:
            joblib.dump(model, os.path.join(out_dir, f"{name}_model.joblib"))
            meta['model_path'] = os.path.join(out_dir, f"{name}_model.joblib")
            meta['framework'] = 'joblib'
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {e}")

    if scaler is not None:
        scaler_path = os.path.join(out_dir, f"{name}_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        meta['scaler_path'] = scaler_path

    # save predictors list (ordered)
    if predictors is not None:
        try:
            preds_path = os.path.join(out_dir, f"{name}_predictors.json")
            with open(preds_path, 'w') as f:
                json.dump(predictors, f)
            meta['predictors_path'] = preds_path
        except Exception as e:
            print(f"Warning: failed to save predictors list: {e}")

    return meta


def run_pipeline(
    data_path: str,
    lags: List[int],
    model_type: str,
    target: str,
    test_size: float,
    preserve_time: bool,
    out_dir: str,
    use_optuna: bool = False,
    n_trials: int = 20,
    n_splits: int = 5,
    initial_train_size: Optional[int] = None
) -> Dict[str, Any]:
    df = load_features(data_path)
    df = create_next_day_targets(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['next_open', 'next_close']
    base_cols = [c for c in ['Open', 'Close'] if c in numeric_cols]
    other_cols = [c for c in numeric_cols if c not in base_cols + exclude]
    lag_cols = base_cols + other_cols

    df = add_lag_features(df, lag_cols, lags)
    df = df.dropna().reset_index(drop=True)

    targets: List[str] = []
    if target == 'next_open':
        targets = ['next_open']
    elif target == 'next_close':
        targets = ['next_close']
    elif target == 'both':
        targets = ['next_open', 'next_close']

    summary: Dict[str, Any] = {}

    for t in targets:
        X_cols = get_predictor_columns(df, exclude=[t])
        X = df[X_cols].values
        y = df[t].values

        split_idx = int(len(X) * (1 - test_size))
        X_train_all, X_test = X[:split_idx], X[split_idx:]
        y_train_all, y_test = y[:split_idx], y[split_idx:]

        scaler = StandardScaler()
        X_train_all_s = scaler.fit_transform(X_train_all)
        X_test_s = scaler.transform(X_test)

        model_results: Dict[str, Any] = {}

        if model_type in ('xgb', 'all'):
            if not XGB_AVAILABLE:
                print('xgboost not installed — skipping xgb training')
            else:
                best_params: Optional[Dict[str, Any]] = None
                if use_optuna and OPTUNA_AVAILABLE:
                    print('Running Optuna tuning (this may take a while)...')
                    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
                    study.optimize(lambda trial: optuna_objective(trial, X_train_all_s, y_train_all,
                                                                  n_splits=max(2, n_splits // 2),
                                                                  initial_train_size=initial_train_size),
                                   n_trials=n_trials)
                    best_params = study.best_params
                    print('Optuna best params:', best_params)
                elif use_optuna and not OPTUNA_AVAILABLE:
                    print('Optuna not installed. Running with default xgboost params.')

                xgb_params = best_params or {
                    'n_estimators': 500,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': SEED
                }

                folds, _ = run_walk_forward_cv(X_train_all_s, y_train_all, n_splits=n_splits,
                                               initial_train_size=initial_train_size, params=xgb_params)

                final_model = train_xgb(X_train_all_s, y_train_all, params=xgb_params)
                preds_test = final_model.predict(X_test_s)
                test_metrics = eval_metrics(y_test, preds_test)

                model_results['xgb'] = {'cv_folds': folds, 'test_metrics': test_metrics, 'best_params': xgb_params}
                # save artifacts and predictor list
                save_dir = os.path.join(out_dir, t)
                meta = save_artifacts(final_model, scaler, out_dir=save_dir, name=f'{t}_xgb', predictors=X_cols)
                model_results['xgb']['saved'] = meta

        if model_type in ('rf', 'all'):
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
            rf.fit(X_train_all_s, y_train_all)
            preds = rf.predict(X_test_s)
            model_results['rf'] = {'test_metrics': eval_metrics(y_test, preds)}
            meta = save_artifacts(rf, scaler, out_dir=os.path.join(out_dir, t), name=f'{t}_rf', predictors=X_cols)
            model_results['rf']['saved'] = meta

        summary[t] = model_results
        print(f"Finished target={t}. Summary:")
        print(json.dumps(model_results.get('xgb', model_results), indent=2, default=str))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='reliance.csv',
                        help='Path to engineered features CSV (must contain Open and Close)')
    parser.add_argument('--lags', type=int, nargs='+', default=[1, 2, 3, 5],
                        help='Lag periods to generate (e.g. --lags 1 2 3 5)')
    parser.add_argument('--model', type=str, default='xgb', choices=['xgb', 'rf', 'all'],
                        help='Model to train (xgb recommended)')
    parser.add_argument('--target', type=str, default='next_close', choices=['next_open', 'next_close', 'both'],
                        help='Which target to train')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion (chronological)')
    parser.add_argument('--preserve_time', action='store_true', help='Use chronological split (recommended)')
    parser.add_argument('--out_dir', type=str, default='models', help='Output folder to save models and scalers')
    parser.add_argument('--use_optuna', action='store_true', help='Enable Optuna tuning for XGBoost')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for walk-forward CV')
    parser.add_argument('--initial_train_size', type=int, default=None,
                        help='Initial train size (rows) for walk-forward CV (optional)')

    args, unknown = parser.parse_known_args()

    run_pipeline(
        data_path=args.data,
        lags=args.lags,
        model_type=args.model,
        target=args.target,
        test_size=args.test_size,
        preserve_time=args.preserve_time,
        out_dir=args.out_dir,
        use_optuna=args.use_optuna,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        initial_train_size=args.initial_train_size
    )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




