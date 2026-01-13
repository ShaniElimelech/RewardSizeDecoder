from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy as np


def no_resample(X, y):
    return X, y


def random_undersample_and_smote_oversample(X_train, y_train):
    # Define resampling steps
    over = SMOTE(sampling_strategy=0.4, random_state=42)  # minority/majority = ~40% by increasing minority
    under = RandomUnderSampler(sampling_strategy=0.6, random_state=42)  # minority/majority = ~60% by decreasing majority

    # Combine in a pipeline
    pipeline = Pipeline(steps=[('o', over), ('u', under)])
    X_res, y_res = pipeline.fit_resample(X_train, y_train)
    return X_res, y_res


def random_undersample(X_train, y_train):
    # Define resampling steps
    under = RandomUnderSampler(sampling_strategy=1.0, random_state=42)  # Reduce majority class to ~50%
    X_res, y_res = under.fit_resample(X_train, y_train)
    return X_res, y_res


def predict_ensemble(models, X_test):
    # Predict by majority vote of all trained classifiers
    preds = np.array([clf.predict(X_test) for clf in models])
    final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
    return final_preds


def predict_ensemble_proba(models, X_test):
    # collect probability predictions from all models
    probas = [clf.predict_proba(X_test) for clf in models]
    # average across models
    return np.mean(probas, axis=0)


def custom_resample(X_train, y_train, regular_keep_trials_idx):
    all_idx = np.arange(len(X_train))
    regular_idx = np.where(y_train == 0)[0]
    n_large = len(np.where(y_train == 1)[0])
    n_regular = len(regular_idx)
    regular_idx = regular_idx[~np.isin(regular_idx, regular_keep_trials_idx)]
    n_remove = min(n_regular - n_large, len(regular_idx))
    remove_idx = np.random.choice(regular_idx, size=n_remove, replace=False)
    res_idx = np.delete(all_idx, remove_idx)
    X_res = X_train[res_idx]
    y_res = y_train[res_idx]

    return X_res, y_res



def custom_resample2(X_train, y_train):
    all_idx = np.arange(len(X_train))
    regular_idx = np.where(y_train == 0)[0]
    n_large = len(np.where(y_train == 1)[0])
    n_regular = len(regular_idx)
    n_remove = n_regular - n_large
    remove_idx = np.linspace(0, n_regular - 1, n_remove, dtype=int)
    res_idx = np.delete(all_idx, remove_idx)
    X_res = X_train[res_idx]
    y_res = y_train[res_idx]

    return X_res, y_res






















