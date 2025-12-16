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






