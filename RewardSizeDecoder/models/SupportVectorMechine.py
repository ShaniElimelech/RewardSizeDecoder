import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, brier_score_loss
import optuna
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SVM:
    def __init__(self, *,
                 c=1.0,                     # C ∈ ℝ⁺: Regularization parameter (controls trade-off between margin size and classification error)
                 kernel='rbf',              # Kernel function: {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
                 degree=3,                  # d ∈ ℕ: Degree of polynomial kernel (if kernel='poly')
                 gamma='scale',             # γ ∈ ℝ⁺ or {'scale', 'auto'}: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
                 coef0=0.0,                 # coef₀ ∈ ℝ: Independent term in 'poly' and 'sigmoid' kernels
                 probability=True,         # Boolean: Whether to enable probability estimates (sigmoid calibrated)
                 tol=0.001,                 # ε ∈ ℝ⁺: Tolerance for stopping criterion (used in KKT condition evaluation)
                 class_weight=None,         # {dict, 'balanced', None}: Weights for handling imbalanced data
                 custom_class_weights=True,    # if true tuning on class weights is done
                 max_iter=100,                  # ℓ ∈ ℕ ∪ {-1}: Max number of iterations (-1 means no limit)
                 decision_function_shape='ovr'  # {'ovr', 'ovo'}: One-vs-Rest or One-vs-One strategy for multi-class
                 ):

        self.params = {
            "C": c,
            "kernel": kernel,
            "degree": degree,
            "gamma": gamma,
            "coef0": coef0,
            "probability": probability,
            "tol": tol,
            "class_weight": class_weight,
            "max_iter": max_iter,
            "decision_function_shape": decision_function_shape
        }

        self.custom_class_weights = custom_class_weights
        # Initialize the SVC model using the params dictionary
        self.model = SVC(**self.params)
        self.y_probs = None
        self.y_pred = None

    def train(self, xtrain, ytrain):
        self.model.fit(xtrain, ytrain)

    def predict_proba(self, xtest):
        if self.params['probability']:
            probs_class1 = self.model.predict_proba(xtest)[:, 1]  # probability for class 1
            return probs_class1
        else:
            print('parameter probability should be set to True')
            return 0

    def predict(self, xtest):
        self.y_pred = self.model.predict(xtest)
        if self.params['probability']:
            self.y_probs = self.model.predict_proba(xtest)[:, 1]  # probability for class 1
        return self.y_pred

    def make_objective(self, x, y):
        def objective(trial):
            # Suggest kernel type
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
            self.params['kernel'] = kernel

            # Suggest hyperparameters based on the kernel choice
            self.params['C'] = trial.suggest_float('C', 1e-4, 1e1, log=True)
            self.params['max_iter'] = trial.suggest_int('max_iter', 100, 1000)
            self.params['tol'] = trial.suggest_float('tol', 1e-5, 1e-2, log=True)
            self.params['degree'] = trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3
            if self.custom_class_weights:
                # Here we would tune numeric class weights (like {0: 90, 1: 10})
                # Making the class weights complementary - assuming more weight for minority class
                class_0_weight = trial.suggest_int('class_0_weight', 1, 50)
                # Set class 1's weight as a fraction or complementary value of class 0 weight
                class_1_weight = 100 - class_0_weight
                self.params['class_weight'] = {0: class_0_weight, 1: class_1_weight}

            # Re-initialize model with the current hyperparameters
            self.model = SVC(**self.params)

            # Build pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', self.model)
            ])

            # Cross-validation and calculate AUC
            auc = cross_val_score(pipeline, x, y, cv=5, scoring='roc_auc').mean()
            return auc

        return objective

    def find_best_params(self, xtrain, ytrain):
        """
        Using Optuna for hyperparameter tuning.
        Hyperparameters: C, kernel, degree, gamma, coef0, class_weight
        """
        objective = self.make_objective(xtrain, ytrain)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        # Get the best parameters from the study
        self.params['C'] = study.best_params['C']
        self.params['max_iter'] = study.best_params['max_iter']
        self.params['tol'] = study.best_params['tol']
        self.params['kernel'] = study.best_params['kernel']
        if self.params['kernel'] == 'poly':
            self.params['degree'] = study.best_params['degree']
        # Update class_weight only if custom weights were tuned
        if self.custom_class_weights:
            class_0_weight = study.best_params['class_0_weight']
            class_1_weight = 100 - class_0_weight
            self.params['class_weight'] = {0: class_0_weight, 1: class_1_weight}

        # update final model with the best parameters
        self.model = SVC(**self.params)

    def get_best_params(self):
        if self.custom_class_weights:
            return {'C': self.params['C'],
                    'max_iter': self.params['max_iter'],
                    'tol': self.params['tol'],
                    'kernel': self.params['kernel'],
                    'degree': self.params['degree'],
                    'class_weight(positive)': self.params['class_weight'][1]}
        else:
            return {'C': self.params['C'],
                    'max_iter': self.params['max_iter'],
                    'tol': self.params['tol'],
                    'kernel': self.params['kernel'],
                    'degree': self.params['degree']}

    def compute_metrics(self, y_true):
        """Compute accuracy, precision, recall, F1 score, and specificity using confusion matrix."""

        tn, fp, fn, tp = confusion_matrix(y_true, self.y_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0  # Avoid division by zero
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity
        }

    def roc(self, y_true):
        """Compute ROC and AUC, if probability=True."""
        if self.y_probs is None:
            raise ValueError("Model was not initialized with probability=True, or predict() not called yet.")
        fpr, tpr, thresholds = roc_curve(y_true, self.y_probs)
        auc_score = roc_auc_score(y_true, self.y_probs)
        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": auc_score
        }

