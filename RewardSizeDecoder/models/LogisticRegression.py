import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, brier_score_loss
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve, CalibrationDisplay
#import ml_insights as mli


class LogisticRegressionModel:
    def __init__(self, *,
                 penalty='l2',
                 solver='lbfgs',
                 C=1.0,
                 tol=1e-4,
                 max_iter=500,
                 class_weight=None,
                 custom_class_weights=True,
                 thresh=0.5
                 ):

        self.params = {
            "C": C,
            "tol": tol,
            "max_iter": max_iter,
            "class_weight": class_weight,
            "solver": solver,  # 'lbfgs' default solver (quasi newton, supports l2 regularization and multiclass)
            "penalty": penalty,  # Regularization penalty
            "random_state": None
        }

        self.custom_class_weights = custom_class_weights

        # Initialize the LogisticRegression model using the params dictionary
        self.model = LogisticRegression(**self.params)
        self.thresh = thresh  # Default value of threshold is 0.5
        self.y_pred = None
        self.y_probs = None

    def train(self, xtrain, ytrain):
        self.model.fit(xtrain, ytrain)

    def predict_proba(self, xtest):
        probs = self.model.predict_proba(xtest)[:, 1]
        return probs

    def predict(self, xtest):
        # Predict classes based on the threshold
        self.y_probs = self.predict_proba(xtest)
        self.y_pred = (self.y_probs >= self.thresh).astype(int)
        return self.y_pred

    def make_objective(self, x, y):
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        def objective(trial):
            # Sample hyperparameters
            self.params['C'] = trial.suggest_float('C', 1e-4, 1e1, log=True)
            self.params['max_iter'] = trial.suggest_int('max_iter', 50, 500)
            self.params['tol'] = trial.suggest_float('tol', 1e-5, 1e-1, log=True)
            if self.custom_class_weights:
                # Here we would tune numeric class weights (like {0: 90, 1: 10})
                # Making the class weights complementary - assuming more weight for minority class
                class_0_weight = trial.suggest_int('class_0_weight', 1, 50)
                # Set class 1's weight as a fraction or complementary value of class 0 weight
                class_1_weight = 100 - class_0_weight
                self.params['class_weight'] = {0: class_0_weight, 1: class_1_weight}

            aucs = []
            thresholds = []

            for train_idx, val_idx in cv.split(x):
                x_train, x_val = x[train_idx], x[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Re-initialize model with the current hyperparameters
                self.model = LogisticRegression(**self.params)
                self.train(x_train, y_train)

                # Predict probabilities
                probs = self.predict_proba(x_val)

                # ROC-based threshold optimization
                fpr, tpr, threshold_vals = roc_curve(y_val, probs)
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                best_threshold = threshold_vals[best_idx]

                auc = roc_auc_score(y_val, probs)
                aucs.append(auc)
                thresholds.append(best_threshold)

            avg_thresh = float(np.mean(thresholds))
            trial.set_user_attr("best_threshold", avg_thresh)

            # Return average AUC over folds
            return float(np.mean(aucs))

        return objective

    def find_best_params(self, xtrain, ytrain):
        """
        Using Optuna for hyperparameters search.
        Learning parameters: w, b
        Hyperparameters: C (regularization strength), tol, max_iter, class_weight
        """
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        objective = self.make_objective(xtrain, ytrain)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        self.params['C'] = study.best_params['C']
        self.params['max_iter'] = study.best_params['max_iter']
        self.params['tol'] = study.best_params['tol']
        # Update class_weight only if custom weights were tuned
        if self.custom_class_weights:
            class_0_weight = study.best_params['class_0_weight']
            class_1_weight = 100 - class_0_weight
            self.params['class_weight'] = {0: class_0_weight, 1: class_1_weight}

        self.thresh = study.best_trial.user_attrs["best_threshold"]
        # update model with the best parameters
        self.model = LogisticRegression(**self.params)

    def get_best_params(self):
        if self.custom_class_weights:
            return {'C': self.params['C'],
                    'max_iter': self.params['max_iter'],
                    'tol': self.params['tol'],
                    'threshold': self.thresh,
                    'class_weight(positive)': self.params['class_weight'][1]}
        else:
            return {'C': self.params['C'],
                    'max_iter': self.params['max_iter'],
                    'tol': self.params['tol'],
                    'threshold': self.thresh}

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
        """Compute ROC and AUC."""

        if self.y_probs is None:
            raise ValueError("LogisticRegressionModel.predict() not called yet.")
        fpr, tpr, thresholds = roc_curve(y_true, self.y_probs)
        auc_score = roc_auc_score(y_true, self.y_probs)
        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": auc_score
        }



















