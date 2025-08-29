import numpy as np
from numpy.linalg import solve
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


class LDA:
    def __init__(self, shrunk=True):
        """
        Initialize the LDA model. By default, use shrinkage (Ledoit estimator for covariance).
        """
        self.shrunk = shrunk
        self.w_neg = None
        self.w_pos = None
        self.b_neg = None
        self.b_pos = None
        self.y_pred = None
        self.y_probs = None

    def train(self, Xtrain, Ytrain):
        """
        Train the LDA model using training data (Xtrain, Ytrain).
        This will compute class means, covariance matrices, and the parameters of the model.
        """
        positive_idx = np.where(Ytrain == 1)[0]
        negative_idx = np.where(Ytrain == 0)[0]
        Xtrain_pos = Xtrain[positive_idx]
        Xtrain_neg = Xtrain[negative_idx]
        n_pos = Xtrain_pos.shape[0]
        n_neg = Xtrain_neg.shape[0]
        n_total = n_pos + n_neg

        # Compute class mean
        m_pos = np.mean(Xtrain_pos, axis=0)
        m_neg = np.mean(Xtrain_neg, axis=0)

        # Compute class covariance matrices
        if self.shrunk:
            s_pos = LedoitWolf().fit(Xtrain_pos).covariance_
            s_neg = LedoitWolf().fit(Xtrain_neg).covariance_
        else:
            s_pos = np.corrcoef(Xtrain_pos - m_pos, Xtrain_pos - m_pos, rowvar=False)
            s_neg = np.corrcoef(Xtrain_neg - m_neg, Xtrain_neg - m_neg, rowvar=False)

        # Pooled covariance
        Sw = ((n_pos - 1) * s_pos + (n_neg - 1) * s_neg) / (n_total - 2)

        # Priors probabilities
        pi_pos = n_pos / n_total
        pi_neg = n_neg / n_total

        # Solve Sw w_k = m_k for each class
        self.w_pos = solve(Sw, m_pos)
        self.w_neg = solve(Sw, m_neg)

        # Bias terms (intercepts)
        self.b_neg = -0.5 * m_neg.T @ self.w_neg + np.log(pi_neg)
        self.b_pos = -0.5 * m_pos.T @ self.w_pos + np.log(pi_pos)

    def predict(self, Xtest):
        """
        Predict labels and probabilities for the test data (Xtest).
        """
        # Compute discriminant scores for each class
        scores0 = Xtest @ self.w_neg + self.b_neg
        scores1 = Xtest @ self.w_pos + self.b_pos
        # Prediction
        self.y_pred = np.array((scores1 > scores0)).astype(int)
        self.y_probs = self.predict_proba(Xtest)
        return self.y_pred

    def predict_proba(self,  Xtest):
        # Compute discriminant scores for each class
        scores0 = Xtest @ self.w_neg + self.b_neg
        scores1 = Xtest @ self.w_pos + self.b_pos
        # Compute probabilities for binary case - for non-binary use softmax
        logit = scores1 - scores0
        probs_class1 = 1 / (1 + np.exp(-logit))
        return probs_class1

    def compute_metrics(self, y_true, y_pred):
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
            raise ValueError("LDA.predict_proba() not called yet.")
        fpr, tpr, thresholds = roc_curve(y_true, self.y_probs)
        auc_score = roc_auc_score(y_true, self.y_probs)
        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": auc_score}


