import statistics

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from utils import sigmoid


class ModelIdentification:
    def __init__(self, features, h1n1_labels, seas_labels, cv_folds):
        self.features = features
        self.h1n1_labels = h1n1_labels
        self.seas_labels = seas_labels
        self.cv_folds = cv_folds
        self.candidates = {"h1n1": list(), "seas": list()}  # stored tuples (model, perf, pars)

    def model_selection(self):
        self.lm()
        self.tree()

        for k in ["h1n1", "seas"]:
            print("{} performance:".format(k))
            for m, auc, pars in sorted(self.candidates[k], reverse=True, key=lambda x: statistics.mean(x[1])):
                ModelIdentification.display_training_result(m, auc, pars)

    def parametric_identification(self, model_h1n1, model_seas, is_reg_model=False):
        """
        Generic loop training the provided model on the dataset and assessing performance.
        """

        n_rows_fold = len(self.features) // self.cv_folds
        h1n1_auc, seas_auc = list(), list()
        thr_h1n1, fpr_h1n1, tpr_h1n1, thr_seas, fpr_seas, tpr_seas = [None] * 6

        for i in range(self.cv_folds):
            X_i = self.features[self.features.columns.to_list()[:]]
            X_i_tr = pd.concat([X_i.iloc[: n_rows_fold * i], X_i.iloc[n_rows_fold * (i + 1):]], axis=0,
                               ignore_index=True)
            X_i_ts = X_i.iloc[n_rows_fold * i: n_rows_fold * (i + 1)]

            y_i_h1n1 = self.h1n1_labels
            y_i_h1n1_tr = pd.concat([y_i_h1n1.iloc[: n_rows_fold * i], y_i_h1n1.iloc[n_rows_fold * (i + 1):]], axis=0,
                                    ignore_index=True)
            y_i_h1n1_ts = y_i_h1n1.iloc[n_rows_fold * i: n_rows_fold * (i + 1)].astype(float)
            y_i_seas = self.seas_labels
            y_i_seas_tr = pd.concat([y_i_seas.iloc[: n_rows_fold * i], y_i_seas.iloc[n_rows_fold * (i + 1):]], axis=0,
                                    ignore_index=True)
            y_i_seas_ts = y_i_seas.iloc[n_rows_fold * i: n_rows_fold * (i + 1)].astype(float)

            # train + predict probabilities
            model_h1n1.fit(X_i_tr, y_i_h1n1_tr)
            y_i_h1n1_pred_prob = sigmoid(model_h1n1.predict(X_i_ts)) if is_reg_model else model_h1n1.predict_proba(
                X_i_ts)[:, 1]
            model_seas.fit(X_i_tr, y_i_seas_tr)
            y_i_seas_pred_prob = sigmoid(model_seas.predict(X_i_ts)) if is_reg_model else model_seas.predict_proba(
                X_i_ts)[:, 1]

            # compute ROC and AUC
            fpr_h1n1, tpr_h1n1, thr_h1n1 = roc_curve(y_i_h1n1_ts, y_i_h1n1_pred_prob)
            fpr_seas, tpr_seas, thr_seas = roc_curve(y_i_seas_ts, y_i_seas_pred_prob)
            h1n1_auc.append(roc_auc_score(y_i_h1n1_ts, y_i_h1n1_pred_prob))
            seas_auc.append(roc_auc_score(y_i_seas_ts, y_i_seas_pred_prob))

        return h1n1_auc, thr_h1n1, fpr_h1n1, tpr_h1n1, seas_auc, thr_seas, fpr_seas, tpr_seas

    @staticmethod
    def display_training_result(model, auc, pars):

        print("{} ({})".format(str(type(model)).split('.')[-1], pars))
        print("avg auc: {}, stddev auc: {}, max auc {}, min auc: {}".format(statistics.mean(auc),
                                                                            statistics.stdev(auc),
                                                                            max(auc),
                                                                            min(auc)))

    @staticmethod
    def plot_tpr_fpr(header, thr_h1n1, fpr_h1n1, tpr_h1n1, thr_seas, fpr_seas, tpr_seas):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(thr_h1n1, fpr_h1n1, color='blue', linewidth=2, linestyle='--', label="FPR")
        ax1.plot(thr_h1n1, tpr_h1n1, color='orange', linewidth=2, linestyle=':', label="TPR")
        ax1.set_title("FPR and TPR for h1n1 predictions ({})".format(header))
        ax1.legend()
        ax2.plot(thr_seas, fpr_seas, color='blue', linewidth=2, linestyle='--', label="FPR")
        ax2.plot(thr_seas, tpr_seas, color='orange', linewidth=2, linestyle=':', label="TPR")
        ax2.set_title("FPR and TPR for seasonal flu predictions ({})".format(header))
        ax2.legend()
        plt.show()

    # Methods corresponding to the "structural identification" step for the different models

    def lm(self):
        # Structural and parametric identification
        lm_h1n1 = LinearRegression()
        lm_seas = LinearRegression()
        ret = self.parametric_identification(lm_h1n1, lm_seas, True)
        self.candidates["h1n1"].append((lm_h1n1, ret[0], str()))
        self.candidates["seas"].append((lm_seas, ret[4], str()))

        # Ridge is a basis of linear model we put it here
        for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]:
            ridge_h1n1 = RidgeClassifier(alpha=alpha)
            ridge_seas = RidgeClassifier(alpha=alpha)
            ret = self.parametric_identification(ridge_h1n1, ridge_seas, True)
            self.candidates["h1n1"].append((ridge_h1n1, ret[0], alpha))
            self.candidates["seas"].append((ridge_seas, ret[4], alpha))

        """
        # Ensemble model
        n = 10
        lm_ens_h1n1 = VotingClassifier([("h1n1_" + str(i), LinearRegression()) for i in range(n)])
        lm_ens_seas = VotingClassifier([("seas_" + str(i), LinearRegression()) for i in range(n)])
        ret = self.parametric_identification(lm_ens_h1n1, lm_ens_seas, False)
        print(lm_ens_h1n1.estimators_[0].coef_)
        print(lm_ens_h1n1.estimators_[0].intercept_)
        ModelIdentification.display_training_result(str(type(lm_ens_h1n1)) + " " + str(n), *ret)
        """

    def tree(self):
        for c in ["entropy", "gini", "log_loss"]:
            dtree_h1n1 = DecisionTreeClassifier(criterion=c)
            dtree_seas = DecisionTreeClassifier(criterion=c)
            ret = self.parametric_identification(dtree_h1n1, dtree_seas, False)
            self.candidates["h1n1"].append((dtree_h1n1, ret[0], c))
            self.candidates["seas"].append((dtree_seas, ret[4], c))

        for c in ["entropy", "gini", "log_loss"]:
            for n in [10, 20, 50, 100]:
                rf_h1n1 = RandomForestClassifier(criterion=c, n_estimators=n)
                rf_seas = RandomForestClassifier(criterion=c, n_estimators=n)
                ret = self.parametric_identification(rf_h1n1, rf_seas, False)
                self.candidates["h1n1"].append((rf_h1n1, ret[0], (c, n)))
                self.candidates["seas"].append((rf_seas, ret[4], (c, n)))
