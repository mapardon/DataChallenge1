import statistics

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MachineLearningProcedure:
    """
        Machine Learning procedure for the "flu shot learning" data challenge
    """

    def __init__(self):
        self.flu_features = None
        self.seas_labels = None
        self.h1n1_labels = None
        self.cv_folds = 5
        self.candidates = {"h1n1": list(), "seas": list()}  # stored tuples (model, perf, pars)

    def proc(self):
        self.load_data()
        # self.exploratory_analysis()
        self.feature_engineering()
        #self.lm()
        self.tree()
        self.model_selection()

    def load_data(self):
        self.flu_features = pd.read_csv("data/training_set_features.csv")
        flu_labels = pd.read_csv("data/training_set_labels.csv")

        # shuffle dataset
        ds = self.flu_features
        ds[["h1n1_vaccine", "seasonal_vaccine"]] = flu_labels[["h1n1_vaccine", "seasonal_vaccine"]]
        ds = ds.sample(frac=1)

        self.flu_features = ds[ds.columns.to_list()[:-2]]
        self.seas_labels = ds[["respondent_id", "seasonal_vaccine"]]
        self.h1n1_labels = ds[["respondent_id", "h1n1_vaccine"]]
        # TODO train/validation sets

    def exploratory_analysis(self):
        print(" * Features dimension:\n{}".format(self.flu_features.shape))
        print("\n * Example record:\n{}".format(self.flu_features.head(1)))
        print("\n * Features summary:")
        self.flu_features.info()
        print("\n * Numeric features info:\n{}".format(self.flu_features.describe()))

        # plots (features)
        to_plot = self.flu_features.iloc[:, 1:]  # remove respondent_id
        ax_count = int()
        axs = list()

        for feature in to_plot.columns.to_list():
            if not ax_count:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                axs = [ax1, ax2, ax3, ax4]

            if len(pd.unique(to_plot[feature])) <= 10:
                # if not many different values, consider as str for better display
                elements = sorted([(str(k), v) for k, v in to_plot[feature].value_counts().to_dict().items()],
                                  key=lambda x: x[0])
            else:
                elements = sorted([(k, v) for k, v in to_plot[feature].value_counts().to_dict().items()],
                                  key=lambda x: x[0])

            axs[ax_count].bar([e[0] for e in elements], [e[1] for e in elements], width=0.3)
            axs[ax_count].set_title(feature)
            ax_count += 1

            if not ax_count % 4:
                plt.show()
                ax_count *= 0

        if ax_count % 4:
            plt.show()

        # plots (labels)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for df, feature, ax in [(self.h1n1_labels, "h1n1_vaccine", ax1), (self.seas_labels, "seasonal_vaccine", ax2)]:
            elements = sorted([(str(k), v) for k, v in df[feature].value_counts().to_dict().items()],
                              key=lambda x: x[0])
            ax.bar([e[0] for e in elements], [e[1] for e in elements], width=0.3, color=["tab:blue", "orange"])
            ax.set_title(feature.capitalize())
        plt.show()

    def feature_engineering(self):
        ds = self.flu_features
        ds["h1n1_vaccine"] = self.h1n1_labels["h1n1_vaccine"]
        ds["seasonal_vaccine"] = self.seas_labels["seasonal_vaccine"]

        # for now, we just remove missing data and non-numeric columns
        ds = ds[~ds.isnull().any(axis=1)]
        ds = ds.select_dtypes([np.number])
        # self.flu_features = ds[["respondent_id", "doctor_recc_h1n1", "opinion_h1n1_risk", "opinion_h1n1_vacc_effective", "opinion_seas_risk"]]
        self.flu_features = ds.iloc[:, 1:-3]
        self.seas_labels = ds["seasonal_vaccine"]
        self.h1n1_labels = ds["h1n1_vaccine"]

    def model_selection(self):
        for k in ["h1n1", "seas"]:
            print("{} performance:".format(k))
            for m, auc, pars in sorted(self.candidates[k], reverse=True, key=lambda x: x[1]):
                print("{} ({}): {}".format(str(type(m)), str(pars), auc))

    def parametric_identification(self, model_h1n1, model_seas, is_reg_model=False):
        """
        Generic loop training the provided model on the dataset and assessing performance.
        """

        n_rows_fold = len(self.flu_features) // self.cv_folds
        h1n1_auc, seas_auc = list(), list()
        thr_h1n1, fpr_h1n1, tpr_h1n1, thr_seas, fpr_seas, tpr_seas = [None] * 6

        for i in range(self.cv_folds):
            X_i = self.flu_features[self.flu_features.columns.to_list()[:]]
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
            y_i_h1n1_pred_prob = sigmoid(model_h1n1.predict(X_i_ts)) if is_reg_model else model_h1n1.predict_proba(X_i_ts)[:, 1]
            model_seas.fit(X_i_tr, y_i_seas_tr)
            y_i_seas_pred_prob = sigmoid(model_seas.predict(X_i_ts)) if is_reg_model else model_seas.predict_proba(X_i_ts)[:, 1]

            # compute ROC and AUC
            fpr_h1n1, tpr_h1n1, thr_h1n1 = roc_curve(y_i_h1n1_ts, y_i_h1n1_pred_prob)
            fpr_seas, tpr_seas, thr_seas = roc_curve(y_i_seas_ts, y_i_seas_pred_prob)
            h1n1_auc.append(roc_auc_score(y_i_h1n1_ts, y_i_h1n1_pred_prob))
            seas_auc.append(roc_auc_score(y_i_seas_ts, y_i_seas_pred_prob))

        return h1n1_auc, thr_h1n1, fpr_h1n1, tpr_h1n1, seas_auc, thr_seas, fpr_seas, tpr_seas

    @staticmethod
    def display_training_result(header, h1n1_auc, thr_h1n1, fpr_h1n1, tpr_h1n1, seas_auc, thr_seas, fpr_seas, tpr_seas, plot=False):
        # Plot ROC (values from last round)
        if plot:
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

        print(header)
        print("h1n1 max auc: {}, min auc: {}, avg auc {}, stddev auc: {}".format(max(h1n1_auc),
                                                                                 min(h1n1_auc),
                                                                                 statistics.mean(h1n1_auc),
                                                                                 statistics.stdev(h1n1_auc)))
        print("seas max auc: {}, min auc: {}, avg auc {}, stddev auc: {}".format(max(seas_auc),
                                                                                 min(seas_auc),
                                                                                 statistics.mean(seas_auc),
                                                                                 statistics.stdev(seas_auc)), end='\n\n')

    def lm(self):
        # Structural and parametric identification
        lm_h1n1 = LinearRegression()
        lm_seas = LinearRegression()
        ret = self.parametric_identification(lm_h1n1, lm_seas, True)
        MachineLearningProcedure.display_training_result(str(type(lm_h1n1)), *ret)
        print(lm_h1n1.coef_)
        print(lm_h1n1.intercept_)

        # Ridge is a basis of linear model we put it here
        for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]:
            ridge_h1n1 = RidgeClassifier(alpha=alpha)
            ridge_seas = RidgeClassifier(alpha=alpha)
            ret = self.parametric_identification(ridge_h1n1, ridge_seas, True)
            MachineLearningProcedure.display_training_result(str(type(ridge_h1n1)) + " " + "alpha={}".format(alpha), *ret)

        # Ensemble model
        n = 10
        lm_ens_h1n1 = VotingClassifier([("h1n1_" + str(i), LinearRegression()) for i in range(n)])
        lm_ens_seas = VotingClassifier([("seas_" + str(i), LinearRegression()) for i in range(n)])
        ret = self.parametric_identification(lm_ens_h1n1, lm_ens_seas, False)
        print(lm_ens_h1n1.estimators_[0].coef_)
        print(lm_ens_h1n1.estimators_[0].intercept_)
        MachineLearningProcedure.display_training_result(str(type(lm_ens_h1n1)) + " " + str(n), *ret)

    def tree(self):
        for c in ["entropy", "gini", "log_loss"]:
            dtree_h1n1 = DecisionTreeClassifier(criterion=c)
            dtree_seas = DecisionTreeClassifier(criterion=c)
            ret = self.parametric_identification(dtree_h1n1, dtree_seas, False)
            #MachineLearningProcedure.display_training_result(str(type(dtree_h1n1)) + " @ " + c, *ret)
            self.candidates["h1n1"].append((dtree_h1n1, ret[0], c))
            self.candidates["seas"].append((dtree_seas, ret[4], c))

        for c in ["entropy", "gini", "log_loss"]:
            for n in [10, 20, 50, 100]:
                rf_h1n1 = RandomForestClassifier(criterion=c, n_estimators=n)
                rf_seas = RandomForestClassifier(criterion=c, n_estimators=n)
                ret = self.parametric_identification(rf_h1n1, rf_seas, False)
                #MachineLearningProcedure.display_training_result(str(type(rf_h1n1)) + " ({}, {})".format(c, n), *ret)
                self.candidates["h1n1"].append((rf_h1n1, ret[0], (c, n)))
                self.candidates["seas"].append((rf_seas, ret[4], (c, n)))


def dummy_df():
    """ Debug and experiment purpose only """
    df = pd.DataFrame({
        "a": [0.5, 0.3, 0.98],
        "b": [1, 2, 3]
    })
    return df


def dummy_array():
    a = np.random.rand(4, 4)
    return a


if __name__ == '__main__':
    MachineLearningProcedure().proc()
