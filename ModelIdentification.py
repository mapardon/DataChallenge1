import statistics

import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve


class ModelIdentification:
    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, test_features: pd.DataFrame,
                 test_labels: pd.DataFrame, cv_folds: int, verbose=False):
        self.train_features = train_features
        self.train_labels = train_labels

        self.test_features = test_features
        self.test_labels = test_labels

        self.cv_folds = cv_folds
        self.candidates = list()  # stored tuples: (model, [perf], [pars], is_reg_model)
        self.verbose = verbose

    @staticmethod
    def model_exploitation(model, test_features: pd.DataFrame):
        """
            Use final model to predict challenge data
        """

        is_reg_model = type(model) is LinearRegression
        return expit(model.predict(test_features)) if is_reg_model else model.predict_proba(test_features)[:, 1]

    def model_testing(self):
        """
            Train selected candidates on complete training set and assess performance on unused test set
        """

        for i in range(len(self.candidates)):
            m = self.candidates[i][0]
            m.fit(self.train_features, self.train_labels)
            y_i_ts_pred_prob = expit(m.predict(self.test_features)) if self.candidates[i][3] else m.predict_proba(self.test_features)[:, 1]
            auc = roc_auc_score(self.test_labels, y_i_ts_pred_prob)
            self.candidates[i] = (m, auc)
        self.candidates.sort(reverse=True, key=lambda x: x[1])

        # print results of testing of most promising models
        if self.verbose:
            print("\n * MODEL TESTING *")
            for k in ["h1n1", "seas"]:
                print("\n -> {} performance:".format(k))
                for m, auc, pars, _ in self.candidates:
                    ModelIdentification.display_training_result(m, auc, pars)

        return self.candidates

    def model_selection(self, n=10):
        """
            Select most promising models based on performance on validation sets.
            For now, we simply take the 10 best performing algorithms, but we could consider creation of heterogeneous
            ensemble model
        """
        # Keep max 10 best models
        self.candidates = sorted(self.candidates, reverse=True, key=lambda x: statistics.mean(x[1]))[:min(n, len(self.candidates))]

    def parametric_identification_cv(self, model, is_reg_model=False):
        """
            Generic loop training the provided model on the training set (split in training/validation
            cross-validation folds) and assessing performance.
        """

        n_rows_fold = len(self.train_features) // self.cv_folds
        auc = list()
        thr, fpr, tpr = [None] * 3

        for i in range(self.cv_folds):
            X_i = self.train_features[self.train_features.columns.to_list()[:]]
            X_i_tr = pd.concat([X_i.iloc[: n_rows_fold * i], X_i.iloc[n_rows_fold * (i + 1):]], axis=0,
                               ignore_index=True)
            X_i_vs = X_i.iloc[n_rows_fold * i: n_rows_fold * (i + 1)]

            y_i = self.train_labels
            y_i_tr = pd.concat([y_i.iloc[: n_rows_fold * i], y_i.iloc[n_rows_fold * (i + 1):]], axis=0,
                               ignore_index=True)
            y_i_vs = y_i.iloc[n_rows_fold * i: n_rows_fold * (i + 1)].astype(float)

            # train + predict probabilities
            model.fit(X_i_tr, y_i_tr)
            y_i_pred_prob = expit(model.predict(X_i_vs)) if is_reg_model else model.predict_proba(X_i_vs)[:, 1]

            # compute ROC and AUC
            fpr, tpr, thr = roc_curve(y_i_vs, y_i_pred_prob)
            auc.append(roc_auc_score(y_i_vs, y_i_pred_prob))

        # print results of model identification operations
        if self.verbose:
            print(" * MODEL IDENTIFICATION *")
            print("\n -> CV performance:")
            for m, auc, pars, _ in sorted(self.candidates, reverse=True, key=lambda x: statistics.mean(x[1])):
                ModelIdentification.display_training_result(m, auc, pars)

        return auc, thr, fpr, tpr

    # Methods corresponding to the "structural identification" step for the different model types
    
    def model_identification(self, models):
        for m in models:
            {"lm": self.lm, "ridge": self.ridge, "svm": self.svm, "tree": self.tree, "nn": self.nn}[m]()

    def lm(self):
        # Structural and parametric identification
        lm = LinearRegression()
        ret = self.parametric_identification_cv(lm, True)
        self.candidates.append((lm, ret[0], str(), True))

    def ridge(self):
        for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]:
            ridge = RidgeClassifier(alpha=alpha)
            ret = self.parametric_identification_cv(ridge, True)
            self.candidates.append((ridge, ret[0], "alpha={}".format(alpha), True))

    def tree(self):
        for c in ["entropy", "gini", "log_loss"]:
            dtree = DecisionTreeClassifier(criterion=c)
            ret = self.parametric_identification_cv(dtree, False)
            self.candidates.append((dtree, ret[0], "c={}".format(c), False))

        for c in ["entropy", "gini", "log_loss"]:
            for n in [10, 20, 50, 100]:
                rf = RandomForestClassifier(criterion=c, n_estimators=n)
                ret = self.parametric_identification_cv(rf, False)
                self.candidates.append((rf, ret[0], "(c={}, n={})".format(c, n), False))

    def svm(self):
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
            svc = SVC(kernel=kernel, probability=True)
            ret = self.parametric_identification_cv(svc, False)
            self.candidates.append((svc, ret[0], "kernel={}".format(kernel), False))

    def nn(self):
        best_perf = -1.0
        best_conf = [None, None, None]
        for size in [50, 100, 200, 500]:
            for act_f in ['logistic', 'tanh', 'relu']:
                for solver in ['lbfgs', 'sgd', 'adam']:
                    nn = MLPClassifier(hidden_layer_sizes=[size], activation=act_f, solver=solver)
                    ret = self.parametric_identification_cv(nn, False)
                    self.candidates.append((nn, ret[0], "hidd_lay_sz={}, act_f={}, solver={}".format(size, act_f, solver), False))

                    tmp_perf = statistics.mean(list(ret[0]))
                    if tmp_perf > best_perf:
                        best_perf = tmp_perf
                        best_conf = [size, act_f, solver]

        # best configuration: try an ensemble model
        nn_ens = [("nn-{}".format(str(i)), MLPClassifier(hidden_layer_sizes=best_conf[0], activation=best_conf[1], solver=best_conf[2])) for i in range(50)]
        vc = VotingClassifier(estimators=nn_ens, voting="soft")
        ret = self.parametric_identification_cv(vc, False)
        self.candidates.append((vc, ret[0], "hidd_lay_sz={}, act_f={}, solver={}".format(*best_conf), False))

    # Display results

    @staticmethod
    def display_training_result(model, auc, pars):
        if type(auc) is list:
            print("{} ({})".format(str(type(model)).split('.')[-1], pars))
            print("avg auc: {}, stddev auc: {}, max auc {}, min auc: {}".format(statistics.mean(auc),
                                                                                statistics.stdev(auc),
                                                                                max(auc),
                                                                                min(auc)))
        else:
            print("{} ({})".format(str(type(model)).split('.')[-1], pars))
            print("avg auc: {}".format(auc))

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
