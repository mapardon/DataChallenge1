import statistics

import pandas as pd
from scipy.special import expit
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score


class Candidate:
    def __init__(self, model, auc, pars, is_reg_model):
        self.model = model
        self.auc: float = auc
        self.pars: str = pars
        self.is_reg_model: bool = is_reg_model

    def __repr__(self):
        return "{} ({}): {}".format(str(type(self.model)).split('.')[-1].split("'")[0], self.pars, self.auc)


class ModelIdentification:
    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, test_features: pd.DataFrame,
                 test_labels: pd.DataFrame, cv_folds: int, verbose=False):
        self.train_features = train_features
        self.train_labels = train_labels

        self.test_features = test_features
        self.test_labels = test_labels

        self.cv_folds = cv_folds
        self.candidates: list[Candidate] = list()
        self.verbose = verbose

    @staticmethod
    def model_exploitation(model, test_features: pd.DataFrame):
        """
            Use final model to predict challenge data
        """

        is_reg_model = type(model) is LinearRegression or type(model) is RidgeClassifier
        return expit(model.predict(test_features)) if is_reg_model else model.predict_proba(test_features)[:, 1]

    def model_testing(self):
        """
            Train selected candidates on complete training set and assess performance on unused test set
        """

        for i in range(len(self.candidates)):
            m, pars = self.candidates[i].model, self.candidates[i].pars
            m.fit(self.train_features, self.train_labels)
            y_i_ts_pred_prob = expit(m.predict(self.test_features)) if self.candidates[i].is_reg_model else m.predict_proba(self.test_features)[:, 1]
            auc = roc_auc_score(self.test_labels, y_i_ts_pred_prob)
            self.candidates[i].model, self.candidates[i].auc, self.candidates[i].pars = m, auc, pars
        self.candidates.sort(reverse=True, key=lambda x: x.auc)

        # print results of testing of most promising models
        if self.verbose:
            print("\n * MODEL TESTING *")
            print(" -> performance:")
            for c in self.candidates:
                ModelIdentification.display_training_result(c)

        return self.candidates

    def model_selection(self, n=10):
        """
            Select most promising models based on performance on validation sets.
            For now, we simply take the 10 best performing algorithms, but we could consider creation of heterogeneous
            ensemble model
        """
        # Keep max n best models
        self.candidates = sorted(self.candidates, reverse=True, key=lambda x: statistics.mean(x.auc))[:min(n, len(self.candidates))]

    def parametric_identification_cv(self, model, is_reg_model=False):
        """
            Generic loop training the provided model on the training set (split in training/validation
            folds) and assessing performance.

            :return: AUC of the models of the different loops,
        """

        n_rows_fold = len(self.train_features) // self.cv_folds
        auc = list()

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

            # compute AUC
            auc.append(roc_auc_score(y_i_vs, y_i_pred_prob))

        return auc

    # Methods corresponding to the "structural identification" step for the different model types
    
    def model_identification(self, models):
        for m in models:
            {"lm": self.lm, "lr": self.lr, "ridge": self.ridge, "gnb": self.gnb, "gpc": self.gpc, "tree": self.tree,
             "forest": self.forest, "ada": self.ada, "gbc": self.gbc, "bc": self.bc, "svm": self.svm, "nn": self.nn}[m]()

        # print results of model identification operations
        if self.verbose:
            print("\n  * MODEL IDENTIFICATION *")
            print(" -> CV performance:")
            for c in sorted(self.candidates, reverse=True, key=lambda x: statistics.mean(x.auc)):
                ModelIdentification.display_training_result(c)

    def lm(self):
        # Structural and parametric identification
        lm = LinearRegression()
        auc = self.parametric_identification_cv(lm, True)
        self.candidates.append(Candidate(lm, auc, str(), True))

    def lr(self):
        lr = LogisticRegression(max_iter=100000)
        auc = self.parametric_identification_cv(lr, False)
        self.candidates.append(Candidate(lr, auc, str(), False))

    def ridge(self):
        for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]:
            ridge = RidgeClassifier(alpha=alpha)
            auc = self.parametric_identification_cv(ridge, True)
            self.candidates.append(Candidate(ridge, auc, "alpha={}".format(alpha), True))

    def gnb(self):
        gnb = GaussianNB()
        auc = self.parametric_identification_cv(gnb, False)
        self.candidates.append(Candidate(gnb, auc, str(), False))

    def gpc(self):
        gpc = GaussianProcessClassifier()
        auc = self.parametric_identification_cv(gpc, False)
        self.candidates.append(Candidate(gpc, auc, str(), False))

    def tree(self):
        for c in ["entropy", "gini", "log_loss"]:
            for s in ["best", "random"]:
                dtree = DecisionTreeClassifier(criterion=c, splitter=s)
                auc = self.parametric_identification_cv(dtree, False)
                self.candidates.append(Candidate(dtree, auc, "c={}, s={}".format(c, s), False))

    def forest(self):
        for c in ["entropy", "gini", "log_loss"]:
            for n in [10, 20, 50, 100]:
                rf = RandomForestClassifier(criterion=c, n_estimators=n)
                auc = self.parametric_identification_cv(rf, False)
                self.candidates.append(Candidate(rf, auc, "(c={}, n={})".format(c, n), False))

    def ada(self):
        for n in (300, 400, 500):
            ada = AdaBoostClassifier(estimator=None, n_estimators=n, algorithm="SAMME")
            auc = self.parametric_identification_cv(ada, False)
            self.candidates.append(Candidate(ada, auc, "n={}".format(n), False))

    def gbc(self):
        for s in [0.75, 1.0]:
            gbc = GradientBoostingClassifier(loss="log_loss", n_estimators=250, subsample=s,
                                             min_samples_split=2, max_depth=3)
            auc = self.parametric_identification_cv(gbc, False)
            self.candidates.append(Candidate(gbc, auc, "subsample:{}".format(s), False))

    def bc(self):
        estimator = GradientBoostingClassifier(loss="log_loss", n_estimators=200, subsample=0.75, min_samples_split=2, max_depth=3)
        bc = BaggingClassifier(estimator=estimator, n_estimators=10)
        auc = self.parametric_identification_cv(bc, False)
        self.candidates.append(Candidate(bc, auc, "estimators={}".format("GBC"), False))

    def svm(self):
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
            svc = SVC(kernel=kernel, probability=True)
            auc = self.parametric_identification_cv(svc, False)
            self.candidates.append(Candidate(svc, auc, "kernel={}".format(kernel), False))

    def nn(self):
        best_perf = -1.0
        best_conf = [None, None, None]
        for size in [50, 100, 200, 500]:
            for act_f in ['logistic', 'tanh', 'relu']:
                for solver in ['lbfgs', 'sgd', 'adam']:
                    nn = MLPClassifier(hidden_layer_sizes=[size], activation=act_f, solver=solver)
                    auc = self.parametric_identification_cv(nn, False)
                    self.candidates.append(Candidate(nn, auc, "hidd_lay_sz={}, act_f={}, solver={}".format(size, act_f, solver), False))

                    tmp_perf = statistics.mean(list(auc))
                    if tmp_perf > best_perf:
                        best_perf = tmp_perf
                        best_conf = [size, act_f, solver]

        # best configuration: try an ensemble model
        nn_ens = [("nn-{}".format(str(i)), MLPClassifier(hidden_layer_sizes=best_conf[0], activation=best_conf[1], solver=best_conf[2])) for i in range(50)]
        vc = VotingClassifier(estimators=nn_ens, voting="soft")
        auc = self.parametric_identification_cv(vc, False)
        self.candidates.append(Candidate(vc, auc, "hidd_lay_sz={}, act_f={}, solver={}".format(*best_conf), False))

    # Display results

    @staticmethod
    def display_training_result(candidate):
        if type(candidate.auc) is list:
            print("{} ({})".format(str(type(candidate.model)).split('.')[-1][:-2], candidate.pars))
            print("avg candidate.auc: {}, stddev candidate.auc: {}, max candidate.auc {}, min candidate.auc: {}".format(statistics.mean(candidate.auc),
                                                                                                                        statistics.stdev(candidate.auc),
                                                                                                                        max(candidate.auc),
                                                                                                                        min(candidate.auc)))
        else:
            print("{} ({})".format(str(type(candidate.model)).split('.')[-1][:-2], candidate.pars))
            print("candidate.auc: {}".format(candidate.auc))
