import statistics

import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    BaggingClassifier
from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from ModelIdentification import ExperimentResult, ModelIdentification


class ExperimentResultSi(ExperimentResult):
    def __init__(self, model_tag, model, mi_auc, test_auc, par_tag, par_value, is_reg_model, is_bag=False, estimator_tag=None):
        super().__init__(model_tag, model, par_tag, par_value, is_reg_model, is_bag, estimator_tag)
        self.mi_auc: list[float] | None = mi_auc
        self.test_auc: float | None = test_auc

    def __repr__(self):
        return "{} ({}: {}): {}".format(self.model_tag, self.par_tag, self.par_value, self.test_auc)


class StructuralIdentification(ModelIdentification):
    """
        Train [and compare] most promising models with hyperparameters defined in model identification
    """
    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, test_features: pd.DataFrame,
                 test_labels: pd.DataFrame, cv_folds: int, verbose=False):
        super().__init__(train_features, train_labels, test_features, test_labels, cv_folds, verbose)
        self.candidates: list[ExperimentResultSi] = list()

    @staticmethod
    def model_exploitation(model, test_features: pd.DataFrame):
        """
            Use final model to predict challenge data
        """

        is_reg_model = type(model) is LinearRegression or type(model) is RidgeClassifier
        return expit(model.predict(test_features)) if is_reg_model else model.predict_proba(test_features)[:, 1]

    def model_testing(self, variant=None):
        """ Train selected candidates on complete training set and assess performance on unused test set. """

        for i in range(len(self.candidates)):
            m = self.candidates[i].model
            m.fit(self.train_features, self.train_labels)
            y_i_ts_pred_prob = expit(m.predict(self.test_features)) if self.candidates[i].is_reg_model else m.predict_proba(self.test_features)[:, 1]
            auc = roc_auc_score(self.test_labels, y_i_ts_pred_prob)
            self.candidates[i].model, self.candidates[i].test_auc = m, auc
        self.candidates.sort(reverse=True, key=lambda x: x.test_auc)

        # print results of testing of most promising models
        if self.verbose:
            print("\n * MODEL TESTING ({}) *".format(variant))
            print(" -> performance:")
            for c in self.candidates:
                StructuralIdentification.display_training_result(c, "test")

        return self.candidates

    def model_selection(self, n=10):
        """ Select most promising models based on performance on validation sets. """

        # Keep max n best models
        self.candidates = sorted(self.candidates, reverse=True, key=lambda x: statistics.mean(x.mi_auc))[:min(n, len(self.candidates))]

    def preprocessing_model_identification(self, model):
        """ For preprocessing experiments, don't compute parametric identification loop (suppose we already know the
         parameters we want) """

        m, par_tag, par_val = {"lm": (LinearRegression(), str(), None),
                   "lr": (LogisticRegression(max_iter=10000), "max_iter", 10000),
                   "gbc": (GradientBoostingClassifier(loss="log_loss", n_estimators=250,
                                                      subsample=0.75, min_samples_split=2, max_depth=3), "loss", "log_loss"),
                   "bc": (BaggingClassifier(estimator=GradientBoostingClassifier(loss="log_loss", n_estimators=200, subsample=0.75, min_samples_split=2, max_depth=3), n_estimators=10),
                          "# estimators", 10)}[model]
        self.candidates.append(ExperimentResultSi(model, m, float(), float(), par_tag, par_val, type(m) is LinearRegression, model == "bc", "gbc" if m == "bc" else None))

    # Methods corresponding to the "structural identification" step for the different model types

    def parametric_identification(self, models, variant=None):
        for m in models:
            if m in ["lm", "lr", "ridge", "tree", "forest", "ada", "nn"]:
                {"lm": self.lm, "lr": self.lr, "ridge": self.ridge, "tree": self.tree, "forest": self.forest,
                 "ada": self.ada, "nn": self.nn}[m]()
            else:
                {"bc": self.bc, "gbc": self.gbc}[m](variant)

                # print results of model identification operations
        if self.verbose:
            print("\n  * MODEL IDENTIFICATION *")
            print(" -> CV performance:")
            for c in sorted(self.candidates, reverse=True, key=lambda x: statistics.mean(x.mi_auc)):
                StructuralIdentification.display_training_result(c, "mi")

    def lm(self):
        # Structural and parametric identification
        lm = LinearRegression()
        auc = self.parametric_identification_cv(lm, True)
        self.candidates.append(ExperimentResultSi("lm", lm, auc, None, str(), None, True, False, None))

    def lr(self):
        lr = LogisticRegression(max_iter=100000)
        auc = self.parametric_identification_cv(lr, False)
        self.candidates.append(ExperimentResultSi("lr", lr, auc, None, "max_iter", 100000, False, False, None))

    def ridge(self):
        for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]:
            ridge = RidgeClassifier(alpha=alpha)
            auc = self.parametric_identification_cv(ridge, True)
            self.candidates.append(ExperimentResultSi("ridge", ridge, auc, None, "alpha", alpha, True, False, None))

    def tree(self):
        for c in ["entropy", "gini", "log_loss"]:
            for s in ["best", "random"]:
                dtree = DecisionTreeClassifier(criterion=c, splitter=s)
                auc = self.parametric_identification_cv(dtree, False)
                self.candidates.append(ExperimentResultSi("dtree", dtree, auc, None, "criterion-splitter", "{}, {}".format(c, s), False, False, None))

    def forest(self):
        for c in ["entropy", "gini", "log_loss"]:
            for n in [10, 20, 50, 100]:
                rf = RandomForestClassifier(criterion=c, n_estimators=n)
                auc = self.parametric_identification_cv(rf, False)
                self.candidates.append(ExperimentResultSi("randforest", rf, auc, None, "criterion-ntrees", "{}, {}".format(c, n), False, False, None))

    def ada(self):
        for n in (300, 400, 500):
            ada = AdaBoostClassifier(estimator=None, n_estimators=n, algorithm="SAMME")
            auc = self.parametric_identification_cv(ada, False)
            self.candidates.append(ExperimentResultSi("ada", ada, auc, None, "n_estimators", n, False, False, None))

    def gbc(self, variant):
        if variant == "h1n1":
            gbc = GradientBoostingClassifier(loss="log_loss", n_estimators=100, subsample=0.5, min_samples_split=3, max_depth=3)
            #gbc = GradientBoostingClassifier(loss="log_loss", subsample=0.5)
        else:
            gbc = GradientBoostingClassifier(loss="log_loss", n_estimators=200, subsample=0.5, min_samples_split=3, max_depth=4)
            gbc = GradientBoostingClassifier(loss="log_loss", subsample=0.5)
        auc = self.parametric_identification_cv(gbc, False)
        self.candidates.append(ExperimentResultSi("gbc", gbc, auc, None, "gbc_si", str(), False, False, None))

    def bc(self, variant):
        if variant == "h1n1":
            estimator = GradientBoostingClassifier(loss="log_loss", n_estimators=100, subsample=0.5, min_samples_split=3, max_depth=3)
            bc = BaggingClassifier(estimator=estimator, max_features=1.0, n_estimators=60)
        else:
            estimator = GradientBoostingClassifier(loss="log_loss", subsample=0.5)
            bc = BaggingClassifier(estimator=estimator, max_features=1.0, n_estimators=40)
        auc = self.parametric_identification_cv(bc, False)
        self.candidates.append(ExperimentResultSi("bc", bc, auc, None, "bc_gbc_si", str(), False, True, "GBC"))

    def nn(self):
        nn = MLPClassifier(hidden_layer_sizes=[100], activation="logistic", solver="sgd", max_iter=300)
        auc = self.parametric_identification_cv(nn, False)
        self.candidates.append(ExperimentResultSi("nn", nn, auc, None, "act_f", "logistic", False, False, None))

        # TODO ensemble model with nn

    # Display results

    @staticmethod
    def display_training_result(candidate, mode):
        if mode == "mi":
            print("{} ({})".format(candidate.model_tag, candidate.par_tag))
            msg = "avg candidate.auc: {}, stddev candidate.auc: {}, max candidate.auc {}, min candidate.auc: {}"
            print(msg.format(round(statistics.mean(candidate.mi_auc), 5), round(statistics.stdev(candidate.mi_auc), 5),
                             round(max(candidate.mi_auc), 5), round(min(candidate.mi_auc), 5)))
        elif mode == "test":
            print("{} ({})".format(candidate.model_tag, candidate.par_tag))
            print("candidate.auc: {}".format(round(candidate.test_auc, 5)))
