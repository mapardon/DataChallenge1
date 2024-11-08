import pickle
import statistics

import pandas as pd
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, \
    AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from ModelIdentification import ModelIdentification, ExperimentResult


class ExperimentResultPi(ExperimentResult):
    def __init__(self, model_tag, model, mi_auc, par_tag, par_value, is_reg_model, is_bag=False, estimator_tag=None):
        super().__init__(model_tag, model, par_tag, par_value, is_reg_model, is_bag, estimator_tag)
        self.mi_auc: list[float] | None = mi_auc

    def __repr__(self):
        return "{} ({}: {}): {}".format(self.model_tag, self.par_tag, self.par_value, statistics.mean(self.mi_auc))


class ParametricIdentification(ModelIdentification):
    """
        Model identification phase, test models with different hyperparameters and perform additional experiments
        (in/out-sample overfitting evaluation...)
    """
    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, test_features: pd.DataFrame,
                 test_labels: pd.DataFrame, cv_folds: int, verbose: bool = False):
        super().__init__(train_features, train_labels, test_features, test_labels, cv_folds, verbose)
        self.candidates: list[ExperimentResultPi] = list()

        self.models_dict = {"nn": self.nn, "ada": self.ada, "gbc": self.gbc, "hgb": self.hgb, "bc": self.bc,
                            "bcl": self.bc_long}

    def parametric_identification(self, model: str, par: str):
        self.models_dict[model]() if par is None else self.models_dict[model](par)
        return self.candidates

    def in_out_parametric_identification(self, model):
        # split train.test features in 50/50 in out sets
        tmp_features, tmp_labels = (pd.concat([self.train_features, self.test_features], axis="rows"),
                                    pd.concat([self.train_labels, self.test_labels], axis="rows"))
        in_features, out_features, = tmp_features[:len(tmp_features) // 2], tmp_features[len(tmp_features) // 2:]
        in_labels, out_labels = tmp_labels[:len(tmp_labels) // 2], tmp_labels[len(tmp_labels) // 2:]

        model.fit(in_features, in_labels)

        # compute AUCs
        y_i_pred_prob_in = model.predict_proba(in_features)[:, 1]
        in_auc = roc_auc_score(in_labels, y_i_pred_prob_in)
        y_i_pred_prob_out = model.predict_proba(out_features)[:, 1]
        out_auc = roc_auc_score(out_labels, y_i_pred_prob_out)

        return in_auc, out_auc

    def gbc(self, par="n_estimators"):
        """ :param par: possible values: n_estimators, subsample, min_sample_split, max_depth """
        if par == "n_estimators":
            for n in [50, 100, 200, 300, 400, 500]:
                gbc = GradientBoostingClassifier(loss="log_loss", n_estimators=n)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(ExperimentResultPi("gbc", gbc, auc, "n_estimators", n, False, False))

        elif par == "subsample":
            for s in [0.1, 0.5, 0.75, 0.9, 1.0]:
                gbc = GradientBoostingClassifier(loss="log_loss", subsample=s)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(ExperimentResultPi("gbc", gbc, auc, "subsample", s, False, False))

        elif par == "min_sample_split":
            for mss in [2, 3, 5, 10]:
                gbc = GradientBoostingClassifier(loss="log_loss", min_samples_split=mss)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(ExperimentResultPi("gbc", gbc, auc, "min_sample_split", mss, False, False))

        elif par == "max_depth":
            for max_depth in [2, 3, 4, 5, 10]:
                gbc = GradientBoostingClassifier(loss="log_loss", max_depth=max_depth)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(ExperimentResultPi("gbc", gbc, auc, "max_depth", max_depth, False, False))

        elif par == "inout":
            gbc = GradientBoostingClassifier(loss="log_loss", n_estimators=250, subsample=0.75, min_samples_split=2, max_depth=3)
            in_auc, out_auc = self.in_out_parametric_identification(gbc)
            self.candidates += [ExperimentResultPi("gbc", gbc, [in_auc], "inout", "in", False, False),
                                ExperimentResultPi("gbc", gbc, [out_auc], "inout", "out", False, False)]

    def hgb(self, par="max_iter"):
        if par == "max_iter":
            for mxi in [50, 100, 500, 1000, 5000, 10**4, 10**5, 10**6]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", max_iter=mxi)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(ExperimentResultPi("hgb", hgb, auc, "max_iter", mxi, False, False))

        elif par == "l2":
            for l2 in [0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 5.0]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", l2_regularization=l2)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(ExperimentResultPi("hgb", hgb, auc, "l2_reg", l2, False, False))

        elif par == "min_samples_leaf":
            for msl in [10, 20, 30, 40, 50]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", min_samples_leaf=msl)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(ExperimentResultPi("hgb", hgb, auc, "min_sample_leaf", msl, False, False))

        elif par == "max_features":
            for mf in [0.5, 0.75, 0.9, 1.0]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", max_features=mf)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(ExperimentResultPi("hgb", hgb, auc, "max_features", mf, False, False))

    def nn(self, par="hl1"):
        if par == "hl1":
            for hl in [(50,), (100,), (150,), (200,)]:
                nn = MLPClassifier(hidden_layer_sizes=hl, activation="logistic", max_iter=300)
                auc = self.parametric_identification_cv(nn, False)
                self.candidates.append(ExperimentResultPi("nn", nn, auc, "hidden_layers", hl[0], False, False))

        elif par == "hl2":
            for hls in [(50, 50), (100, 50)]:
                nn = MLPClassifier(hidden_layer_sizes=hls, activation="logistic", max_iter=300)
                auc = self.parametric_identification_cv(nn, False)
                self.candidates.append(ExperimentResultPi("nn", nn, auc, "hidden_layers", str(hls), False, False))

        elif par == "act_f":
            for actf in ['logistic', 'tanh', 'relu']:
                nn = MLPClassifier(hidden_layer_sizes=(50,), activation=actf, max_iter=300)
                auc = self.parametric_identification_cv(nn, False)
                self.candidates.append(ExperimentResultPi("nn", nn, auc, "act_f", actf, False, False))

        elif par == "solver":
            for solver in ['lbfgs', 'sgd', 'adam']:
                nn = MLPClassifier(hidden_layer_sizes=(50,), activation="logistic", solver=solver, max_iter=300)
                auc = self.parametric_identification_cv(nn, False)
                self.candidates.append(ExperimentResultPi("nn", nn, auc, "solver", solver, False, False))

        elif par == "miter":
            for miter in [100, 200, 300, 400, 500]:
                nn = MLPClassifier(hidden_layer_sizes=(50,), activation="logistic", solver="sgd", max_iter=miter)
                auc = self.parametric_identification_cv(nn, False)
                self.candidates.append(ExperimentResultPi("nn", nn, auc, "max_iter", miter, False, False))

    # Ensemble of estimators

    def ada(self, par="_uni"):
        if "_uni" in par:
            estimators, names = [None, DecisionTreeClassifier(criterion="log_loss", splitter="best"),
                                 DecisionTreeClassifier(criterion="log_loss", splitter="random"),
                                 LogisticRegression(max_iter=100000)][:1], ['None', 'DTC-best', 'DTC-random', 'LR']
        else:  # "_ens" in par
            estimators, names = [GradientBoostingClassifier(loss="log_loss", n_estimators=200, subsample=0.75,
                                 min_samples_split=2, max_depth=3)], ['GBC']

        for e, n in zip(estimators, names):
            if par == "n_estimators_uni":
                for ne in [100, 200, 300, 500][2:]:
                    ada = AdaBoostClassifier(estimator=e, n_estimators=ne, algorithm="SAMME")
                    auc = self.parametric_identification_cv(ada, False)
                    self.candidates.append(ExperimentResultPi("ada", ada, auc, "n_estimators", ne, False, True, n))

            elif par == "n_estimators_ens":
                for ne in [10, 15, 20]:
                    ada = AdaBoostClassifier(estimator=e, n_estimators=ne, algorithm="SAMME")
                    auc = self.parametric_identification_cv(ada, False)
                    self.candidates.append(ExperimentResultPi("ada", ada, auc, "n_estimators", ne, False, True, n))

    def bc(self, par="max_features_lr"):
        if "lr" in par:
            estimator, name = LogisticRegression(max_iter=100000), 'LR'
        elif "tree" in par:
            estimator, name = DecisionTreeClassifier(criterion="log_loss", splitter="best"), 'DTC-best'
        elif "gbc" in par:
            estimator, name = (GradientBoostingClassifier(loss="log_loss", n_estimators=100, subsample=0.5, min_samples_split=3, max_depth=3),
                               'GBC')
        else:
            raise ValueError("Parametric identification for bagging classifier must specify an estimator class")

        if "max_features" in par:
            for mf in [0.5, 0.75, 1.0]:
                bc = BaggingClassifier(estimator=estimator, max_features=mf)
                auc = self.parametric_identification_cv(bc, False)
                self.candidates.append(ExperimentResultPi("bc", bc, auc, "max_features", mf, False, True, str(estimator)))

        elif "n_estimators" in par:
            for ne in [10, 20, 40, 60]:
                bc = BaggingClassifier(estimator=estimator, n_estimators=ne)
                auc = self.parametric_identification_cv(bc, False)
                self.candidates.append(ExperimentResultPi("bc", bc, auc, "n_estimators", ne, False, True, str(estimator)))

    def bc_long(self, par):
        # par = h1n1/seas
        e = GradientBoostingClassifier(loss="log_loss", n_estimators=200, subsample=0.75, min_samples_split=2, max_depth=3)
        ne = 45
        bc = BaggingClassifier(estimator=e, n_estimators=ne)
        auc = self.parametric_identification_cv(bc, False)
        self.candidates.append(ExperimentResultPi("bc", bc, auc, "n_estimators", ne, False, True, "GBC"))

        pickle.dump(bc, open("bc_long_save_{}".format(par), "wb"))
