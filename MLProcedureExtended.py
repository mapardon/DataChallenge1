import pickle
import statistics
import time
from multiprocessing import Process

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier, \
    AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from DataPreprocessing import DataPreprocessing
from ModelIdentification import ModelIdentification, Candidate


class SpecificCandidate(Candidate):
    def __init__(self, model, auc, pars, is_reg_model, is_bag=False, estimator_tag=None):
        super().__init__(model, auc, pars, is_reg_model)
        self.is_bag = is_bag  # if the method uses an estimator (other than itself)
        self.estimator_tag = estimator_tag


class ModelIdentificationExtended(ModelIdentification):
    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, test_features: pd.DataFrame,
                 test_labels: pd.DataFrame, cv_folds: int, verbose: bool = False):
        super().__init__(train_features, train_labels, test_features, test_labels, cv_folds, verbose)

    def gbc(self, par="n_estimators"):
        """ :param par: possible values: n_estimators, subsample, min_sample_split, max_depth """
        # cool params: subsample (<1 -> stochastic boosting); min_samples_split (default=2); max_depth (default=3)
        if par == "n_estimators":
            for n in [50, 100, 200, 300, 400, 500]:
                gbc = GradientBoostingClassifier(loss="log_loss", n_estimators=n)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(SpecificCandidate(gbc, auc, "n_estimators={}".format(n), False, True))

        elif par == "subsample":
            for s in [0.1, 0.5, 0.75, 0.9, 1.0]:
                gbc = GradientBoostingClassifier(loss="log_loss", subsample=s)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(SpecificCandidate(gbc, auc, "subsample={}".format(s), False, False))

        elif par == "min_sample_split":
            for mss in [2, 3, 5, 10]:
                gbc = GradientBoostingClassifier(loss="log_loss", min_samples_split=mss)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(SpecificCandidate(gbc, auc, "min_sample_split={}".format(mss), False, False))

        elif par == "max_depth":
            for max_depth in [2, 3, 4, 5, 10, 20]:
                gbc = GradientBoostingClassifier(loss="log_loss", max_depth=max_depth)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(SpecificCandidate(gbc, auc, "max_depth={}".format(max_depth), False, False))

    def hgb(self, par="max_iter"):
        if par == "max_iter":
            for mxi in [50, 100, 500, 1000, 5000, 10**4, 10**5, 10**6]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", max_iter=mxi)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(SpecificCandidate(hgb, auc, "max_iter={}".format(mxi), False, False))

        elif par == "l2":
            for l2 in [0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 5.0]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", l2_regularization=l2)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(SpecificCandidate(hgb, auc, "l2_reg={}".format(l2), False, False))

        elif par == "min_samples_leaf":
            for msl in [10, 20, 30, 40, 50]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", min_samples_leaf=msl)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(SpecificCandidate(hgb, auc, "min_sample_leaf={}".format(msl), False, False))

        elif par == "max_features":
            for mf in [0.5, 0.75, 0.9, 1.0]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", max_features=mf)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(SpecificCandidate(hgb, auc, "max_features={}".format(mf), False, False))

    def nn(self, par="hl1"):
        if par == "hl1":
            for hl in [(50,), (100,), (200,), (500,)]:
                nn = MLPClassifier(hidden_layer_sizes=hl, activation="logistic")
                auc = self.parametric_identification_cv(nn, False)
                self.candidates.append(SpecificCandidate(nn, auc, "hidden_layers={}".format(hl[0]), False, False))

        elif par == "hl2":
            for hls in [(50, 50), (100, 100)]:
                nn = MLPClassifier(hidden_layer_sizes=hls, activation="logistic")
                auc = self.parametric_identification_cv(nn, False)
                self.candidates.append(SpecificCandidate(nn, auc, "hidden_layers={}".format(hls), False, False))

        elif par == "act_f":
            for actf in ['logistic', 'tanh', 'relu']:
                nn = MLPClassifier(hidden_layer_sizes=(50,), activation=actf)
                auc = self.parametric_identification_cv(nn, False)
                self.candidates.append(SpecificCandidate(nn, auc, "act_f={}".format(actf), False, False))

        elif par == "solver":
            for solver in ['lbfgs', 'sgd', 'adam']:
                nn = MLPClassifier(hidden_layer_sizes=(50,), activation="logistic", solver=solver)
                auc = self.parametric_identification_cv(nn, False)
                self.candidates.append(SpecificCandidate(nn, auc, "solver={}".format(solver), False, False))

    def nn_short(self, par=None):
        nn = MLPClassifier(hidden_layer_sizes=(50,), activation="logistic", solver="sgd", max_iter=100)
        auc = self.parametric_identification_cv(nn, False)
        self.candidates.append(SpecificCandidate(nn, auc, "nn_short_test={}".format(50), False, False))

    def nn_long(self):
        nn = MLPClassifier(hidden_layer_sizes=(150,), activation="logistic", solver="sgd", learning_rate="adaptive",
                           max_iter=500, early_stopping=True)
        auc = self.parametric_identification_cv(nn, False)
        self.candidates.append(SpecificCandidate(nn, auc, "nn_full_stuff={}".format((150,)), False, False))

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
                    self.candidates.append(SpecificCandidate(ada, auc, "n_estimators={}".format(ne), False, True, n))

            elif par == "n_estimators_ens":
                for ne in [10, 15, 20]:
                    ada = AdaBoostClassifier(estimator=e, n_estimators=ne, algorithm="SAMME")
                    auc = self.parametric_identification_cv(ada, False)
                    self.candidates.append(SpecificCandidate(ada, auc, "n_estimators={}".format(ne), False, True, n))

    def bc(self, par="_uni"):
        if "_uni" in par:
            estimators, names = [None, DecisionTreeClassifier(criterion="log_loss", splitter="best"),
                                 DecisionTreeClassifier(criterion="log_loss", splitter="random"),
                                 LogisticRegression(max_iter=100000)], ['None', 'DTC-best', 'DTC-random', 'LR']
        else:  # "_ens" in par
            estimators, names = [GradientBoostingClassifier(loss="log_loss", n_estimators=200, subsample=0.75,
                                 min_samples_split=2, max_depth=3)], ['GBC']

        for e, n in zip(estimators, names):
            if "max_features" in par:
                for mf in [0.5, 0.75, 1.0]:
                    bc = BaggingClassifier(estimator=e, max_features=mf)
                    auc = self.parametric_identification_cv(bc, False)
                    self.candidates.append(SpecificCandidate(bc, auc, "max_features={}".format(mf), False, True, n))

            elif par == "n_estimators_uni":
                for ne in [10, 20, 50, 100, 200]:
                    bc = BaggingClassifier(estimator=e, n_estimators=ne)
                    auc = self.parametric_identification_cv(bc, False)
                    self.candidates.append(SpecificCandidate(bc, auc, "n_estimators={}".format(ne), False, True, n))

            elif par == "n_estimators_ens":
                for ne in [10, 15, 20]:
                    bc = BaggingClassifier(estimator=e, n_estimators=ne)
                    auc = self.parametric_identification_cv(bc, False)
                    self.candidates.append(SpecificCandidate(bc, auc, "n_estimators={}".format(ne), False, True, n))

    def bc_long(self, par):
        # par = h1n1/seas
        e = GradientBoostingClassifier(loss="log_loss", n_estimators=200, subsample=0.75, min_samples_split=2, max_depth=3)
        ne = 45
        bc = BaggingClassifier(estimator=e, n_estimators=ne)
        auc = self.parametric_identification_cv(bc, False)
        self.candidates.append(SpecificCandidate(bc, auc, "n_estimators={}".format(ne), False, True, "GBC"))

        pickle.dump(bc, open("bc_long_save_{}".format(par), "wb"))


class MLProcedureExtended:
    """
        Alternative to model identification step for highly-parametric models to run experiments outside the
        complete procedure loop
    """

    def __init__(self, exp_rounds, variants, models_pars=(("bcl", None),), dp_short=False):
        self.exp_rounds = exp_rounds
        self.variants = variants
        self.models_pars = models_pars
        self.dp_short = dp_short

    def main(self):
        for variant in self.variants:
            for model, par in self.models_pars:
                self.specific_identification(variant, model, par)

    def specific_identification(self, variant, model, par):
        f1, f2 = ("serialized_df/features_{}".format(variant), "serialized_df/labels_{}".format(variant))
        if self.dp_short:
            f1, f2 = f1 + "_short", f2 + "_short"

        candidates = list()
        for i in range(self.exp_rounds):
            # Load & reshuffle preprocessed dataset
            features, labels = pd.read_pickle(f1), pd.read_pickle(f2)
            dp = DataPreprocessing(self.dp_short)
            dp.shuffle_datasets(features, pd.DataFrame(labels))
            train_features, train_labels, test_features, test_labels = dp.get_train_test_datasets()

            # Train models with CV and test performance on unused test set
            mi = ModelIdentificationExtended(train_features, train_labels, test_features, test_labels, cv_folds=5, verbose=True)
            mods_dict = {"nns": mi.nn_short, "nn": mi.nn, "ada": mi.ada, "gbc": mi.gbc, "hgb": mi.hgb, "bc": mi.bc, "bcl": mi.bc_long}
            mods_dict[model]() if par is None else mods_dict[model](par)
            mi.model_selection(n=1000)
            candidates += mi.model_testing()

        # print all candidates
        print("\nFinal {} candidates".format(variant))
        for c in sorted(candidates, reverse=True, key=lambda x: x.auc):
            print(c)

        self.plot_results(candidates, model, variant, par)

    def plot_results(self, candidates: list[SpecificCandidate], model, variant, par):

        # group results by parameter value for plotting
        if not candidates[0].is_bag:
            candidates = [candidates]
        else:
            candidates.sort(key=lambda x: x.estimator_type)
            candidates2 = [[candidates[0]]]
            for i in range(1, len(candidates)):
                if candidates[i].estimator_tag != candidates[i-1].estimator_tag:
                    candidates2.append(list())
                candidates2[-1].append(candidates[i])
            candidates = candidates2

        # plot results
        for category in candidates:  # manage candidates using estimator
            if floatable(category[0].pars.split("=")[1]):
                plot = "plot"
                category_tmp = [SpecificCandidate(c.model, c.auc, float(c.pars.split("=")[1]), c.is_reg_model, c.is_bag, c.estimator_tag) for c in category]
            else:
                plot = "hist"
                category_tmp = [SpecificCandidate(c.model, c.auc, c.pars.split("=")[1], c.is_reg_model, c.is_bag, c.estimator_tag) for c in category]
            category_tmp.sort(key=lambda x: x.pars)

            x, y = list(), list()
            for i in range(0, len(category_tmp), self.exp_rounds):
                x.append(category_tmp[i].pars)
                y.append(statistics.mean([category_tmp[j].auc for j in range(i, min(len(category_tmp), i + self.exp_rounds))]))
            fig, ax = plt.subplots()

            if plot == "hist":
                ax.bar(x, y)
            elif plot == "plot":
                ax.plot(x, y)

            title = 'Experiment ({}, {}'.format(model, variant) + (", {})".format(category[0].estimator_tag) if category[0].is_bag else ")")
            ax.set(xlabel=par, ylabel='AUC', title=title)
            ax.grid()
            #plt.savefig("figures/{}-{}-{}-{}.png".format(variant, model, par, category[0].pars[-1]))
            plt.show()


def floatable(f):
    is_float = True
    try:
        float(f)
    except ValueError:
        is_float = False
    return is_float


def multi_proc():
    confs = [("nn", "act_f"), ("nn", "solver"), ("nn", "hl1"), ("nn", "hl2")]
    variants = ["h1n1",]
    procs = list()
    for v in variants:
        procs += [Process(target=MLProcedureExtended(1, (v,), (c,), False).main) for c in confs]

    for p in procs:
        p.start()

    for p in procs:
        p.join()


def uni_proc():
    confs = [("nn", "act_f"), ("nn", "solver"), ("nn", "hl1"), ("nn", "hl2")]
    variants = ["h1n1",]
    MLProcedureExtended(1, variants, confs, dp_short=False).main()


if __name__ == '__main__':

    uni_proc()
