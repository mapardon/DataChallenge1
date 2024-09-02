import statistics
from multiprocessing import Process

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ModelIdentification import ModelIdentification, Candidate


class ModelIdentificationSpecific(ModelIdentification):
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
                self.candidates.append(Candidate(gbc, auc, "n_estimators={}".format(n), False))

        elif par == "subsample":
            for s in [0.1, 0.5, 0.75, 0.9, 1.0]:
                gbc = GradientBoostingClassifier(loss="log_loss", subsample=s)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(Candidate(gbc, auc, "subsample={}".format(s), False))

        elif par == "min_sample_split":
            for mss in [2, 3, 5, 10]:
                gbc = GradientBoostingClassifier(loss="log_loss", min_samples_split=mss)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(Candidate(gbc, auc, "min_sample_split={}".format(mss), False))

        elif par == "max_depth":
            for max_depth in [2, 3, 4, 5, 10, 20]:
                gbc = GradientBoostingClassifier(loss="log_loss", max_depth=max_depth)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(Candidate(gbc, auc, "max_depth={}".format(max_depth), False))

        elif par == "init":
            pass

    def hgb(self, par="max_iter"):
        if par == "max_iter":
            for mxi in [50, 100, 500, 1000, 5000, 10**4, 10**5, 10**6]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", max_iter=mxi)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(Candidate(hgb, auc, "max_iter={}".format(mxi), False))

        elif par == "l2":
            for l2 in [0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 5.0]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", l2_regularization=l2)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(Candidate(hgb, auc, "l2_reg={}".format(l2), False))

        elif par == "min_samples_leaf":
            for msl in [10, 20, 30, 40, 50]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", min_samples_leaf=msl)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(Candidate(hgb, auc, "min_sample_leaf={}".format(msl), False))

        elif par == "max_features":
            for mf in [0.5, 0.75, 0.9, 1.0]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", max_features=mf)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(Candidate(hgb, auc, "max_features={}".format(mf), False))

    def bc(self, par):
        for e in [None, DecisionTreeClassifier(criterion="log_loss", splitter="best"),
                  DecisionTreeClassifier(criterion="log_loss", splitter="random"),
                  LogisticRegression(max_iter=100000),
                  GradientBoostingClassifier(loss="log_loss", n_estimators=300, subsample=0.75,
                                             min_samples_split=2, max_depth=4)][:-1]:

            if par == "max_features":
                for mf in [0.5, 0.75, 1.0]:
                    bc = BaggingClassifier(estimator=e, max_features=mf)
                    auc = self.parametric_identification_cv(bc, False)
                    self.candidates.append(Candidate(bc, auc, "estimator={}, max_features={}".format(type(e), mf), False))

            elif par == "oob_score":
                for oob in [False, True]:
                    bc = BaggingClassifier(estimator=e, oob_score=oob)
                    auc = self.parametric_identification_cv(bc, False)
                    self.candidates.append(Candidate(bc, auc, "estimator={}, oob_score={}".format(type(e), oob), False))

            elif par == "n_estimators":
                ns = [10, 15, 20] if type(e) is GradientBoostingClassifier() else [10, 20, 50, 100, 200]
                for n in ns:
                    bc = BaggingClassifier(estimator=e, n_estimators=n)
                    auc = self.parametric_identification_cv(bc, False)
                    self.candidates.append(Candidate(bc, auc, "estimator={}, n_estimators={}".format(type(e), n), False))


class SpecificIdentification:
    """
        In-depth identification for highly-parametric models
    """

    def __init__(self, exp_rounds, variants, models_pars=(("ada", None),)):
        self.exp_rounds = exp_rounds
        self.variants = variants
        self.models_pars = models_pars

    def main(self):
        for variant in self.variants:
            for model, par in self.models_pars:
                self.specific_identification(variant, model, par)

    def specific_identification(self, variant, model, par):
        final_train_sets, final_test_sets = list(), list()
        for i in range(self.exp_rounds):
            final_train_sets.append((pd.read_pickle("serialized_df/trs_{}_features_{}".format(variant, str(i)))[:500],
                                     pd.read_pickle("serialized_df/trs_{}_labels_{}".format(variant, str(i)))[:500]))
            final_test_sets.append((pd.read_pickle("serialized_df/tss_{}_features_{}".format(variant, str(i)))[:500],
                                    pd.read_pickle("serialized_df/tss_{}_labels_{}".format(variant, str(i)))[:500]))

        candidates = list()
        for i in range(self.exp_rounds):
            mi = ModelIdentificationSpecific(*final_train_sets[i], *final_test_sets[i], cv_folds=5, verbose=True)
            mods_dict = {"ada": mi.ada, "gbc": mi.gbc, "hgb": mi.hgb, "bc": mi.bc}
            mods_dict[model]() if par is None else mods_dict[model](par)
            mi.model_selection(n=1000)
            candidates += mi.model_testing()

        print("\nFinal {} candidates".format(variant))
        for c in sorted(candidates, reverse=True, key=lambda x: x.auc):
            print(c)

        # plotting bc its cool
        candidates_num = list()
        for c in candidates:
            c.pars = float(c.pars.split('=')[1])
            candidates_num.append(c)
        candidates_num.sort(key=lambda x: x.pars)

        print(candidates_num)
        x, y = list(), list()
        for i in range(0, len(candidates_num), self.exp_rounds):
            x.append(candidates_num[i].pars)
            y.append(statistics.mean([candidates_num[j].auc for j in range(i, min(len(candidates_num), i+self.exp_rounds))]))
        print(x, y)
        fig, ax = plt.subplots()
        ax.plot(x, y)

        ax.set(xlabel=par, ylabel='AUC', title='Bigger is better ({}, {})?'.format(model, variant))
        ax.grid()
        plt.show()
        #plt.savefig("{}-{}-{}.png".format(variant, model, par))


def multi_proc():
    #confs = [("gbc", "n_estimators",), ("gbc", "subsample",), ("gbc", "min_sample_split",), ("gbc", "max_depth",)]
    confs = [("bc", "max_features",), ("bc", "oob_score",), ("bc", "n_estimators",)]
    procs = [Process(target=SpecificIdentification(1, ("h1n1",), (c,)).main) for c in confs]

    for p in procs:
        p.start()

    for p in procs:
        p.join()


def uni_proc():
    SpecificIdentification(1, ("h1n1",), (("gbc", "n_estimators"),)).main()


if __name__ == '__main__':

    multi_proc()
