import statistics
from multiprocessing import Process

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier, \
    AdaBoostClassifier
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
                self.candidates.append(Candidate(gbc, auc, ["n_estimators={}".format(n)], False))

        elif par == "subsample":
            for s in [0.1, 0.5, 0.75, 0.9, 1.0]:
                gbc = GradientBoostingClassifier(loss="log_loss", subsample=s)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(Candidate(gbc, auc, ["subsample={}".format(s)], False))

        elif par == "min_sample_split":
            for mss in [2, 3, 5, 10]:
                gbc = GradientBoostingClassifier(loss="log_loss", min_samples_split=mss)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(Candidate(gbc, auc, ["min_sample_split={}".format(mss)], False))

        elif par == "max_depth":
            for max_depth in [2, 3, 4, 5, 10, 20]:
                gbc = GradientBoostingClassifier(loss="log_loss", max_depth=max_depth)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append(Candidate(gbc, auc, ["max_depth={}".format(max_depth)], False))

    def hgb(self, par="max_iter"):
        if par == "max_iter":
            for mxi in [50, 100, 500, 1000, 5000, 10**4, 10**5, 10**6]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", max_iter=mxi)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(Candidate(hgb, auc, ["max_iter={}".format(mxi)], False))

        elif par == "l2":
            for l2 in [0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 5.0]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", l2_regularization=l2)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(Candidate(hgb, auc, ["l2_reg={}".format(l2)], False))

        elif par == "min_samples_leaf":
            for msl in [10, 20, 30, 40, 50]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", min_samples_leaf=msl)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(Candidate(hgb, auc, ["min_sample_leaf={}".format(msl)], False))

        elif par == "max_features":
            for mf in [0.5, 0.75, 0.9, 1.0]:
                hgb = HistGradientBoostingClassifier(loss="log_loss", max_features=mf)
                auc = self.parametric_identification_cv(hgb, False)
                self.candidates.append(Candidate(hgb, auc, ["max_features={}".format(mf)], False))

    def ada(self, par):
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
                    self.candidates.append(Candidate(ada, auc, ["n_estimators={}".format(ne), n], False))

            elif par == "n_estimators_ens":
                for ne in [10, 15, 20]:
                    ada = AdaBoostClassifier(estimator=e, n_estimators=ne, algorithm="SAMME")
                    auc = self.parametric_identification_cv(ada, False)
                    self.candidates.append(Candidate(ada, auc, ["n_estimators={}".format(ne), n], False))

    def bc(self, par):
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
                    self.candidates.append(Candidate(bc, auc, ["max_features={}".format(mf), n], False))

            elif par == "n_estimators_uni":
                for ne in [10, 20, 50, 100, 200]:
                    bc = BaggingClassifier(estimator=e, n_estimators=ne)
                    auc = self.parametric_identification_cv(bc, False)
                    self.candidates.append(Candidate(bc, auc, ["n_estimators={}".format(ne), n], False))

            elif par == "n_estimators_ens":
                for ne in [10, 15, 20]:
                    bc = BaggingClassifier(estimator=e, n_estimators=ne)
                    auc = self.parametric_identification_cv(bc, False)
                    self.candidates.append(Candidate(bc, auc, ["n_estimators={}".format(ne), n], False))


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
            final_train_sets.append((pd.read_pickle("serialized_df/trs_{}_features_{}".format(variant, str(i))),
                                     pd.read_pickle("serialized_df/trs_{}_labels_{}".format(variant, str(i)))))
            final_test_sets.append((pd.read_pickle("serialized_df/tss_{}_features_{}".format(variant, str(i))),
                                    pd.read_pickle("serialized_df/tss_{}_labels_{}".format(variant, str(i)))))

        # get experiment results
        candidates = list()
        for i in range(self.exp_rounds):
            mi = ModelIdentificationSpecific(*final_train_sets[i], *final_test_sets[i], cv_folds=5, verbose=True)
            mods_dict = {"ada": mi.ada, "gbc": mi.gbc, "hgb": mi.hgb, "bc": mi.bc}
            mods_dict[model]() if par is None else mods_dict[model](par)
            mi.model_selection(n=1000)
            candidates += mi.model_testing()

        # print all candidates
        print("\nFinal {} candidates".format(variant))
        for c in sorted(candidates, reverse=True, key=lambda x: x.auc):
            print(c)

        # group results by parameters for plotting
        candidates.sort(key=lambda x: x.pars[1])
        if len(candidates[0].pars) == 1:
            candidates = [candidates]
        elif len(candidates[0].pars) == 2:
            candidates2 = [[candidates[0]]]
            for i in range(1, len(candidates)):
                if candidates[i].pars[1] != candidates[i-1].pars[1]:
                    candidates2.append(list())
                candidates2[-1].append(candidates[i])
            candidates = candidates2

        # plotting bc its cool
        for category in candidates:
            category_num = list()
            for c in category:
                num_par = c.pars[0].split('=')[-1]
                num_par = float(num_par) if num_par not in ["True", "False"] else {"True": 1.0, "False": 0.0}[num_par]
                category_num.append(Candidate(c.model, c.auc, num_par, c.is_reg_model))
            category_num.sort(key=lambda x: x.pars)

            x, y = list(), list()
            for i in range(0, len(category_num), self.exp_rounds):
                x.append(category_num[i].pars)
                y.append(statistics.mean([category_num[j].auc for j in range(i, min(len(category_num), i+self.exp_rounds))]))
            fig, ax = plt.subplots()
            ax.plot(x, y)

            ax.set(xlabel=par, ylabel='AUC', title='Bigger is better ({}, {}, {})?'.format(model, variant, category[0].pars[-1]))
            ax.grid()
            plt.savefig("figures/{}-{}-{}-{}.png".format(variant, model, par, category[0].pars[-1]))
            plt.show()


def multi_proc():
    confs = [("bc", "max_features_uni",), ("bc", "max_features_ens",), ("bc", "n_estimators_uni",), ("bc", "n_estimators_ens",)]
    confs = [("ada", "n_estimators_ens",), ("bc", "n_estimators_ens",)]
    confs = [("ada", "n_estimators_uni",)]
    procs = [Process(target=SpecificIdentification(1, ("h1n1",), (c,)).main) for c in confs]

    for p in procs:
        p.start()

    for p in procs:
        p.join()


def uni_proc():
    SpecificIdentification(1, ("h1n1",), (("gbc", "n_estimators"),)).main()


if __name__ == '__main__':

    multi_proc()
