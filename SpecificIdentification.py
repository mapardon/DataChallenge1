import statistics
from multiprocessing import Process

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

from ModelIdentification import ModelIdentification


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
                self.candidates.append((gbc, auc, "n_estimators={}".format(n), False))

        elif par == "subsample":
            for s in [0.1, 0.5, 0.75, 0.9, 1.0]:
                gbc = GradientBoostingClassifier(loss="log_loss", subsample=s)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append((gbc, auc, "subsample={}".format(s), False))

        elif par == "min_sample_split":
            for mss in [2, 3, 5, 10]:
                gbc = GradientBoostingClassifier(loss="log_loss", min_samples_split=mss)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append((gbc, auc, "min_sample_split={}".format(mss), False))

        elif par == "max_depth":
            for max_depth in [2, 3, 4, 5, 10, 20]:
                gbc = GradientBoostingClassifier(loss="log_loss", max_depth=max_depth)
                auc = self.parametric_identification_cv(gbc, False)
                self.candidates.append((gbc, auc, "max_depth={}".format(max_depth), False))

        elif par == "init":
            pass


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

        candidates = list()
        for i in range(self.exp_rounds):
            mi = ModelIdentificationSpecific(*final_train_sets[i], *final_test_sets[i], cv_folds=5, verbose=True)
            mods_dict = {"ada": mi.ada, "gbc": mi.gbc}
            mods_dict[model]() if par is None else mods_dict[model](par)
            mi.model_selection(n=1000)
            candidates += mi.model_testing()

        print("\nFinal {} candidates".format(variant))
        for c in sorted(candidates, reverse=True, key=lambda x: x[1]):
            print(c)

        # plotting bc its cool
        candidates_num = list()
        for c in candidates:
            c = c[:2] + tuple([float(c[2].split('=')[1])]) + c[3:]
            candidates_num.append(c)
        candidates_num.sort(key=lambda x: x[2])

        print(candidates_num)
        x, y = list(), list()
        for i in range(0, len(candidates_num), self.exp_rounds):
            x.append(candidates_num[i][2])
            y.append(statistics.mean([candidates_num[j][1] for j in range(i, min(len(candidates_num), i+self.exp_rounds))]))
        print(x, y)
        fig, ax = plt.subplots()
        ax.plot(x, y)

        ax.set(xlabel=candidates[0][2].split('=')[0], ylabel='AUC', title='Bigger is better ({}, {})?'.format(model, variant))
        ax.grid()
        plt.show()
        #plt.savefig("{}-{}-{}.png".format(variant, model, par))


def multi_proc():
    confs = [("gbc", "n_estimators",), ("gbc", "subsample",), ("gbc", "min_sample_split",), ("gbc", "max_depth",)]
    procs = [Process(target=SpecificIdentification(5, ("h1n1",), (c,)).main) for c in confs]

    for p in procs:
        p.start()

    for p in procs:
        p.join()


def uni_proc():
    SpecificIdentification(5, ("h1n1",), (("gbc", "min_sample_split"),)).main()


if __name__ == '__main__':

    uni_proc()