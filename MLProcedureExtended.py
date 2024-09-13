import statistics
from multiprocessing import Process

import pandas as pd
from matplotlib import pyplot as plt

from DataPreprocessing import DataPreprocessing
from ModelIdentificationExtended import ModelIdentificationExtended, SpecificCandidate


class MLProcedureExtended:
    """
        Alternative to model identification step for in-depth algorithms analysis
    """

    def __init__(self, exp_rounds, variants, models_pars=(("bcl", None),), dp_short=False):
        """

        :param exp_rounds: number of experiments to run
        :param variants: variants to tests (h1n1/seasonal)
        :param models_pars: name of experiment and eventual specification
        :param dp_short: whether to use short dataset for debug
        """

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
            mods_dict = {"nns": mi.nn_short, "nnl": mi.nn_long, "nn": mi.nn, "ada": mi.ada, "gbc": mi.gbc, "hgb": mi.hgb, "bc": mi.bc, "bcl": mi.bc_long}
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

    # UTILS #


def floatable(f):
    is_float = True
    try:
        float(f)
    except ValueError:
        is_float = False
    return is_float

    # RUN #


def multi_proc():
    confs = [("nn", "act_f"), ("nn", "solver"), ("nn", "hl1")]
    variants = ["h1n1",]
    procs = list()
    for v in variants:
        procs += [Process(target=MLProcedureExtended(3, (v,), (c,), False).main) for c in confs]

    for p in procs:
        p.start()

    for p in procs:
        p.join()


def uni_proc():
    confs = (("nnl", None),)
    variants = ("h1n1",)
    MLProcedureExtended(1, variants, confs, dp_short=False).main()


if __name__ == '__main__':

    uni_proc()
