import pickle
import statistics
from itertools import groupby
from multiprocessing import Process
from operator import attrgetter

import pandas as pd
from matplotlib import pyplot as plt

from DataPreprocessing import DataPreprocessing
from ModelIdentification import ModelIdentification, ExperimentResultMi
from ModelSelection import ModelSelection

PREPROC_SAVE = "preproc_config_save_{}"
MODELS_SAVE = "models_trained_save_{}"


class PreprocCandidate:
    def __init__(self, dp_model=None, auc=None, imp_num=None, imp_obj=None, out_detect=None, out_detect_res=None,
                 scaler=None, numerizer=None, feat_selector=None, corr_features_out=None, selected_features=None):
        self.dp_model = dp_model
        self.auc = auc

        self.imp_num = imp_num
        self.imp_obj = imp_obj
        self.out_detect = out_detect
        self.out_detect_res = out_detect_res
        self.scaler = scaler
        self.numerizer = numerizer
        self.feat_selector = feat_selector
        self.corr_features_out = corr_features_out
        self.selected_features = selected_features

    def __repr__(self):
        return "model={}, imp_num={}, imp_obj={}, knn={}, numerizer={}, scaler={}, feat_select={}".format(
            str(self.dp_model), self.imp_num, self.imp_obj, self.out_detect, self.numerizer, self.scaler,
            self.feat_selector)


class MachineLearningProcedure:
    """
        Main class initiating different steps of the complete procedure
    """

    def __init__(self, exp_rounds=5, variants=("h1n1", "flu"), steps=("pre", "id", "exp"), store=False,
                 dp_model_tag="lm", mi_pars=(("gbc", "*"),), ms_models=("lm",), short_ds=False):
        """
        :param exp_rounds: How many times the whole procedure must be run
        :param variants: Which variant to treat in the procedure
        :param steps: Procedure steps to be executed (can be executed independently due to storage of processed data)
        :param store: Store results (preprocessed data and experiments results)
        :param dp_model_tag: Short name to indicate which model to use for preprocessing experiments
        :param mi_pars: Parameters for model identification experiments
        :param ms_models: Models to analyze for model identification phase
        :param short_ds: Consider short version of the dataset (debug purpose)
        """

        self.exp_rounds = exp_rounds
        self.variants = variants
        self.steps = steps
        self.store = store
        self.short_ds = short_ds

        # preproc
        self.dp_model_tag = dp_model_tag
        self.preproc_features = None
        self.preproc_labels = None
        self.final_confs = {"h1n1": PreprocCandidate(), "seas": PreprocCandidate()}

        # model identification
        self.mi_pars = mi_pars

        # model selection
        self.ms_models = ms_models
        self.final_models = {"h1n1": None, "seas": None}

    def main(self):
        procs = list()
        if "pre" in self.steps:
            for variant in self.variants:
                procs.append(Process(target=self.preprocessing, args=(variant,)))
            for p in procs: p.start()
            for p in procs: p.join()

        procs = list()
        if "ms" in self.steps:
            for variant in self.variants:
                procs.append(Process(target=self.model_selection, args=(variant,)))
            for p in procs: p.start()
            for p in procs: p.join()

        procs = list()
        if "mi" in self.steps:
            for variant in self.variants:
                procs.append(Process(target=self.model_identification, args=(variant,)))
            for p in procs: p.start()
            for p in procs: p.join()

        if "exp" in self.steps:
            self.model_exploitation()

    # PREPROCESSING

    def preprocessing(self, variant):
        """
            First we evaluate the influence of different preprocessing parameters in order to preprocess the final dataset
        """

        print("Preprocessing - {}".format(variant))

        default_imp_num = "median"
        default_imp_obj = "most_frequent"
        default_nn = int()
        default_scaler = None
        default_numerizer = "remove"
        default_feat_select = None

        imputation_res = dict()
        outliers_res = dict()
        scaling_res = dict()
        feat_select_res = dict()
        combination_res = dict()

        for i in range(self.exp_rounds):

            # init data for experiments
            dp = DataPreprocessing(self.short_ds, self.dp_model_tag)
            dp.load_datasets("data/training_set_features.csv", "data/training_set_labels_{}.csv".format(variant))
            self.preproc_features, self.preproc_labels = dp.get_features_labels()

            # Imputation of missing values
            for imp_num in ["remove", "knn", "mean", "median"][1:]:
                for imp_obj in ["remove", "most_frequent"][1:]:
                    pc = PreprocCandidate(imp_num=imp_num, imp_obj=imp_obj, out_detect=default_nn, numerizer=default_numerizer, scaler=default_scaler, feat_selector=default_feat_select)
                    self.preprocessing_exp(pc)

                    if (imp_num, imp_obj) in imputation_res:
                        imputation_res[(imp_num, imp_obj)].append(pc)
                    else:
                        imputation_res[(imp_num, imp_obj)] = [pc]

            # Outliers detection
            for nn in [0, 2, 25]:
                pc = PreprocCandidate(imp_num=default_imp_num, imp_obj=default_imp_obj, out_detect=nn, numerizer=default_numerizer, scaler=default_scaler, feat_selector=default_feat_select)
                self.preprocessing_exp(pc)

                if nn in outliers_res:
                    outliers_res[nn].append(pc)
                else:
                    outliers_res[nn] = [pc]

            # Numeric features scaling
            for scaler in [None, "minmax"]:
                pc = PreprocCandidate(imp_num=default_imp_num, imp_obj=default_imp_obj, out_detect=default_nn, numerizer=default_numerizer, scaler=scaler, feat_selector=default_feat_select)
                self.preprocessing_exp(pc)

                if scaler in scaling_res:
                    scaling_res[scaler].append(pc)
                else:
                    scaling_res[scaler] = [pc]

            # Numerize categorical features & Features selection
            for numerizer in ["remove", "one-hot"]:
                for feat_select in [None, "mut_inf", "f_stat", "RFE"]:
                    pc = PreprocCandidate(imp_num=default_imp_num, imp_obj=default_imp_obj, out_detect=default_nn, numerizer=numerizer, scaler=default_scaler, feat_selector=feat_select)
                    self.preprocessing_exp(pc)

                    if (numerizer, feat_select) in feat_select_res:
                        feat_select_res[(numerizer, feat_select)].append(pc)
                    else:
                        feat_select_res[(numerizer, feat_select)] = [pc]

        # Change experiments results to list because we need order on the results
        imputation_res = sorted([imputation_res[k] for k in imputation_res], reverse=True, key=lambda x: statistics.mean([c.auc for c in x]))
        outliers_res = sorted([outliers_res[k] for k in outliers_res], reverse=True, key=lambda x: statistics.mean([c.auc for c in x]))
        scaling_res = sorted([scaling_res[k] for k in scaling_res], reverse=True, key=lambda x: statistics.mean([c.auc for c in x]))
        feat_select_res = sorted([feat_select_res[k] for k in feat_select_res], reverse=True, key=lambda x: statistics.mean([c.auc for c in x]))
        combination_res = sorted([combination_res[k] for k in combination_res], reverse=True, key=lambda x: statistics.mean([c.auc for c in x]))

        # Attempt with combination of the best parameters value of previous experiments
        pc = PreprocCandidate(imp_num=imputation_res[0][0].imp_num, imp_obj=imputation_res[0][0].imp_obj,
                              out_detect=outliers_res[0][0].out_detect, scaler=scaling_res[0][0].scaler,
                              numerizer=feat_select_res[0][0].numerizer, selected_features=feat_select_res[0][0].selected_features)
        self.preprocessing_exp(pc)
        combination_res.append([pc])

        # Display results
        for title, res in zip([" * Imputation of missing values", "\n * Outliers detection",
                               "\n * Numeric features scaling",
                               "\n * Numerize categorical features & Features selection",
                               "\n * Combination of best parameters"],
                              [imputation_res, outliers_res, scaling_res, feat_select_res, combination_res]):
            print(title)
            self.format_data_exp_output(res)

        # Store configuration having shown the highest performance average over experiments
        self.final_confs[variant] = max(imputation_res + outliers_res + scaling_res + feat_select_res + combination_res,
                                        key=lambda x: statistics.mean([c.auc for c in x]) if len(x) > 1 else x[0].auc)[0]

        print("\n * Final configuration")
        print(self.final_confs[variant], self.final_confs[variant].selected_features)

        if self.store:
            pickle.dump(self.final_confs[variant], open(PREPROC_SAVE.format(variant), "wb"))
            self.store_datasets(variant, self.final_confs[variant])

    def preprocessing_exp(self, preproc_conf):
        """
        Train a linear model with the provided preprocessing parameters to evaluate their influence on model performance

        :params: parameters for different preprocessing phases

        :return: the object is accessed by reference (normally)
        """

        # Data preprocessing
        pc = preproc_conf
        dp = DataPreprocessing(short=self.short_ds)
        ds = dp.training_preprocessing_pipeline(self.preproc_features.copy(deep=True), self.preproc_labels.copy(deep=True),
                                                pc.imp_num, pc.imp_obj, pc.out_detect, pc.numerizer, pc.scaler, pc.feat_selector)
        # how many outliers/correlated features/useless features are removed
        pc.out_detect_res = dp.get_outlier_detection_res()
        pc.corr_features_out, _, pc.selected_features = dp.get_feature_selection_res()

        # Model identification and validation
        mi = ModelSelection(*ds, cv_folds=5, verbose=False)
        mi.preprocessing_model_identification(self.dp_model_tag)
        candidates = mi.model_testing()  # 1 single candidate is returned

        pc.dp_model = candidates[0].model
        pc.auc = candidates[0].test_auc

    @staticmethod
    def store_datasets(variant, conf):
        """ Apply preprocessing operations (conf) on dataset and store it on disk to allow future fast loading of
        preprocessed dataset """
        ds = DataPreprocessing().training_preprocessing_pipeline("data/training_set_features.csv",
                                                                 "data/training_set_labels.csv",
                                                                 conf.imp_num, conf.imp_obj, conf.out_detect,
                                                                 conf.numerizer, conf.scaler, conf.selected_features)

        features, labels = pd.concat([ds[0], ds[2]], axis="rows"), pd.concat([ds[1], ds[3]], axis="rows")
        features.to_pickle("serialized_df/features_{}".format(variant))
        labels.to_pickle("serialized_df/labels_{}".format(variant))
        features[:200].to_pickle("serialized_df/features_{}_short".format(variant))
        labels[:200].to_pickle("serialized_df/labels_{}_short".format(variant))

    @staticmethod
    def format_data_exp_output(preproc_candidates):
        """ :param preproc_candidates: list of lists of PreprocCandidates (all candidates of different
        configurations) """

        for pc in preproc_candidates:
            print(pc[0])
            if len(pc) == 1:
                print("\t-> score: {}".format(round(pc[0].auc, 5)))
            else:
                aucs = [pc.auc for pc in pc]
                outliers_removed = "outliers_removed: {}/{}".format(min(pc, key=attrgetter('out_detect_res')).out_detect_res,
                                                                    max(pc, key=attrgetter('out_detect_res')).out_detect_res)
                corr_features_removed = "corr_features_removed: {}/{}".format(min(pc, key=attrgetter('corr_features_out')).corr_features_out,
                                                                              max(pc, key=attrgetter('corr_features_out')).corr_features_out)
                n_features_final = "final_#_features {}/{}".format(len(min(pc, key=lambda x: len(x.selected_features)).selected_features),
                                                                   len(max(pc, key=lambda x: len(x.selected_features)).selected_features))
                print("\t-> avg: {}, stdev: {}".format(round(statistics.mean(aucs), 5), round(statistics.stdev(aucs), 5)),
                      outliers_removed, corr_features_removed, n_features_final, sep=", ")

    # MODEL SELECTION

    def model_selection(self, variant):
        """
            Using the datasets pre-processed with the previously chosen configuration, we now test the performance of
            different learning algorithms
        """

        f1, f2 = ("serialized_df/features_{}".format(variant), "serialized_df/labels_{}".format(variant))
        if self.short_ds:
            f1, f2 = f1 + "_short", f2 + "_short"

        candidates = list()
        for i in range(self.exp_rounds):
            # Load & reshuffle preprocessed dataset
            features, labels = pd.read_pickle(f1), pd.read_pickle(f2)
            dp = DataPreprocessing(self.short_ds)
            dp.shuffle_datasets(features, pd.DataFrame(labels))
            train_features, train_labels, test_features, test_labels = dp.get_train_test_datasets()

            # Train models with CV and test performance on unused test set
            ms = ModelSelection(train_features, train_labels, test_features, test_labels, cv_folds=10, verbose=True)
            for m in self.ms_models:
                ms.model_identification((m,))
            ms.model_selection()
            candidates += ms.model_testing()

        candidates = [list(g) for _, g in groupby(sorted(candidates, key=lambda x: str(x.model)), lambda x: str(x.model))]
        candidates = sorted([sorted(l, reverse=True, key=lambda c: c.test_auc) for l in candidates], reverse=True, key=lambda x: statistics.mean(c.test_auc for c in x))

        # Display results
        MachineLearningProcedure.format_model_exp_output(variant, candidates)

        self.final_models[variant] = candidates[0][0]

        if self.store:
            pickle.dump(self.final_models[variant], open(MODELS_SAVE.format(variant), "wb"))

    @staticmethod
    def format_model_exp_output(variant, candidates_lists):
        """ :param candidates_lists: list of lists of candidates (several experiments with the same estimator) """
        print("\n * Final {} candidates".format(variant))
        for l in candidates_lists:
            tmp_auc = [c.test_auc for c in l]
            print("model: {}".format(str(l[0].model)))
            print("\tPerformance (AUC) -> avg: {}, stdev: {}, min: {}, max: {}".format(
                round(statistics.mean(tmp_auc), 5), round(statistics.stdev(tmp_auc), 5), round(min(tmp_auc), 5), round(max(tmp_auc), 5)))

    # MODEL IDENTIFICATION

    def model_identification(self, variant):
        f1, f2 = ("serialized_df/features_{}".format(variant) + "_short" * self.short_ds,
                  "serialized_df/labels_{}".format(variant) + "_short" * self.short_ds)

        res = dict()
        for i in range(self.exp_rounds):
            # Load & reshuffle preprocessed dataset
            # TODO: optimize to avoid reloading each round
            features, labels = pd.read_pickle(f1), pd.read_pickle(f2)
            dp = DataPreprocessing(self.short_ds)
            dp.shuffle_datasets(features, pd.DataFrame(labels))
            train_features, train_labels, test_features, test_labels = dp.get_train_test_datasets()

            # Train models with CV and test performance on unused test set
            for model, par in self.mi_pars:
                mi = ModelIdentification(train_features, train_labels, test_features, test_labels, cv_folds=5, verbose=True)
                if model in res:
                    if par in res[model]:
                        res[model][par] += mi.model_identification(model, par)
                    else:
                        res[model][par] = mi.model_identification(model, par)
                else:
                    res[model] = dict()
                    res[model][par] = mi.model_identification(model, par)

        # split results by parameter value (easier manipulation to plot results)
        res2 = dict()
        for m in res:
            res2[m] = dict()
            for par in res[m]:
                res2[m][par] = dict()
                vals = set([r.par_value for r in res[m][par]])
                for v in vals:
                    res2[m][par][v] = list()
                    for r in res[m][par]:
                        if r.par_value == v:
                            res2[m][par][v].append(r)
        res = res2

        # print all candidates
        print("\n{} Model identification results".format(variant))
        for m in res:
            for par in res[m]:
                print("\n * Experiment: {} - {}".format(m, par))
                tmp_res = sorted(res[m][par].items(), reverse=True, key=lambda x: statistics.mean([statistics.mean(r.mi_auc) for r in x[1]]))
                for par_val, results in tmp_res:
                    tmp_auc = [statistics.mean(r.mi_auc) for r in results]
                    print("model: {} {}={}".format(m, par, par_val))
                    print("\tPerformance (AUC) -> avg: {}, stdev: {}, min: {}, max: {}".format(
                        round(statistics.mean(tmp_auc), 5), round(statistics.stdev(tmp_auc), 5), round(min(tmp_auc), 5),
                        round(max(tmp_auc), 5)))

        for m in res:
            for p in res[m]:
                self.plot_results(res[m][p], variant)

    def plot_results(self, results: dict, variant: str):

        ex = list(results.items())[0][1][0]
        plot = "plot" if floatable(ex.par_value) else "hist"
        results = sorted(results.items(), key=lambda x: x[0])
        x, y = list(), list()
        for par, res in results:
            x.append(par)
            y.append(statistics.mean([statistics.mean(r.mi_auc) for r in res]))
        fig, ax = plt.subplots()

        if plot == "hist":
            ax.bar(x, y)
        elif plot == "plot":
            ax.plot(x, y)

        title = 'Experiment: {}, {}{}, {}'.format(variant, ex.model_tag, " ({})".format(ex.estimator_tag) if ex.is_bag else str(), ex.par_tag)
        ax.set(xlabel=ex.par_tag, ylabel='AUC', title=title)
        ax.grid()
        if self.store:
            plt.savefig("figures/{}-{}{}-{}.png".format(variant, ex.model_tag, '-' + ex.estimator_tag if ex.is_bag else str(), ex.par_tag))
        else:
            plt.show()

    # MODEL EXPLOITATION

    def model_exploitation(self):
        """ Finally, we use the models and preprocessing parameters having shown the best performance during training
        to predict challenge data """

        out = {
            "id": None,
            "h1n1": None,
            "seas": None
        }

        for variant in ["h1n1", "seas"]:
            conf = pickle.load(open("preproc_config_save_{}".format(variant), 'rb'))
            model = pickle.load(open("models_trained_save_{}".format(variant), 'rb')).model

            # Pre-process challenge data
            dp = DataPreprocessing()
            resp_id, features = dp.challenge_preprocessing_pipeline("data/test_set_features.csv",
                                                                    conf.imp_num, conf.imp_obj, conf.out_detect,
                                                                    conf.numerizer, conf.scaler, conf.selected_features)

            # Use previously trained model on processed challenge data
            out["id"] = resp_id
            out[variant] = ModelSelection.model_exploitation(model, features)

        pd.DataFrame({
            "respondent_id": out["id"],
            "h1n1_vaccine": out["h1n1"],
            "seasonal_vaccine": out["seas"]
        }).to_csv("data/submission.csv", index=False)

        print("\n * Model exploitation complete")

    # UTILS #


def floatable(f):
    is_float = True
    try:
        float(f)
    except ValueError:
        is_float = False
    return is_float
