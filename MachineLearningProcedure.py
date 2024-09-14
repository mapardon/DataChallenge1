import pickle
import statistics
from itertools import groupby
from multiprocessing import Process

import pandas as pd

from DataPreprocessing import DataPreprocessing
from ModelIdentification import ModelIdentification

PREPROC_SAVE = "preproc_config_save"
MODELS_SAVE = "models_trained_save"


class MachineLearningProcedure:
    """
        Main class initiating different steps of the complete procedure
    """

    def __init__(self, exp_rounds=5, variants=("h1n1", "flu"), steps=("pre", "id", "exp"), store=False,
                 dp_model="lm", mi_models=("lm",), short_ds=False):
        """
        :param exp_rounds: How many times the whole procedure must be run
        :param variants: Which variant to treat in the procedure
        :param steps: Procedure steps to be executed (can be executed independently due to storage of processed data)
        :param store: Store results (preprocessed data and experiments results)
        :param mi_models: Models to analyze for model identification phase
        :param short_ds: Consider short version of the dataset (debug purpose)
        """

        self.exp_rounds = exp_rounds
        self.variants = variants
        self.steps = steps
        self.store = store
        self.dp_model = dp_model
        self.mi_models = mi_models
        self.short_ds = short_ds

        self.preproc_features = None
        self.preproc_labels = None
        self.final_confs = {
            "h1n1": {
                "model": self.dp_model,
                "imp_num": None,
                "imp_obj": None,
                "out_detect": None,
                "scaler": None,
                "numerizer": None,
                "feat_selector": None,
                "selected_features": list()
             },
            "seas": {
                "model": self.dp_model,
                "imp_num": None,
                "imp_obj": None,
                "out_detect": None,
                "scaler": None,
                "numerizer": None,
                "feat_selector": None,
                "selected_features": list()
             }
        }
        self.final_models = {"h1n1": None, "seas": None}

    def main(self):
        procs = list()
        for variant in self.variants:
            if "pre" in self.steps:
                procs.append(Process(target=self.preprocessing, args=(variant,)))
            if "mi" in self.steps:
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
            dp = DataPreprocessing(self.short_ds, self.dp_model)
            dp.load_datasets("data/training_set_features.csv", "data/training_set_labels_{}.csv".format(variant))
            self.preproc_features, self.preproc_labels = dp.get_features_labels()

            # Imputation of missing values
            for imp_num in ["remove", "knn", "mean", "median"][1:]:
                for imp_obj in ["remove", "most_frequent"][1:]:
                    conf = [imp_num, imp_obj, default_nn, default_numerizer, default_scaler, default_feat_select]
                    best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(*conf)

                    if (imp_num, imp_obj) in imputation_res:
                        imputation_res[(imp_num, imp_obj)][2].append(best_models_perfs[0])
                    else:
                        imputation_res[(imp_num, imp_obj)] = [best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                           best_models_perfs]

            # Outliers detection
            for nn in [0, 2, 25]:
                conf = [default_imp_num, default_imp_obj, nn, default_numerizer, default_scaler, default_feat_select]
                best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(*conf)

                if nn in outliers_res:
                    outliers_res[nn][2].append(best_models_perfs[0])
                else:
                    outliers_res[nn] = [best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                        best_models_perfs]

            # Numeric features scaling
            for scaler in [None, "minmax"]:
                conf = [default_imp_num, default_imp_obj, default_nn, default_numerizer, scaler, default_feat_select]
                best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(*conf)

                if scaler in scaling_res:
                    scaling_res[scaler][2].append(best_models_perfs[0])
                else:
                    scaling_res[scaler] = [best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                           best_models_perfs]

            # Numerize categorical features & Features selection
            for numerizer in ["remove", "one-hot"]:
                for feat_select in [None, "mut_inf", "f_stat", "RFE"]:
                    conf = [default_imp_num, default_imp_obj, default_nn, numerizer, default_scaler, feat_select]
                    best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(*conf)

                    if (numerizer, feat_select) in feat_select_res:
                        feat_select_res[(numerizer, feat_select)][2].append(best_models_perfs[0])
                    else:
                        feat_select_res[(numerizer, feat_select)] = [best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                                                     best_models_perfs]

        # Change experiments results to list because we need order on the results
        imputation_res = sorted([imputation_res[k] for k in imputation_res], reverse=True, key=lambda x: x[2])
        outliers_res = sorted([outliers_res[k] for k in outliers_res], reverse=True, key=lambda x: x[2])
        scaling_res = sorted([scaling_res[k] for k in scaling_res], reverse=True, key=lambda x: x[2])
        feat_select_res = sorted([feat_select_res[k] for k in feat_select_res], reverse=True, key=lambda x: x[2])
        combination_res = sorted([combination_res[k] for k in combination_res], reverse=True, key=lambda x: x[2])

        # Attempt with combination of the best parameters value of previous experiments
        conf = [imputation_res[0][1][0], imputation_res[0][1][1], outliers_res[0][1][2], feat_select_res[0][1][4], scaling_res[0][1][5], feat_select_res[0][1][6]]
        best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(*conf)

        combination_res.append([best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out], best_models_perfs])

        # Display results
        for title, res in zip([" * Imputation of missing values", "\n * Outliers detection",
                               "\n * Numeric features scaling",
                               "\n * Numerize categorical features & Features selection",
                               "\n * Combination of best parameters"],
                              [imputation_res, outliers_res, scaling_res, feat_select_res, combination_res]):
            print(title)
            if "Combination" in title:  # for the experiment with all best other features: print configuration
                print("conf: ", conf)
            self.format_data_exp_output(res)

        # Store configuration having shown the highest performance average over experiments
        conf = self.final_confs["h1n1"] if variant == "h1n1" else self.final_confs["seas"]
        (conf["imp_num"], conf["imp_obj"], conf["out_detect"], _, conf["numerizer"],
         conf["scaler"], conf["feat_selector"], conf["selected_features"]) = sorted(
            imputation_res + outliers_res + scaling_res + feat_select_res + combination_res,
            reverse=True, key=lambda x: statistics.mean(x[2]) if len(x[2]) > 1 else x[2])[0][1]

        # TODO make struct-like object for configuration results?
        conf["selected_features"] = conf["selected_features"][2]  # retrieve features list and discard output info
        print("\n * Final configuration")
        print(", ".join(["{}: {}".format(k, self.final_confs[variant][k]) for k in self.final_confs[variant]]))

        if self.store:
            pickle.dump(self.final_confs[variant], open(PREPROC_SAVE + "_" + variant, "wb"))
            self.store_datasets(variant, self.final_confs[variant])

    def preprocessing_exp(self, imp_num, imp_obj, nn, numerizer, scaler, feat_selector):
        """
        Train a linear model with the provided preprocessing parameters to evaluate their influence on model performance

        :params: parameters for different preprocessing phases

        :return: list of models, str indicating output of operations for outlier detection/feature
            selection..., performance of the returned models on the validation phase
        """

        best_models, best_models_perfs, out_removed, corr_removed = list(), list(), list(), list()

        # Data preprocessing
        dp = DataPreprocessing(short=self.short_ds)
        ds = dp.training_preprocessing_pipeline(self.preproc_features.copy(deep=True), self.preproc_labels.copy(deep=True),
                                                imp_num, imp_obj, nn, numerizer, scaler, feat_selector)
        out_removed.append(dp.get_outlier_detection_res())
        corr_removed.append(dp.get_feature_selection_res())

        # Model identification and validation
        mi = ModelIdentification(*ds, cv_folds=5, verbose=False)
        mi.preprocessing_model_identification(self.dp_model)
        candidates = mi.model_testing()
        best_models.append(candidates[0].model)
        best_models_perfs.append(candidates[0].auc)

        # how many outliers/correlated features/useless features removed
        outlier_detect_res = "{}-{}".format(min(out_removed), max(out_removed))
        corr_removed_res = "{}-{}".format(min(corr_removed, key=lambda x: x[0])[0], max(corr_removed, key=lambda x: x[0])[0])
        feat_select_res = "{}-{}".format(min(corr_removed, key=lambda x: x[1])[1], max(corr_removed, key=lambda x: x[1])[1])
        selected_features = corr_removed[best_models_perfs.index(max(best_models_perfs))][2]

        return best_models, outlier_detect_res, (corr_removed_res, feat_select_res, selected_features), best_models_perfs

    @staticmethod
    def store_datasets(variant, conf):
        """ Apply preprocessing operations (conf) on dataset and store it on disk to allow future fast loading of
        preprocessed dataset """
        ds = DataPreprocessing().training_preprocessing_pipeline("data/training_set_features.csv",
                                                                 "data/training_set_labels.csv",
                                                                 conf["imp_num"], conf["imp_obj"], conf["out_detect"],
                                                                 conf["numerizer"], conf["scaler"], conf["selected_features"])

        features, labels = pd.concat([ds[0], ds[2]], axis="rows"), pd.concat([ds[1], ds[3]], axis="rows")
        features.to_pickle("serialized_df/features_{}".format(variant))
        labels.to_pickle("serialized_df/labels_{}".format(variant))
        features[:200].to_pickle("serialized_df/features_{}_short".format(variant))
        labels[:200].to_pickle("serialized_df/labels_{}_short".format(variant))

    @staticmethod
    def format_data_exp_output(conf_perf):
        """ format of conf_perf : [[(m1, m2), (p1, p2, p3), (perf1, perf2)], [...]] """
        for models, conf, perf in conf_perf:
            print("imp_num={}, imp_obj={}, knn={}(#rem.:{}), numerizer={}, scaler={}, feat_select={}(#corr_rem.:{}/#select:{})".format(
                conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6], conf[7][0], conf[7][1]))
            if len(perf) == 1:
                print("\t-> score: {}".format(round(perf, 5)))
            else:
                print("\t-> avg: {}, stdev: {}".format(round(statistics.mean(perf), 5), round(statistics.stdev(perf), 5)))

    # MODEL IDENTIFICATION

    def model_identification(self, variant):
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
            for m in self.mi_models:
                mi = ModelIdentification(train_features, train_labels, test_features, test_labels, cv_folds=10, verbose=True)
                mi.model_identification((m,))
                mi.model_selection()
                candidates += mi.model_testing()

        candidates = [list(g) for _, g in groupby(sorted(candidates, key=lambda x: str(x.model)), lambda x: str(x.model))]
        candidates = sorted([sorted(l, reverse=True, key=lambda c: c.auc) for l in candidates], reverse=True, key=lambda x: statistics.mean(c.auc for c in x))

        # Display results
        MachineLearningProcedure.format_model_exp_output(variant, candidates)

        self.final_models[variant] = candidates[0][0].model
        print(candidates[0][0])

        if self.store:
            pickle.dump(self.final_models[variant], open(MODELS_SAVE + "_" + variant, "wb"))

    @staticmethod
    def format_model_exp_output(variant, candidates_lists):
        """ :param candidates_lists: list of lists of candidates (several experiments with the same estimator) """
        print("\n * Final {} candidates".format(variant))
        for l in candidates_lists:
            tmp_auc = [c.auc for c in l]
            print("model: {}".format(str(l[0].model)))
            print("\tPerformance (AUC) -> avg: {}, stdev: {}".format(round(statistics.mean(tmp_auc), 5),
                                                                   round(statistics.stdev(tmp_auc), 5)))

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
            model = pickle.load(open("models_trained_save_{}".format(variant), 'rb'))

            # Pre-process challenge data
            dp = DataPreprocessing()
            resp_id, features = dp.challenge_preprocessing_pipeline("data/test_set_features.csv",
                                                                    conf["imp_num"], conf["imp_obj"], conf["out_detect"],
                                                                    conf["numerizer"], conf["scaler"],
                                                                    conf["selected_features"])

            # Use previously trained model on processed challenge data
            out["id"] = resp_id
            out[variant] = ModelIdentification.model_exploitation(model, features)

        pd.DataFrame({
            "respondent_id": out["id"],
            "h1n1_vaccine": out["h1n1"],
            "seasonal_vaccine": out["seas"]
        }).to_csv("data/submission.csv", index=False)
