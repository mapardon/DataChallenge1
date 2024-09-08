import pickle
import statistics
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

    def __init__(self, exp_rounds=5, variant=("h1n1", "flu"), steps=("pre", "id", "exp"), store=False, mi_models=("lm",), dp_short=False):
        self.exp_rounds = exp_rounds
        self.final_confs = {
            "h1n1": {
                "imp_num": None,
                "imp_obj": None,
                "out_detect": None,
                "scaler": None,
                "numerizer": None,
                "feat_selector": None,
                "selected_features": list()
             },
            "seas": {
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

        self.variant = variant
        self.steps = steps
        self.store = store
        self.mi_models = mi_models
        self.dp_short = dp_short

    def main(self):
        procs = list()
        for variant in self.variant:
            if "pre" in self.steps:
                procs.append(Process(target=self.preprocessing, args=(variant,)))
            if "mi" in self.steps:
                procs.append(Process(target=self.model_identification, args=(variant,)))

        for p in procs:
            p.start()

        for p in procs:
            p.join()

        if "exp" in self.steps:
            self.exploitation_loop()

    def preprocessing(self, variant):
        """
            First we evaluate the influence of different preprocessing parameters (in combination of a fast-training
            model) in order to preprocess the final dataset
        """

        print("Preprocessing - {}".format(variant))

        default_imp_num = "median"
        default_imp_obj = "most_frequent"
        default_nn = int()
        default_scaler = None
        default_numerizer = "remove"
        default_feat_select = None

        # Imputation of missing values
        print(" * Imputation of missing values")
        imputation_res = list()
        for imp_num in ["remove", "knn", "mean", "median"][1:]:
            for imp_obj in ["remove", "most_frequent"][1:]:
                conf = [imp_num, imp_obj, default_nn, default_numerizer, default_scaler, default_feat_select]
                best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(*conf, variant)

                imputation_res.append([best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                       best_models_perfs])

        imputation_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(imputation_res)

        # Outliers detection
        print("\n * Outliers detection")
        outliers_res = list()
        for nn in [0, 2, 25]:
            conf = [default_imp_num, default_imp_obj, nn, default_numerizer, default_scaler, default_feat_select]
            best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(*conf, variant)

            outliers_res.append([best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                 best_models_perfs])

        outliers_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(outliers_res)

        # Numeric features scaling
        print("\n * Numeric features scaling")
        scaling_res = list()
        for scaler in ["minmax"]:
            conf = [default_imp_num, default_imp_obj, default_nn, default_numerizer, scaler, default_feat_select]
            best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(*conf, variant)

            scaling_res.append([best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                best_models_perfs])

        scaling_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(scaling_res)

        # Numerize categorical features & Features selection
        print("\n * Numerize categorical features & Features selection")
        feat_select_res = list()
        for numerizer in ["remove", "one-hot"]:
            for feat_select in [None, "mut_inf", "f_stat"]:
                conf = [default_imp_num, default_imp_obj, default_nn, numerizer, default_scaler, feat_select]
                best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(*conf, variant)

                feat_select_res.append([best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                        best_models_perfs])

        feat_select_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(feat_select_res)

        # Attempt with combination of the best parameters value of previous experiments
        print("\n * Combination of best parameters")
        conf = [imputation_res[0][1][0], imputation_res[0][1][1], outliers_res[0][1][2], feat_select_res[0][1][4], scaling_res[0][1][5], feat_select_res[0][1][6]]
        print(conf)
        best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(*conf, variant)

        combination_res = [[best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out], best_models_perfs]]
        self.format_data_exp_output(combination_res)

        # Store configuration having shown the highest performance average over experiments
        conf = self.final_confs["h1n1"] if variant == "h1n1" else self.final_confs["seas"]
        (conf["imp_num"], conf["imp_obj"], conf["out_detect"], _, conf["numerizer"],
         conf["scaler"], _, conf["selected_features"]) = sorted(
            imputation_res + outliers_res + scaling_res + feat_select_res + combination_res,
            reverse=True, key=lambda x: statistics.mean(x[2]))[0][1]
        conf["selected_features"] = conf["selected_features"][2]  # retrieve features list
        print("\n * Final configuration")
        print(", ".join(["{}: {}".format(k, self.final_confs[variant][k]) for k in self.final_confs[variant]]))

        pickle.dump(self.final_confs, open(PREPROC_SAVE + "_" + variant, "wb"))

        if self.store:
            self.store_datasets(variant, conf)

    def preprocessing_exp(self, imp_num, imp_obj, nn, numerizer, scaler, feat_selector, variant):
        """
        Train a linear model with the provided preprocessing parameters to evaluate their influence on model performance

        :params: parameters for different preprocessing phases

        :return: list of models, str indicating output of operations for outlier detection/feature
            selection..., performance of the returned models on the validation phase
        """

        best_models, best_models_perfs, out_removed, corr_removed = list(), list(), list(), list()

        for _ in range(self.exp_rounds):

            # Data preprocessing
            dp = DataPreprocessing(short=self.dp_short)
            ds = dp.training_preprocessing_pipeline("data/training_set_features.csv",
                                                    "data/training_set_labels_{}.csv".format(variant),
                                                    imp_num, imp_obj, nn, numerizer, scaler, feat_selector)
            out_removed.append(dp.get_outlier_detection_res())
            corr_removed.append(dp.get_feature_selection_res())

            # Model identification and validation
            mi = ModelIdentification(*ds, cv_folds=5)
            mi.lm()
            mi.model_selection()
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
        # Prepare datasets with parameters previously defined to avoid recomputing them unnecessarily
        ds = DataPreprocessing().training_preprocessing_pipeline("data/training_set_features.csv",
                                                                 "data/training_set_labels.csv",
                                                                 conf["imp_num"], conf["imp_obj"], conf["out_detect"],
                                                                 conf["numerizer"], conf["scaler"], conf["selected_features"])

        features, labels = pd.concat([ds[0], ds[2]], axis="rows"), pd.concat([ds[1], ds[3]], axis="rows")
        features.to_pickle("serialized_df/features_{}".format(variant))
        labels.to_pickle("serialized_df/labels_{}".format(variant))

    @staticmethod
    def format_data_exp_output(conf_perf):
        """ format of conf_perf : [[(m1, m2), (p1, p2, p3), (perf1, perf2)], [...]] """
        for models, conf, perf in conf_perf:
            print("imp_num={}, imp_obj={}, knn={}(#rem.:{}), numerizer={}, scaler={}, feat_select={}(#corr_rem.:{}/#select:{})".format(
                conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6], conf[7][0], conf[7][1]))
            print("\t-> avg: {}, stdev: {}".format(round(statistics.mean(perf), 5), round(statistics.stdev(perf), 5)))

    def model_identification(self, variant):
        """
            Using the pre-processed datasets with the previously chosen parameters, we now test the performance of
            different learning algorithms
        """

        f1, f2 = ("serialized_df/features_{}".format(variant), "serialized_df/labels_{}".format(variant))
        if self.dp_short:
            f1, f2 = f1 + "_short", f2 + "_short"

        candidates = list()
        for i in range(self.exp_rounds):
            # load & reshuffle preprocessed dataset
            features, labels = pd.read_pickle(f1), pd.read_pickle(f2)
            dp = DataPreprocessing(self.dp_short)
            dp.shuffle_datasets(features, labels)
            train_features, train_labels, test_features, test_labels = dp.get_train_test_datasets()

            # Train models with CV and test performance on unused test set
            mi = ModelIdentification(train_features, train_labels, test_features, test_labels, cv_folds=5, verbose=True)
            mi.model_identification(self.mi_models)
            mi.model_selection()
            candidates += mi.model_testing()

        print("\nFinal {} candidates".format(variant))
        for c in sorted(candidates, reverse=True, key=lambda x: x.auc):
            print(c)

        self.final_models[variant] = candidates[0].model

        if self.store:
            pickle.dump(self.final_models, open(MODELS_SAVE + "_" + variant, "wb"))

    @staticmethod
    def format_model_exp_output(models_perf):
        for models, perfs in models_perf:
            for m in models:
                print("{}: {}".format(m[0], m[1]))
            print("Performance (ROC) -> avg: {}, stdev: {}".format(round(statistics.mean(perfs), 5),
                                                                   round(statistics.stdev(perfs), 5)))

    def exploitation_loop(self):
        """ Finally, we use the models and preprocessing parameters having shown the best performance during training
        to predict challenge data """

        self.final_confs = pickle.load(open("preproc_config_save", 'rb'))
        self.final_models = pickle.load(open("models_trained_save", 'rb'))

        out = {
            "id": None,
            "h1n1": None,
            "seas": None
        }

        for variant in ["h1n1", "seas"]:
            conf = self.final_confs[variant]

            # Must predict all challenge tuples
            final_imp_num = conf["imp_num"] if conf["imp_num"] != "remove" else "median"
            final_imp_obj = conf["imp_obj"] if conf["imp_obj"] != "remove" else "most_frequent"

            # Pre-process challenge data
            dp = DataPreprocessing()
            resp_id, features = dp.challenge_preprocessing_pipeline("data/test_set_features.csv",
                                                                    final_imp_num, final_imp_obj, conf["out_detect"],
                                                                    conf["numerizer"], conf["scaler"],
                                                                    conf["selected_features"])

            # Use previously trained model on processed challenge data
            out["id"] = resp_id
            out[variant] = ModelIdentification.model_exploitation(self.final_models[variant], features)

        pd.DataFrame({
            "respondent_id": out["id"],
            "h1n1_vaccine": out["h1n1"],
            "seasonal_vaccine": out["seas"]
        }).to_csv("data/submission.csv", index=False)
