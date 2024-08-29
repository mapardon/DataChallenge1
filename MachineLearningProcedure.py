import statistics

import pandas as pd

from DataPreprocessing import DataPreprocessing
from ModelIdentification import ModelIdentification


class MachineLearningProcedure:
    """
        Main class initiating different steps of the complete procedure
    """

    def __init__(self):
        self.exp_rounds = 5
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

    def main(self):
        for variant in ["h1n1", "seas"][:1]:
            self.preprocessing(variant)
            self.model_identification(variant)
        #self.exploitation_loop()

        res0, res1, res2 = int(), int(), int()
        for i in range(5):
            tmp = pd.read_pickle("serialized_df/trs_{}_features_{}".format("h1n1", str(i)))
            pmt = pd.read_pickle("serialized_df/tss_{}_features_{}".format("h1n1", str(i)))

            res0 += tmp.shape[0] + pmt.shape[0]
            res1 += tmp.shape[0]
            res2 += pmt.shape[0]

        print(res0, res0/5, res1, res2)

    def preprocessing(self, variant):
        """
            First we evaluate the influence of different preprocessing parameters (in combination of a fast-training
            model) in order to preprocess the final dataset
        """

        print("Preprocessing - {}".format(variant))

        default_imp_num = "median"
        default_imp_obj = "median"
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
                best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf, variant)

                imputation_res.append([best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                       best_models_perfs])

        imputation_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(imputation_res)

        # Outliers detection
        print("\n * Outliers detection")
        outliers_res = list()
        for nn in [0, 2, 25]:
            conf = [default_imp_num, default_imp_obj, nn, default_numerizer, default_scaler, default_feat_select]
            best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf, variant)

            outliers_res.append([best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                 best_models_perfs])

        outliers_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(outliers_res)

        # Numerize categorical features
        # TODO: keep?
        print("\n * Numerize categorical features")
        numerize_res = list()
        for numerizer in ["remove", "one-hot"]:
            conf = [default_imp_num, default_imp_obj, default_nn, numerizer, default_scaler, default_feat_select]
            best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf, variant)

            numerize_res.append([best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                 best_models_perfs])

        numerize_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(numerize_res)

        # Numeric features scaling
        print("\n * Numeric features scaling")
        scaling_res = list()
        for scaler in ["minmax"]:
            conf = [default_imp_num, default_imp_obj, default_nn, default_numerizer, scaler, default_feat_select]
            best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf, variant)

            scaling_res.append([best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                best_models_perfs])

        scaling_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(scaling_res)

        # Features selection
        print("\n * Features selection")
        feat_select_res = list()
        for feat_select, numerizer in [(None, default_numerizer), ("mut_inf", "remove"), ("mut_inf", "one-hot"), ("f_stat", "remove"), ("f_stat", "one-hot")]:
            conf = [default_imp_num, default_imp_obj, default_nn, numerizer, default_scaler, feat_select]
            best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf, variant)

            feat_select_res.append([best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                    best_models_perfs])

        feat_select_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(feat_select_res)

        # Attempt with combination of the best parameters value of previous experiments
        print("\n * Combination of best parameters")
        conf = [imputation_res[0][1][0], imputation_res[0][1][1], outliers_res[0][1][2], numerize_res[0][1][4], scaling_res[0][1][5], feat_select_res[0][1][6]]
        best_models, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf, variant)

        combination_res = [[best_models, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out], best_models_perfs]]
        self.format_data_exp_output(combination_res)

        # Store configuration having shown the highest performance average over experiments
        conf = self.final_confs["h1n1"] if variant == "h1n1" else self.final_confs["seas"]
        (conf["imp_num"], conf["imp_obj"], conf["out_detect"], _, conf["numerizer"],
         conf["scaler"], _, conf["selected_features"]) = sorted(
            imputation_res + outliers_res + numerize_res + scaling_res + feat_select_res + combination_res,
            reverse=True, key=lambda x: statistics.mean(x[2]))[0][1]
        conf["selected_features"] = conf["selected_features"][2]  # retrieve features list
        print("\n * Final configuration")
        print(", ".join(["{}: {}".format(k, self.final_confs[variant][k]) for k in self.final_confs[variant]]))

        store = True
        if store:
            self.store_datasets(variant, conf)

    def store_datasets(self, variant, conf):
        # Prepare datasets with parameters previously defined to avoid recomputing them unnecessarily
        final_train_sets = list()
        final_test_sets = list()
        for _ in range(self.exp_rounds):
            ds = DataPreprocessing().training_preprocessing_pipeline(variant, "data/training_set_features.csv",
                                                                 "data/training_set_labels.csv",
                                                                 conf["imp_num"], conf["imp_obj"], conf["out_detect"],
                                                                 conf["numerizer"], conf["scaler"], conf["selected_features"])
            final_train_sets.append(ds[:2])
            final_test_sets.append(ds[2:])

        for i in range(self.exp_rounds):
            for df, name in zip(final_train_sets[i], ["features", "labels"]):
                df.to_pickle("serialized_df/trs_{}_{}_{}".format(variant, name, str(i)))
            for df, name in zip(final_test_sets[i], ["features", "labels"]):
                df.to_pickle("serialized_df/tss_{}_{}_{}".format(variant, name, str(i)))

    @staticmethod
    def preprocessing_exp(n_iter, imp_num, imp_obj, nn, numerizer, scaler, feat_selector, variant):
        """
        Train a linear model with the provided preprocessing parameters to evaluate their influence on model performance

        :params: parameters for different preprocessing phases

        :return: list of pairs of models, str indicating output of operations for outlier detection/feature
            selection..., performance of the returned models on the validation phase
        """

        best_models, best_models_perfs, out_removed, corr_removed = list(), list(), list(), list()

        for _ in range(n_iter):

            # Data preprocessing
            dp = DataPreprocessing()
            ds = dp.training_preprocessing_pipeline(variant, "data/training_set_features.csv",
                                                    "data/training_set_labels.csv",
                                                    imp_num, imp_obj, nn, numerizer, scaler, feat_selector)
            out_removed.append(dp.get_outlier_detection_res())
            corr_removed.append(dp.get_feature_selection_res())

            print(imp_num, imp_obj, nn, numerizer, scaler, feat_selector)
            print(ds[0].shape, ds[1].shape, ds[2].shape, ds[3].shape)
            # Model identification and validation
            mi = ModelIdentification(*ds, cv_folds=5)
            mi.lm()
            mi.model_selection()
            candidates = mi.model_testing()
            best_models.append((candidates[0][0]))
            best_models_perfs.append(candidates[0][1])

        # how many outliers/correlated features/useless features removed
        outlier_detect_res = "{}-{}".format(min(out_removed), max(out_removed))
        corr_removed_res = "{}-{}".format(min(corr_removed, key=lambda x: x[0])[0], max(corr_removed, key=lambda x: x[0])[0])
        feat_select_res = "{}-{}".format(min(corr_removed, key=lambda x: x[1])[1], max(corr_removed, key=lambda x: x[1])[1])
        selected_features = corr_removed[best_models_perfs.index(max(best_models_perfs))][2]

        return best_models, outlier_detect_res, (corr_removed_res, feat_select_res, selected_features), best_models_perfs

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

        # use pre-preprocessed datasets to avoid recomputing them every time
        final_train_sets, final_test_sets = list(), list()
        for i in range(self.exp_rounds):
            final_train_sets.append((pd.read_pickle("serialized_df/trs_{}_features_{}".format(variant, str(i))),
                                     pd.read_pickle("serialized_df/trs_{}_labels_{}".format(variant, str(i)))))
            final_test_sets.append((pd.read_pickle("serialized_df/tss_{}_features_{}".format(variant, str(i))),
                                    pd.read_pickle("serialized_df/tss_{}_labels_{}".format(variant, str(i)))))

        # Train models with CV and test performance on unused test set
        models = ["lm", "lr", "ridge", "gnb", "gpc", "tree", "ada", "gbc", "svm", "nn"][6:8]
        models = ["lm"]
        candidates = list()
        for i in range(self.exp_rounds):
            mi = ModelIdentification(*final_train_sets[i], *final_test_sets[i], cv_folds=5, verbose=True)
            mi.model_identification(models)
            mi.model_selection()
            res = mi.model_testing()
            candidates += res

        print("\nFinal {} candidates".format(variant))
        for c in sorted(candidates, reverse=True, key=lambda x: x[1]):
            print(c)

        self.final_models[variant] = candidates[0][0]

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
