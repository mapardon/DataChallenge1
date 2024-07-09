import statistics

import pandas as pd

from DataPreprocessing import DataPreprocessing
from ModelIdentification import ModelIdentification


class MachineLearningProcedure:
    """
        Main function initiating different steps of the complete procedure
    """
    def __init__(self):
        self.exp_rounds = 5
        self.final_imp_num = None
        self.final_imp_obj = None
        self.final_out_detect = None
        self.final_scaling = None
        self.final_numerizer = None
        self.final_feat_selector = None

        self.final_train_sets = list()
        self.final_test_sets = list()

        self.final_h1n1_model = None
        self.final_seas_model = None

    def main(self):
        self.preprocessing()
        #self.model_identification()
        #self.exploitation_loop()

    def preprocessing(self):
        """
            First we evaluate the influence of different preprocessing parameters (in combination of a fast-training
            model) in order to preprocess the final dataset
        """

        default_imp_num = "remove"
        default_imp_obj = "remove"
        default_nn = int()
        default_scaling = False
        default_numerizer = "remove"
        default_feat_select = None

        # Imputation of missing values
        print(" * Imputation of missing values")
        imputation_res = list()
        for imp_num in ["knn", "remove", "mean", "median"]:
            for imp_obj in ["remove", "most_frequent"]:
                conf = [imp_num, imp_obj, default_nn, default_numerizer, default_scaling, default_feat_select]
                best_models_pairs, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf)

                imputation_res.append([best_models_pairs, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                       best_models_perfs])

        imputation_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(imputation_res)

        # Outliers detection
        print("\n * Outliers detection")
        outliers_res = list()
        for nn in [0, 2, 25]:
            conf = [default_imp_num, default_imp_obj, nn, default_numerizer, default_scaling, default_feat_select]
            best_models_pairs, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf)

            outliers_res.append([best_models_pairs, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                 best_models_perfs])

        outliers_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(outliers_res)

        # Numerize categorical features
        print("\n * Numerize categorical features")
        numerize_res = list()
        for numerizer in ["remove", "one-hot"]:
            conf = [default_imp_num, default_imp_obj, default_nn, numerizer, default_scaling, default_feat_select]
            best_models_pairs, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf)

            numerize_res.append([best_models_pairs, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                 best_models_perfs])

        numerize_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(numerize_res)

        # Numeric features scaling
        print("\n * Numeric features scaling")
        scaling_res = list()
        for scaling in [True, False]:
            conf = [default_imp_num, default_imp_obj, default_nn, default_numerizer, scaling, default_feat_select]
            best_models_pairs, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf)

            scaling_res.append([best_models_pairs, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                best_models_perfs])

        scaling_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(scaling_res)

        # Features selection
        print("\n * Features selection")
        feat_select_res = list()
        for feat_select in ["remove", "one-hot"]:
            conf = [default_imp_num, default_imp_obj, default_nn, default_numerizer, default_scaling, feat_select]
            best_models_pairs, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf)

            feat_select_res.append([best_models_pairs, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out],
                                    best_models_perfs])

        numerize_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(numerize_res)

        # Attempt with combination of the best parameters value of previous experiments
        print("\n * Combination of best parameters")
        conf = [imputation_res[0][1][0], imputation_res[0][1][1], outliers_res[0][1][2], numerize_res[0][1][4], scaling_res[0][1][5], feat_select_res[0][1][6]]
        best_models_pairs, outlier_detect_out, feat_select_out, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf)

        combination_res = [[best_models_pairs, conf[:3] + [outlier_detect_out] + conf[3:] + [feat_select_out], best_models_perfs]]
        self.format_data_exp_output(combination_res)

        # Select configuration having shown the highest performance average
        (self.final_imp_num, self.final_imp_obj, self.final_out_detect, _,
         self.final_numerizer, self.final_scaling, self.final_feat_selector, _) = sorted(
            imputation_res + outliers_res + numerize_res + scaling_res + feat_select_res + combination_res,
            reverse=True, key=lambda x: statistics.mean(x[2]))[0][1]

        # Prepare datasets with parameters previously defined to avoid recomputing them unnecessarily
        for _ in range(self.exp_rounds):
            ds = DataPreprocessing().data_preprocessing_pipeline("data/training_set_features.csv",
                                                                 "data/training_set_labels.csv",
                                                                 self.final_imp_num, self.final_imp_obj,
                                                                 self.final_out_detect, self.final_numerizer,
                                                                 self.final_scaling, self.final_feat_selector)
            self.final_train_sets.append(ds[:3])
            self.final_test_sets.append(ds[3:])

        for i in range(self.exp_rounds):
            for df, name in zip(self.final_train_sets[i], ["labels", "h1n1", "seas"]):
                df.to_pickle("serialized_df/trs_" + name + str(i))
            for df, name in zip(self.final_test_sets[i], ["labels", "h1n1", "seas"]):
                df.to_pickle("serialized_df/tss_" + name + str(i))

    @staticmethod
    def preprocessing_exp(n_iter, imp_num, imp_obj, nn, numerizer, scaling, feat_selector):

        best_models_pairs, perfs, out_removed, corr_removed = list(), list(), list(), list()

        for _ in range(n_iter):
            # Data preprocessing
            dp = DataPreprocessing()
            dp.load_train_test_datasets("data/training_set_features.csv", "data/training_set_labels.csv")

            # Feature engineering
            dp.missing_values_imputation(imp_num, imp_obj)
            out_removed.append(dp.outlier_detection(nn))
            dp.numerize_categorical_features(numerizer)
            if scaling:
                dp.features_scaling()
            corr_removed.append(dp.feature_selection(feat_selector))

            # Model identification and validation
            mi = ModelIdentification(*dp.get_train_test_datasets(), cv_folds=5)
            mi.lm()
            mi.model_selection()
            candidates = mi.model_testing()
            best_models_pairs.append((candidates["h1n1"][0][0], candidates["seas"][0][0]))
            perfs.append((statistics.mean([candidates["h1n1"][0][1], candidates["seas"][0][1]])))

        outlier_detect_res = "{}-{}".format(min(out_removed), max(out_removed))
        corr_removed_res = "{}-{}".format(min(corr_removed, key=lambda x: x[0])[0], max(corr_removed, key=lambda x: x[0])[0])
        feat_select_res = "{}-{}".format(min(corr_removed, key=lambda x: x[1])[1], max(corr_removed, key=lambda x: x[1])[1])

        return best_models_pairs, outlier_detect_res, (corr_removed_res, feat_select_res), perfs

    @staticmethod
    def format_data_exp_output(conf_perf):
        """ conf_perf : [[((m1, m2), (m1, m2)), (p1, p2, p3), (perf1, perf2)], ...] """
        for models, conf, perf in conf_perf:
            print("imp_num={}, imp_obj={}, knn={}(#rem.:{}), scaling={}, numerizer={}, feat_select={}(#corr_rem.:{}/#select:{})".format(
                conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6], conf[7][0], conf[7][1]))
            print("\t-> avg: {}, stdev: {}".format(round(statistics.mean(perf), 5), round(statistics.stdev(perf), 5)))

    def model_identification(self):
        """
            Using the pre-processed datasets with the previously chosen parameters, we now test the performance of
            different learning algorithms
        """

        for i in range(self.exp_rounds):
            self.final_train_sets.append((pd.read_pickle("serialized_df/trs_labels" + str(i)),
                                          pd.read_pickle("serialized_df/trs_h1n1" + str(i)),
                                          pd.read_pickle("serialized_df/trs_seas" + str(i))))
            self.final_test_sets.append((pd.read_pickle("serialized_df/tss_labels" + str(i)),
                                         pd.read_pickle("serialized_df/tss_h1n1" + str(i)),
                                         pd.read_pickle("serialized_df/tss_seas" + str(i))))

        # Train models with CV and test performance on unused test set
        models = ["lm", "ridge", "tree"]
        models = ["lm"]
        candidates = {"h1n1": list(), "seas": list()}
        for i in range(self.exp_rounds):
            mi = ModelIdentification(*self.final_train_sets[i], *self.final_test_sets[i], cv_folds=5)
            mi.model_identification(models)
            mi.model_selection()
            res = mi.model_testing()
            candidates["h1n1"] += res["h1n1"]
            candidates["seas"] += res["seas"]

        for k in candidates:
            print(k)
            for c in sorted(candidates[k], reverse=True, key=lambda x: x[1]):
                print(c)

        self.final_h1n1_model = candidates["h1n1"][0][0]
        self.final_seas_model = candidates["seas"][0][0]

        self.final_train_sets.clear()
        self.final_test_sets.clear()

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

        # Must predict all challenge tuples
        self.final_imp_num = self.final_imp_num if self.final_imp_num != "remove" else "median"
        self.final_imp_obj = self.final_imp_obj if self.final_imp_obj != "remove" else "most_frequent"

        # Pre-process challenge data
        dp = DataPreprocessing()
        resp_id, features = dp.data_preprocessing_pipeline("data/test_set_features.csv", None,
                                                           self.final_imp_num, self.final_imp_obj,
                                                           self.final_out_detect, self.final_numerizer,
                                                           self.final_scaling, self.final_feat_selector)

        # Use previously trained model on processed challenge data
        ModelIdentification.model_exploitation("data/submission.csv", self.final_h1n1_model, self.final_seas_model, features, resp_id)
