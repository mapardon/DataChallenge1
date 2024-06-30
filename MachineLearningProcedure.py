import shelve
import statistics

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

        self.final_train_sets = list()
        self.final_test_sets = list()

    def main(self):
        self.preprocessing()

    def preprocessing(self):
        """
            First we evaluate the influence of different preprocessing parameters (in combination of a fast-training
            model) in order to preprocess the final dataset
        """

        default_imp_num = "remove"
        default_imp_obj = "remove"
        default_nn = int()
        default_scaling = False

        # Imputation of missing values
        print(" * Imputation of missing values")
        imputation_res = list()
        for imp_num in ["knn", "remove", "mean", "median"]:
            for imp_obj in ["remove", "most_frequent"]:
                conf = [imp_num, imp_obj, default_nn, default_scaling]
                best_models_pairs, outlier_detect_res, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf)

                imputation_res.append([best_models_pairs, conf[:3] + [outlier_detect_res] + conf[3:], best_models_perfs])

        imputation_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(imputation_res)

        # Outliers detection
        print("\n * Outliers detection")
        outliers_res = list()
        for nn in [0, 2, 25]:
            conf = [default_imp_num, default_imp_obj, nn, default_scaling]
            best_models_pairs, outlier_detect_res, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf)

            outliers_res.append([best_models_pairs, conf[:3] + [outlier_detect_res] + conf[3:], best_models_perfs])

        outliers_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(outliers_res)

        # Numeric features scaling
        print("\n * Numeric features scaling")
        scaling_res = list()
        for scaling in [True, False]:
            conf = [default_imp_num, default_imp_obj, default_nn, scaling]
            best_models_pairs, outlier_detect_res, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf)

            scaling_res.append([best_models_pairs, conf[:3] + [outlier_detect_res] + conf[3:], best_models_perfs])

        scaling_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        self.format_data_exp_output(scaling_res)

        # Attempt with combination of the best parameters value of previous experiments
        print("\n * Combination of best parameters")
        conf = [imputation_res[0][1][0], imputation_res[0][1][1], outliers_res[0][1][2], scaling_res[0][1][4]]
        best_models_pairs, outlier_detect_res, best_models_perfs = self.preprocessing_exp(self.exp_rounds, *conf)

        combination_res = [[best_models_pairs, conf[:3] + [outlier_detect_res] + conf[3:], best_models_perfs]]
        self.format_data_exp_output(combination_res)

        # Select configuration having shown the highest performance average
        self.final_imp_num, self.final_imp_obj, self.final_out_detect, _, self.final_scaling = sorted(imputation_res + outliers_res + scaling_res + combination_res,
                                                                                                      reverse=True, key=lambda x: statistics.mean(x[2]))[0][1]

        # Prepare datasets with parameters previously defined to avoid recomputing them unnecessarily
        for _ in range(self.exp_rounds):
            ds = DataPreprocessing().data_preprocessing_pipeline("data/training_set_features.csv",
                                                                 "data/training_set_labels.csv",
                                                                 self.final_imp_num, self.final_imp_obj,
                                                                 self.final_out_detect, self.final_scaling)[:3]
            self.final_train_sets.append(ds[:3])
            self.final_test_sets.append(ds[3:])

        with shelve.open("db") as db:
            db["final_train_sets"] = self.final_train_sets
            db["final_test_sets"] = self.final_test_sets

    @staticmethod
    def preprocessing_exp(n_iter, imp_num, imp_obj, nn, scaling):

        best_models_pairs, perfs, n_removed = list(), list(), list()

        for _ in range(n_iter):
            # Data preprocessing
            dp = DataPreprocessing()
            dp.load_train_test_datasets("data/training_set_features.csv", "data/training_set_labels.csv")

            # Feature engineering
            dp.missing_values_imputation(imp_num, imp_obj)
            n_removed.append(dp.outlier_detection(nn))
            dp.numerize_categorical_features()
            if scaling:
                dp.features_scaling()
            dp.feature_selection()

            # Model identification and validation
            mi = ModelIdentification(*dp.get_train_test_datasets(), cv_folds=5)
            mi.lm()
            mi.model_selection()
            best_models_pair, best_models_perfs = mi.model_testing()
            best_models_pairs.append(best_models_pair)
            perfs.append(best_models_perfs)

        outlier_detect_res = "{}-{}".format(min(n_removed), max(n_removed))

        return best_models_pairs, outlier_detect_res, perfs

    @staticmethod
    def format_data_exp_output(conf_perf):
        """ conf_perf : [[((m1, m2), (m1, m2)), (p1, p2, p3), (perf1, perf2)], ...] """
        for models, conf, perf in conf_perf:
            print("imp_num={}, imp_obj={}, knn={}(#removed:{}), scaling={} -> avg: {}, stdev: {}".format(conf[0], conf[1],
                                                                                                         conf[2], conf[3], conf[4],
                                                                                                         round(statistics.mean(perf), 5),
                                                                                                         round(statistics.stdev(perf), 5)))

    def model_identification(self):
        """
            Using the pre-processed datasets with the previously chosen parameters, we now test the performance of
            different learning algorithms
        """

        # Plain old linear model
        lm_res = self.model_identification_exp("lm")
        self.format_model_exp_output(lm_res)

        #

        self.final_train_sets.clear()
        self.final_test_sets.clear()

    def model_identification_exp(self, model_name):
        best_models_pairs_perfs = list()

        for i in range(self.exp_rounds):
            mi = ModelIdentification(*self.final_train_sets[i], *self.final_test_sets[i], cv_folds=5)
            {"lm": mi.lm, "ridge": mi.ridge, "svg": mi.svm, "tree": mi.tree}[model_name]()
            mi.model_selection()
            best_models_pair, best_models_perf = mi.model_testing()
            best_models_pairs_perfs.append((best_models_pair, best_models_perf))

        return best_models_pairs_perfs

    @staticmethod
    def format_model_exp_output(models_perf):
        for models, perfs in models_perf:
            for m in models:
                print("{}: {}".format(m[0], m[1]))
            print("Performance (ROC) -> avg: {}, stdev: {}".format(round(statistics.mean(perfs), 5),
                                                                   round(statistics.stdev(perfs), 5)))

    def exploitation_loop(self, models, imp_num, imp_obj, scaling):
        """ Finally, we use the models and preprocessing parameters having shown the best performance during training
        to predict challenge data """

        # Pre-process challenge data
        dp = DataPreprocessing()
        dp.load_challenge_dataset("data/test_set_features.csv")
        dp.missing_values_imputation(imp_num, imp_obj)
        dp.numerize_categorical_features()
        if scaling:
            dp.features_scaling()
        dp.feature_selection()
        resp_id, features = dp.get_challenge_dataset()

        # Use previously trained model on processed challenge data
        ModelIdentification.model_exploitation(models[0], models[1], features, resp_id)

        # Exploit best model on challenge data
        # TODO choose h1n1/seas model independently
        print("\n * Model exploitation")

        candidate = [[1], [2], [3], [4]]

        self.exploitation_loop(candidate[0],
                          "median" if candidate[1][0] == "remove" else candidate[1][0],
                          "most_frequent" if candidate[1][1] == "remove" else candidate[1][1], candidate[1][3])
