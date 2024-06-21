import statistics

from DataPreprocessing import DataPreprocessing
from ModelIdentification import ModelIdentification


def machine_learning_loop(n_iter, imp_num, imp_obj, nn, scaling):
    conf, perf = list(), list()
    n_removed = list()
    models = list()

    for _ in range(n_iter):
        # Data preprocessing
        dp = DataPreprocessing()
        dp.load_data()
        # dp.exploratory_analysis()

        # Feature engineering
        dp.missing_values_imputation(imp_num, imp_obj)
        n_removed.append(dp.outlier_detection(nn))
        dp.numerize_categorical_features()
        if scaling:
            dp.features_scaling()
        dp.feature_selection()

        # Model identification
        mi = ModelIdentification(*dp.get_datasets(), 5)
        mi.model_identification(["lm"])
        mi.model_selection()
        best_model_names, best_perf = mi.model_testing()
        models.append(best_model_names)
        perf.append(best_perf)

    models = sorted([(str(m), models.count(m)) for m in set(models)], reverse=True, key=lambda x: x[1])
    n_rem_summary = "{}-{}".format(min(n_removed), max(n_removed))
    conf = [imp_num, imp_obj, nn, n_rem_summary, scaling]

    return models, conf, perf


def format_loop_output(conf_perf):
    for models, conf, perf in conf_perf:
        for m in models:
            print("{}: {}".format(m[0], m[1]))
        print("imp_num={}, imp_pbj={}, knn={}({}), scaling={} -> avg: {}, stdev: {}".format(conf[0], conf[1],
                                                                                            conf[2], conf[3], conf[4],
                                                                                            round(statistics.mean(perf), 5),
                                                                                            round(statistics.stdev(perf), 5)))


def machine_learning_procedure():
    N = 5

    # imputation of missing values
    print(" * Imputation of missing values")
    imp_res = list()
    for imp_num in ["knn", "remove", "mean", "median", "most_frequent"]:
        for imp_obj in ["remove", "most_frequent"]:
            imp_res.append(machine_learning_loop(N, imp_num, imp_obj, 0, False))
    imp_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
    format_loop_output(imp_res)

    # outliers detection
    print("\n * Outliers detection")
    outliers_res = list()
    for nn in [0, 2, 25]:
        outliers_res.append(machine_learning_loop(N, imp_num, imp_obj, nn, False))
    outliers_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
    format_loop_output(outliers_res)

    # numeric features scaling
    print("\n * Numeric features scaling")
    scaling_res = list()
    for scaling in [True, False]:
        scaling_res.append(machine_learning_loop(N, imp_num, imp_obj, 0, scaling))
    scaling_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
    format_loop_output(scaling_res)

    # attempt with all best performing parameters
    print("\n * Combination of best parameters")
    best_params = [imp_res[0][1][0], imp_res[0][1][1], outliers_res[0][1][2], scaling_res[0][1][4]]
    format_loop_output([machine_learning_loop(N, *best_params)])


if __name__ == '__main__':

    machine_learning_procedure()
