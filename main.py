import statistics

from DataPreprocessing import DataPreprocessing
from ModelIdentification import ModelIdentification


def experiment_loop(n_iter, imp_num, imp_obj, nn, scaling):
    best_models_pairs, perf, n_removed = list(), list(), list()

    for _ in range(n_iter):
        # Data preprocessing
        dp = DataPreprocessing()
        dp.load_train_set("data/training_set_features.csv", "data/training_set_labels.csv")
        # dp.exploratory_analysis()

        # Feature engineering
        dp.missing_values_imputation(imp_num, imp_obj)
        n_removed.append(dp.outlier_detection(nn))
        dp.numerize_categorical_features()
        if scaling:
            dp.features_scaling()
        dp.feature_selection()

        # Model identification
        mi = ModelIdentification(*dp.get_train_datasets(), cv_folds=5)
        mi.model_identification(["lm"])
        mi.model_selection()
        best_models_pair, best_perf = mi.model_testing()
        best_models_pairs.append(best_models_pair)
        perf.append(best_perf)

    n_rem_summary = "{}-{}".format(min(n_removed), max(n_removed))

    return best_models_pairs, n_rem_summary, perf


def exploitation_loop(model, imp_num, imp_obj, scaling):

    # challenge data pre-processing
    dp = DataPreprocessing()
    dp.load_test_set("data/test_set_features.csv")
    dp.missing_values_imputation("median", "most_frequent")
    dp.numerize_categorical_features()
    if scaling:
        dp.features_scaling()
    dp.feature_selection()

    # use previously trained model on processed challenge data
    model.predict


def machine_learning_procedure():
    N = 5
    all_models_conf = list()
    default_imp_num = "remove"
    default_imp_obj = "remove"
    default_nn = int()
    default_scaling = False

    # imputation of missing values
    print(" * Imputation of missing values")
    imp_res = list()
    for imp_num in ["knn", "remove", "mean", "median"]:
        for imp_obj in ["remove", "most_frequent"]:
            conf = [imp_num, imp_obj, default_nn, default_scaling]
            best_models_pairs, n_rem_summary, perf = experiment_loop(N, *conf)

            model_names = sorted([(m, [str(bp) for bp in best_models_pairs].count(m)) for m in set([str(bp) for bp in best_models_pairs])],
                                 reverse=True, key=lambda x: x[1])
            imp_res.append([model_names, conf[:3] + [n_rem_summary] + conf[3:], perf])
            all_models_conf.append([best_models_pairs, conf, perf, statistics.mean(perf)])

    imp_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
    format_loop_output(imp_res)

    # outliers detection
    print("\n * Outliers detection")
    outliers_res = list()
    for nn in [0, 2, 25]:
        conf = [default_imp_num, default_imp_obj, nn, default_scaling]
        best_models_pairs, n_rem_summary, perf = experiment_loop(N, *conf)

        model_names = sorted([(m, [str(bp) for bp in best_models_pairs].count(m)) for m in set([str(bp) for bp in best_models_pairs])],
                             reverse=True, key=lambda x: x[1])
        outliers_res.append([model_names, conf[:3] + [n_rem_summary] + conf[3:], perf])
        all_models_conf.append([best_models_pairs, conf, perf, statistics.mean(perf)])

    outliers_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
    format_loop_output(outliers_res)

    # numeric features scaling
    print("\n * Numeric features scaling")
    scaling_res = list()
    for scaling in [True, False]:
        conf = [default_imp_num, default_imp_obj, default_nn, scaling]
        best_models_pairs, n_rem_summary, perf = experiment_loop(N, *conf)

        model_names = sorted([(m, [str(bp) for bp in best_models_pairs].count(m)) for m in set([str(bp) for bp in best_models_pairs])],
                             reverse=True, key=lambda x: x[1])
        scaling_res.append([model_names, conf[:3] + [n_rem_summary] + conf[3:], perf])
        all_models_conf.append([best_models_pairs, conf, perf, statistics.mean(perf)])

    scaling_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
    format_loop_output(scaling_res)

    # attempt with all best performing parameters
    print("\n * Combination of best parameters")
    conf = [imp_res[0][1][0], imp_res[0][1][1], outliers_res[0][1][2], scaling_res[0][1][4]]
    #conf = ["remove", "most_frequent", 25, True]
    best_models_pairs, n_rem_summary, perf = experiment_loop(N, *conf)

    model_names = sorted([(m, [str(bp) for bp in best_models_pairs].count(m)) for m in set([str(bp) for bp in best_models_pairs])],
                         reverse=True, key=lambda x: x[1])
    conf_res = [model_names, conf[:3] + [n_rem_summary] + conf[3:], perf]
    all_models_conf.append([best_models_pairs, conf, perf, statistics.mean(perf)])
    format_loop_output([conf_res])

    # exploit best model on challenge data -> overall best or best of group of highest 5-rounds average?
    all_models_conf.sort(reverse=True, key=lambda x: x[3])
    # TODO pick best model
    #exploitation_loop


def format_loop_output(conf_perf):
    for models, conf, perf in conf_perf:
        for m in models:
            print("{}: {}".format(m[0], m[1]))
        print("imp_num={}, imp_pbj={}, knn={}({}), scaling={} -> avg: {}, stdev: {}".format(conf[0], conf[1],
                                                                                            conf[2], conf[3], conf[4],
                                                                                            round(statistics.mean(perf), 5),
                                                                                            round(statistics.stdev(perf), 5)))


if __name__ == '__main__':

    machine_learning_procedure()
