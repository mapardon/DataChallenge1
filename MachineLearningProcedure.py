import shelve
import statistics

from DataPreprocessing import DataPreprocessing
from ModelIdentification import ModelIdentification


def machine_learning_procedure():
    """
        Main function initiating different steps of the complete procedure
    """

    x = False
    if x:
        N = 5
        all_models_conf = list()
        default_imp_num = "remove"
        default_imp_obj = "remove"
        default_nn = int()
        default_scaling = False

        # Imputation of missing values
        print(" * Imputation of missing values")
        imp_res = list()
        for imp_num in ["knn", "remove", "mean", "median"]:
            for imp_obj in ["remove", "most_frequent"]:
                conf = [imp_num, imp_obj, default_nn, default_scaling]
                best_models_pairs, n_rem_summary, perfs = experiment_loop(N, *conf)

                model_names = sorted([(m, [str(bp) for bp in best_models_pairs].count(m)) for m in set([str(bp) for bp in best_models_pairs])],
                                     reverse=True, key=lambda x: x[1])
                imp_res.append([model_names, conf[:3] + [n_rem_summary] + conf[3:], perfs])
                all_models_conf.append([best_models_pairs, conf, perfs, statistics.mean(perfs)])

        imp_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        format_loop_output(imp_res)

        # Outliers detection
        print("\n * Outliers detection")
        outliers_res = list()
        for nn in [0, 2, 25]:
            conf = [default_imp_num, default_imp_obj, nn, default_scaling]
            best_models_pairs, n_rem_summary, perfs = experiment_loop(N, *conf)

            model_names = sorted([(m, [str(bp) for bp in best_models_pairs].count(m)) for m in set([str(bp) for bp in best_models_pairs])],
                                 reverse=True, key=lambda x: x[1])
            outliers_res.append([model_names, conf[:3] + [n_rem_summary] + conf[3:], perfs])
            all_models_conf.append([best_models_pairs, conf, perfs, statistics.mean(perfs)])

        outliers_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        format_loop_output(outliers_res)

        # Numeric features scaling
        print("\n * Numeric features scaling")
        scaling_res = list()
        for scaling in [True, False]:
            conf = [default_imp_num, default_imp_obj, default_nn, scaling]
            best_models_pairs, n_rem_summary, perfs = experiment_loop(N, *conf)

            model_names = sorted([(m, [str(bp) for bp in best_models_pairs].count(m)) for m in set([str(bp) for bp in best_models_pairs])],
                                 reverse=True, key=lambda x: x[1])
            scaling_res.append([model_names, conf[:3] + [n_rem_summary] + conf[3:], perfs])
            all_models_conf.append([best_models_pairs, conf, perfs, statistics.mean(perfs)])

        scaling_res.sort(reverse=True, key=lambda x: statistics.mean(x[2]))
        format_loop_output(scaling_res)

        # Attempt with the best parameters value of previous experiments
        print("\n * Combination of best parameters")
        conf = [imp_res[0][1][0], imp_res[0][1][1], outliers_res[0][1][2], scaling_res[0][1][4]]
        best_models_pairs, n_rem_summary, perfs = experiment_loop(N, *conf)

        model_names = sorted([(m, [str(bp) for bp in best_models_pairs].count(m)) for m in set([str(bp) for bp in best_models_pairs])],
                             reverse=True, key=lambda x: x[1])
        conf_res = [model_names, conf[:3] + [n_rem_summary] + conf[3:], perfs]
        all_models_conf.append([best_models_pairs, conf, perfs, statistics.mean(perfs)])
        format_loop_output([conf_res])

        with shelve.open("db") as db:
            db["all_models_conf"] = all_models_conf

    with shelve.open("db") as db:
        all_models_conf = db["all_models_conf"]

    # Exploit best model on challenge data -> best among configurations within 0.01 of the highest average
    # TODO choose h1n1/seas model independently
    print("\n * Model exploitation")
    all_models_conf.sort(reverse=True, key=lambda x: x[3])
    candidates = list()
    for candidate in [c for c in all_models_conf if all_models_conf[0][3] - c[3] < 0.01]:
        for models, perf in zip(candidate[0], candidate[2]):
            candidates.append((models, candidate[1], perf))

    candidate = sorted(candidates, reverse=True, key=lambda x: x[2])[0]
    exploitation_loop(candidate[0],
                      "median" if candidate[1][0] == "remove" else candidate[1][0],
                      "most_frequent" if candidate[1][1] == "remove" else candidate[1][1], candidate[1][3])


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

        # Model identification and validation
        mi = ModelIdentification(*dp.get_train_datasets(), cv_folds=5)
        mi.model_identification(["lm"])
        mi.model_selection()
        best_models_pair, best_perf = mi.model_testing()
        best_models_pairs.append(best_models_pair)
        perf.append(best_perf)

    n_rem_summary = "{}-{}".format(min(n_removed), max(n_removed))

    return best_models_pairs, n_rem_summary, perf


def exploitation_loop(models, imp_num, imp_obj, scaling):

    # Pre-process challenge data
    dp = DataPreprocessing()
    dp.load_test_set("data/test_set_features.csv")
    dp.missing_values_imputation(imp_num, imp_obj)
    dp.numerize_categorical_features()
    if scaling:
        dp.features_scaling()
    dp.feature_selection()
    resp_id, features = dp.get_test_dataset()
    print()

    print(models[0].__dict__)
    print(models[1].__dict__)
    print(features.columns.to_list())

    # Use previously trained model on processed challenge data
    ModelIdentification.model_exploitation(models[0], models[1], features, resp_id)


def format_loop_output(conf_perf):
    for models, conf, perf in conf_perf:
        for m in models:
            print("{}: {}".format(m[0], m[1]))
        print("imp_num={}, imp_pbj={}, knn={}({}), scaling={} -> avg: {}, stdev: {}".format(conf[0], conf[1],
                                                                                            conf[2], conf[3], conf[4],
                                                                                            round(statistics.mean(perf), 5),
                                                                                            round(statistics.stdev(perf), 5)))
