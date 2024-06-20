import statistics

from DataPreprocessing import DataPreprocessing
from ModelIdentification import ModelIdentification


def machine_learning_procedure():
    N = 5
    analytics = list()

    for imp_num in ["knn", "remove", "mean", "median", "most_frequent"]:
        for imp_obj in ["remove", "most_frequent"]:
            for nn in [0, 2, 25]:
                for scaling in [True, False]:
                    analytics.append([[], []])
                    n_removed = list()
                    models = list()

                    for _ in range(N):
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
                        best_models, best_perf = mi.model_testing()
                        models.append(best_models)
                        analytics[-1][-1].append(best_perf)

                    best_model = max(set(models), key=models.count)
                    n_rem_summary = "{}-{}".format(min(n_removed), max(n_removed))
                    analytics[-1][0] += [best_model, imp_num, imp_obj, nn, n_rem_summary, scaling]

    for conf, res in sorted(analytics, reverse=True, key=lambda x: statistics.mean(x[1])):
        print("imp_num={}, imp_pbj={}, knn={}({}), scaling={}, {} -> avg: {}, stdev: {}".format(conf[1], conf[2], conf[3],
                                                                                                conf[4], conf[5], conf[0],
                                                                                                round(statistics.mean(res), 5),
                                                                                                round(statistics.stdev(res), 5)))


if __name__ == '__main__':
    machine_learning_procedure()
