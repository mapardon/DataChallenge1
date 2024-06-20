import statistics

from DataPreprocessing import DataPreprocessing
from ModelIdentification import ModelIdentification


def machine_learning_procedure():
    N = 5
    analytics = list()

    for imp_num in ["knn", "remove", "mean", "median", "most_frequent"]:
        for imp_obj in ["remove", "most_frequent"]:
            for nn in [0, 2, 25]:
                analytics.append([[imp_num, imp_obj, nn], []])
                for _ in range(N):
                    # Data preprocessing
                    dp = DataPreprocessing()
                    dp.load_data()
                    # dp.exploratory_analysis()

                    # Feature engineering
                    dp.missing_values_imputation(imp_num, imp_obj)
                    n_removed = dp.outlier_detection(nn)
                    dp.numerize_categorical_features()
                    dp.features_scaling()
                    dp.feature_selection()

                    # Model identification
                    mi = ModelIdentification(*dp.get_datasets(), 5)
                    mi.model_identification(["lm"])
                    mi.model_selection()
                    analytics[-1][0].append(n_removed)
                    analytics[-1][-1].append(mi.model_testing())

    for conf, res in sorted(analytics, reverse=True, key=lambda x: statistics.mean(x[1])):
        print("imp_num={}, imp_pbj={}, knn={}({}) -> avg: {}, stdev: {}".format(conf[0], conf[1], conf[2], conf[3],
                                                                                round(statistics.mean(res), 5),
                                                                                round(statistics.stdev(res), 5)))


if __name__ == '__main__':
    machine_learning_procedure()
