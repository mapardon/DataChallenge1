import statistics

from DataPreprocessing import DataPreprocessing
from ModelIdentification import ModelIdentification


def machine_learning_procedure():
    N = 5
    analytics = list()

    for imp_num in ["knn", "remove", "mean", "median", "most_frequent"]:
        for imp_obj in ["remove", "most_frequent"]:
            analytics.append(list())
            for _ in range(N):

                # Data preprocessing
                dp = DataPreprocessing()
                dp.load_data()
                #dp.exploratory_analysis()

                # Feature engineering
                dp.missing_values_imputation(imp_num, imp_obj)
                dp.outlier_detection()
                dp.numerize_categorical_features()
                dp.features_scaling()
                dp.feature_selection()
                print(dp.features.shape)

                # Model identification
                mi = ModelIdentification(*dp.get_datasets(), 5)
                mi.model_identification(["lm"])
                mi.model_selection()
                analytics[-1].append(mi.model_testing())

            print("imp_num={}, imp_pbj={} -> avg: {}, stdev: {}".format(imp_num, imp_obj, round(statistics.mean(analytics[-1]), 5), round(statistics.stdev(analytics[-1]), 5)))


if __name__ == '__main__':

    machine_learning_procedure()
