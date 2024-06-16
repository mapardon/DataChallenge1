import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer


class DataPreprocessing:
    def __init__(self):
        self.features = None
        self.h1n1_labels = None
        self.seas_labels = None

    def data_preprocessing(self, m1, m2):
        self.load_data()
        #self.exploratory_analysis()

        # feature engineering
        self.missing_values_imputation(m1, m2)
        self.outlier_detection()
        self.numerize_categorical_features()
        self.features_scaling()
        self.feature_selection()

        return self.features, self.h1n1_labels, self.seas_labels

    def load_data(self):
        self.features = pd.read_csv("data/training_set_features.csv")
        flu_labels = pd.read_csv("data/training_set_labels.csv")

        # shuffle dataset
        ds = self.features
        ds[["h1n1_vaccine", "seasonal_vaccine"]] = flu_labels[["h1n1_vaccine", "seasonal_vaccine"]]
        ds = ds.sample(frac=1)

        self.features = ds[ds.columns.to_list()[:-2]]
        self.h1n1_labels = ds["h1n1_vaccine"]
        self.seas_labels = ds["seasonal_vaccine"]

    def exploratory_analysis(self):
        print(" * Features dimension:\n{}".format(self.features.shape))
        print("\n * Example record:\n{}".format(self.features.head(1)))
        print("\n * Features summary:")
        self.features.info()
        print("\n * Numeric features info:\n{}".format(self.features.describe()))

        # plots (features)
        to_plot = self.features.iloc[:, 1:]  # remove respondent_id
        ax_count = int()
        axs = list()

        for feature in to_plot.columns.to_list():
            if not ax_count:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                axs = [ax1, ax2, ax3, ax4]

            if len(pd.unique(to_plot[feature])) <= 10:
                # if not many different values, consider as str for better display
                elements = sorted([(str(k), v) for k, v in to_plot[feature].value_counts().to_dict().items()],
                                  key=lambda x: x[0])
            else:
                elements = sorted([(k, v) for k, v in to_plot[feature].value_counts().to_dict().items()],
                                  key=lambda x: x[0])

            axs[ax_count].bar([e[0] for e in elements], [e[1] for e in elements], width=0.3)
            axs[ax_count].set_title(feature)
            ax_count += 1

            if not ax_count % 4:
                plt.show()
                ax_count *= 0

        if ax_count % 4:
            plt.show()

        # plots (labels)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for df, feature, ax in [(self.h1n1_labels, "h1n1_vaccine", ax1), (self.seas_labels, "seasonal_vaccine", ax2)]:
            elements = sorted([(str(k), v) for k, v in df[feature].value_counts().to_dict().items()],
                              key=lambda x: x[0])
            ax.bar([e[0] for e in elements], [e[1] for e in elements], width=0.3, color=["tab:blue", "orange"])
            ax.set_title(feature.capitalize())
        plt.show()

    def missing_values_imputation(self, num_strat="remove", obj_strat="remove"):

        if num_strat == "remove" or obj_strat == "remove":
            self.features = self.features[~self.features.isnull().any(axis=1)]

        else:
            num_features = self.features.select_dtypes(["number"])
            obj_features = self.features.select_dtypes(["object"])

            if num_strat in ["mean", "median", "most_frequent"]:
                imp = SimpleImputer(missing_values=np.nan, strategy=num_strat)
                imp.fit(num_features)
                num_features = pd.DataFrame(imp.transform(num_features))

            elif num_strat == "knn":
                imp = KNNImputer(n_neighbors=5)
                num_features = pd.DataFrame(imp.fit_transform(num_features))

            if obj_strat == "most_frequent":
                imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
                imp.fit(obj_features)
                obj_features = pd.DataFrame(imp.transform(obj_features))

            self.features = pd.concat([num_features, obj_features], axis="columns")
        print(self.features.shape)

    def outlier_detection(self):
        pass

    def numerize_categorical_features(self):
        pass

    def features_scaling(self):
        pass

    def feature_selection(self):
        ds = self.features
        ds["h1n1_vaccine"] = self.h1n1_labels
        ds["seasonal_vaccine"] = self.seas_labels

        # For now, we just remove non-numeric columns
        ds = ds.select_dtypes([np.number])

        self.features = ds.iloc[:, 1:-3]
        self.h1n1_labels = ds["h1n1_vaccine"]
        self.seas_labels = ds["seasonal_vaccine"]
