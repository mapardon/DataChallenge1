import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessing:
    def __init__(self):
        self.features = None
        self.h1n1_labels = None
        self.seas_labels = None

    def get_datasets(self):
        return self.features, self.h1n1_labels, self.seas_labels

    def load_data(self):
        self.features = pd.read_csv("data/training_set_features.csv")
        flu_labels = pd.read_csv("data/training_set_labels.csv")

        # shuffle dataset (and reset indexes)
        ds = self.features
        ds[["h1n1_vaccine", "seasonal_vaccine"]] = flu_labels[["h1n1_vaccine", "seasonal_vaccine"]]
        ds = ds.sample(frac=1)
        ds.reset_index(inplace=True, drop=True)

        self.features = ds[ds.columns.to_list()[1:-2]]
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

        if num_strat == "remove" and obj_strat == "remove":
            na_idx = self.features.isnull().any(axis="columns")
            self.features = self.features[~na_idx].reset_index(drop=True)
            self.h1n1_labels = self.h1n1_labels[~na_idx].reset_index(drop=True)
            self.seas_labels = self.seas_labels[~na_idx].reset_index(drop=True)

        else:
            if num_strat == "remove":
                na_idx = self.features.select_dtypes([np.number]).isnull().any(axis="columns")
                self.features = self.features[~na_idx].reset_index(drop=True)
                self.h1n1_labels = self.h1n1_labels[~na_idx].reset_index(drop=True)
                self.seas_labels = self.seas_labels[~na_idx].reset_index(drop=True)

            else:
                num_features = self.features.select_dtypes(["number"])
                num_features_name = num_features.columns.to_list()
                obj_features = self.features.select_dtypes(["object"])

                if num_strat in ["mean", "median", "most_frequent"]:
                    imp = SimpleImputer(missing_values=np.nan, strategy=num_strat)
                    imp.fit(num_features)
                    num_features = pd.DataFrame(imp.transform(num_features), columns=num_features_name)

                elif num_strat == "knn":
                    imp = KNNImputer(n_neighbors=5)
                    num_features = pd.DataFrame(imp.fit_transform(num_features), columns=num_features_name)

                self.features = pd.concat([num_features, obj_features], axis="columns")

            if obj_strat == "remove":
                na_idx = self.features.select_dtypes(["object"]).isnull().any(axis="columns")
                self.features = self.features[~na_idx].reset_index(drop=True)
                self.h1n1_labels = self.h1n1_labels[~na_idx].reset_index(drop=True)
                self.seas_labels = self.seas_labels[~na_idx].reset_index(drop=True)

            else:
                num_features = self.features.select_dtypes(["number"])
                obj_features = self.features.select_dtypes(["object"])
                obj_features_name = obj_features.columns.to_list()

                if obj_strat == "most_frequent":
                    imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
                    imp.fit(obj_features)
                    obj_features = pd.DataFrame(imp.transform(obj_features), columns=obj_features_name)

                self.features = pd.concat([num_features, obj_features], axis="columns", ignore_index=True)

    def outlier_detection(self, n=0):
        removed = int()
        if n > 1:
            num_features = self.features.select_dtypes([np.number])
            lof = LocalOutlierFactor(n_neighbors=n)
            idx = np.where(lof.fit_predict(num_features) > 0, True, False)  # lof returns -1/1 which we change to use results as indexes
            removed = num_features.shape[0] - np.sum(idx)
            self.features = self.features[idx].reset_index(drop=True)
            self.h1n1_labels = self.h1n1_labels[idx].reset_index(drop=True)
            self.seas_labels = self.seas_labels[idx].reset_index(drop=True)
        return removed

    def numerize_categorical_features(self):
        pass

    def features_scaling(self):
        scaler = MinMaxScaler()
        num_features = pd.DataFrame(scaler.fit_transform(self.features.select_dtypes([np.number])))
        self.features = pd.concat([num_features, self.features.select_dtypes(["object"])], axis="columns", ignore_index=True)

    def feature_selection(self):
        # For now, we just remove non-numeric columns
        self.features = self.features.select_dtypes([np.number])
