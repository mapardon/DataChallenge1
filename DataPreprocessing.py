import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class DataPreprocessing:
    def __init__(self):
        self.features = None
        self.h1n1_labels = None
        self.seas_labels = None

    def data_preprocessing(self):
        self.load_data()
        #self.exploratory_analysis()

        # feature engineering
        self.feature_engineering()
        return self.features, self.h1n1_labels, self.seas_labels

    def load_data(self):
        self.features = pd.read_csv("data/training_set_features.csv")
        flu_labels = pd.read_csv("data/training_set_labels.csv")

        # shuffle dataset
        ds = self.features
        ds[["h1n1_vaccine", "seasonal_vaccine"]] = flu_labels[["h1n1_vaccine", "seasonal_vaccine"]]
        ds = ds.sample(frac=1)

        self.features = ds[ds.columns.to_list()[:-2]]
        self.seas_labels = ds[["respondent_id", "seasonal_vaccine"]]
        self.h1n1_labels = ds[["respondent_id", "h1n1_vaccine"]]

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

    def feature_engineering(self):
        ds = self.features
        ds["h1n1_vaccine"] = self.h1n1_labels["h1n1_vaccine"]
        ds["seasonal_vaccine"] = self.seas_labels["seasonal_vaccine"]

        # for now, we just remove missing data and non-numeric columns
        ds = ds[~ds.isnull().any(axis=1)]
        ds = ds.select_dtypes([np.number])
        # self.features = ds[["respondent_id", "doctor_recc_h1n1", "opinion_h1n1_risk", "opinion_h1n1_vacc_effective", "opinion_seas_risk"]]
        self.features = ds.iloc[:, 1:-3]
        self.seas_labels = ds["seasonal_vaccine"]
        self.h1n1_labels = ds["h1n1_vaccine"]
