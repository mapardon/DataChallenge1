import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class DataPreprocessing:
    def __init__(self):
        self.features: pd.DataFrame | None = None
        self.labels: pd.DataFrame | None = None
        self.resp_id: pd.DataFrame | None = None

    def data_preprocessing_pipeline(self, variant, features_src, labels_src, imp_num, imp_obj, nn, numerizer, scaling, feat_selector):
        """ Shortcut for loading and applying all preprocessing operations and returning processed dataset """
        
        if labels_src is not None:
            self.load_train_test_datasets(features_src, labels_src, variant)
        else:
            self.load_challenge_dataset(features_src)

        self.missing_values_imputation(imp_num, imp_obj)
        _ = self.outlier_detection(nn)
        self.numerize_categorical_features(numerizer)
        if scaling:
            self.features_scaling()
        _ = self.feature_selection(feat_selector)

        return self.get_train_test_datasets() if labels_src is not None else self.get_challenge_dataset()

    def get_train_test_datasets(self):
        train_features = self.features.iloc[:round(len(self.features) * 0.75), :]
        train_labels = self.labels[:round(len(self.features) * 0.75)]

        test_features = self.features.iloc[round(len(self.features) * 0.75):, :]
        test_labels = self.labels[round(len(self.features) * 0.75):]

        return train_features, train_labels, test_features, test_labels

    def get_challenge_dataset(self):
        return self.resp_id, self.features

    def load_train_test_datasets(self, features_src, labels_src, variant):
        features = pd.read_csv(features_src)
        flu_labels = pd.read_csv(labels_src)
        variant = "h1n1_vaccine" if variant == "h1n1" else "seasonal_vaccine"

        # shuffle dataset (and reset indexes)
        ds = features

        ds[[variant]] = flu_labels[[variant]]
        ds = ds.sample(frac=1)
        ds.reset_index(inplace=True, drop=True)

        # split train-test sets
        short = False
        if short:
            ds = ds[:750]

        self.features = ds[ds.columns.to_list()[1:-2]]
        self.labels = ds[variant]

    def load_challenge_dataset(self, features_src):
        self.features = pd.read_csv(features_src)
        self.resp_id = self.features.pop("respondent_id")
        self.labels = None

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
        for df, feature, ax in [(self.labels, "h1n1_vaccine", ax1), (self.labels, "seasonal_vaccine", ax2)]:
            elements = sorted([(str(k), v) for k, v in df[feature].value_counts().to_dict().items()],
                              key=lambda x: x[0])
            ax.bar([e[0] for e in elements], [e[1] for e in elements], width=0.3, color=["tab:blue", "orange"])
            ax.set_title(feature.capitalize())
        plt.show()

    def missing_values_imputation(self, imp_num="remove", imp_obj="remove"):

        if imp_num == "remove" and imp_obj == "remove":
            na_idx = self.features.isnull().any(axis="columns")
            self.features = self.features[~na_idx].reset_index(drop=True)
            self.labels = self.labels[~na_idx].reset_index(drop=True)

        else:
            if imp_num == "remove":
                na_idx = self.features.select_dtypes([np.number]).isnull().any(axis="columns")
                self.features = self.features[~na_idx].reset_index(drop=True)
                self.labels = self.labels[~na_idx].reset_index(drop=True)

            else:
                num_features = self.features.select_dtypes(["number"])
                num_features_name = num_features.columns.to_list()
                obj_features = self.features.select_dtypes(["object"])

                if imp_num in ["mean", "median", "most_frequent"]:
                    imp = SimpleImputer(missing_values=np.nan, strategy=imp_num)
                    imp.fit(num_features)
                    num_features = pd.DataFrame(imp.transform(num_features), columns=num_features_name)

                elif imp_num == "knn":
                    imp = KNNImputer(n_neighbors=5)
                    num_features = pd.DataFrame(imp.fit_transform(num_features), columns=num_features_name)

                self.features = pd.concat([num_features, obj_features], axis="columns")

            if imp_obj == "remove":
                na_idx = self.features.select_dtypes(["object"]).isnull().any(axis="columns")
                self.features = self.features[~na_idx].reset_index(drop=True)
                self.labels = self.labels[~na_idx].reset_index(drop=True)

            else:
                num_features = self.features.select_dtypes(["number"])
                obj_features = self.features.select_dtypes(["object"])
                obj_features_name = obj_features.columns.to_list()

                if imp_obj == "most_frequent":
                    imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
                    imp.fit(obj_features)
                    obj_features = pd.DataFrame(imp.transform(obj_features), columns=obj_features_name)

                self.features = pd.concat([num_features, obj_features], axis="columns")

    def outlier_detection(self, nn=0):

        removed = int()
        if nn > 1:
            # test set should not be outlier processed
            train_features, train_labels, test_features, test_labels = self.get_train_test_datasets()

            num_features = train_features.select_dtypes([np.number])
            lof = LocalOutlierFactor(n_neighbors=nn)
            idx = np.where(lof.fit_predict(num_features) > 0, True, False)  # lof returns -1/1 which we change to use results as indexes
            removed = num_features.shape[0] - np.sum(idx)
            train_features = train_features[idx].reset_index(drop=True)
            train_labels = train_labels[idx].reset_index(drop=True)

            # reconstitute datasets
            self.features = pd.concat([train_features, test_features], axis="rows", ignore_index=True)
            self.labels = pd.concat([train_labels, test_labels], axis="rows", ignore_index=True)

        return removed

    def numerize_categorical_features(self, numerizer="remove"):
        if numerizer == "remove":
            self.features = self.features.select_dtypes([np.number])

        elif numerizer == "one-hot":
            num_features = self.features.select_dtypes(["number"])
            obj_features = self.features.select_dtypes(["object"])
            enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None)
            enc.set_output(transform="pandas")
            obj_features = enc.fit_transform(obj_features)
            self.features = pd.concat([num_features, obj_features], axis="columns")

    def features_scaling(self):
        scaler = MinMaxScaler()
        self.features[self.features.select_dtypes([np.number]).columns] = scaler.fit_transform(self.features.select_dtypes([np.number]))

    def remove_corr_features(self):
        """ Remove highly correlated features (useless and error source)
        :returns: Number of correlated features removed """

        corr_matrix = self.features.corr(numeric_only=True).to_dict()
        correlated_features = dict()
        for k1 in corr_matrix:
            for k2 in list(corr_matrix.keys())[:list(corr_matrix.keys()).index(k1) + 1]:
                corr_matrix[k1][k2] = round(corr_matrix[k1][k2], 4 if corr_matrix[k1][k2] < 0 else 5)

                if abs(corr_matrix[k1][k2]) > 0.6 and k1 != k2:
                    if k1 not in correlated_features and k2 not in correlated_features:
                        correlated_features[k1] = {k2}
                    elif k1 in correlated_features:
                        correlated_features[k1].add(k2)
                    elif k2 in correlated_features:
                        correlated_features[k2].add(k1)

        self.features.drop(columns=correlated_features.keys(), inplace=True)
        return len(correlated_features)

    def feature_selection(self, feat_selector=None):
        """ Calls the procedure removing highly correlated features (similar idea as this function) then calls the
        adequate feature_selection procedure depending on the type of the feat_selector parameter

        :returns: number of correlated features removed, final number of features selected by procedure, list of
         selected features """

        n_corr_removed = int()
        if type(feat_selector) is str:
            n_corr_removed = self.remove_corr_features()
            self.feature_selection_proc(feat_selector)

        elif type(feat_selector) is list:
            self.feature_selection_list(feat_selector)

        return n_corr_removed, len(self.features.columns.to_list()), self.features.columns.to_list()

    def feature_selection_proc(self, feat_selector):
        """ Runs a feature selection algorithm """

        if feat_selector == "mut_inf":
            feat_ranking = sorted([(name, info) for name, info in zip(self.features.columns.to_list(),
                                                                      mutual_info_classif(self.features, self.labels))],
                                  reverse=True, key=lambda x: x[1])
            self.features = self.features[[f[0] for f in feat_ranking if f[1] > 0]]

        elif feat_selector == "f_stat":
            feat_ranking = sorted([(name, p_val) for name, p_val in zip(self.features.columns.to_list(),
                                                                        f_classif(self.features, self.labels)[1])],
                                  key=lambda x: x[1])
            self.features = self.features[[f[0] for f in feat_ranking if f[1] < 0.05]]

    def feature_selection_list(self, selected_features):
        """ Select features specified in the list parameter """

        self.features = self.features[selected_features]
