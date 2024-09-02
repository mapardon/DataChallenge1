import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class DataPreprocessing:
    def __init__(self, short=False):
        self.features: pd.DataFrame | None = None
        self.labels: pd.DataFrame | None = None
        self.test_features: pd.DataFrame | None = None
        self.test_labels: pd.DataFrame | None = None
        self.resp_id: pd.DataFrame | None = None

        self.out_detect_res = None
        self.feat_sel_res = None

        self.short = short

    def training_preprocessing_pipeline(self, variant, features_src, labels_src, imp_num, imp_obj, nn, numerizer, scaler, feat_selector):
        """ Shortcut for loading and applying all preprocessing operations and returning processed dataset.
        Train/test sets are separated to apply only relevant treatments on test set (e.g., no outlier removal) """

        self.load_train_test_datasets(features_src, labels_src, variant)
        n_test = len(self.test_features)

        self.features = pd.concat([self.features, self.test_features], copy=False, ignore_index=True)
        self.labels = pd.concat([self.labels, self.test_labels], copy=False, ignore_index=True)
        self.missing_values_imputation(imp_num, imp_obj)
        self.numerize_categorical_features(numerizer)
        self.features_scaling(scaler)
        self.feature_selection(feat_selector)

        self.test_features = self.features[len(self.features) - n_test:]
        self.features = self.features[:len(self.features) - n_test]
        self.test_labels = self.labels[len(self.labels) - n_test:]
        self.labels = self.labels[:len(self.labels) - n_test]

        self.outlier_detection(nn)

        return self.get_train_test_datasets()

    def challenge_preprocessing_pipeline(self, features_src, imp_num, imp_obj, nn, numerizer, scaler, feat_selector):
        self.load_challenge_dataset(features_src)

        self.missing_values_imputation(imp_num, imp_obj)
        self.outlier_detection(nn)
        self.numerize_categorical_features(numerizer)
        self.features_scaling(scaler)
        self.feature_selection(feat_selector)

        return self.get_challenge_dataset()

    def get_train_test_datasets(self):
        """
            :returns: split dataset in 75/25% train set and test set
        """
        if self.test_features is None:
            train_features = self.features.iloc[:round(len(self.features) * 0.75), :]
            train_labels = self.labels[:round(len(self.features) * 0.75)]
            test_features = self.features.iloc[round(len(self.features) * 0.75):, :]
            test_labels = self.labels[round(len(self.features) * 0.75):]

        else:
            train_features = self.features
            train_labels = self.labels
            test_features = self.test_features
            test_labels = self.test_labels

        return train_features, train_labels, test_features, test_labels

    def get_challenge_dataset(self):
        """ Manage featureless challenge dataset """
        return self.resp_id, self.features

    def load_train_test_datasets(self, features_src, labels_src, variant):
        features = pd.read_csv(features_src)
        labels = pd.read_csv(labels_src)
        variant = "h1n1_vaccine" if variant == "h1n1" else "seasonal_vaccine"

        # shuffle dataset (and reset indexes)
        ds = features

        ds[[variant]] = labels[[variant]]
        ds = ds.sample(frac=1)
        ds.reset_index(inplace=True, drop=True)

        # debug purpose
        # TODO: remove in final version
        if self.short:
            ds = ds[:500]

        # split features/labels and remove respondent_id
        features = ds[ds.columns.to_list()[1:-2]]
        labels = ds[variant]

        # split train-test sets
        self.features = features[:round(len(features) * 0.75)]
        self.labels = labels[:round(len(labels) * 0.75)]
        self.test_features = features[round(len(features) * 0.75):]
        self.test_labels = labels[round(len(labels) * 0.75):]

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
        """ :returns: Number of outliers removed """

        n_removed = int()
        if nn > 1:

            num_features = self.features.select_dtypes([np.number])
            lof = LocalOutlierFactor(n_neighbors=nn)
            idx = np.where(lof.fit_predict(num_features) > 0, True, False)  # lof returns -1/1 which we convert to use results as indexes
            n_removed = num_features.shape[0] - np.sum(idx)
            self.features = self.features[idx].reset_index(drop=True)
            self.labels = self.labels[idx].reset_index(drop=True)

        self.out_detect_res = n_removed

    def get_outlier_detection_res(self):
        return self.out_detect_res

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

    def features_scaling(self, scaler=None):
        if scaler is not None:
            if scaler == "minmax":
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
        if feat_selector in ["mut_inf", "f_stat"]:
            n_corr_removed = self.remove_corr_features()
            self.feature_selection_proc(feat_selector)

        elif type(feat_selector) is list:
            self.feature_selection_list(feat_selector)

        self.feat_sel_res = n_corr_removed, len(self.features.columns.to_list()), self.features.columns.to_list()

    def get_feature_selection_res(self):
        return self.feat_sel_res

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