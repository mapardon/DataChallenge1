"""
    Other experiments on learning algorithms and dataset
"""
import statistics

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier


def in_out_overfitting_test():
    """
        Test overfitting of the model by comparing the performance on train set vs test set
    """

    short = True
    if short:
        features = pd.read_pickle("serialized_df/features_h1n1_short")
        labels = pd.read_pickle("serialized_df/labels_h1n1_short")
    else:
        features = pd.read_pickle("serialized_df/features_h1n1")
        labels = pd.read_pickle("serialized_df/labels_h1n1")

    estimator = GradientBoostingClassifier(loss="log_loss", n_estimators=200, subsample=0.75, min_samples_split=2, max_depth=3)
    model = MLPClassifier(hidden_layer_sizes=[500,], activation="logistic", solver="sgd", max_iter=300)
    model.fit(features[:round(len(features) * 0.5)], labels[:round(len(features) * 0.5)])

    # compute AUCs
    print("In sample AUC:")
    y_i_pred_prob_in = model.predict_proba(features[:round(len(features) * 0.5)])[:, 1]
    print(roc_auc_score(labels[:round(len(features) * 0.5)], y_i_pred_prob_in))

    print("Out sample AUC:")
    y_i_pred_prob_out = model.predict_proba(features[round(len(features) * 0.5):])[:, 1]
    print(roc_auc_score(labels[round(len(features) * 0.5):], y_i_pred_prob_out))


def dp_frac_perf():
    """
        Plot the performance (on a test set) of a model trained on different fractions of the dataset
    """

    def data():
        features = pd.read_pickle("serialized_df/features_h1n1")
        labels = pd.DataFrame(pd.read_pickle("serialized_df/labels_h1n1"))

        # shuffle dataset (and reset indexes)
        ds = features
        ds[[labels.columns.to_list()[-1]]] = labels[[labels.columns.to_list()[-1]]]
        ds = ds.sample(frac=1)
        ds.reset_index(inplace=True, drop=True)

        # split features/labels and remove respondent_id
        labels = ds[ds.columns.to_list()[-1]]
        features = ds[ds.columns.to_list()[:-1]]

        test_features = features[round(len(features) * 0.9):]
        test_labels = labels[round(len(labels) * 0.9):]
        features = features[:round(len(features) * 0.9)]
        labels = labels[:round(len(labels) * 0.9)]

        return features, labels, test_features, test_labels

    model = LogisticRegression()

    res = list()
    n_exp = 100
    for i in range(10):
        res.append(list())

        for _ in range(n_exp):
            features, labels, test_features, test_labels = data()

            model.fit(features[:round(len(test_features) * ((i + 1) / 10))], labels[:round(len(test_features) * ((i + 1) / 10))])
            y_i_pred_prob = model.predict(test_features)
            res[-1].append(roc_auc_score(test_labels, y_i_pred_prob))

    fig, ax = plt.subplots()
    ax.plot([i for i in range(1, 11)], [statistics.mean(r) for r in res])
    title = "Experiment"
    ax.set(xlabel="%", ylabel='AUC', title=title)
    ax.grid()
    plt.show()

    for i in range(len(res)):
        print("{}: {}".format((i + 1) / 10, res[i]))

    print('>>', [statistics.mean(r) for r in res])


def rfe_prop():
    """
        Plot the performance of a model trained with dataset preprocessed with RFE and varying % of features selected
    """

    features_src = pd.read_pickle("serialized_df/features_h1n1")
    labels_src = pd.read_pickle("serialized_df/labels_h1n1")

    res, tmp = list(), list()

    for i in range(1, 9):

        tmp.clear()

        for _ in range(5):
            features = features_src.copy(deep=True)
            labels = labels_src.copy(deep=True)

            dp_model = LogisticRegression(max_iter=10000)
            selector = RFE(dp_model, n_features_to_select=i/8, step=1)
            selector = selector.fit(features, labels)

            features = features[[c for c, cond in zip(features.columns.to_list(), selector.support_) if cond]]

            model = LogisticRegression()
            model = GradientBoostingClassifier(n_estimators=250, subsample=0.75)
            model.fit(features[:len(features) // 2], labels[:len(features) // 2])
            pred = model.predict(features[len(features) // 2:])
            auc = roc_auc_score(labels[len(features) // 2:], pred)

            tmp.append(auc)

        print(tmp)
        res.append((i, statistics.mean(tmp)))

    print(res)

    fig, ax = plt.subplots()
    ax.plot([i[0] for i in res], [i[1] for i in res])

    title = "Best features proportion for RFE"
    ax.set(xlabel="Features %", ylabel='AUC', title=title)
    ax.grid()
    plt.show()
