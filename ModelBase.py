from abc import ABC

import pandas as pd


class ModelCandidate:
    def __init__(self, model, auc, pars, is_reg_model):
        self.model = model
        self.auc: float = auc
        self.pars: str = pars
        self.is_reg_model: bool = is_reg_model

    def __repr__(self):
        return "{} ({}): {}".format(str(type(self.model)).split('.')[-1].split("'")[0], self.pars, self.auc)


class ModelBase(ABC):
    """
        Base class for model identification and model selection phases
    """
    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, test_features: pd.DataFrame,
                 test_labels: pd.DataFrame, cv_folds: int, verbose=False):
        self.train_features = train_features
        self.train_labels = train_labels

        self.test_features = test_features
        self.test_labels = test_labels

        self.cv_folds = max(cv_folds, 2)
        self.candidates: list[ModelCandidate] = list()
        self.verbose = verbose

    def parametric_identification_cv(self, model, is_reg_model=False):
        """
            Generic loop training the provided model on the training set (split in training/validation
            folds) and assessing performance.

            :return: AUC of the models of the different loops
        """

        n_rows_fold = len(self.train_features) // self.cv_folds
        auc = list()

        for i in range(self.cv_folds):
            X_i = self.train_features[self.train_features.columns.to_list()[:]]
            X_i_tr = pd.concat([X_i.iloc[: n_rows_fold * i], X_i.iloc[n_rows_fold * (i + 1):]], axis=0,
                               ignore_index=True)
            X_i_vs = X_i.iloc[n_rows_fold * i: n_rows_fold * (i + 1)]

            y_i = self.train_labels
            y_i_tr = pd.concat([y_i.iloc[: n_rows_fold * i], y_i.iloc[n_rows_fold * (i + 1):]], axis=0,
                               ignore_index=True)
            y_i_vs = y_i.iloc[n_rows_fold * i: n_rows_fold * (i + 1)].astype(float)

            # train + predict probabilities
            model.fit(X_i_tr, y_i_tr)
            y_i_pred_prob = expit(model.predict(X_i_vs)) if is_reg_model else model.predict_proba(X_i_vs)[:, 1]

            # compute AUC
            auc.append(roc_auc_score(y_i_vs, y_i_pred_prob))

        return auc
