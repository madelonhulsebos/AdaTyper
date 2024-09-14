import collections
import itertools
import json
import os
import random
import re
import typing

import joblib
import numpy as np
import pandas as pd
import requests
import spacy_universal_sentence_encoder
import torch
import tqdm
import urllib3

from sklearn import base, preprocessing, ensemble

from typetabert.typetabert import typetabert
from adatyper import settings, utils, value_matching
from table_bert.table import Column, Table

# For the KB lookup.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class HeaderSyntacticEstimator(base.BaseEstimator, base.ClassifierMixin):
    """
    Estimator for inferring types from column headers.
    It compares a cleaned column name with the type ontology.
    If they match, it is predicted to be of the matching type.

    Attributes
    ----------
    class_labels
        List of labels with types per datapoint, should cover all classes.
    """

    def __init__(self, class_labels):
        super().__init__()
        self.class_labels = class_labels
        self.class_labels_ = None

    def fit(self, X, y):
        print("Fitting Syntactic Header estimator.")

        self.classes_ = np.unique(self.class_labels).tolist()
        self.le = preprocessing.LabelEncoder().fit(self.classes_)
        
        self.use = spacy_universal_sentence_encoder.load_model('en_use_lg')
        self.type_embeddings = [self.use(_type) for _type in self.classes_]

        return self

    def predict(self, X, threshold):
        probs = self.predict_proba(X)
        y_hat = probs.idxmax(axis=1)

        y_hat_probs = probs.max(axis=1)
        null_idx = y_hat_probs < threshold
        y_hat[null_idx] = "null"

        return y_hat

    def predict_proba(self, X):
        X = X["values"]
        probs = pd.DataFrame(columns=self.classes_, index=np.arange(len(X)))

        for i, index in enumerate(X.index):
            column_name = index.lower().replace("_", " ").replace("-", " ")
            for j, _type in enumerate(self.classes_):
                # match string syntactically
                if column_name == _type:
                    probs.loc[i, _type] = 100
                else:
                    # match string semantically
                    probs.loc[i, _type] = self.use(column_name).similarity(self.type_embeddings[j]) * 100

        return probs.fillna(0).astype(float)


class ValuesRegexEstimator(base.BaseEstimator, base.ClassifierMixin):
    """
    Estimator for inferring types based on regular expression matching.
    It compares a column values with predefined regular expressions.
    The type yielding most matches between its regex and the column values,
    is the predicted type.

    Attributes
    ----------
    class_labels
        List of labels with types per datapoint, should cover all classes.
    """

    def __init__(self, class_labels):
        super().__init__()
        self.class_labels = class_labels
        self.class_labels_ = None

    def fit(self, X, y):
        print("Fitting RegEx estimator.")

        self.classes_ = np.unique(y).tolist()
        self.le = preprocessing.LabelEncoder().fit(self.class_labels)
        self.class_labels_ = np.unique(self.class_labels)

        return self

    def predict(self, X, threshold):
        probs = self.predict_proba(X)
        y_hat = probs.idxmax(axis=1)

        y_hat_probs = probs.max(axis=1)
        null_idx = y_hat_probs < threshold
        y_hat[null_idx] = "null"

        return y_hat

    def predict_proba(self, X):
        X = X["values"]
        probs = pd.DataFrame(columns=self.class_labels_, index=np.arange(len(X)))
        regex_dict = value_matching.get_regular_expression_dict()

        for i, col in enumerate(X):
            # Each row of the series is assumed to consist of a list of values.
            values = sample_values_from_column(col, n=20)
            values = [str(val) for val in col]
            for _type, regexes in regex_dict.items():
                if _type in self.class_labels_:
                    highest_matches = 0
                    for regex in regexes["regular_expressions"]:
                        compiled_regex = re.compile(regex)
                        num_matches = len(list(filter(compiled_regex.match, values)))
                        if num_matches > highest_matches:
                            probs.loc[i, _type] = (num_matches / len(values)) * 100
                            highest_percentage = num_matches

        return probs.fillna(0).astype(float)


class ValuesLabelingFunctionEstimator(base.BaseEstimator, base.ClassifierMixin):
    """
    Estimator for inferring types based on regular expression matching.
    It compares a column values with predefined regular expressions.
    The type yielding most matches between its regex and the column values,
    is the predicted type.

    Attributes
    ----------
    class_labels
        List of labels with types per datapoint, should cover all classes.
    """

    def __init__(self, class_labels):
        super().__init__()
        self.class_labels = class_labels
        self.class_labels_ = None

    def fit(self, X, y):
        print("Fitting LF estimator.")

        X = X["values"]
        self.classes_ = np.unique(y).tolist()
        self.le = preprocessing.LabelEncoder().fit(self.class_labels)
        self.class_labels_ = np.unique(self.class_labels)

        return self

    # TODO: compile labeling functions from config files of newly added types and add here.
    def predict(self, X, threshold):
        probs = self.predict_proba(X)
        y_hat = probs.idxmax(axis=1)

        y_hat_probs = probs.max(axis=1)
        null_idx = y_hat_probs < threshold
        y_hat[null_idx] = "null"

        return y_hat

    def predict_proba(self, X):
        X = X["values"]
        probs = pd.DataFrame(columns=self.class_labels_, index=np.arange(len(X)))
        rule_dict = value_matching.get_type_function_dict()

        for i, col in enumerate(X):
            values = sample_values_from_column(col, n=20)
            for _type, rule_list in rule_dict.items():
                if _type in self.class_labels_:
                    for rule in rule_list:
                        num_matches = sum([rule(value) for value in values])
                        if num_matches > 0:
                            probs.loc[i, _type] = (num_matches / len(values)) * 100

        return probs.fillna(0).astype(float)


class ValuesKnowledgeBaseEstimator(base.BaseEstimator, base.ClassifierMixin):
    def __init__(self, class_labels):
        super().__init__()
        self.class_labels = class_labels
        self.class_labels_ = None

    def fit(self, X, y):
        print("Fitting KB estimator.")

        X = X["values"]
        self.classes_ = np.unique(y).tolist()
        self.le = preprocessing.LabelEncoder().fit(self.class_labels)
        self.class_labels_ = np.unique(self.class_labels)

        return self

    def predict(self, X, threshold):
        probs = self.predict_proba(X)
    
        y_hat = probs.idxmax(axis=1)

        y_hat_probs = probs.max(axis=1)
        null_idx = y_hat_probs < threshold
        y_hat[null_idx] = "null"

        return y_hat

    def predict_proba(self, X):
        X = X["values"]
        probs = pd.DataFrame(columns=self.class_labels_, index=np.arange(len(X)))

        for i, col in enumerate(X):
            j = 0
            num_matches = 0
            for value in col:  # values:
                # Lookup is costly, only lookup two values.
                if j > 3:
                    break
                # Do not look up non-string values.
                if not isinstance(value, str):
                    continue
                j += 1
                response = requests.get(
                    f"https://lookup.dbpedia.org/api/search?format=JSON&query={value}&MaxHits=1",
                    verify=False
                )
                dbpedia_type_data = response.json()
                if not "docs" in dbpedia_type_data.keys():
                    continue
                if not len(dbpedia_type_data["docs"]) > 0:
                    continue
                if "typeName" in dbpedia_type_data["docs"][0].keys():
                    for dbpedia_class in dbpedia_type_data["docs"][0]["typeName"]:
                        _type = (
                            re.sub(r"([A-Z])", r" \1", dbpedia_class).lower().strip()
                        )
                        if _type in self.class_labels_:
                            num_matches += 1
                        else:
                            continue
                else:
                    continue

            if num_matches > 0:
                probs.loc[i, _type] = num_matches / j * 100

        return probs.fillna(0).astype(float)


class ValuesDictionaryEstimator(base.BaseEstimator, base.ClassifierMixin):

    # TODO: serialize dictionary and adapt based on data generated from model adaptation.
    # This could be achieved by storing the dictionary as a json on `fit`, loading it on `predict`, loading and adapting on `fit(adapt=True)` or with additional method "adapt".
    # Preferably the majority voting classifier is replaced with a pipeline conditioned on confidence per prediction.

    # The pretrained dictionary should contain roughly equal numbers of samples per type,
    # otherwise overrepresented types will dominate classifier.
    def __init__(self, class_labels):
        super().__init__()
        self.class_labels = class_labels
        self.class_labels_ = None
        self.dictionary_ = None

    def fit(self, X, y):
        print("Fitting dictionary estimator.")

        X = X["values"]
        self.class_labels_ = np.unique(y).tolist()
        # self.le = preprocessing.LabelEncoder().fit(self.classes)
        # self.classes = np.unique(self.class_labels)
        # column_labels = self.le.inverse_transform(y)

        type_dictionary = {}
        for _type in self.class_labels_:
            type_dictionary[_type] = []

        for i, column in enumerate(X):
            _type = y.iloc[i]
            most_freq_values = [
                _tuple[0] for _tuple in collections.Counter(column).most_common(25)
            ]
            type_dictionary[_type] += most_freq_values

        self.dictionary_ = type_dictionary

        return self

    def predict(self, X, threshold):
        probs = self.predict_proba(X)
        y_hat = probs.idxmax(axis=1)

        y_hat_probs = probs.max(axis=1)
        null_idx = y_hat_probs < threshold
        y_hat[null_idx] = "null"

        return y_hat

    def predict_proba(self, X):
        X = X["values"]
        probs = pd.DataFrame(columns=self.class_labels_, index=np.arange(len(X)))

        for i, column in enumerate(X):
            values = sample_values_from_column(column, n=20)
            for _type, common_values in self.dictionary_.items():
                try:
                    num_matches = len(set(common_values).intersection(set(values)))
                except:
                    num_matches = 0
                if num_matches > 0:
                    probs.loc[i, _type] = (num_matches / len(values)) * 100

        return probs.fillna(0).astype(float)



class TableMachineLearningExperimentEstimator(base.BaseEstimator, base.ClassifierMixin):

    def __init__(self, class_labels):
        super().__init__()
        self.class_labels = class_labels
        self.class_labels_ = None
        self.rf_ = None

        model_path = "typetabert/typetabert/models/"
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)

        # TODO: persist this model in memory.
        self.typetabert_model = typetabert.TypeTaBERTModel(
            config["num_classes"],
            "cpu",
            joblib.load(os.path.join(model_path, "label_encoder.joblib")),
        )
        self.typetabert_model.load_state_dict(
            torch.load(
                os.path.join(model_path, "model_epoch0.bin"),
                map_location="cpu"
            )
        )
        self.typetabert_model.eval()

        self.dtype_mapping = {
            "object": "text",
            "float64": "real",
            "int64": "real",
            "bool": "real",
            "datetime64[ns]": "text",
            "datetime64": "text",
            "timedelta[ns]": "text",
            "category": "text",
        }

    def fit(self, X_encoded: pd.DataFrame, y_encoded: pd.Series):
        print("Fitting ML estimator.")

        self.le = preprocessing.LabelEncoder().fit(self.class_labels)
        self.class_labels_ = np.unique(self.class_labels)

        # The classes in the training data may be incomplete regarding the type catalog.
        all_class_frequencies = pd.Series([0]*len(self.class_labels_), index=self.class_labels_)
        all_class_frequencies.loc[y_encoded.value_counts().index] = y_encoded.value_counts(normalize=True)

        # The weights are used to account for the heavy imbalance between types.
        # For example, we train TypeTaBERT on a background class which occurs very often.
        weights = 1 - np.array(all_class_frequencies.tolist())
        classes = all_class_frequencies.index.tolist()
        
        class_weights = dict(zip(classes, weights))

        rf = ensemble.RandomForestClassifier(n_estimators=500, random_state=settings.RANDOM_STATE, class_weight=class_weights)
        rf.fit(X_encoded, y_encoded)

        self.rf_ = rf

        return self


    def predict(self, X_encoded: pd.DataFrame, threshold):
        """."""
        probs = self.predict_proba(X_encoded)
        y_hat = self.rf_.predict(X_encoded)

        y_hat_probs = probs.max(axis=1)

        null_idx = y_hat_probs < threshold
        y_hat[null_idx] = "null"

        return y_hat


    def predict_proba(self, X_encoded: pd.DataFrame):
        probs = pd.DataFrame(
            self.rf_.predict_proba(X_encoded) * 100,
            columns=self.rf_.classes_
        ).astype(float)

        return probs


def sample_values_from_column(column: typing.List, n: int):
    random.seed(settings.RANDOM_STATE)

    num_samples = n if len(column) >= n else len(column)
    values = random.sample(column, num_samples)

    return values
