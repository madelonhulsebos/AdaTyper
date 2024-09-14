import argparse
import datetime
import joblib
import json
import os
import time
import typing

import numpy as np
import pandas as pd
import torch
import tqdm

from joblib import dump, load
from sklearn import ensemble, metrics, model_selection
from sklearn.utils.validation import check_is_fitted

from typetabert.typetabert import typetabert
from adatyper import adaptation, base_estimator, settings, utils
from table_bert.table import Column, Table



class SequentialPipeline():

    def __init__(self, estimator_names, model_key, pipeline_identifier):
        pipeline = {}
        for estimator_name in estimator_names:
            estimator = load(f"models/SingleTypeClassifier_{model_key}_{pipeline_identifier}_{estimator_name}.joblib")
            pipeline[estimator_name] = estimator
        
        self.pipeline = pipeline

    def predict(self, X, X_encoded):
        """Method for predicting types for the given columns in X and X_encoded (which should be aligned and have the same shape).
        Note that y and y_encoded should correspond to the training labels, to inform the classes of the estimators.

        Arguments
        ---------
        X
        X_encoded
        y
        y_encoded
        """
        predictions = {}
        y_pred = []
        estimator_config = get_estimators()
        for i, (column, column_encoding) in tqdm.tqdm(enumerate(zip(X.iterrows(), X_encoded.iterrows()))):
            no_prediction = True
            estimator_step = 0
            while no_prediction and estimator_step < len(estimator_config):
                estimator_name = estimator_config[estimator_step]["name"]
                estimator = self.pipeline[estimator_name]
                if estimator_name == "TableMachineLearningExperimentEstimator":
                    # the first item of the tuple is the index, the second the row as a series
                    type_prediction = estimator.predict(pd.DataFrame(column_encoding[1]).transpose(), estimator_config[estimator_step]["threshold"])
                    no_prediction = False
                else:
                    # the first item of the tuple is the index, the second the row as a series
                    type_prediction = estimator.predict(pd.DataFrame(column[1]).transpose(), estimator_config[estimator_step]["threshold"])
                    if type_prediction[0] != "null":
                        no_prediction = False
            
                estimator_step += 1
            
            y_pred.append(type_prediction[0])

        return y_pred




def train_type_classifier(
    train_set_id: str,
    test_set_id: str,
    pipeline_identifier: str,
):
    """
    Train type detection pipeline consisting out of the follow steps:
    1) Header: syntactic string matching per column name,
    2) Values: regular expression matching,
    3) Values: labeling function evaluation,
    4) Values: dictionary (populated by training data) lookup,
    6) Table: machine learning model.

    Parameters
    ----------
    data_identifier
        Identifier of the training data to train the pipeline on.
    pipeline_identifier
        Identifier to store the trained pipeline with.
    """
    X_train, X_train_encoded, X_test, X_test_encoded, y_train, y_train_encoded, y_test, y_test_encoded = utils.load_training_data(train_set_id, test_set_id)
    
    estimators = fit_and_store_model(X_train, X_train_encoded, y_train, y_train_encoded, pipeline_identifier)

    model_key = "experiments"

    with open(
        f"models/version_metadata/model_metadata_{model_key}_{pipeline_identifier}.json",
        "w+",
    ) as f:
        metadata = {
            "datetime": str(datetime.date.today()),
            "model_key": f"SequentialTypeClassifier_{model_key}_{pipeline_identifier}",
            "train_data_id": train_set_id,
            "test_data_id": test_set_id,
            "train_size": str(X_train.shape[0]),
            "train_encoded_size": str(X_train_encoded.shape[0]),
            "test_size": str(X_test.shape[0]),
            "test_encoded_size": str(X_test_encoded.shape[0]),
            "types": np.unique(y_train).tolist(),
        }
        json.dump(metadata, f)


def fit_and_store_model(
    X_train: pd.DataFrame, X_train_encoded: pd.DataFrame, y_train: pd.Series, y_train_encoded: pd.Series,
    pipeline_identifier: str, with_adaptation_types=False,
):
    """
    Train and store the type detection pipeline, which is based on
    a (weighted) majority vote from the individual base estimators.

    Parameters
    ----------
    pipeline_identifier
        Identifier of the updated model.
    """
    model_key = "experiments"

    class_labels = [str(type_) for type_ in json.load(open("adatyper/types.json")).keys()]
    estimators = get_estimators(class_labels, y_train_encoded)
    training_runtimes = {}
    for i, estimator in enumerate(estimators):
        estimator_name = estimator["name"]
        if estimator_name == "TableMachineLearningExperimentEstimator":
            start = time.time()
            estimator["estimator"].fit(X_train_encoded, y_train_encoded)
            total_time = time.time() - start
        else:
            start = time.time()
            estimator["estimator"].fit(X_train, y_train)
            total_time = time.time() - start
        
        training_runtimes[estimator_name] = total_time
        dump(estimator["estimator"], f"models/SingleTypeClassifier_{model_key}_{pipeline_identifier}_{estimator_name}.joblib")
        estimators[i] = estimator

    # dump(estimators, f"models/SequentialTypeClassifier_{model_key}_{pipeline_identifier}.joblib")
    json.dump(training_runtimes, open(f"evaluation/results/training_runtimes_{pipeline_identifier}.json", "w+"))

    return estimators


def evaluate_estimator_thresholds(X_test, X_test_encoded, y_test, y_test_encoded, estimator_names, model_key, pipeline_identifier):

    roc_curves = {}
    for estimator_name in estimator_names:

        estimator = load(f"models/SingleTypeClassifier_{model_key}_{pipeline_identifier}_{estimator_name}.joblib")

        print(check_is_fitted(estimator))

        if estimator_name == "TableMachineLearningExperimentEstimator":
            y_pred = estimator.predict_proba(X_test_encoded)
        else:
            y_pred = estimator.predict_proba(X_test)

        y_pred = y_pred/100
        # Focus on "foreground" classes, i.e. the target types without background class (null).
        y_pred_target = y_pred.drop("null", axis=1)

        y_test_oh = pd.DataFrame(np.zeros_like(y_pred_target))
        y_test_oh.columns = y_pred_target.columns
        for i, class_ in enumerate(y_test):
            y_test_oh.iloc[i][class_] = 1

        y_test_oh = y_test_oh.to_numpy()
        y_pred_target = y_pred_target.to_numpy()

        fpr = dict()
        tpr = dict()
        for i in range(y_pred_target.shape[1]):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_test_oh[:, i], y_pred_target[:, i])

        fpr["micro"], tpr["micro"], thresholds = metrics.roc_curve(y_test_oh.ravel(), y_pred_target.ravel())

        roc_curves[estimator_name] = {"fpr_micro": fpr["micro"].tolist(), "tpr_micro": tpr["micro"].tolist(), "thresholds": thresholds.tolist()}

    return roc_curves


def evaluate_configured_estimators(X_test, X_test_encoded, y_test, y_test_encoded, estimator_names, model_key, pipeline_identifier, individual_mode=True, sequential_mode=False):
    
    clsf_report, precision_recall_scores, runtimes = None, None, None

    if individual_mode:
        # Get the config
        precision_recall_scores = {}
        runtimes = {}
        class_labels = [type_ for type_ in json.load(open("adatyper/types.json")).keys()]
        pipeline = get_estimators(class_labels, y_test_encoded)
        for i, estimator_config in enumerate(pipeline):
            estimator_name = estimator_config["name"]
            estimator = load(f"models/SingleTypeClassifier_{model_key}_{pipeline_identifier}_{estimator_name}.joblib")

            if estimator_name == "TableMachineLearningExperimentEstimator":
                start = time.time()
                y_pred = estimator.predict(X_test_encoded, estimator_config["threshold"])
                total_time = time.time() - start
                precision = metrics.precision_score(y_test_encoded, y_pred, average="macro")
                recall = metrics.recall_score(y_test_encoded, y_pred, average="macro")

                clsf_report = metrics.classification_report(y_test, y_pred)
            else:
                start = time.time()
                y_pred = estimator.predict(X_test, estimator_config["threshold"])
                total_time = time.time() - start
                precision = metrics.precision_score(y_test, y_pred, average="macro")
                recall = metrics.recall_score(y_test, y_pred, average="macro")

            runtimes[estimator_name] = total_time
            precision_recall_scores[estimator_name] = {"precision": precision, "recall": recall}

    if sequential_mode:

        sequential_pipeline = SequentialPipeline(estimator_names, model_key, pipeline_identifier)
        y_pred = sequential_pipeline.predict(X_test, X_test_encoded)

        clsf_report = metrics.classification_report(y_test.astype(str), pd.Series(y_pred).fillna("null").astype(str), output_dict=True)

    return precision_recall_scores, runtimes, clsf_report



def get_type_predictions(table: pd.DataFrame, model_key: str, pipeline_identifier: str) -> typing.Dict:
    """
    Get semantic type predictions for each table column along with confidence scores per prediction.
    The pipeline used to make predictions can be pretrained or customized.

    Parameters
    ----------
    table
        Table for which predictions per column should be returned.
    model_key
        ID of the model.
    pipeline_identifier
        ID (number) of the trained pipeline files to use for predictions.

    Returns
    -------
    predictions
        Nested dictionary with the html formatted table and predictions with column names as keys.
        Each column key has a dictionary with keys "type" and "confidence".
    """
    column_values = []
    column_names = []
    column_dtypes = []
    table_numbers = []
    for col in table:
        values = table[col].tolist()
        column_values.append(values)
        column_names.append(col)
        column_dtypes.append(str(table[col].dtype))
        table_numbers.append(1)

    X = pd.DataFrame(pd.Series(column_values, index=column_names, name="values"))
    X["dtype"] = pd.Series(column_dtypes, index=column_names, name="dtype")
    X["table_number"] = pd.Series(
        table_numbers, index=column_names, name="table_number"
    )

    tabert_table = []
    columns = table.columns
    dtypes = table.dtypes
    for k, column in enumerate(columns):
        values = table[column].tolist()
        sample_value = values[0]
        tabert_table.append({"column_name": column, "values": values, "sample_value": sample_value, "dtype": str(dtypes[k])})

    typetabert_model = utils.load_typetabert_model()
    X_encoded = utils.encode_tabert_table(typetabert_model, tabert_table, "1")

    model_key = "experiments"
    pipeline_identifier = "19"
    estimator_names = [estimator["name"] for estimator in get_estimators()]
    sequential_pipeline = SequentialPipeline(estimator_names, model_key, pipeline_identifier)
    y_pred = sequential_pipeline.predict(X, pd.DataFrame(X_encoded).reset_index(drop=False))

    predictions = {**{"table": table.head().to_html()}, **{"prediction": y_pred}}

    return predictions, X_encoded


def adapt_pipeline(column_embedding, example_value: str, train_set_id: str, test_set_id: str, _type: str, regular_expression: str = None):
    
    X_train, X_train_encoded, X_test, X_test_encoded, y_train, y_test, y_train_encoded, y_test_encoded = utils.load_training_data(train_set_id, test_set_id, _type)
    initial_table_number = int(np.nanmax(X_train["table_number"].unique().tolist() + X_test["table_number"].unique().tolist()))

    generated_training_data, query_time, recall = adaptation.generate_training_data_from_embeddings(
        initial_table_number,
        column_embedding,
        _type,
    )

    num_new_samples = len(new_training_data)
    print(f"The number of new training samples for type {_type} is: {num_new_samples}")

    # TODO: update new training table numbers for typetabert.
    X_train_encoded = X_train_encoded.append(new_training_data[0][0].apply(pd.Series))
    y_train_encoded = y_train_encoded.append(np.repeat(new_type, num_new_samples))

    X_train_encoded.to_pickle(
        f"data/training_data/raw/training_set_encoded_latest.pickle",
        protocol=4,
    )

    # Adapt pipeline to new_training_data
    estimators = fit_and_store_model(
        X_train,
        X_train_encoded,
        y_train,
        y_train_encoded,
        pipeline_identifier="latest",
        with_adaptation_types=True
    )

    if regular_expression:
        adaptation.adapt_regular_expression_dict(_type, example_value, regular_expression)

    model_key = "experiments"
    estimator_names = [estimator["name"] for estimator in estimators]
    sequential_pipeline = SequentialPipeline(estimator_names, model_key, "latest")
    # TODO: check if labels are consistent across y_train and y_test!
    y_pred = sequential_pipeline.predict(X_test, X_test_encoded)

    with open("adatyper/types.json") as j:
        types = json.load(j)
        if type_ not in types:
            types[_type] = "adaptation"

    json.dump(types, "adatyper/types.json")


def get_estimators(class_labels: pd.Series = None, y_train_encoded: pd.Series = None):
    """
    Get named base estimators and weights associated with each base estimator.

    Parameters
    ----------
    y_train
        Class labels (strings) of each sample in the training dataset.
    """
    estimators = [
        {
            "name": "HeaderSyntacticEstimator",
            "estimator": base_estimator.HeaderSyntacticEstimator(class_labels=class_labels),
            "threshold": 70,
        },
        {
            "name": "ValuesRegexEstimator",
            "estimator": base_estimator.ValuesRegexEstimator(class_labels=class_labels),
            "threshold": 1,
        },
        # {
        #     "name": "ValuesLabelingFunctionEstimator",
        #     "estimator": base_estimator.ValuesLabelingFunctionEstimator(class_labels=class_labels),
        #     "threshold": 80,
        # },
        {
            "name": "ValuesDictionaryEstimator",
            "estimator": base_estimator.ValuesDictionaryEstimator(class_labels=class_labels),
            "threshold": 35,
        },
        # The KB estimator is currently heavy due to dependency on web API (DBpedia search).
        # (
        #     "Values knowledge base estimator",
        #     base_estimator.ValuesKnowledgeBaseEstimator(class_labels=class_labels)
        # ),
        # (
        #     "Table machine learning estimator",
        #     base_estimator.TableMachineLearningEstimator(class_labels=class_labels),
        # ),
        {
            "name": "TableMachineLearningExperimentEstimator",
            # experiment estimator uses a random forest classifier instead of a retrained tabert model
            "estimator": base_estimator.TableMachineLearningExperimentEstimator(class_labels=y_train_encoded),
            "threshold": 7,
        },
    ]

    return estimators


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for training the type detection pipeline."
    )

    parser.add_argument(
        "--train_set_id",
        type=int,
        help="""
            Identifier for training data to be used for training the type detectoin pipeline.
            Training data with this id should exist in the 'data/raw/' directory.
        """,
    )
    parser.add_argument(
        "--test_set_id",
        type=int,
        help="""
            Identifier for test data to be used for training the type detectoin pipeline.
            Test data with this id should exist in the 'data/raw/' directory.
        """,
    )
    parser.add_argument(
        "--pipeline_identifier",
        type=int,
        help="""
            Identifier for newly trained type detection pipeline.
            The model will be stored as 'models/SequentialTypeClassifier_<model key>_<pipeline identifier>.joblib'.
        """,
    )
    args = parser.parse_args()
    train_set_id = args.train_set_id
    test_set_id = args.test_set_id
    pipeline_identifier = args.pipeline_identifier

    train_type_classifier(train_set_id, test_set_id, pipeline_identifier)
