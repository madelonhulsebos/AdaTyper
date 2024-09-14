import itertools
import joblib
import json
import os
import typing

import pandas as pd
import torch

from sklearn import model_selection
from table_bert.table import Column, Table

from typetabert import typetabert
from adatyper import settings


def load_adaptation_training_data(data_identifier: int):

    X_train, X_train_encoded, _, _, y_train, y_train_encoded, _, _ = load_training_data(data_identifier, data_identifier)

    last_table_nr = int(X_train["table_number"].max())
    last_table_nr_encoded = int(X_train_encoded.reset_index()["table_number"].max())

    # Remove the types to which we adapt from the catalog.
    scoped_types = json.load(open("adatyper/types.json"))
    types = [type_ for type_, category in scoped_types.items()]

    X_train = X_train[y_train.isin(types)]
    y_train = y_train[y_train.isin(types)]

    X_train_encoded = X_train_encoded[y_train_encoded.isin(types)]
    y_train_encoded = y_train_encoded[y_train_encoded.isin(types)]

    return X_train, X_train_encoded, y_train, y_train_encoded, last_table_nr, last_table_nr_encoded


def load_ctu_evaluation_data():

    df = pd.read_pickle("data/ctu_evaluation_data/processed/ctu_dataset.pickle")
    df_encoded = pd.read_pickle("data/ctu_evaluation_data/processed/ctu_dataset_encoded.pickle")

    scoped_types = json.load(open("adatyper/types.json"))
    types = [type_ for type_, category in scoped_types.items()]

    df = df[df["type"].isin(types)]
    df_encoded = df_encoded[df_encoded["types"].isin(types)]

    X_test = df.drop("type", axis=1)
    y_test = df["type"]

    X_test_encoded = df_encoded["table_encoded"].apply(pd.Series)
    X_test_encoded.index = df_encoded["table_number"].drop("table_number", axis=1)
    y_test_encoded = df_encoded["types"]

    return X_test, X_test_encoded, y_test, y_test_encoded


def load_training_data(training_set_id: str, test_set_id: str, new_type: str = None):
    """
    Load training.

    Parameters
    ----------
    training_data_id
        Identifier of the training data to train the pipeline on.
    test_data_id
        Identifier of the test data to train the pipeline on.
    new_type
        Type that is added to the catalog, is used for filtering the test set.
    """
    with open("adatyper/types.json") as f:
        types = json.load(f)

    dataset = pd.read_pickle(f"data/training_data/raw/training_set_{training_set_id}.pickle")
    dataset_encoded = pd.read_pickle(f"data/training_data/raw/training_set_encoded_{training_set_id}.pickle")

    X_train, X_train_encoded, y_train, y_train_encoded = _sample_filter_dataset_on_types(
        dataset, dataset_encoded, types, n=250
    )

    # Save the unique table numbers dataset loader for training typetabert.
    X_train_encoded['table_number'].to_csv(f"data/training_data/raw/training_set_encoded_table_numbers_{training_set_id}.csv")
    X_train_encoded = X_train_encoded.set_index('table_number')

    test_dataset = pd.read_pickle(f"data/training_data/raw/test_set_{training_set_id}.pickle")
    test_dataset_encoded = pd.read_pickle(f"data/training_data/raw/test_set_encoded_{training_set_id}.pickle")

    X_test, X_test_encoded, y_test, y_test_encoded = _sample_filter_dataset_on_types(
        # n is much higher as tables are aligned (i.e. some tables will be removed) after downsampling.
        test_dataset, test_dataset_encoded, types, n=1000
    )
    
    # Alignment of embedded and raw test sets, such that each row corresponds to the same column.
    # This enables evaluating the sequential pipeline of the type predictor,
    # where we may need both representations (raw and embedded) for a given column.
    
    # First, get overlapping tables (i.e. such that each column is represent by its raw values and the embedding)
    test_table_numbers = X_test["table_number"].unique()
    test_encoded_table_numbers = X_test_encoded.reset_index(drop=False)["table_number"].unique()
    overlapping_table_numbers = set(test_table_numbers).intersection(set(test_encoded_table_numbers))

    # Get the numbers of tables from which all columns have an embedding.
    equal_sized_tables = (
        X_test
        .query("table_number in @overlapping_table_numbers")
        .groupby("table_number")
        .apply(len)
        ==
        X_test_encoded
        .reset_index(drop=False)
        .query("table_number in @overlapping_table_numbers")
        .groupby("table_number")
        .apply(len)
    )
    equal_sized_tables = equal_sized_tables[equal_sized_tables].index

    # Filter on tables with the same number of columns.
    test_idx = X_test["table_number"].isin(equal_sized_tables)
    test_encoded_idx = X_test_encoded.reset_index(drop=False)["table_number"].isin(equal_sized_tables)
    X_test, y_test = X_test[test_idx], y_test.loc[test_idx]
    X_test_encoded = X_test_encoded.reset_index(drop=False)[test_encoded_idx].set_index("table_number").drop("table_number", axis=1)

    y_test_encoded = y_test.copy()

    del dataset, dataset_encoded, test_dataset, test_dataset_encoded

    return X_train, X_train_encoded, X_test, X_test_encoded, y_train, y_train_encoded, y_test, y_test_encoded


def _sample_filter_dataset_on_types(dataset, dataset_encoded, types, new_type=None, n=1000):
    """Sample columns and tables based on column types that are targeted.
    These training datasets do not correspond to the same data, hence columns and tables are sampled separately.
    """

    default_types = [_type for _type, key in types.items() if key == "default"]
    
    # Filter on tables with at least one column of a type in the type catalog.
    table_numbers = dataset[
        dataset["type"].isin(default_types + [new_type])
    ]["table_number"]

    unique_target_table_numbers = dataset[dataset["table_number"].isin(table_numbers)]["table_number"].astype(int).unique().tolist()

    # Remove duplicate tables
    X = dataset[dataset["table_number"].isin(unique_target_table_numbers)].drop("type", axis=1)
    X["table_number"] = pd.to_numeric(X["table_number"], downcast="integer", errors='ignore')
    y = dataset[dataset["table_number"].isin(unique_target_table_numbers)]["type"]
    y[~y.isin(default_types + [new_type]).values] = "null"

    dataset_encoded.index = dataset_encoded["table_number"]
    X_encoded = dataset_encoded[dataset_encoded["table_number"].isin(unique_target_table_numbers)]["table_encoded"].apply(pd.Series)
    y_encoded = dataset_encoded[dataset_encoded["table_number"].isin(unique_target_table_numbers)]["types"].str.lower().replace('[^a-zA-Z]', ' ')
    # y_encoded = dataset_encoded["types"].explode().str.lower().replace('[^a-zA-Z]', ' ')
    y_encoded[~y_encoded.isin(default_types + [new_type]).values] = "null"

    # Downsample to n samples per class
    y_capped = y.reset_index(drop=False).groupby("type").head(n=n).index
    y_encoded_capped = y_encoded.reset_index(drop=False).groupby("types").head(n=n).index

    y = y.reset_index(drop=False).loc[y_capped].set_index("index")["type"]
    X = X.reset_index(drop=False).loc[y_capped].set_index("index")

    y_encoded = y_encoded.reset_index(drop=False).loc[y_encoded_capped].set_index("table_number")["types"]
    X_encoded = X_encoded.reset_index(drop=False).loc[y_encoded_capped]

    return X, X_encoded, y, y_encoded


def load_typetabert_model():
    model_path = "typetabert/typetabert/models/"
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)

    # TODO: persist this model in memory.
    typetabert_model = typetabert.typetabert.TypeTaBERTModel(
        config["num_classes"],
        "cpu",
        joblib.load(os.path.join(model_path, "label_encoder.joblib")),
    )
    typetabert_model.load_state_dict(
        torch.load(
            os.path.join(model_path, "model_epoch0.bin"),
            map_location="cpu"
        )
    )
    typetabert_model.eval()

    return typetabert_model


def encode_tabert_table(typetabert_model, tabert_table: typing.List, table_number: int):

    dtype_mapping = {
        "object": "text",
        "O": "text",
        "float64": "real",
        "int64": "real",
        "bool": "real",
        "datetime64": "text",
        "datetime64[ns]": "text",
        "timedelta[ns]": "text",
        "category": "text",
    }

    # columns are oriented across the rows
    header = []
    for row in tabert_table:
        tabert_column = Column(
            # original column name is stored in index
            row["column_name"],
            dtype_mapping[row["dtype"]],
            sample_value=str(row["sample_value"]),
        )
        header.append(tabert_column)

    data = [
        list(row) for row in itertools.zip_longest(*[x["values"] for x in tabert_table])
    ]

    tabert_context = ""
    tabert_table = Table(id=table_number, header=header, data=data).tokenize(
        typetabert_model._tabert_model.tokenizer
    )

    _, column_encoding, _ = typetabert_model._tabert_model.encode(
        contexts=[typetabert_model._tabert_model.tokenizer.tokenize(tabert_context)],
        tables=[tabert_table],
    )

    return column_encoding.numpy()[0]
