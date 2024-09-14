import argparse
import json
import os
import traceback
import typing

import joblib
import pandas as pd
import pyarrow.parquet as pq
import tqdm

from sklearn import model_selection

from typetabert.typetabert import typetabert
from adatyper import settings, utils


def get_ontology() -> typing.List:
    """Read custom ontology.
    Currently only geographic types are in scope.

    Returns
    -------
    geo_types
        List with only fine-grained flat geographic types.
    """
    # The code below can be used to expand the types.
    custom_types = (
        pd.read_csv("ontologies/FlatSemanticTypes.csv")["Semantic Type "]
        .str.strip()
        .str.lower()
        .tolist()
    )
    dbpedia_types = pd.read_csv("ontologies/dbpedia_20210528.csv", index_col=0)[
        "cleaned_label"
    ].tolist()
    ontology = list(set(custom_types + dbpedia_types))

    return ontology


def generate_training_data_gittables(topic_subsets: typing.List):
    """
    Parameters
    ----------
    topic_subset
        Indicator for which directory to take, corresponding to a subset of tables from a certain topic from GitTables.
    """
    ontology = get_ontology()

    model_path = "typetabert/typetabert/models/"
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)
    typetabert_model = typetabert.TypeTaBERTModel(
        config["num_classes"],
        "cpu",
        joblib.load(os.path.join(model_path, "label_encoder.joblib")),
    )

    column_values = []
    column_names = []
    column_types = []
    right_column_names = []
    left_column_names = []
    column_dtypes = []
    table_numbers = []

    encoded_table_numbers =[]
    encoded_tables = []
    encoded_types = []

    print(f"Generating data from {topic_subsets} topics.")

    # Iterate over tables and extract columns of target types
    table_number = 0
    for i, topic_subset in tqdm.tqdm(enumerate(topic_subsets)):

        topic_dir = f"data/{topic_subset}"
        print(f"Generating data for topic: {topic_subset}.")
        if not os.path.exists(topic_dir):
            # Better to raise error
            print(
                """
                No source data available.
                Make sure there is a directory: data/<topic_subset>/ from the root.
                This directory should consist of parquet files with tables from GitTables, or similarly formatted.
                """
            )
        topic_tables = os.listdir(topic_dir)[:100000]
        print("Processing tables from topic dir: ", topic_subset)
        for j, table_filename in tqdm.tqdm(enumerate(topic_tables)):

            try:
                tabert_table = []
                table_number += 1
                table = pq.read_table(os.path.join(topic_dir, table_filename))
                metadata = json.loads(table.schema.metadata[b"gittables"])
                # Use dbpedia annotations to facilitate dbpedia lookups
                columns_metadata = metadata["dbpedia_syntactic_column_types"]
                columns = table.column_names
                dtypes = table.to_pandas().dtypes
                current_tables_numbers = []
                current_values = []
                current_column_names = []
                current_column_types = []
                current_column_dtypes = []
                current_right_column_names = []
                current_left_column_names = []
                for k, column in enumerate(columns):
                    values = table.column(column).to_pylist()
                    sample_value = values[0]
                    tabert_table.append({"column_name": column, "values": values, "sample_value": sample_value, "dtype": str(dtypes[k])})
                    # for _type in ontology:
                        # We are not bound by dbpedia types,
                        # e.g. latitude/longitude are not present so we 'annotate' here.
                    column_clean_name = column.lower()
                    column_annotation = "null"
                    if column in columns_metadata:
                        column_annotation = columns_metadata[column][
                            "cleaned_label"
                        ]
                        metadata["dtypes"][column]
                    column_id = columns.index(column)
                    if column_id < len(columns) - 1:
                        right_column_name = columns[column_id + 1].lower()
                    else:
                        # No right column for the last column
                        right_column_name = None
                    if column_id > 0:
                        left_column_name = columns[column_id - 1].lower()
                    else:
                        # No left column for the first column
                        left_column_name = None

                    current_tables_numbers.append(table_number)
                    current_values.append(values)
                    current_column_names.append(column)
                    current_column_types.append(column_annotation)
                    current_column_dtypes.append(dtypes[k])
                    current_right_column_names.append(right_column_name)
                    current_left_column_names.append(left_column_name)

                table_numbers += current_tables_numbers
                column_values += current_values
                column_names += current_column_names
                column_types += current_column_types
                column_dtypes += current_column_dtypes
                right_column_names += current_right_column_names
                left_column_names += current_left_column_names

                encoded_table = utils.encode_tabert_table(typetabert_model, tabert_table, table_number)
                encoded_tables.append(encoded_table)
                # note: due to the varying types in scope, we leave all column names as-is, to be processed later.
                encoded_types.append(columns)
                encoded_table_numbers.append(table_number)

                ## TODO: get embedding from table for every column in the table, if possible, else skip.
            except Exception as e:
                print(e)
                continue

    dataset, duplicated_idx = merge_and_resample(
        table_numbers,
        column_types,
        column_values,
        column_dtypes,
        column_names,
        right_column_names,
        left_column_names,
    )

    dataset_encoded = (
        pd.DataFrame([encoded_tables, encoded_types, encoded_table_numbers])
        .transpose()
    )
    dataset_encoded.columns = ["table_encoded", "types", "table_number"]
    dataset_encoded_tables = dataset_encoded.explode("table_encoded").reset_index(drop=True)[~duplicated_idx]
    dataset_encoded_tables["types"] = dataset_encoded["types"].explode().reset_index(drop=True)[~duplicated_idx]

    train_table_numbers, test_table_numbers = model_selection.train_test_split(
        dataset["table_number"].unique(),
        test_size=0.2,
        random_state=settings.RANDOM_STATE
    )

    train_set = dataset[dataset["table_number"].isin(train_table_numbers)]
    test_set = dataset[dataset["table_number"].isin(test_table_numbers)]
    encoded_tables_train = dataset_encoded_tables[dataset_encoded_tables["table_number"].isin(train_table_numbers)]
    encoded_tables_test = dataset_encoded_tables[dataset_encoded_tables["table_number"].isin(test_table_numbers)]

    _write_to_file(dataset, train_set, test_set, dataset_encoded_tables, encoded_tables_train, encoded_tables_test)


def generate_evaluation_data_ctu():
    
    ontology = get_ontology()

    model_path = "typetabert/typetabert/models/"
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)
    typetabert_model = typetabert.TypeTaBERTModel(
        config["num_classes"],
        "cpu",
        joblib.load(os.path.join(model_path, "label_encoder.joblib")),
    )

    column_values = []
    column_names = []
    column_types = []
    right_column_names = []
    left_column_names = []
    column_dtypes = []
    table_numbers = []

    encoded_table_numbers =[]
    encoded_tables = []
    encoded_types = []

    table_number = 0

    base_dir = "data/ctu_evaluation_data"
    tables_dir = os.path.join(base_dir, "tables")
    processed_dir = os.path.join(base_dir, "processed")

    if not os.path.exists(processed_dir):
        # create separate directory for processed data
        os.makedirs(processed_dir)

    for db in tqdm.tqdm(os.listdir(tables_dir)):
        db_table_dir = os.path.join(tables_dir, db)
        for j, table_filename in enumerate(os.listdir(db_table_dir)):
            try:
                tabert_table = []
                table_number += 1
                table = pq.read_table(os.path.join(db_table_dir, table_filename))
                # These are the aggregated annotations from MTurk.
                annotations_filename = table_filename.replace(".parquet", ".csv")
                column_annotations = pd.read_csv(
                    os.path.join(base_dir, f"annotations/{db}/{annotations_filename}")
                )["type"].tolist()
                columns = table.column_names
                dtypes = table.to_pandas().dtypes
                current_tables_numbers = []
                current_values = []
                current_column_names = []
                current_column_types = []
                current_column_dtypes = []
                current_right_column_names = []
                current_left_column_names = []
                for k, column in enumerate(columns):
                    values = table.column(column).to_pylist()
                    sample_value = values[0]
                    tabert_table.append({"column_name": column, "values": values, "sample_value": sample_value, "dtype": str(dtypes[k])})
                    column_annotation = column_annotations[k]
                    column_id = columns.index(column)
                    if column_id < len(columns) - 1:
                        right_column_name = columns[column_id + 1].lower()
                    else:
                        # No right column for the last column
                        right_column_name = None
                    if column_id > 0:
                        left_column_name = columns[column_id - 1].lower()
                    else:
                        # No left column for the first column
                        left_column_name = None

                    current_tables_numbers.append(table_number)
                    current_values.append(values)
                    current_column_names.append(column)
                    current_column_types.append(column_annotation)
                    current_column_dtypes.append(dtypes[k])
                    current_right_column_names.append(right_column_name)
                    current_left_column_names.append(left_column_name)

                encoded_table = utils.encode_tabert_table(typetabert_model, tabert_table, table_number)
                
                # Ensure the number of columns remain the same for raw/encoded before adding tables.
                assert len(encoded_table) == len(current_tables_numbers)

                table_numbers += current_tables_numbers
                column_values += current_values
                column_names += current_column_names
                column_types += current_column_types
                column_dtypes += current_column_dtypes
                right_column_names += current_right_column_names
                left_column_names += current_left_column_names

                encoded_tables.append(encoded_table)
                # note: due to the varying types in scope, we leave all column names as-is, to be processed later.
                encoded_types.append(column_annotations)
                encoded_table_numbers.append(table_number)

            except Exception as e:
                print(e)
                
                continue

    dataset, duplicated_idx = merge_and_resample(
        table_numbers,
        column_types,
        column_values,
        column_dtypes,
        column_names,
        right_column_names,
        left_column_names,
    )
    # the encoded and raw columns should stay exactly aligned.
    dataset_encoded = (
        pd.DataFrame([encoded_tables, encoded_types, encoded_table_numbers])
        .transpose()
        # remove duplicate columns based on the raw values, to keep datasets aligned
    )
    dataset_encoded.columns = ["table_encoded", "types", "table_number"]
    dataset_encoded_tables = dataset_encoded.explode("table_encoded").reset_index(drop=True)[~duplicated_idx]
    dataset_encoded_tables["types"] = dataset_encoded["types"].explode().reset_index(drop=True)[~duplicated_idx]

    assert dataset_encoded_tables.shape[0] == dataset.shape[0]

    dataset.to_pickle(os.path.join(base_dir, "processed/ctu_dataset.pickle"))
    dataset_encoded_tables.to_pickle(os.path.join(base_dir, "processed/ctu_dataset_encoded.pickle"))


def merge_and_resample(
    table_numbers: typing.List,
    column_types: typing.List,
    column_values: typing.List,
    column_dtypes: typing.List,
    column_names: typing.List,
    right_column_names: typing.List,
    left_column_names: typing.List,
) -> pd.DataFrame:
    """
    Merge and deduplicate all column data (type, values, names, etc) into one table.
    Samples in this table are rebalanced so that each type has the exact same number of occurrences in the training data.
    The sample rate is set to the type count that is minimally represent.
    """
    df = (
        pd.DataFrame(
            [
                table_numbers,
                column_types,
                column_values,
                column_dtypes,
                left_column_names,
                right_column_names,
            ],
        )
        .transpose()
        .rename(
            {
                0: "table_number",
                1: "type",
                2: "values",
                3: "dtype",
                4: "left_column_name",
                5: "right_column_name",
            },
            axis=1,
        )
    )
    df.index = column_names

    duplicated_idx = df.astype(str).duplicated(subset=["type", "values", "dtype", "left_column_name", "right_column_name"])
    df_deduplicated = df[~duplicated_idx]

    return df_deduplicated, duplicated_idx.reset_index(drop=True)


def _write_to_file(dataset: pd.DataFrame, train_set: pd.DataFrame, test_set: pd.DataFrame, dataset_encoded: pd.DataFrame, train_set_encoded: pd.DataFrame, test_set_encoded: pd.DataFrame):
    """
    Write raw training data to a file.
    A file identifier is added based on the count of training files present in the directory.

    Parameters
    ----------
    training_data
        Dataframe with raw column data to store.
    """
    training_data_dir = "data/training_data/raw"
    training_data_id = max([int(id_.split("_")[-1].replace(".pickle","").replace(".csv","")) for id_ in os.listdir(training_data_dir)]) + 1

    dataset_filepath = os.path.join(
        training_data_dir, f"training_data_{training_data_id}.pickle"
    )

    if os.path.exists(dataset_filepath):
        raise ValueError(
            f"""Cannot automatically infer a new training file.
            Please, rename the file at: {dataset_filepath}."""
        )
    
    train_set.to_pickle(os.path.join(training_data_dir, f"training_set_{training_data_id}.pickle"), protocol=4)
    test_set.to_pickle(os.path.join(training_data_dir, f"test_set_{training_data_id}.pickle"), protocol=4)  

    train_set_encoded.to_pickle(os.path.join(training_data_dir, f"training_set_encoded_{training_data_id}.pickle"), protocol=4)
    test_set_encoded.to_pickle(os.path.join(training_data_dir, f"test_set_encoded_{training_data_id}.pickle"), protocol=4)

    # The unique training numbers are needed to determine #samples w/o loading the full dataset for TypeTaBERT 
    train_set_encoded["table_number"].to_csv(os.path.join(training_data_dir, f"training_set_encoded_table_numbers_{training_data_id}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for training the type detection pipeline."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="""
            The dataset to process, can be 'gittables' for training and in-distribution evaluation
            or 'ctu' for out-of-distribution evaluation.
        """,
    )
    parser.add_argument(
        "--dirs",
        type=str,
        required=False,
        nargs="+",
        default=[
            "id",
            "object",
        ],  # Corresponds to the for training data with ID 0.
        help="""
            List of directory names with tables (stored in parquet files),
            available in the 'data/' directory.
        """,
    )
    args = parser.parse_args()
    training_data_dirs = args.dirs

    if args.dataset == "gittables":
        generate_training_data_gittables(training_data_dirs)
    if args.dataset == "ctu":
        generate_evaluation_data_ctu()
