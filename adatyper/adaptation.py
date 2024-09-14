import copy
import collections
import json
import os
import random
import re
import time
import traceback
import typing
import sys

import hnswlib
import numpy as np
import pandas as pd
import pickle
import psutil
import pyarrow.parquet as pq
import spacy_universal_sentence_encoder
import tqdm

from scipy import stats, spatial

from adatyper import preprocess, settings



def build_and_store_hnsw_index(data_identifier, ef=150, ef_construction=150, M=8, evaluation=False):
    """
    Source code: https://github.com/nmslib/hnswlib

    Arguments
    ---------
    encoded_dataset
        Dataset with column-level table encodings (columns can be grouped by table number).
    size
        The number of column vectors to build the index for.
    """
    num_samples = 300000
    encoded_dataset = (
        pd.read_pickle(f"data/training_data/raw/training_set_encoded_{data_identifier}.pickle")
    ).sample(n=num_samples, random_state=settings.RANDOM_STATE)
    ids = encoded_dataset.index.tolist()
    data = np.array([x[0].tolist() for x in encoded_dataset.to_numpy()])

    # We index 300 k
    num_elements = encoded_dataset.shape[0]

    # Get process id to measure memory and initial mem
    process = psutil.Process(os.getpid())
    start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    start_time = time.time()
    index = hnswlib.Index(space="cosine", dim=len(data[0]))
    # Initializing index - the maximum number of elements should be known beforehand
    # ef: 
    # M: 
    # Construction complexity: O(NlogN)
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)

    index.add_items(data, ids)
    # Controlling the recall by setting ef
    index.set_ef(ef) # ef should always be > k (number query)

    memory_fp = (psutil.Process(os.getpid()).memory_info().vms / (1024 ** 2)) - start_mem
    index_build_time = time.time() - start_time

    with open("data/adaptation/hnsw_index/hnsw_index.pickle", "wb+") as f:
        pickle.dump(index, f)

    bf_index = None
    if evaluation:
        bf_index = hnswlib.BFIndex(space="cosine", dim=len(data[0]))
        bf_index.init_index(max_elements=num_elements)
        bf_index.add_items(data, ids)

        index_size = sys.getsizeof(bf_index)
        index_type = type(bf_index)

    return index_build_time, memory_fp, bf_index


def generate_training_data_from_embeddings(
    initial_table_number: int,
    column_embedding: typing.List,
    _type: str,
    k=50,
    bf_index=None,
    evaluation=False,
):
    """
    Generate embedded training columns for the new type to adapt the ML model.
    The new data is generated through lookups of similar columns in the embedded training dataset.
    This lookup is conducted by an approximate nearest neighbor search algorithmm Hierarchical Navigable Small World (HNSW).

    Arguments
    ---------
    training_data_directory
        Directory where the training data can be found.
    initial_table_number
        The ID for the table, each table used to generate IDs for the new tables to be added.
    column_embedding
        Table holding the user-labeled column.
    _type
        Type of the column, labeled by a user.
    k
        The number of most similar columns to return, defaults to 50.
    bf_index
        The brute-force index used for evaluating HNSW performance on the column vectors (only for evaluation).
    evaluation
        Boolean to indicate whether function is called for evaluation purposes (will calculate and return the recall).
    """

    if evaluation and not bf_index:
        raise ValueError(
            "Evaluation mode requires a brute-force index to be passed to the function as well."
        )

    with open("data/adaptation/hnsw_index/hnsw_index.pickle", "rb") as f:
        index = pickle.load(f)

    # Query complexity: O(logN)
    start = time.time()
    labels, distances = index.knn_query(column_embedding, k=k)
    query_time = time.time() - start

    print(f"querying hnsw index took: {query_time}")

    recall = None
    if evaluation == True:

        labels_bf, distances_bf = bf_index.knn_query(column_embedding, k)

        # Measure recall
        correct = 0
        # for i in range(nun_queries):
        for label in labels[0]: #should be [i] for multiple example vectors
            if label in labels_bf[0]: correct += 1
        
        # if evaluating on multiple example vectors, increase num_examples
        num_examples = 1
        recall = float(correct)/(k*num_examples)

    ## The labels are indices, and the original vectors should be retrieved thorugh "index.get_items()".
    nearest_column_vectors = index.get_items(labels[0])
    table_numbers = pd.Series(
        range(
            initial_table_number,
            initial_table_number+len(nearest_column_vectors)
        )
    )
    types = pd.Series([_type]*len(nearest_column_vectors))

    generated_training_data = pd.concat(
        [pd.Series(nearest_column_vectors), types],
        axis=1
    )

    return generated_training_data, distances, query_time, recall


def adapt_regular_expression_dict(_type, example_string: str, regular_expression: str = None):

    type_regex_dict = json.load("type_regex.json")

    if not regular_expression:
        type_regex_dict[_type]["sources"] = ["user"]
    else:
        try:
            re.compile(regular_expression)
            if not re.fullmatch(regular_expression, example_string):
                raise ValueError("The regular expression is invalid or does not match the .")
        except:
            # TODO: return this message somewhere.
            raise ValueError("The regular expression is invalid or does not match the .")

        if _type not in type_regex_dict:
            # Add new type along with the regular expression
            type_regex_dict[_type]["regular_expressions"] = [regular_expression]
            type_regex_dict[_type]["sources"] = ["user"]
        else:
            type_regex_dict[_type]["regular_expressions"].append(regular_expression)
            type_regex_dict[_type]["sources"].append("user")

    json.dump(type_regex_dict, "adatyper/type_regex_latest.json")

    return type_regex_dict


def generate_training_data_columns_features(
    training_data_directory: str,
    topic: str,
    column: typing.List,
    column_id: int,
    header: typing.List,
    _type: str,
    initial_table_number: int,
    numerical_lf: typing.Dict,
    general_lf: typing.Dict,
    example_vector: typing.List
):
    """
    Scan columns from data directory to evaluate them against labeling functions.
    If at least 3 LFs return True, it is found to be a match for the type corresponding to the LF.

    Parameters
    ----------
    training_data_directory
        Directory with training data which
    column_lf
        Labeling functions to evaluate at column-level
    table_lf
        Labeling functions to evaluate at table-level
    """
    table_numbers = [initial_table_number + 1]
    column_values = [column.values]
    column_names = [column.name]
    column_types = [_type]
    column_dtypes = [column.dtype]
    left_column_name, right_column_name = _get_neighbouring_column_names(header, column_id)
    left_column_names = [left_column_name]
    right_column_names = [right_column_name]

    ##### CHANGE PER EXAMPLE
    # available: id, object, thing, abstraction, whole_tables # per file take diff. topic
    # topic = "id"
    ##### CHANGE PER EXAMPLE
    training_data_directory = os.path.join(training_data_directory, topic)

    table_files = os.listdir(training_data_directory)
    column_vector_similarities = []
    original_types = []
    feature_vectors = []
    table_increment = 2  # set to 2 to account for the initial sample being added above
    lf_collection = general_lf #numerical_lf |
    random.seed(settings.RANDOM_STATE)
    # expand to 5000
    for table_file in tqdm.tqdm(random.sample(table_files, 1000)):
        try:
            table = pq.read_table(os.path.join(training_data_directory, table_file))
            metadata = json.loads(table.schema.metadata[b"gittables"])
            df = table.to_pandas().dropna(axis=1, how="all")
            header = df.columns
            # Use dbpedia annotations to facilitate dbpedia lookups
            columns_metadata = metadata["dbpedia_embedding_column_types"]
            for column_id, column_name in enumerate(header):
                if not df[column_name].tolist():
                    continue
                column_vector = []
                for lf in lf_collection:
                    func_value = lf_collection[lf]
                    function, args = map(func_value.get, ("function", "params"))
                    # Evaluate labeling function on column
                    lf_vector = function(df, column_id, args)
                    column_vector += lf_vector
                
                sim = 1 - spatial.distance.cosine(column_vector, example_vector)
                # print(column_vector, example_vector)
                column_vector_similarities.append(sim)
                if sim > 0.7:
                    new_column = df[column_name]
                    # TODO: convert additional sample from column-level to entire table by
                    # adding all (labeled) columns, not only the ones matching this type.
                    table_numbers.append(int(initial_table_number + table_increment))
                    column_values.append(new_column.tolist())
                    column_names.append(column_name)
                    column_types.append(_type)
                    column_dtypes.append(str(new_column.dtype))
                    left_column_name, right_column_name = _get_neighbouring_column_names(header, column_id)
                    left_column_names.append(left_column_name)
                    right_column_names.append(right_column_name)
                    column_annotation = ""
                    if column_name in columns_metadata:
                        column_annotation = columns_metadata[column_name][
                            "cleaned_label"
                        ]
                    original_types.append(column_annotation)
                    feature_vectors.append(column_vector)
            table_increment += 1
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            continue

    feature_vectors = np.unique(feature_vectors).tolist()
    pd.Series(original_types).to_csv("models/evaluation/adaptation_ground_truth_types.csv")
    pd.DataFrame(feature_vectors).to_csv("models/evaluation/adaptation_feature_vectors.csv")

    return preprocess.merge_and_resample(
        table_numbers,
        column_types,
        column_values,
        column_dtypes,
        column_names,
        right_column_names,
        left_column_names,
    ), column_vector_similarities



def infer_numerical_column_lf_parameters(df: pd.DataFrame, column_id: str, _type: str):
    """
    Infer column parameters for the labeling functions from an example column for a specified type.
    Extract statistics, like the minimal, maximal, value from each column.

    Sep 2021: Only numerical types/columns are in scope.

    Parameters
    ----------
    df
        Table holding a relabeled column.
    column_id
        0-indexed identifier of the target column in the table.
    _type
        Type of the column, relabeled by an end-user.
    """
    numerical_column_lfs = collections.OrderedDict({
        "min": {"params": {"value": 0, "offset": 0}, "function": _min_value_lf},
        "max": {"params": {"value": 0, "offset": 0}, "function": _max_value_lf},
        "mean": {"params": {"value": 0, "offset": 0}, "function": _mean_value_lf},
        "median": {"params": {"value": 0, "offset": 0}, "function": _median_value_lf},
    })

    column_values = df.iloc[:, column_id].tolist()
    if not _check_numerical_type(column_values):
        return numerical_column_lfs, [0,0,0,0]

    values = np.array(column_values).astype(float)

    min_value = values.min()
    max_value = values.max()
    mean_value = values.mean()
    median_value = np.median(values)
    # For now, the standard deviation is taken as offset for every statistic.
    std = values.std()

    numerical_column_lfs["min"]["params"] = {"value": min_value, "offset": std}
    numerical_column_lfs["max"]["params"] = {"value": max_value, "offset": std}
    numerical_column_lfs["mean"]["params"] = {"value": mean_value, "offset": std}
    numerical_column_lfs["median"]["params"] = {"value": median_value, "offset": std}

    base_dir = f"data/adaptation/lf_config"
    # _save_labeling_functions(numerical_column_lfs, "numerical", base_dir, _type)

    # TODO: make binary: all 1, with range check for LF templates: 0 if far away, 1 if within range.
    numerical_lf_vector = [1,1,1,1] #[min_value, max_value, mean_value, median_value]

    return numerical_column_lfs, numerical_lf_vector


def infer_general_lf_parameters(df: pd.DataFrame, column_id: str, _type: str):
    """
    Infer parameters for the labeling functions from an example column for a specified type.
    Extracts LF parameters:
        - regardless of a columns datatype,
        - for signals across the entire table.

    Parameters
    ----------
    df
        Table holding the user-labeled column.
    column_id
        0-indexed identifier of the target column in the table.
    _type
        Type of the column, labeled by a user.
    """
    header = df.columns
    column_values = df.iloc[:, column_id].tolist()
    use = spacy_universal_sentence_encoder.load_model('en_use_lg')

    left_column_name, right_column_name = _get_neighbouring_column_names(header, column_id)

    column_name = _clean_string(header[column_id])
    column_entropy = _column_entropy(column_values)
    mean_value_length, std_value_length = _calculate_value_length(column_values)
    frequent_values = _get_frequent_values(column_values)
    mean_fraction_numbers = _calculate_fraction_numbers(column_values)
    mean_fraction_special_characters = _calculate_fraction_special_characters(column_values)
    column_name_embedding = _get_use_embedding(column_name, use)
    if right_column_name == None:
        # there is no right column
        neighboring_columns = _clean_string(left_column_name)
    elif left_column_name == None:
        # there is no left column
        neighboring_columns = _clean_string(right_column_name)
    else:
        neighboring_columns = _clean_string(left_column_name) + _clean_string(right_column_name)
    neighboring_columns_embedding = _get_use_embedding(neighboring_columns, use)
    column_values_embedding = _get_use_embedding(" ".join([str(val) for val in column_values[:10]]), use)

    general_lfs = collections.OrderedDict({
        # "right_column": {"params": {"right_column_name": right_column_name}, "function": _right_column_lf},
        # "left_column": {"params": {"left_column_name": left_column_name}, "function": _left_column_lf},
        # "column_name": {"params": {"column_name": column_name, "type": _type}, "function": _column_name_lf},
        "column_entropy": {"params": {"entropy": column_entropy}, "function": _column_entropy_lf},
        "value_length": {
            "params": {"mean_value_length": mean_value_length, "std_value_length": std_value_length},
            "function": _value_length_lf
        },
        "frequent_values": {"params": {"frequent_values": frequent_values}, "function": _frequent_values_lf},
        # TODO: frequent values -> embedding of a sample of column values as a sentence.
        "fraction_numbers": {"params": {"mean_fraction_numbers": mean_fraction_numbers}, "function": _fraction_numbers_lf},
        "fraction_special_characters": {"params": {"mean_fraction_special_characters": mean_fraction_special_characters}, "function": _fraction_special_characters_lf},
        "use_embeddings":{
            "params":
                {"column_name_embedding": column_name_embedding, "neighboring_columns_embedding": neighboring_columns_embedding, "column_values_embedding": column_values_embedding, "use": use},
                "function": _use_similarity_lf
            },
    })
    # 1 here represents the example its similarity with its own column name embedding and neighboring column name embeddings.
    general_lf_vector = [1, 1, 1, mean_fraction_numbers, mean_fraction_special_characters, 1, 1, 1] # column_entropy, mean_value_length,#+ column_name_embedding.vector.tolist() #+ neighboring_columns_embedding.vector.tolist()

    base_dir = f"data/adaptation/lf_config"
    # _save_labeling_functions(general_lfs, "general", base_dir, _type)

    return general_lfs, general_lf_vector



def _save_labeling_functions(
    lf_dict: typing.Dict, lf_type: str, base_dir: str, _type: str
):
    """
    Save inferred labeling functions parameters and remove (unserializable) labeling functions.

    Parameters
    ----------
    lf_dict
        Labeling function dictionary, each lf exists of parameter values and a Python function.
    lf_type
        Identifier (column or table) corresponding to type of LF, to id file with.
    base_dir
        Directory for storing labeling function parameters.
    _type
        Type corresponding to the LFs.
    """
    lf_config_id = len(os.listdir(base_dir))
    filename = f"{_type}_{lf_type}_lf_config_{lf_config_id}.json"
    with open(os.path.join(base_dir, filename), "w+") as f:

        lf_parameters = copy.deepcopy(lf_dict)
        for key in lf_parameters:
            lf_parameters[key].pop("function")

        json.dump(lf_parameters, f)


def _get_neighbouring_column_names(header: typing.List, column_id: int):
    if column_id == len(header) - 1:
        # Column is last column in table, no right column exists.
        right_column_name = None
    else:
        right_column_name = _clean_string(header[column_id + 1])
    if column_id > 0:
        left_column_name = _clean_string(header[column_id - 1])
    else:
        # Column is first column in table, no left column exists.
        left_column_name = None
    
    return left_column_name, right_column_name



def _min_value_lf(df: pd.DataFrame, column_id: int, params):
    column_values = df.iloc[:, column_id].tolist()
    if not _check_numerical_type(column_values):
        return [0]

    min_value = params["value"]
    offset = params["offset"]

    observed_min = np.min(column_values)
    if (min_value - offset <= observed_min <= min_value + offset):
        return [1]
    return [0]


def _max_value_lf(df: pd.DataFrame, column_id: int, params):
    column_values = df.iloc[:, column_id].tolist()
    if not _check_numerical_type(column_values):
        return [0]

    max_value = params["value"]
    offset = params["offset"]

    observed_max = np.max(column_values)
    if (max_value - offset <= observed_max <= max_value + offset):
        return [1]
    return [0]


def _median_value_lf(df: pd.DataFrame, column_id: int, params):
    column_values = df.iloc[:, column_id].tolist()
    if not _check_numerical_type(column_values):
        return [0]

    median_value = params["value"]
    offset = params["offset"]

    observed_median = np.median(column_values)
    if (median_value - offset <= observed_median <= median_value + offset):
        return [1]
    return [0]


def _mean_value_lf(df: pd.DataFrame, column_id: int, params):
    column_values = df.iloc[:, column_id].tolist()
    if not _check_numerical_type(column_values):
        return [0]

    mean_value = params["value"]
    offset = params["offset"]

    observed_mean = np.mean(column_values)
    if (mean_value - offset <= observed_mean <= mean_value + offset):
        return [1]
    return [0]


def _column_entropy_lf(df: pd.DataFrame, column_id: int, params):
    column_values = df.iloc[:, column_id].tolist()
    observed_entropy = _column_entropy(column_values)

    column_entropy = params["entropy"]

    if (column_entropy - 0.1 <= observed_entropy <= column_entropy + 0.1):
        return [1]
    return [0]


def _value_length_lf(df: pd.DataFrame, column_id: int, params):
    column_values = df.iloc[:, column_id].tolist()
    observed_mean_length, _ = _calculate_value_length(column_values)

    mean_length = params["mean_value_length"]
    std_length = params["std_value_length"]

    if (mean_length - std_length <= observed_mean_length <= mean_length + std_length):
        return [1]
    return [0]



def _frequent_values_lf(df: pd.DataFrame, column_id: int, params):
    column_values = df.iloc[:, column_id].astype(str).tolist()
    observed_frequent_values = _get_frequent_values(column_values)

    frequent_values = params["frequent_values"]
    number_overlap = len(np.intersect1d(observed_frequent_values, frequent_values))
    if number_overlap >= 2:
        return [1]
    return [0]


def _left_column_lf(df: pd.DataFrame, column_id: int, params):
    columns = df.columns
    left_column_name = params["left_column_name"]
    if column_id > 0:
        observed_left_column = _clean_string(columns[column_id - 1])
    else:
        # If the target column is the first of the table, there is no left column.
        observed_left_column = None

    if observed_left_column == left_column_name:
        return True
    return False


def _right_column_lf(df: pd.DataFrame, column_id: int, params):
    columns = df.columns
    right_column_name = params["right_column_name"]
    if len(columns) - 1 != column_id:
        observed_right_column = _clean_string(columns[column_id + 1])
    else:
        # If the target column is the last of the table, there is no right column.
        observed_right_column = None

    if observed_right_column == right_column_name:
        return True
    return False


def _column_name_lf(df: pd.DataFrame, column_id: int, params: typing.Dict):
    columns = df.columns
    column_name = params["column_name"]
    _type = params["type"]
    observed_column_name = _clean_string(columns[column_id])

    if (observed_column_name == column_name) or (observed_column_name == _type):
        return True

    return False


def _fraction_numbers_lf(df: pd.DataFrame, column_id: int, params: typing.Dict):
    column_values = df.iloc[:, column_id].astype(str).tolist()
    mean_fraction_numbers = params["mean_fraction_numbers"]

    observed_fraction_numbers = _calculate_fraction_numbers(column_values)

    return [observed_fraction_numbers]

    # if (mean_fraction_numbers - 0.05 <= observed_fraction_numbers <= mean_fraction_numbers + 0.05):
    #     return [1]
    # return [0]


def _fraction_special_characters_lf(df: pd.DataFrame, column_id: int, params: typing.Dict):
    column_values = df.iloc[:, column_id].astype(str).tolist()
    mean_fraction_special_characters = params["mean_fraction_special_characters"]

    observed_fraction_special_characters = _calculate_fraction_special_characters(column_values)

    return [observed_fraction_special_characters]

    # if (mean_fraction_special_characters - 0.05 <= observed_fraction_special_characters <= mean_fraction_special_characters + 0.05):
    #     return [1]
    # return [0]


def _use_similarity_lf(df: pd.DataFrame, column_id: int, params: typing.Dict):
    column_name = df.columns[column_id]
    header = " ".join(df.columns.astype(str).tolist())
    column_values = df.iloc[:,column_id].astype(str).tolist()[:10]
    column_name_embedding = params["column_name_embedding"]
    neighboring_columns_embedding = params["neighboring_columns_embedding"]
    column_values_embedding = params["column_values_embedding"]
    use = params["use"]

    observed_column_name_embedding = _get_use_embedding(column_name, use)

    left_column_name, right_column_name = _get_neighbouring_column_names(header, column_id)

    if right_column_name == None:
        # there is no right column
        neighboring_columns = _clean_string(left_column_name)
    elif left_column_name == None:
        # there is no left column
        neighboring_columns = _clean_string(right_column_name)
    else:
        neighboring_columns = _clean_string(left_column_name) + _clean_string(right_column_name)
    observed_neighboring_columns_embedding = _get_use_embedding(neighboring_columns, use)
    observed_column_values_embedding = _get_use_embedding(" ".join(column_values), use)

    column_name_similarity = column_name_embedding.similarity(observed_column_name_embedding)
    neighboring_columns_similarity = neighboring_columns_embedding.similarity(observed_neighboring_columns_embedding)
    column_values_similarity = column_values_embedding.similarity(observed_column_values_embedding)

    return [column_name_similarity, neighboring_columns_similarity, column_values_similarity]


def _check_numerical_type(column_values: typing.List):
    if all((isinstance(value, float) or isinstance(value, int)) for value in column_values):
        return True
    return False


def _column_entropy(column_values: typing.List):
    """Calculate entropy of list of values."""
    frequencies = list(collections.Counter(column_values).values())
    value_probabilities = [freq / len(column_values) for freq in frequencies]
    entropy = stats.entropy(value_probabilities)
    
    return entropy


def _get_frequent_values(column_values: typing.List):
    frequent_values = list(collections.Counter(column_values).keys())[:100]
    return frequent_values


def _calculate_value_length(column_values: typing.List):
    value_lengths = [len(str(column_value)) for column_value in column_values]
    mean_value_length = np.mean(value_lengths)
    std_value_length = np.std(value_lengths)
    
    return mean_value_length, std_value_length


def _calculate_fraction_numbers(column_values: typing.List):
    fraction_numbers = [len(re.findall(r'\d', str(value)))/len(str(value)) for value in column_values]
    return np.mean(fraction_numbers)


def _calculate_fraction_special_characters(column_values: typing.List):
    # The trailing space is intended to include spaces.
    fraction_special_characters = [len(re.findall(r"[-.,_:; ]", str(value)))/len(str(value)) for value in column_values]
    return np.mean(fraction_special_characters)


def _get_use_embedding(string: str, use):
    return use(string)


def _clean_string(_string: str) -> str:
    return re.sub(
        "[^a-zA-Z0-9]",
        " ",
        _string.lower()
    )
