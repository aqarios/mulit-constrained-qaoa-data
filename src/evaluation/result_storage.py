from __future__ import annotations
import base64
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.single_run import ExperimentResultRow

PROJECT_PATH = Path("../").absolute()
while not (PROJECT_PATH / ".git").is_dir():
    PROJECT_PATH = PROJECT_PATH.parent

RESULTS = PROJECT_PATH / "results/"


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # if isinstance(obj, ExecutionSpans):
        #     exec_list = []
        #     for exec_span in obj:
        #         exec_dict = {
        #             "start": exec_span.start.timestamp(),
        #             "stop": exec_span.stop.timestamp(),
        #         }
        #         exec_list.append(exec_dict)
        #     return exec_list  # Assuming ExecutionSpans has a `to_dict` method
        if isinstance(obj, bytes):
            # Convert bytes to a base64 string
            return base64.b64encode(obj).decode("utf-8")
        return json.JSONEncoder.default(self, obj)


def _serialize_df(df: pd.DataFrame, dict_cols: list[str]):
    df = df.copy()

    for col in dict_cols:
        df[col] = df[col].apply(lambda x: json.dumps(x, cls=CustomEncoder))
    return df


def convert_deltas(entry):
    if "init_deltas" in entry:
        entry["init_deltas"] = tuple(entry["init_deltas"])
    return entry


def _deserialize_df(df: pd.DataFrame, dict_cols: list[str]):
    df = df.copy()
    for col in dict_cols:
        # Check if column already contains dictionaries
        if df[col].dtype == "object" and df[col].apply(type).eq(dict).all():
            continue  # Skip conversion if already dictionaries
        try:
            # Attempt JSON parsing if not already dictionaries
            df[col] = df[col].apply(json.loads)
        except (json.JSONDecodeError, TypeError) as e:
            raise RuntimeError(f"Error parsing column '{col}'. Skipping column.", e)
            # Optionally, you could add more robust error handling here

    # Assuming convert_deltas is a defined function
    df["optimizer_params"] = df["optimizer_params"].apply(convert_deltas)

    return df


class ResultStorage:
    """
    Manages storage and retrieval of optimization results in CSV or Parquet files.

    This class provides functionality to store optimization run results, load existing data,
    normalize nested data structures, and check for redundant entries. It supports both
    CSV and Parquet file formats with automatic serialization and deserialization of
    complex data types.

    Parameters
    ----------
    file_path : str
        Path to the storage file (.csv or .parquet.gz).
    dry_run : bool, optional
        If True, prevents writing to disk. Defaults to False.

    Attributes
    ----------
    file_path : str
        Path to the storage file.
    RowClass : ExperimentSetupRow
        Class defining the structure of result entries.
    """

    file_path: str
    _dry_run: bool
    _df: pd.DataFrame
    _current_index: int
    _dict_cols: list  # list of all dict columns

    _idx_columns = ["id", "pipeline_name", "sublayer"]
    RowClass = ExperimentResultRow.init_empty()

    def _determine_dict_columns(self):
        dict_cols = []
        for i_key, i_type in self.RowClass.result_table_types():
            if i_type == "dict":
                dict_cols.append(i_key)
        self._dict_cols = dict_cols

    def __init__(self, file_path: str, dry_run: bool = False):
        self._determine_dict_columns()

        self.file_path = str(file_path)
        self._initialize_results_table()
        self._dry_run = dry_run

    def _check_table_path(self):
        # check if directory exits and create table if not
        directory_path = os.path.dirname(self.file_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def _initialize_results_table(self):
        self._check_table_path()

        if self.file_path.endswith(".csv"):
            self._read_csv()
        elif self.file_path.endswith(".parquet.gz"):
            self._read_parquet()
        else:
            raise ValueError("Filepath must be either '.csv' or '.parquet.gz'")

    def _read_parquet(self):
        self._df = pd.read_parquet(self.file_path)

    def _read_csv(self):
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=self.RowClass.result_table_columns())
            df.to_csv(self.file_path, header=True, index=False)
        df = pd.read_csv(self.file_path)

        print(f"Loading '{self.file_path}'... ")
        self._df = _deserialize_df(df, self._dict_cols)

        self._current_index = (
            (max(self._df.index) + 1) if len(self._df.index) > 0 else 0
        )
        print("Done", self._current_index)

    def save_to_new(self):
        """
        Saves the current DataFrame to a new file, overwriting any existing content.

        The data is serialized according to the file format:
        - CSV: Saved without index
        - Parquet: Saved with gzip compression

        Raises
            ValueError: If file extension is neither .csv nor .parquet.gz
        """

        df = _serialize_df(self._df, self._dict_cols)
        if self.file_path.endswith(".csv"):
            df.to_csv(self.file_path, index=False)
        elif self.file_path.endswith(".parquet.gz"):
            df.to_parquet(self.file_path, compression="gzip")
        else:
            raise ValueError("Files can only be stored in .csv or .parquet.gz format.")

    def get_df(
        self,
        *filter_keys,
        strict: bool = False,
        retrieve_all: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieves a filtered and normalized DataFrame based on specified keys.

        Extracts and normalizes nested dictionary data, applying filters according to the
        provided keys. Supports both strict and loose filtering of rows.

        Parameters
        ----------
        *filter_keys
            Keys to filter the DataFrame.
        strict : bool, optional
            If True, requires all filter keys to be present.
            If False, accepts rows with any filter key.
            Defaults to False.
        retrieve_all : bool, optional
            If True, includes all columns in output.
            If False, returns only filtered columns.
            Defaults to False.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing requested columns and rows.
        """

        # convert table
        json_columns = list(self._dict_cols)

        extracted_table = self._normalize_dict_columns(json_columns)
        idx_columns = self._idx_columns

        if retrieve_all:
            idx_columns = extracted_table.columns.tolist()

        # Combine index columns with filter keys
        columns_to_retain = list(set(idx_columns + list(filter_keys)))
        # Apply filtering based on filter_keys
        if filter_keys:
            if strict:
                # Retain only rows where ALL filter_keys are present
                mask = extracted_table[list(filter_keys)].notna().all(axis=1)
            else:
                # Retain rows where ANY filter_keys are present
                mask = extracted_table[list(filter_keys)].notna().any(axis=1)

            extracted_table = extracted_table[mask]

        # Return DataFrame with only specified columns
        return extracted_table[columns_to_retain]

    def _normalize_dict_columns(self, dict_columns: list[str]):
        """
        Normalize dictionary columns in a DataFrame by flattening nested dictionaries
        into separate columns.

        Args:
            dict_columns (list[str]): List of column names containing dictionaries

        Returns:
            pd.DataFrame: DataFrame with normalized dictionary columns
        """

        df_deserialized = _deserialize_df(self._df, self._dict_cols)
        normalized_df = df_deserialized.reset_index(drop=True)

        for dict_col in dict_columns:
            # Ensure the column contains dictionaries
            if not all(isinstance(x, dict) for x in normalized_df[dict_col].dropna()):
                raise ValueError(f"Column {dict_col} must contain only dictionaries")

            # Use json_normalize directly on the dictionary column
            normalized = pd.json_normalize(normalized_df[dict_col].tolist())

            # Join with original DataFrame
            normalized_df = pd.concat(
                [normalized_df.drop(dict_col, axis=1), normalized], axis=1
            )

        return normalized_df

    def _save_addition_to_df(self, additional_df):
        if len(self._df) == 0:
            self._df = additional_df
        else:
            self._df = pd.concat([self._df, additional_df], sort=True)

        dfx = _serialize_df(additional_df, self._dict_cols)

        if not self._dry_run:
            dfx.to_csv(self.file_path, mode="a", header=False, index=False)


# def get_df_from_sqlitedict(sqlitedict_path: str) -> pd.DataFrame:
#     # load sqlite dict and transform
#     os.path.isfile(sqlitedict_path)
#     result_dict = SqliteDict(sqlitedict_path)
#     df = pd.DataFrame.from_dict(result_dict, orient="index")
#     df.reset_index(inplace=True)
#
#     return df
#
#
# def merge_metadata_from_optimizer(row, optimizer_metadata_column: str):
#     # Check if either field is NaN/float
#     if isinstance(row["metadata"], float) or isinstance(
#         row[optimizer_metadata_column], float
#     ):
#         # Return empty dict if both are NaN, or the non-NaN value
#         if isinstance(row["metadata"], float) and isinstance(
#             row[optimizer_metadata_column], float
#         ):
#             return {}
#         elif isinstance(row["metadata"], float):
#             return {optimizer_metadata_column: row[optimizer_metadata_column]}
#         else:
#             return row["metadata"]
#     # Normal case - both are dictionaries
#     return {**row["metadata"], **row[optimizer_metadata_column]}
#
#
# def get_num_layers(params):
#     """
#     Determines the number of layers based on the input parameters.
#
#     Parameters:
#         params (Union[float, list, np.ndarray]): Input value which may represent
#             either a scalar, a list, or a numpy array.
#
#     Returns:
#         Union[int, pd.NA]: Number of layers for array-like inputs, or pd.NA for
#         scalar NA and other unsupported cases.
#     """
#     if isinstance(params, float) and pd.isna(params):  # Handle scalar NA
#         return pd.NA
#     if isinstance(params, (list, np.ndarray)):  # Handle array types
#         return len(params) // 2
#     return pd.NA  # Handle other cases
#
#
# def merge_metadata(row: pd.Series) -> dict:
#     """
#     Merge metadata dictionaries safely
#
#     Parameters:
#     row : pandas Series containing metadata columns
#
#     Returns:
#     dict: Combined metadata dictionary
#     """
#     initial_metadata = row["metadata_x"] if isinstance(row["metadata_x"], dict) else {}
#     additional_metadata = (
#         row["metadata_y"] if isinstance(row["metadata_y"], dict) else {}
#     )
#
#     return {**initial_metadata, **additional_metadata}
