from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from datetime import datetime, timezone

import pandas as pd


@dataclass
class SingleRun(ABC):
    """
    Abstract base class defining the interface for single run operations.

    The SingleRun class provides a blueprint for creating specific run objects
    that can be converted to row entries and have unique columns defined.
    Subclasses are required to implement methods for obtaining unique columns
    and converting instances to row entries.

    Methods
    -------
    to_row_entry()
        Abstract method that must be implemented by subclasses. This method
        should provide the logic to convert the run object into a
        representation suitable for a row entry.

    get_unique_columns() -> list[str]
        Abstract method that must be implemented by subclasses. This method
        returns a list of unique column names and run attributes that need to
        coincide to detect duplicates.
    """

    @abstractmethod
    def to_row_entry(self, **kwargs) -> pd.DataFrame:
        """
        to_row_entry(self) -> pd.DataFrame
            Converts the optimization results into a pandas DataFrame, aggregating key performance
            metrics and metadata. It processes each layer's results, evaluates the circuit performance,
            and ensures that data types are JSON serializable.

            Returns
            -------
            pandas.DataFrame
                A DataFrame containing the results with each row representing an optimization run and
                its corresponding metrics and metadata.

            Notes
            -----
            Ensure that numpy integer types are converted to Python native integers to maintain JSON
            serializability.
        """
        raise NotImplementedError("Please implement to_row_entry function!")


@dataclass
class ResultTableRow(ABC):
    """
    Class representing a row in a result table.

    This class provides methods to retrieve column names and types for the
    result table, and to convert an instance of the class into a dictionary
    representation. It is meant to be extended by other classes to provide
    specific implementations of result table rows.
    """

    def __init__(self, run: SingleRun, **kwargs):
        for field in fields(self):
            if field.name in dir(run):
                run_attr = getattr(run, field.name)
                setattr(self, field.name, run_attr)
            elif field.name in kwargs:
                run_attr = kwargs[field.name]
                setattr(self, field.name, run_attr)
            elif field.name == "timestamp":
                self.timestamp = datetime.now(timezone.utc)
            else:
                raise KeyError(f"{field.name} is not a valid attribute")

    @classmethod
    def result_table_columns(cls) -> list:
        return [field.name for field in fields(cls)]

    @classmethod
    def result_table_types(cls) -> list[str, type]:
        return [(field.name, field.type) for field in fields(cls)]

    def to_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class ExperimentResultRow(ResultTableRow):
    """
    Represents a row in the result table specific to a configured optimization run.

    The Experiment Setup Row class holds all relevant information about a specific
    optimization run, including identifiers, parameters, and metrics. It serves
    as a container for the results data obtained from the optimization process.

    Attributes
    ----------
    id : str
        Unique identifier for the specific optimization run.
    pipeline_name : str
        Name of the pipeline used in the optimization run.
    passes : dict
        Dictionary containing the passes applied during the optimization.
    optimizer_params : dict
        Dictionary of parameters specific to the optimizer used.
    circuit_generator_params : dict
        Dictionary of parameters for the circuit generator involved.
    metadata : dict
        Dictionary containing additional metadata about the optimization run.
    """

    # consistent for configured optimization run
    id: str
    pipeline_name: str
    passes: dict
    optimizer_params: dict
    circuit_generator_params: dict

    # sub layer params
    metrics: dict
    metadata: dict
    sublayer: int
    timestamp: datetime = None

    @classmethod
    def init_empty(cls) -> ExperimentResultRow:
        """
        Create a new instance with all fields initialized to None.

        Creates a ExperimentSetupRow instance where all dataclass fields are
        set to None, except timestamp which is set to the current UTC time.
        This is useful for creating placeholder rows or testing.

        Returns
        -------
        ExperimentSetupRow
            A new instance with all attributes set to None
        """
        empty_dict = {
            field.name: None for field in fields(cls) if field.name != "timestamp"
        }
        return cls(**empty_dict)  # type: ignore
