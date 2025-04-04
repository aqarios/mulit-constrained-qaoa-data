from dimod import ConstrainedQuadraticModel
from gurobipy import Env, GRB
import os
import pandas as pd
import dimod
from dataclasses import dataclass
from typing import Generator

from utils.solving import build_gurobi_model_from_cqm



def calculate_exact_solution(cqm: ConstrainedQuadraticModel, logfile: str, quiet: bool):
    # calculate exact solution
    env = Env(logfile, empty=True)
    # transform cqm to gurobi model
    env.setParam("OutputFlag", int(not quiet))
    env.start()

    model, _ = build_gurobi_model_from_cqm(cqm=cqm, env=env)

    # solve gurobi model
    model.update()
    model.optimize()

    # save exact solution
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        raise ValueError("Model not feasible")


@dataclass
class SingleInstance:
    exact_solution: float
    file_name: str
    file_path: str

    def load_cqm(self) -> tuple[ConstrainedQuadraticModel, float, str]:
        with open(self.file_path, "r") as file:
            cqm = dimod.lp.load(file)

        return cqm, self.exact_solution, self.file_name


class InstanceManager:
    instance_directory: str
    instances: list[SingleInstance]

    def __init__(self, instance_directory: str, quiet: bool = True, logfile: str = ""):
        # check which path to select based on starting .
        if instance_directory.startswith("./"):
            self.instance_directory = os.path.abspath(
                os.path.join(os.path.dirname(__file__), instance_directory)
            )
        else:
            # get path until parent folder
            project_path = os.path.dirname(os.path.dirname(__file__))
            self.instance_directory = os.path.join(project_path, instance_directory)

        # check if overview.csv exists, if not create one with columns file_name, time created, exact solution
        overview_path = os.path.join(self.instance_directory, "overview.csv")

        if not os.path.exists(overview_path):
            df = pd.DataFrame(
                columns=[
                    "file_name",
                    "time_created",
                    "exact_solution",
                    "num_vars",
                    "num_cons",
                ]
            )
            df.to_csv(overview_path, index=False)
        else:
            df = pd.read_csv(overview_path)

        # iterate through all lp files in instance_directory
        self.instances = []

        for root, _, files in os.walk(self.instance_directory):
            for file in files:
                if not file.endswith(".lp"):
                    continue
                file_path = os.path.join(root, file)

                file_time = os.path.getmtime(file_path)

                # check if filename not in df or file created timestamp differs
                old_entry = df[
                    (df["file_name"] == file) & (df["time_created"] == file_time)
                ]
                if old_entry.empty:
                    with open(file_path, "r") as f:
                        cqm = dimod.lp.load(f)
                    exact_solution = calculate_exact_solution(cqm, logfile, quiet)
                    df = df[df["file_name"] != file]

                    # Creating a new DataFrame with the row to add
                    new_row_df = pd.DataFrame(
                        [
                            {
                                "file_name": file,
                                "time_created": file_time,
                                "exact_solution": exact_solution,
                                "num_vars": len(cqm.variables),
                                "num_cons": len(cqm.constraints),
                            }
                        ]
                    )
                    # Concatenating the new row with the existing DataFrame
                    if df.empty:
                        df = new_row_df
                    else:
                        df = pd.concat([df, new_row_df], ignore_index=True)
                else:
                    exact_solution = old_entry["exact_solution"].values[0]

                # append tuple loaded cqm, and exact solution to self.instances
                instance = SingleInstance(exact_solution, file, file_path)
                self.instances.append(instance)

        self.instances_dict = {
            instance.file_name: instance for instance in self.instances
        }

        # save updated DataFrame to CSV
        df.to_csv(overview_path, index=False)

    def instance_iterator(self) -> Generator[ConstrainedQuadraticModel, float, str]:
        # include a function that gives an iterator over self.instances with yield
        for instance_obj in self.instances:
            yield instance_obj.load_cqm()

    def get(self, instance_label: str) -> tuple[ConstrainedQuadraticModel, float]:
        """
        Function to return a single problem instance, cqm and exact solution
        """
        instance: SingleInstance = self.instances_dict[instance_label]
        return instance.load_cqm()

    def get_metadata(self, instance_label: str) -> tuple[float, str]:
        if instance_label not in self.instances_dict:
            raise ValueError(
                f"There is no instance {instance_label} is not in provided directory."
            )
        instance: SingleInstance = self.instances_dict[instance_label]
        return instance.exact_solution, instance.file_name
