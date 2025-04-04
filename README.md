# Efficient QAOA Architecture for Multi-Constrained Optimization
This repo contains the data, results and evaluation code to reproduce the plots in the paper.
As of right now, the code for simulation can not be provided. Please reach out to the
authors for more information.

## Installation

To install and use, the authors recommend the [uv package manager](https://docs.astral.sh/uv/).
Clone the repo and install uv in the system. Then run
```
uv pip install -e .
```
This should install all dependencies including development tools like jupyter lab

## Running the evaluation notebooks

To run the evaluation notebooks for generating the plots featured in the paper, it is
best to use the provided jupyter lab
```
uv run jupyter lab
```
The notebooks are placed in the notebooks directory.

## Accessing the instances
All instances solved throughout the paper are located in the instances directory. They
are in LP file format. On the python-side, we use `dimod` to load the optimization
problems.

To ease the use of the instances, we provide the instance manager class
```python
from instances import InstanceManager

# instantiate an InstanceManger for all problems located within a directory
im = InstanceManager("instances/prosumer_problem")

# get a single instance
cqm, opt, _ = im.get("Comb0_0_increasing.lp")

# iterate over all instances
for cqm, opt, name in im.instance_iterator():
    print(name, opt)
```


