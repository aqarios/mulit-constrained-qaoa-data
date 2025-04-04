import gurobipy as gp
from dimod import (
    BinaryQuadraticModel,
    ConstrainedQuadraticModel,
    SampleSet,
    Vartype,
    QuadraticModel,
    sym,
)
from gurobipy import GRB


def solve_cqm_gurobi(
    cqm: ConstrainedQuadraticModel,
    timeout: int = 30,
    quiet: bool = False,
    gap: float = None,
    work_limit: float = None,
    objective_stop: float = None,
    method: int = None,
    return_model: bool = False,
    seed: int = None,
    start: dict[str, int | float] = None,
    **_,
) -> SampleSet:
    """Solves a ConstrainedQuadraticModel with Gurobi by first translating the
    model into Gurobi's modelling language and then solving.

    Parameters
    ----------
    cqm : ConstrainedQuadraticModel
        The model to solve.
    timeout : int
        The time limit stopping criterion.
    quiet : bool
        A flag to set Gurobi output quiet.
    gap : float
        The MIPGap stopping criterion.
    work_limit : float
        The Gurobi internal deterministic working time as stopping criterion.
    objective_stop : float
        The solver objective value a stopping criterion.

    Returns
    -------
    SampleSet
        The Solver result wrapped in a Dimod SampleSet
    """
    # Set Gurobi environment variables for quiet output
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", int(not quiet))
    env.start()
    gm, vardict = build_gurobi_model_from_cqm(cqm, env=env, start=start)

    # Set Gurobi solver parameters
    if timeout is not None:
        gm.setParam("TimeLimit", timeout)
    if work_limit is not None:
        gm.setParam("WorkLimit", work_limit)
    if objective_stop is not None:
        gm.setParam("BestObjStop", objective_stop)
    if gap is not None:
        gm.setParam("MIPGap", gap)
    if method is not None:
        print(f"using method {method}")
        gm.setParam("Method", method)
    if seed is not None:
        gm.setParam("Seed", seed)

    try:
        # Solve
        gm.optimize()
    except gp.GurobiError as e:
        if "NonConvex" in e.message:
            gm.setParam("NonConvex", 2)
            gm.optimize()
        else:
            raise e

    # Add metadata
    info = {"solver": "gurobi", "time": gm.Runtime}
    if return_model:
        info["model"] = gm

    def get_var(v):
        if isinstance(v, gp.LinExpr):
            return int(round(2 * v.getVar(0).X - 1))
        else:
            return v.X

    # Wrap in sampleset
    s = SampleSet.from_samples_cqm(
        [{k: get_var(v) for k, v in vardict.items()}],
        cqm,
        info=info,
    )
    return s


def solve_bqm_gurobi(bqm: BinaryQuadraticModel, **kwargs):
    cqm = ConstrainedQuadraticModel()
    cqm.set_objective(bqm)
    return solve_cqm_gurobi(cqm, **kwargs)


def build_gurobi_model_from_cqm(cqm, env, start=None):
    gm = gp.Model(env=env)

    # Translate the variables from dimod to gurobipy
    vardict = convert_decision_variables(cqm, gm)

    if start is not None:
        for var, v in start.items():
            vardict[var].Start = v

    # Translate the constraints
    convert_constraints(cqm, gm, vardict)

    # Translate the objective
    convert_objective(cqm, gm, vardict)

    return gm, vardict


def convert_objective(cqm, gm, vardict):
    qm = cqm.objective
    obj = sum(vardict[name] * bias for name, bias in qm.iter_linear()) + qm.offset
    obj += sum(vardict[n1] * vardict[n2] * bias for n1, n2, bias in qm.iter_quadratic())
    # print("adding constraints and objective", time.perf_counter() - b)
    # print("total", time.perf_counter() - a)
    gm.setObjective(obj)


def convert_constraints(cqm, gm, vardict):
    for label, eq in cqm.constraints.items():
        qm: QuadraticModel = eq.lhs
        lhs = sum(vardict[name] * bias for name, bias in qm.iter_linear()) + qm.offset
        lhs += sum(
            vardict[n1] * vardict[n2] * bias for n1, n2, bias in qm.iter_quadratic()
        )

        if isinstance(eq, sym.Eq):
            gm.addConstr(lhs == eq.rhs, name=label)
        elif isinstance(eq, sym.Le):
            gm.addConstr(lhs <= eq.rhs, name=label)
        elif isinstance(eq, sym.Ge):
            gm.addConstr(lhs >= eq.rhs, name=label)


def convert_decision_variables(cqm, gm):
    vardict = {}
    for var in cqm.variables:
        if cqm.vartype(var) == Vartype.BINARY:
            vardict[var] = gm.addVar(name=str(var), vtype=GRB.BINARY)
        if cqm.vartype(var) == Vartype.SPIN:
            vardict[var] = (2 * gm.addVar(name=str(var), vtype=GRB.BINARY)) - 1
        if cqm.vartype(var) == Vartype.INTEGER:
            vardict[var] = gm.addVar(
                name=str(var),
                vtype=GRB.INTEGER,
                lb=cqm.lower_bound(var),
                ub=cqm.upper_bound(var),
            )
        if cqm.vartype(var) == Vartype.REAL:
            vardict[var] = gm.addVar(
                name=str(var),
                vtype=GRB.CONTINUOUS,
                lb=cqm.lower_bound(var),
                ub=cqm.upper_bound(var),
            )

    return vardict
