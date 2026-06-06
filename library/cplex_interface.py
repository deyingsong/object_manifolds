"""
IBM ILOG CPLEX Python wrappers matching the MATLAB cplexqp / cplexlsqlin API.

Functions
---------
cplexoptimset(**kwargs) -> CplexOptions
    Create an options struct (dataclass) mirroring MATLAB cplexoptimset().
cplexqp(H, f, Aineq, bineq, Aeq, beq, lb, ub, x0, options) -> (x, fval, exitflag, output)
    Solve a convex (or first-order) QP, matching MATLAB cplexqp().
cplexlsqlin(C, d, Aineq, bineq, Aeq, beq, lb, ub, options) -> (x, resnorm, residual, exitflag, output, lambda_struct)
    Solve a constrained least-squares problem, matching MATLAB cplexlsqlin().

Requirements
------------
IBM CPLEX Studio >= 22.1.1, Python binding for the matching Python version.
Install via:
    cd /Applications/CPLEX_Studio2211/cplex/python/3.10/x86-64_osx
    pip install .

CPLEX only ships Python 3.8 / 3.9 / 3.10 binaries.  If you are running
Python 3.12+ you must use a Python 3.10 virtual environment.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# CPLEX import – try the standard path first, then the CPLEX Studio path.
# ---------------------------------------------------------------------------
_CPLEX_STUDIO_PATHS = [
    "/Applications/CPLEX_Studio2211/cplex/python/3.10/x86-64_osx",
    "/Applications/CPLEX_Studio2212/cplex/python/3.10/x86-64_osx",
    "/Applications/CPLEX_Studio221/cplex/python/3.10/x86-64_osx",
    os.environ.get("CPLEX_PYTHON_PATH", ""),
]

_cplex = None
for _path in _CPLEX_STUDIO_PATHS:
    if _path and os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import cplex as _cplex_module
    _cplex = _cplex_module
except (ImportError, Exception):
    # CPLEX raises a plain Exception (not ImportError) when the Python
    # version is incompatible (e.g. Python 3.12 vs CPLEX 3.10 binding).
    _cplex = None

# ---------------------------------------------------------------------------
# Options dataclass
# ---------------------------------------------------------------------------

@dataclass
class CplexOptions:
    """
    Options for CPLEX QP / LS solvers.

    Mirrors the MATLAB struct returned by cplexoptimset().

    Attributes
    ----------
    Display : str
        'off' suppresses all CPLEX output.  Any other value enables it.
    threads : int
        Number of threads (0 = automatic, 1 = sequential).
    feasibility_tolerance : float
        simplex.tolerances.feasibility
    optimality_tolerance : float
        simplex.tolerances.optimality
    max_iterations : int
        simplex.limits.iterations (0 = default / unlimited)
    optimality_target : int
        0 = auto-detect convexity (default)
        1 = require convex (globally optimal)
        2 = first-order optimal (for non-convex QP, MATLAB solutiontarget=2)
        3 = globally optimal (branch-and-bound for non-convex)
    qp_method : int
        0 = automatic, 1 = primal, 2 = dual, 4 = barrier
    """

    Display: str = "off"
    threads: int = 0
    feasibility_tolerance: float = 1e-9
    optimality_tolerance: float = 1e-9
    max_iterations: int = 0
    optimality_target: int = 0
    qp_method: int = 0


def cplexoptimset(**kwargs) -> CplexOptions:
    """
    Create a CplexOptions with default values, overriding with kwargs.

    Mirrors MATLAB::

        options = cplexoptimset();
        options.Display = 'off';
        options.simplex.tolerances.feasibility = 1e-10;

    The Python mapping is::

        options = cplexoptimset(
            Display='off',
            feasibility_tolerance=1e-10,
            optimality_tolerance=1e-10,
            threads=1,
        )
    """
    return CplexOptions(**{k: v for k, v in kwargs.items()
                           if k in CplexOptions.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _LambdaStruct:
    """Lagrange multipliers returned by cplexlsqlin (mirrors lambda.ineqlin)."""
    ineqlin: np.ndarray = field(default_factory=lambda: np.array([]))
    eqlin: np.ndarray = field(default_factory=lambda: np.array([]))
    lower: np.ndarray = field(default_factory=lambda: np.array([]))
    upper: np.ndarray = field(default_factory=lambda: np.array([]))


def _require_cplex() -> None:
    if _cplex is None:
        raise ImportError(
            "IBM CPLEX Python API is not available.\n"
            "CPLEX Studio 2211 is installed at /Applications/CPLEX_Studio2211/ but "
            "its Python bindings only support Python 3.8–3.10.\n"
            "To use CPLEX solvers, run in a Python 3.10 virtual environment and install:\n"
            "  pip install /Applications/CPLEX_Studio2211/cplex/python/3.10/x86-64_osx"
        )


def _apply_options(c, options: CplexOptions) -> None:
    """Apply CplexOptions to a cplex.Cplex() instance."""
    if options.Display == "off":
        c.set_results_stream(None)
        c.set_log_stream(None)
        c.set_warning_stream(None)
        c.set_error_stream(None)
        c.parameters.simplex.display.set(0)
        try:
            c.parameters.barrier.display.set(0)
        except Exception:
            pass

    if options.threads > 0:
        c.parameters.threads.set(options.threads)

    c.parameters.simplex.tolerances.feasibility.set(options.feasibility_tolerance)
    c.parameters.simplex.tolerances.optimality.set(options.optimality_tolerance)

    if options.max_iterations > 0:
        c.parameters.simplex.limits.iterations.set(options.max_iterations)

    if options.optimality_target > 0:
        c.parameters.optimalitytarget.set(options.optimality_target)

    if options.qp_method > 0:
        c.parameters.qpmethod.set(options.qp_method)


def _cplex_status_to_exitflag(status: int) -> int:
    """Map a CPLEX solution status code to a MATLAB-style exit flag."""
    # Status 1  = CPX_STAT_OPTIMAL
    # Status 5  = CPX_STAT_OPTIMAL_INFEAS (optimal within tolerances)
    # Status 6  = CPX_STAT_NUM_BEST (numerical issues but found solution)
    # Status 2  = CPX_STAT_UNBOUNDED
    # Status 3  = CPX_STAT_INFEASIBLE
    # Status 10 = CPX_STAT_ABORT_IT_LIM
    # Status 4  = CPX_STAT_INForUNBD
    if status == 1:
        return 1   # optimal
    if status == 5:
        return 5   # optimal with infeasibility (matches MATLAB flag 5)
    if status == 6:
        return 6   # numerical best
    if status == 10:
        return 0   # iteration limit
    if status in (2, 18):
        return -2  # unbounded
    if status in (3, 4):
        return -1  # infeasible
    if status > 1:
        return 1   # treat other optimal-class codes as optimal
    return -3      # unknown / error


def _dense_H_to_sparsepairs(H: np.ndarray):
    """Convert a full symmetric H matrix to CPLEX SparsePair list (per column)."""
    n = H.shape[0]
    spairs = []
    for j in range(n):
        col = H[:, j]
        nz = np.where(col != 0)[0]
        if len(nz) > 0:
            spairs.append(_cplex.SparsePair(ind=nz.tolist(), val=col[nz].tolist()))
        else:
            spairs.append(_cplex.SparsePair(ind=[], val=[]))
    return spairs


def _row_to_sparsepair(row: np.ndarray):
    nz = np.where(row != 0)[0]
    return _cplex.SparsePair(ind=nz.tolist(), val=row[nz].tolist())


# ---------------------------------------------------------------------------
# cplexqp
# ---------------------------------------------------------------------------

def cplexqp(
    H: np.ndarray,
    f: np.ndarray,
    Aineq=None,
    bineq=None,
    Aeq=None,
    beq=None,
    lb=None,
    ub=None,
    x0=None,
    options: Optional[CplexOptions] = None,
):
    """
    Solve a quadratic programme using IBM ILOG CPLEX.

    Minimises  0.5 x' H x + f' x
    subject to  Aineq x <= bineq
                Aeq   x == beq
                lb <= x <= ub

    Mirrors MATLAB ``cplexqp``.

    Parameters
    ----------
    H      : (n, n) symmetric positive-semidefinite matrix.
    f      : (n,) linear cost vector.
    Aineq  : (m, n) inequality constraint matrix, or None.
    bineq  : (m,)   inequality rhs, or None.
    Aeq    : (p, n) equality constraint matrix, or None.
    beq    : (p,)   equality rhs, or None.
    lb     : (n,) lower bounds on x (None = -inf).
    ub     : (n,) upper bounds on x (None = +inf).
    x0     : (n,) initial guess (ignored by CPLEX; accepted for API parity).
    options: CplexOptions from cplexoptimset().

    Returns
    -------
    x        : (n,) solution vector (may be all-NaN on failure).
    fval     : float  objective value at x.
    exitflag : int    1 = optimal, 0 = iteration limit, <0 = failure.
    output   : dict   with keys 'status', 'message', 'iterations'.
    """
    _require_cplex()

    if options is None:
        options = CplexOptions()

    H = np.asarray(H, dtype=float)
    f = np.asarray(f, dtype=float).ravel()
    n = len(f)
    assert H.shape == (n, n), f"H must be ({n},{n})"

    c = _cplex.Cplex()
    _apply_options(c, options)

    # --- Variables ---
    # Default bounds: -inf to +inf (matches MATLAB cplexqp with empty lb/ub)
    cpx_lb = ([-_cplex.infinity] * n) if lb is None else list(np.asarray(lb, float))
    cpx_ub = ([_cplex.infinity]  * n) if ub is None else list(np.asarray(ub, float))

    c.variables.add(
        obj=list(f),
        lb=cpx_lb,
        ub=cpx_ub,
    )

    # --- Quadratic objective: 0.5 x' H x ---
    c.objective.set_sense(c.objective.sense.minimize)
    c.objective.set_quadratic(_dense_H_to_sparsepairs(H))

    # --- Inequality constraints ---
    if Aineq is not None and bineq is not None:
        Aineq = np.asarray(Aineq, dtype=float)
        bineq = np.asarray(bineq, dtype=float).ravel()
        m = len(bineq)
        lin_expr = [_row_to_sparsepair(Aineq[i]) for i in range(m)]
        c.linear_constraints.add(
            lin_expr=lin_expr,
            senses=["L"] * m,
            rhs=list(bineq),
        )

    # --- Equality constraints ---
    n_ineq = c.linear_constraints.get_num()
    if Aeq is not None and beq is not None:
        Aeq = np.asarray(Aeq, dtype=float)
        beq = np.asarray(beq, dtype=float).ravel()
        p = len(beq)
        lin_expr = [_row_to_sparsepair(Aeq[i]) for i in range(p)]
        c.linear_constraints.add(
            lin_expr=lin_expr,
            senses=["E"] * p,
            rhs=list(beq),
        )

    # --- Solve ---
    try:
        c.solve()
    except _cplex.exceptions.CplexSolverError as err:
        output = {"status": -100, "message": str(err), "iterations": 0}
        return np.full(n, np.nan), np.nan, -100, output

    status = c.solution.get_status()
    exitflag = _cplex_status_to_exitflag(status)
    status_str = c.solution.get_status_string()

    output = {
        "status": status,
        "message": status_str,
        "iterations": c.solution.progress.get_num_iterations(),
    }

    if exitflag >= 0:
        try:
            x = np.array(c.solution.get_values())
            fval = c.solution.get_objective_value()
        except Exception:
            x = np.full(n, np.nan)
            fval = np.nan
            exitflag = -1
    else:
        x = np.full(n, np.nan)
        fval = np.nan

    return x, fval, exitflag, output


# ---------------------------------------------------------------------------
# cplexlsqlin
# ---------------------------------------------------------------------------

def cplexlsqlin(
    C: np.ndarray,
    d: np.ndarray,
    Aineq=None,
    bineq=None,
    Aeq=None,
    beq=None,
    lb=None,
    ub=None,
    options: Optional[CplexOptions] = None,
):
    """
    Solve a constrained least-squares problem using IBM ILOG CPLEX.

    Minimises  0.5 ||Cx - d||^2
    subject to  Aineq x <= bineq
                Aeq   x == beq
                lb <= x <= ub

    Formulated internally as QP::

        min  0.5 x' (C'C) x  -  (C'd)' x
        s.t. same constraints

    Mirrors MATLAB ``cplexlsqlin``.

    Parameters
    ----------
    C      : (m, n) matrix.
    d      : (m,)   right-hand side.
    Aineq  : (p, n) inequality constraint matrix, or None.
    bineq  : (p,)   inequality rhs, or None.
    Aeq    : (q, n) equality constraint matrix, or None.
    beq    : (q,)   equality rhs, or None.
    lb     : (n,) lower bounds on x (None = -inf).
    ub     : (n,) upper bounds on x (None = +inf).
    options: CplexOptions from cplexoptimset().

    Returns
    -------
    x              : (n,) solution vector.
    resnorm        : float  0.5 ||Cx - d||^2 at solution.
    residual       : (m,)  Cx - d at solution.
    exitflag       : int   1 = optimal, 0 = iteration limit, <0 = failure.
    output         : dict  with 'status', 'message', 'iterations'.
    lambda_struct  : _LambdaStruct  Lagrange multipliers (.ineqlin for inequalities).
    """
    _require_cplex()

    if options is None:
        options = CplexOptions()

    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    m, n = C.shape
    assert len(d) == m

    # QP matrices: H = C'C,  f = -(C'd)
    H = C.T @ C
    f = -(C.T @ d)

    x, fval, exitflag, output = cplexqp(
        H, f, Aineq, bineq, Aeq, beq, lb, ub, None, options
    )

    empty_lam = _LambdaStruct()

    if exitflag < 0 or not np.all(np.isfinite(x)):
        residual = np.full(m, np.nan)
        resnorm = np.nan
        return x, resnorm, residual, exitflag, output, empty_lam

    residual = C @ x - d
    resnorm = float(0.5 * residual @ residual)

    # --- Lagrange multipliers for inequality constraints ---
    # Rebuild CPLEX problem to retrieve dual values (cplexqp discards the object)
    # We re-solve cheaply: just rebuild and retrieve duals from the last solve.
    # Since we already solved above, retrieve duals from that solve.

    # We need to get the dual values from the original solve.
    # To do this properly, re-use the solve via a small wrapper.
    c = _cplex.Cplex()
    _apply_options(c, options)

    cpx_lb = ([-_cplex.infinity] * n) if lb is None else list(np.asarray(lb, float))
    cpx_ub = ([_cplex.infinity]  * n) if ub is None else list(np.asarray(ub, float))
    c.variables.add(obj=list(f), lb=cpx_lb, ub=cpx_ub)
    c.objective.set_sense(c.objective.sense.minimize)
    c.objective.set_quadratic(_dense_H_to_sparsepairs(H))

    n_ineq = 0
    if Aineq is not None and bineq is not None:
        Aineq_arr = np.asarray(Aineq, dtype=float)
        bineq_arr = np.asarray(bineq, dtype=float).ravel()
        n_ineq = len(bineq_arr)
        lin_expr = [_row_to_sparsepair(Aineq_arr[i]) for i in range(n_ineq)]
        c.linear_constraints.add(lin_expr=lin_expr, senses=["L"] * n_ineq,
                                  rhs=list(bineq_arr))

    if Aeq is not None and beq is not None:
        Aeq_arr = np.asarray(Aeq, dtype=float)
        beq_arr = np.asarray(beq, dtype=float).ravel()
        n_eq = len(beq_arr)
        lin_expr = [_row_to_sparsepair(Aeq_arr[i]) for i in range(n_eq)]
        c.linear_constraints.add(lin_expr=lin_expr, senses=["E"] * n_eq,
                                  rhs=list(beq_arr))

    try:
        c.solve()
        if n_ineq > 0:
            duals = np.array(c.solution.get_dual_values(0, n_ineq - 1))
        else:
            duals = np.array([])
    except Exception:
        duals = np.zeros(n_ineq)

    lambda_struct = _LambdaStruct(ineqlin=duals)
    return x, resnorm, residual, exitflag, output, lambda_struct
