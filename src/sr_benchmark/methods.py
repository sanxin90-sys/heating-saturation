"""
Uniform wrappers around symbolic-regression engines.

Every wrapper takes (X, y, var_names, time_budget_s, seed) and returns a
`Fit` with a predict function and a human-readable expression. Engines that
cannot be imported (e.g. PySR without a Julia runtime) raise `Unavailable`.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sympy as sp


class Unavailable(RuntimeError):
    pass


@dataclass
class Fit:
    expression: str
    predict: Callable[[np.ndarray], np.ndarray]
    seconds: float


def _sympy_predict(expr: str, var_names: list[str]) -> Callable[[np.ndarray], np.ndarray]:
    syms = sp.symbols(var_names)
    f = sp.lambdify(syms, sp.sympify(expr, locals={v: s for v, s in zip(var_names, syms)}), "numpy")
    return lambda X: np.broadcast_to(np.asarray(f(*X.T), dtype=float), (len(X),)).copy()


def complexity(expr: str) -> int:
    """Number of nodes in the SymPy expression tree."""
    try:
        return sum(1 for _ in sp.preorder_traversal(sp.sympify(expr)))
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# PyOperon (C++ genetic programming with nonlinear least-squares constants)
# ---------------------------------------------------------------------------
def fit_operon(X, y, var_names, time_budget_s=60, seed=0) -> Fit:
    from pyoperon.sklearn import SymbolicRegressor

    X = np.array(X, dtype=float, order="C", copy=True)  # Operon rejects read-only arrays
    y = np.array(y, dtype=float, order="C", copy=True)
    t0 = time.perf_counter()
    reg = SymbolicRegressor(
        allowed_symbols="add,sub,mul,div,exp,log,sqrt,square,constant,variable",
        max_length=40,
        max_depth=10,
        population_size=1000,
        generations=10_000,
        max_evaluations=10_000_000,
        optimizer_iterations=10,
        objectives=["r2", "length"],
        model_selection_criterion="minimum_description_length",
        max_time=time_budget_s,
        n_threads=int(os.environ.get("SR_THREADS", 2)),
        random_state=seed,
    )
    reg.fit(X, y)
    expr = reg.get_model_string(reg.model_, 6, var_names)
    return Fit(expr, reg.predict, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# gplearn (pure-Python genetic programming, Koza-style)
# ---------------------------------------------------------------------------
_GPLEARN_LOCALS = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b, "sqrt": lambda a: sp.sqrt(sp.Abs(a)),
    "log": lambda a: sp.log(sp.Abs(a)), "exp": sp.exp, "neg": lambda a: -a, "inv": lambda a: 1 / a,
}


def fit_gplearn(X, y, var_names, time_budget_s=60, seed=0) -> Fit:
    from gplearn.functions import make_function
    from gplearn.genetic import SymbolicRegressor

    def _pexp(a):
        with np.errstate(over="ignore"):
            return np.where(a < 50, np.exp(np.minimum(a, 50)), 0.0)

    pexp = make_function(function=_pexp, name="exp", arity=1)
    t0 = time.perf_counter()
    best, gen = None, 0
    # gplearn has no wall-clock limit: evolve in warm-started chunks until the budget is spent
    reg = SymbolicRegressor(
        population_size=2000, generations=1, tournament_size=20,
        function_set=("add", "sub", "mul", "div", "sqrt", "log", pexp, "neg", "inv"),
        parsimony_coefficient=1e-3, p_crossover=0.7, p_subtree_mutation=0.1,
        p_hoist_mutation=0.05, p_point_mutation=0.1, max_samples=1.0,
        const_range=(-5, 5), warm_start=True, n_jobs=1, random_state=seed,
    )
    while time.perf_counter() - t0 < time_budget_s and gen < 200:
        gen += 1
        reg.set_params(generations=gen)
        reg.fit(X, y)
        best = reg
    expr_raw = str(best._program)
    expr_raw = re.sub(r"X(\d+)", lambda m: var_names[int(m.group(1))], expr_raw)
    try:
        expr = str(sp.sympify(expr_raw, locals={**_GPLEARN_LOCALS, **{v: sp.Symbol(v) for v in var_names}}))
    except Exception:
        expr = expr_raw
    return Fit(expr, best.predict, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# SINDy-style sparse regression on a fixed candidate library (STLSQ)
# ---------------------------------------------------------------------------
def _library(X, var_names):
    cols, names = [np.ones(len(X))], ["1"]
    n = X.shape[1]
    for i in range(n):
        xi, vi = X[:, i], var_names[i]
        cols += [xi, xi**2, xi**3]
        names += [vi, f"{vi}**2", f"{vi}**3"]
        if np.all(xi > 0):
            cols += [np.log(xi), np.sqrt(xi), 1 / xi]
            names += [f"log({vi})", f"sqrt({vi})", f"1/{vi}"]
        for j in range(i + 1, n):
            cols.append(xi * X[:, j]); names.append(f"{vi}*{var_names[j]}")
            if np.all(X[:, j] != 0):
                cols.append(xi / X[:, j]); names.append(f"{vi}/{var_names[j]}")
            if np.all(xi != 0):
                cols.append(X[:, j] / xi); names.append(f"{var_names[j]}/{vi}")
    return np.column_stack(cols), names


def fit_sindy(X, y, var_names, time_budget_s=60, seed=0, threshold=0.1) -> Fit:
    import pysindy as ps

    t0 = time.perf_counter()
    theta, names = _library(X, var_names)
    scale = np.linalg.norm(theta, axis=0)
    scale[scale == 0] = 1.0
    opt = ps.STLSQ(threshold=threshold * np.linalg.norm(y), alpha=1e-8, max_iter=50)
    opt.fit(theta / scale, y.reshape(-1, 1))
    coef = np.ravel(opt.coef_) / scale
    terms = [f"({c:.6g})*{n}" if n != "1" else f"({c:.6g})" for c, n in zip(coef, names) if c != 0]
    expr = " + ".join(terms) or "0"
    return Fit(expr, _sympy_predict(expr, var_names), time.perf_counter() - t0)


def fit_sindy_poly(X, y, var_names, time_budget_s=60, seed=0, threshold=0.1, degree=2) -> Fit:
    """Classic SINDy: quadratic polynomial library only (Brunton et al., 2016)."""
    import pysindy as ps
    from sklearn.preprocessing import PolynomialFeatures

    t0 = time.perf_counter()
    pf = PolynomialFeatures(degree=degree, include_bias=True)
    theta = pf.fit_transform(X)
    names = [n.replace(" ", "*").replace("^", "**") for n in pf.get_feature_names_out(var_names)]
    scale = np.linalg.norm(theta, axis=0)
    scale[scale == 0] = 1.0
    opt = ps.STLSQ(threshold=threshold * np.linalg.norm(y), alpha=1e-8, max_iter=50)
    opt.fit(theta / scale, y.reshape(-1, 1))
    coef = np.ravel(opt.coef_) / scale
    terms = [f"({c:.6g})*{n}" if n != "1" else f"({c:.6g})" for c, n in zip(coef, names) if c != 0]
    expr = " + ".join(terms) or "0"
    return Fit(expr, _sympy_predict(expr, var_names), time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# PySR (Julia backend). Raises Unavailable when Julia cannot be provisioned.
# ---------------------------------------------------------------------------
def fit_pysr(X, y, var_names, time_budget_s=60, seed=0) -> Fit:
    try:
        from pysr import PySRRegressor
    except Exception as exc:  # juliacall downloads Julia on first import
        raise Unavailable(f"PySR/Julia not available: {type(exc).__name__}") from None

    t0 = time.perf_counter()
    reg = PySRRegressor(
        niterations=1000,
        populations=15,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt", "square"],
        maxsize=30,
        timeout_in_seconds=time_budget_s,
        model_selection="best",
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        progress=False,
        verbosity=0,
    )
    reg.fit(X, y, variable_names=var_names)
    return Fit(str(reg.sympy()), reg.predict, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# AI Feynman 2.0 (run in a subprocess: it writes to its working directory)
# ---------------------------------------------------------------------------
AIF_HOME = Path(os.environ.get("AIFEYNMAN_HOME", Path(__file__).resolve().parent / "_aifeynman"))


def fit_aifeynman(X, y, var_names, time_budget_s=900, seed=0) -> Fit:
    """
    AI Feynman is orders of magnitude slower than the GP engines (its own README
    quotes 10-30 min for its first example), so `time_budget_s` is the total
    wall-clock limit of the subprocess. If the run is cut before the final
    solution file is written, the last Pareto front printed to stdout is used.
    """
    if not (AIF_HOME / "aifeynman" / "__init__.py").exists():
        raise Unavailable(f"AI Feynman sources not found at {AIF_HOME} (run setup_aifeynman.sh)")
    t0 = time.perf_counter()
    work = Path(tempfile.mkdtemp(prefix="aif_"))
    data_dir = work / "data"
    data_dir.mkdir()
    np.savetxt(data_dir / "task.txt", np.column_stack([X, y]))
    bf_time = max(10, time_budget_s // 10)
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from aifeynman import run_aifeynman\n"
        "run_aifeynman('data/', 'task.txt', %d, '14ops.txt', polyfit_deg=3, NN_epochs=300)\n"
    ) % (str(AIF_HOME), bf_time)
    log = work / "log.txt"
    with log.open("w") as fh:
        # own process group, so the Fortran brute-force children are killed on timeout too
        proc = subprocess.Popen(
            [sys.executable, "-c", code], cwd=work, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "1",
                 "PATH": f"{AIF_HOME / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"},
        )
        try:
            proc.wait(timeout=time_budget_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    sol = work / "results" / "solution_task.txt"
    if sol.exists():
        exprs = [" ".join(ln.split()[5:]) for ln in sol.read_text().splitlines() if len(ln.split()) >= 6]
    else:  # last printed Pareto front: lines "complexity  mdl_loss  expression"
        exprs, block = [], []
        for ln in log.read_text(errors="ignore").splitlines():
            if ln.startswith("Complexity #"):
                block = []
            elif re.match(r"^\s*[\d.]+\s+[\d.]+\s+\S", ln):
                block.append(" ".join(ln.split()[2:]))
            if block:
                exprs = block[:]
    shutil.rmtree(work, ignore_errors=True)
    aif_vars = [f"x{i}" for i in range(len(var_names))]
    candidates = []
    for expr in exprs:
        try:
            pred = _sympy_predict(expr, aif_vars)(X)
            err = float(np.mean((pred - y) ** 2))
            if np.isfinite(err):
                candidates.append((err, expr))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("AI Feynman returned no usable expression within the budget")
    _, expr = min(candidates)
    expr_named = str(sp.sympify(expr).subs({sp.Symbol(a): sp.Symbol(v) for a, v in zip(aif_vars, var_names)}))
    return Fit(expr_named, _sympy_predict(expr_named, var_names), time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# PhySO (physics-constrained symbolic optimisation: RNN policy + risk-seeking
# gradient, Tenachi et al. 2023). Runs in-process with parallel_mode off.
# ---------------------------------------------------------------------------
def fit_physo(X, y, var_names, time_budget_s=60, seed=0) -> Fit:
    try:
        import physo
        import torch
    except Exception as exc:
        raise Unavailable(f"PhySO not available: {type(exc).__name__}") from None

    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()
    # PhySO has no wall-clock limit; the budget is mapped to epochs (about 50 s per
    # epoch on 1 000 points with free-constant optimisation, measured on this container).
    epochs = max(2, int(time_budget_s / 50))
    prog, _log = physo.SR(
        np.asarray(X, dtype=float).T.copy(), np.asarray(y, dtype=float).copy(),
        X_names=list(var_names), y_name="y",
        fixed_consts=[1.0], free_consts_names=["c0", "c1", "c2"],
        epochs=epochs, max_n_evaluations=None, stop_after_n_epochs=epochs,
        use_protected_ops=False,  # protected sqrt/log would not survive the SymPy export
        parallel_mode=False, device="cpu",
    )
    sym = prog.get_infix_sympy(evaluate_consts=True)
    if isinstance(sym, (list, tuple, np.ndarray)):
        sym = np.ravel(sym)[0]
    expr = str(sym)
    return Fit(expr, _sympy_predict(expr, var_names), time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Kolmogorov–Arnold network with symbolic snapping (pykan; Liu et al. 2024).
# A small KAN is trained, then each edge is replaced by the closest symbolic
# function from a fixed library.
# ---------------------------------------------------------------------------
def fit_kan(X, y, var_names, time_budget_s=60, seed=0) -> Fit:
    try:
        import torch
        from kan import KAN
    except Exception as exc:
        raise Unavailable(f"pykan not available: {type(exc).__name__}") from None

    torch.manual_seed(seed)
    t0 = time.perf_counter()
    Xt = torch.tensor(np.asarray(X, dtype=float), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y, dtype=float)[:, None], dtype=torch.float32)
    data = {"train_input": Xt, "train_label": yt, "test_input": Xt, "test_label": yt}
    model = KAN(width=[X.shape[1], 3, 1], grid=5, k=3, seed=seed, device="cpu")
    model.fit(data, opt="LBFGS", steps=int(os.environ.get("KAN_STEPS", 30)), lamb=0.001, log=1000)
    model.auto_symbolic()
    syms = sp.symbols(var_names)
    expr = str(model.symbolic_formula(var=list(syms))[0][0])
    return Fit(expr, _sympy_predict(expr, var_names), time.perf_counter() - t0)


def fit_dso(X, y, var_names, time_budget_s=60, seed=0) -> Fit:
    """Deep Symbolic Regression (Petersen et al., 2019) with DSO's fitted-constant token.

    DSO has no wall-clock limit, so the budget is converted to a number of sampled
    expressions (about 100 per second with constants, measured on the static suite).
    Needs the separate dso environment (see LEIA-ME.md).
    """
    try:
        from dso.task.regression.sklearn import DeepSymbolicRegressor
    except ImportError as exc:
        raise Unavailable(f"DSO not importable: {exc}")
    from dso_check import FUNCTION_SET
    config = {
        "task": {"task_type": "regression", "function_set": FUNCTION_SET + ["const"],
                 "metric": "inv_nrmse", "metric_params": [1.0], "protected": True, "threshold": 1e-12},
        "training": {"n_samples": max(2000, int(100 * time_budget_s)), "epsilon": 0.05},
        "experiment": {"seed": seed, "logdir": None, "verbose": False},
    }
    t0 = time.perf_counter()
    model = DeepSymbolicRegressor(config)
    model.fit(X, y)
    seconds = time.perf_counter() - t0
    sym = model.program_.sympy_expr
    expr = str(sym.subs({sp.Symbol(f"x{i + 1}"): sp.Symbol(v) for i, v in enumerate(var_names)}))
    return Fit(expr, lambda Xn: np.broadcast_to(np.asarray(model.predict(Xn), float), (len(Xn),)).copy(),
               seconds)


METHODS: dict[str, Callable[..., Fit]] = {
    "pysr": fit_pysr,
    "operon": fit_operon,
    "gplearn": fit_gplearn,
    "sindy_stlsq": fit_sindy,
    "sindy_poly2": fit_sindy_poly,
    "ai_feynman": fit_aifeynman,
    "physo": fit_physo,
    "kan": fit_kan,
    "dso": fit_dso,
}
