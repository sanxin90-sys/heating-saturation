"""
Derivative estimators for the dynamic (ODE-discovery) tasks.

* finite_difference : second-order central differences (np.gradient).
* savitzky_golay    : local polynomial smoothing + analytic derivative.
* xtfc              : Python port of the black-box X-TFC stage of AI-Lorenz
                      (De Florio, Kevrekidis & Karniadakis, 2024; MATLAB
                      original at github.com/mariodeflorio/AI-Lorenz,
                      `bbxtfc_lorenz.m`). Each sub-domain is fitted with a
                      random-feature (extreme-learning-machine) expansion whose
                      constrained expression pins the left end to the previous
                      sub-domain, so the learned trajectory is continuous and
                      its derivative is analytic.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


def finite_difference(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return y, np.gradient(y, t, axis=0)


def savitzky_golay(
    t: np.ndarray, y: np.ndarray, window: int = 15, polyorder: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    dt = float(np.mean(np.diff(t)))
    ys = savgol_filter(y, window, polyorder, axis=0)
    dy = savgol_filter(y, window, polyorder, deriv=1, delta=dt, axis=0)
    return ys, dy


def xtfc(
    t: np.ndarray,
    y: np.ndarray,
    samples_per_subdomain: int = 50,
    n_neurons: int = 20,
    n_collocation: int | None = None,
    lb: float = -1.0,
    ub: float = 1.0,
    smooth_window: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Black-box X-TFC. Returns smoothed states and derivatives at the original
    sample times `t` (shape (n, d)).

    The constrained expression on each sub-domain is
        y(x) = (h(x) - h(-1)) @ xi + y0,   x in [-1, 1],
    and, because it is linear in xi, the iterative least-squares loop of the
    MATLAB original converges in a single minimum-norm solve.
    """
    y = np.atleast_2d(y.T).T
    n_obs, dim = y.shape
    rng = np.random.default_rng(seed)
    n_col = n_collocation or samples_per_subdomain
    t_step = samples_per_subdomain * float(np.mean(np.diff(t)))
    t_nodes = np.arange(t[0], t[-1] + 1e-12, t_step)
    if t_nodes[-1] < t[-1]:
        t_nodes = np.append(t_nodes, t[-1])
    n_sub = len(t_nodes) - 1

    x = np.linspace(-1.0, 1.0, n_col)
    w = rng.uniform(lb, ub, n_neurons)
    b = rng.uniform(lb, ub, n_neurons)
    h = np.tanh(np.outer(x, w) + b)
    hd = w * (1.0 - h**2)
    h_c = h - h[0]
    pinv = np.linalg.pinv(h_c)

    spline = CubicSpline(t, y, axis=0)
    t_col_all, y_all, dy_all = [], [], []
    y0 = y[0].copy()
    for i in range(n_sub):
        ta, tb = t_nodes[i], t_nodes[i + 1]
        tc = ta + (x + 1.0) * (tb - ta) / 2.0
        data = spline(tc)
        c_i = 2.0 / (tb - ta)
        xi = pinv @ (data - y0)
        yi = h_c @ xi + y0
        dyi = c_i * hd @ xi
        sl = slice(0 if i == 0 else 1, None)  # avoid duplicating shared nodes
        t_col_all.append(tc[sl]); y_all.append(yi[sl]); dy_all.append(dyi[sl])
        y0 = yi[-1]

    tc = np.concatenate(t_col_all)
    ys = np.concatenate(y_all)
    dys = np.concatenate(dy_all)
    if smooth_window:
        ys = savgol_filter(ys, smooth_window, 5, axis=0)
        dys = savgol_filter(dys, smooth_window, 5, axis=0)
    ys_obs = CubicSpline(tc, ys, axis=0)(t)
    dys_obs = CubicSpline(tc, dys, axis=0)(t)
    return ys_obs, dys_obs


ESTIMATORS = {
    "finite_difference": finite_difference,
    "savitzky_golay": savitzky_golay,
    "xtfc_ai_lorenz": xtfc,
}
