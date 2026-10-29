"""
Benchmark tasks: established environmental / micrometeorological equations.

Static tasks map instantaneous state variables to a diagnosed quantity.
Dynamic tasks require the right-hand side of an ODE to be discovered from a
noisy time series, so the derivative must be estimated first.

Every task stores its ground truth as a SymPy-parsable string, so recovered
expressions can be compared symbolically and evaluated on an extrapolation
domain that is wider than the training domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class StaticTask:
    name: str
    formula: str                          # SymPy expression in the variable names
    train_ranges: dict[str, tuple[float, float]]
    test_ranges: dict[str, tuple[float, float]]
    description: str
    difficulty: str                       # "easy" | "medium" | "hard"
    constraint: str | None = None         # optional numpy-evaluable mask

    @property
    def variables(self) -> list[str]:
        return list(self.train_ranges)


# Saturation vapour pressure pieces reused below
_ES_KPA = "0.6108*exp(17.27*T/(T + 237.3))"
_DELTA = f"4098*{_ES_KPA}/(T + 237.3)**2"

STATIC_TASKS: list[StaticTask] = [
    StaticTask(
        "tetens_es",
        "6.112*exp(17.67*T/(T + 243.5))",
        {"T": (5, 35)}, {"T": (0, 45)},
        "Saturation vapour pressure (hPa), Tetens/Bolton",
        "easy",
    ),
    StaticTask(
        "stefan_boltzmann",
        "eps*5.670374e-8*Tk**4",
        {"eps": (0.90, 1.00), "Tk": (280, 310)}, {"eps": (0.85, 1.00), "Tk": (265, 325)},
        "Long-wave emission (W m-2)",
        "easy",
    ),
    StaticTask(
        "monin_obukhov_length",
        "-ustar**3*theta/(0.41*9.81*wth)",
        {"ustar": (0.1, 0.8), "theta": (290, 310), "wth": (0.02, 0.3)},
        {"ustar": (0.05, 1.0), "theta": (285, 315), "wth": (0.01, 0.4)},
        "Obukhov length (m) from friction velocity and kinematic heat flux",
        "easy",
    ),
    StaticTask(
        "log_wind_profile",
        "ustar/0.41*log((z - 20)/2)",
        {"ustar": (0.1, 0.9), "z": (28, 62)}, {"ustar": (0.05, 1.2), "z": (25, 80)},
        "Neutral logarithmic wind profile above a forest (d = 20 m, z0 = 2 m)",
        "medium",
    ),
    StaticTask(
        "bulk_sensible_heat",
        "1.2*1005*0.41**2*u*dT/log(z/0.1)**2",
        {"u": (0.5, 6), "dT": (-2, 8), "z": (5, 60)}, {"u": (0.3, 8), "dT": (-3, 10), "z": (3, 80)},
        "Bulk-aerodynamic sensible heat flux (W m-2), neutral",
        "medium",
    ),
    StaticTask(
        "rh_from_dewpoint",
        "100*exp(17.67*Td/(Td + 243.5) - 17.67*T/(T + 243.5))",
        {"T": (15, 38), "Td": (5, 26)}, {"T": (10, 42), "Td": (2, 28)},
        "Relative humidity (%) from air and dew-point temperature",
        "medium",
        constraint="Td <= T",
    ),
    StaticTask(
        "vapour_pressure_deficit",
        f"{_ES_KPA}*(1 - RH/100)",
        {"T": (10, 38), "RH": (20, 100)}, {"T": (5, 42), "RH": (10, 100)},
        "Vapour-pressure deficit (kPa)",
        "medium",
    ),
    StaticTask(
        "hargreaves_et0",
        "0.0023*0.408*Ra*(T + 17.8)*sqrt(DTR)",
        {"Ra": (25, 40), "T": (18, 32), "DTR": (4, 14)}, {"Ra": (20, 42), "T": (14, 36), "DTR": (2, 16)},
        "Hargreaves reference evapotranspiration (mm d-1)",
        "medium",
    ),
    StaticTask(
        "priestley_taylor",
        f"1.26*({_DELTA})/(({_DELTA}) + 0.066)*(Rn - G)",
        {"T": (15, 35), "Rn": (50, 800), "G": (-20, 80)}, {"T": (10, 40), "Rn": (20, 950), "G": (-30, 100)},
        "Priestley-Taylor latent heat flux (W m-2)",
        "hard",
    ),
    StaticTask(
        "penman_monteith_fao56",
        f"(0.408*({_DELTA})*Rn + 0.066*900/(T + 273)*u2*VPD)/(({_DELTA}) + 0.066*(1 + 0.34*u2))",
        {"T": (15, 35), "Rn": (4, 22), "u2": (0.5, 4), "VPD": (0.2, 2.5)},
        {"T": (10, 40), "Rn": (2, 26), "u2": (0.3, 6), "VPD": (0.1, 3.5)},
        "FAO-56 Penman-Monteith reference ET (mm d-1), G = 0",
        "hard",
    ),
]


def sample_static(
    task: StaticTask, n: int, rng: np.random.Generator, test: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform sampling inside the (training or extrapolation) box, noise-free target."""
    import sympy as sp

    ranges = task.test_ranges if test else task.train_ranges
    cols: dict[str, np.ndarray] = {}
    while True:
        m = 3 * n
        cand = {v: rng.uniform(lo, hi, m) for v, (lo, hi) in ranges.items()}
        mask = np.ones(m, bool) if task.constraint is None else eval(task.constraint, {}, cand)
        for v in cand:
            cols[v] = cand[v][mask][:n]
        if mask.sum() >= n:
            break
    x = np.column_stack([cols[v] for v in task.variables])
    syms = sp.symbols(task.variables)
    f = sp.lambdify(syms, sp.sympify(task.formula), "numpy")
    return x, np.asarray(f(*x.T), dtype=float)


def add_noise(y: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise with sigma = level * std(y) (SRBench convention)."""
    return y + rng.normal(0.0, level * np.std(y), y.shape)


# ---------------------------------------------------------------------------
# Dynamic tasks
# ---------------------------------------------------------------------------
@dataclass
class DynamicTask:
    name: str
    state_names: list[str]
    rhs_formulas: list[str]               # one per state; may reference forcing names
    forcing_names: list[str] = field(default_factory=list)
    description: str = ""


LORENZ = DynamicTask(
    "lorenz63",
    ["x", "y", "z"],
    ["10*(y - x)", "x*(28 - z) - y", "x*y - 8/3*z"],
    description="Lorenz (1963) convection model, chaotic regime",
)

JARU_T = DynamicTask(
    "jaru_temperature",
    ["T"],
    ["0.0055*Rn - 0.30*T + 6.9"],
    forcing_names=["Rn"],
    description="Radiatively forced relaxation of canopy-air temperature (synthetic Jaru)",
)


def simulate_lorenz(dt: float = 0.01, t_final: float = 20.0, x0=(-8.0, 7.0, 27.0)) -> dict:
    t = np.arange(0.0, t_final + dt / 2, dt)

    def rhs(_t, s):
        x, y, z = s
        return [10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z]

    sol = solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t, rtol=1e-10, atol=1e-10, method="DOP853")
    states = sol.y.T
    derivs = np.array([rhs(0, s) for s in states])
    return {"t": t, "states": states, "derivs": derivs, "forcing": np.empty((len(t), 0))}


def simulate_jaru(n_days: int = 60, seed: int = 1) -> dict:
    """Synthetic Jaru record with constant T_eq so the RHS is exactly linear."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_inspection import GeneratorConfig, generate_synthetic

    cfg = GeneratorConfig(seed=seed)
    cfg.site.n_days = n_days
    cfg.physics.t_eq_seasonal_amp = 0.0
    df = generate_synthetic(cfg)
    t_h = np.arange(len(df)) * 0.5
    states = df[["T_ar_62m_true"]].to_numpy()
    rn = df[["R_liq_true"]].to_numpy()
    return {"t": t_h, "states": states, "derivs": df[["dTdt_true_K_per_h"]].to_numpy(), "forcing": rn}
