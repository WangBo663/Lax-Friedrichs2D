"""Two-dimensional Euler solver: Lax-Friedrichs, MacCormack and SHASTA/FCT.

Extends the 1D Lax-Friedrichs.py to 2D using a 1-cell ghost layer around the
physical domain.  The same six initial/boundary-value tests are kept so the
schemes can be compared head-to-head against the 1D reference.

The conservative vector has four components in 2D
    U = (rho, rho*u, rho*v, E)
and the equations are
    dU/dt + dF(U)/dx + dG(U)/dy = 0
with
    F = (rho*u, rho*u^2 + p, rho*u*v, (E+p)*u)
    G = (rho*v, rho*u*v, rho*v^2 + p, (E+p)*v).

The MacCormack artificial viscosity follows equations (41)-(42) of the
reference article (van Albada, van Leer and Roberts), applied in x and y
independently.  The FCT/SHASTA method is the SHASTA of equations (43), (44)
and (46), adapted to two spatial dimensions by performing transport,
diffusion and anti-diffusion in x and y separately.

Examples (PowerShell):
    py ./Euler2D.py --test 4 --method all --nx 100 --cfl 0.4
    py ./Euler2D.py --test 3 --method shasta --nx 200
    py ./Euler2D.py --test 5 --method lax_friedrichs --nx 200 --t-end 0.01 0.1 0.2
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
# TkAgg can fail on Windows for large figures ("not enough free memory for
# image buffer" when the GDI heap is fragmented).  Agg is the safe default;
# users can pass --show to opt into TkAgg for the interactive window.
try:
    matplotlib.use("Agg")
except Exception:
    pass
import matplotlib.pyplot as plt
import numpy as np

GAMMA = 1.4
EPS = 1.0e-12


# ---------------------------------------------------------------------------
# Config and result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulationConfig:
    """Parameters shared by every numerical scheme in a fair comparison."""

    test_case: int = 3
    nx: int = 200
    ny: int = 50
    cfl: float = 0.9
    t_end: float | None = None
    mach: float = 2.0
    viscosity_b: float = 1.0
    shasta_limiter: bool = True


@dataclass
class SimulationResult:
    method: str
    x: np.ndarray
    y: np.ndarray
    U: np.ndarray            # shape (4, ny, nx)
    initial_rho: np.ndarray  # shape (ny, nx) for quality scoring
    time: float
    steps: int
    stable: bool
    message: str = ""


# ---------------------------------------------------------------------------
# Domain and Rankine-Hugoniot helpers
# ---------------------------------------------------------------------------
def domain_for_test(test_case: int, nx: int) -> np.ndarray:
    """Return the original 1D coursework domain, kept here for parity with the 1D code."""
    if test_case == 6:
        return np.linspace(0.0, 1.0, nx)
    return np.linspace(-0.5, 0.5, nx)


def shock_relations(mach: float, gamma: float = GAMMA) -> dict[str, float]:
    """Rankine-Hugoniot values for a shock of Mach `mach` running into gas at rest (rho = p = 1).

    Region 1 is the undisturbed gas, region 2 is the state behind the incident shock, and
    region 5 is the state at the wall after reflection.  These closed-form values are used
    both to choose a physically meaningful end time and to check the computed diagnostics.
    """
    sound = float(np.sqrt(gamma))
    incident_speed = mach * sound
    p2 = (2.0 * gamma * mach**2 - (gamma - 1.0)) / (gamma + 1.0)
    rho2 = ((gamma + 1.0) * mach**2) / ((gamma - 1.0) * mach**2 + 2.0)
    u2 = incident_speed * (rho2 - 1.0) / rho2

    # Reflected shock brings the moving gas of region 2 back to rest at the wall.
    p5_over_p2 = ((3.0 * gamma - 1.0) * p2 - (gamma - 1.0)) / ((gamma - 1.0) * p2 + (gamma + 1.0))
    rho5_over_rho2 = ((gamma + 1.0) * p5_over_p2 + (gamma - 1.0)) / ((gamma - 1.0) * p5_over_p2 + (gamma + 1.0))
    rho5 = rho2 * rho5_over_rho2
    reflected_speed = rho2 * u2 / (rho5 - rho2)

    # Downstream Mach number behind a stationary shock.
    mach2_squared = (1.0 + 0.5 * (gamma - 1.0) * mach**2) / (gamma * mach**2 - 0.5 * (gamma - 1.0))

    return {
        "sound": sound,
        "incident_speed": incident_speed,
        "p2": p2,
        "rho2": rho2,
        "u2": u2,
        "p5": p2 * p5_over_p2,
        "rho5": rho5,
        "reflected_speed": reflected_speed,
        "mach_downstream": float(np.sqrt(mach2_squared)),
    }


def default_nx(test_case: int) -> int:
    """Per-test grid size.

    The 1D reference uses nx=300 for tests 1-5 and nx=10000 for test 6
    (Shu-Osher entropy waves are heavily damped by Lax-Friedrichs and need
    that many points to be visible).  In 2D the same nx spans the same x
    interval but the array is (4, ny+2, nx+2), so we scale the Shu-Osher grid
    down to keep memory under control on modest hardware while still giving
    enough resolution to display the entropy waves.
    """
    if test_case == 6:
        return 300
    if test_case == 5:
        return 200
    return 300


def default_cfl(test_case: int) -> float:
    """Per-test CFL number.  Test 4 reflects a strong shock off a wall and
    the artificial-viscosity corrector of equation (41) needs a smaller step
    than the convective limit; tests 1 and 2 in 1D also use 0.4.  Test 6
    (Shu-Osher) also uses 0.4 to keep the FCT scheme monotone.  Tests 3
    (Sod) and 5 (explosion) are bumped down from the 1D default of 0.9 to
    0.4 because the 2D MacCormack transport blows up on a 2D strong shock
    when the time step is too large — the 1D scheme has no second
    direction to confuse the predictor/corrector.
    """
    if test_case in (1, 2, 3, 4, 5, 6):
        return 0.4
    return 0.9


def default_end_time(test_case: int, mach: float = 2.0) -> float:
    """Physics-based end time so each test finishes in its intended wave configuration."""
    if test_case == 6:
        return 0.178
    if test_case == 2:
        return 0.35 / shock_relations(mach)["incident_speed"]
    if test_case == 4:
        relations = shock_relations(mach)
        # Same formula as the 1D reference: shock reaches the right wall at
        # t = 0.5/incident_speed, then the reflected shock travels back from
        # x = 0.5 to x = 0.2.  For M=2 this gives t = 0.4648, identical to
        # the 1D README's Test 4.
        return 0.5 / relations["incident_speed"] + 0.3 / relations["reflected_speed"]
    return 0.2
    return 0.2


def default_runs(test_case: int) -> list[tuple[int, int, float, float, float]]:
    """Per-test (nx, ny, cfl, t_end, mach) tuples.  Each test number triggers the
    parameter combination used in the 1D reference, so changing --test alone
    switches the picture exactly as the 1D README documents.  Pass any of
    --nx / --ny / --cfl / --t-end / --mach to override a single run instead.
    """
    nx = default_nx(test_case)
    cfl = default_cfl(test_case)
    # Test 6's x interval [0,1] is small, so we keep ny modest (≈ nx/10) to
    # fit the 2D array in memory on a typical laptop.
    if test_case == 6:
        ny = max(20, nx // 10)
    else:
        ny = max(50, nx // 4)
    if test_case == 5:
        return [(nx, ny, cfl, 0.01, 2.0),
                (nx, ny, cfl, 0.1, 2.0),
                (nx, ny, cfl, 0.2, 2.0)]
    mach = 3.0 if test_case == 6 else 2.0
    t_end = default_end_time(test_case, mach)
    return [(nx, ny, cfl, t_end, mach)]


def boundary_kind(test_case: int) -> str:
    """Per-test boundary tag, mirroring the 1D reference cell by cell.

    The 1D code has two physical walls (x = -0.5, x = +0.5) and applies
    per-test rules in `apply_bc`.  We translate those rules into one tag
    per test so that a future change to one test never silently mutates
    another test's boundary.

    - "none":   zero-gradient on every side (tests 1, 2, 3, 6)
    - "right":  reflecting wall on the right edge only (test 4)
    - "both":   reflecting walls on both edges in x (test 5)
    - "all":    reflecting walls on all four sides (only via --reflect all)
    """
    return {
        1: "none",   # left wall: zero-gradient (free outflow)
        2: "none",   # moving shock on [-0.5, 0.5], no wall
        3: "none",   # Sod shock tube, no wall
        4: "right",  # right wall reflects, left wall zero-gradient
        5: "both",   # explosion between two reflecting walls
        6: "none",   # Shu-Osher on [0, 1], periodic-like / open ends
    }.get(test_case, "none")


# ---------------------------------------------------------------------------
# Primitive <-> conservative conversions (2D)
# ---------------------------------------------------------------------------
def prim2cons(rho: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray,
              gamma: float = GAMMA) -> np.ndarray:
    """Convert primitive fields to conservative U with shape (4, ny, nx)."""
    energy = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)
    return np.stack((rho, rho * u, rho * v, energy), axis=0)


def cons2prim(U: np.ndarray, gamma: float = GAMMA) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (rho, u, v, p).  Shape of each is (ny, nx) when U is (4, ny, nx)."""
    rho = U[0]
    inv_rho = np.reciprocal(rho)
    u = U[1] * inv_rho
    v = U[2] * inv_rho
    p = (gamma - 1.0) * (U[3] - 0.5 * (U[1] * u + U[2] * v))
    return rho, u, v, p


def flux_x(U: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """x-direction flux F(U), shape (4, ny, nx).  Builds a single (4, ny, nx) array."""
    rho = U[0]
    inv_rho = np.reciprocal(rho)
    u = U[1] * inv_rho
    v = U[2] * inv_rho
    p = (gamma - 1.0) * (U[3] - 0.5 * (U[1] * u + U[2] * v))
    out = np.empty_like(U)
    out[0] = rho * u
    out[1] = rho * u * u + p
    out[2] = rho * u * v
    out[3] = (U[3] + p) * u
    return out


def flux_y(U: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """y-direction flux G(U), shape (4, ny, nx).  Builds a single (4, ny, nx) array."""
    rho = U[0]
    inv_rho = np.reciprocal(rho)
    u = U[1] * inv_rho
    v = U[2] * inv_rho
    p = (gamma - 1.0) * (U[3] - 0.5 * (U[1] * u + U[2] * v))
    out = np.empty_like(U)
    out[0] = rho * v
    out[1] = rho * u * v
    out[2] = rho * v * v + p
    out[3] = (U[3] + p) * v
    return out


# ---------------------------------------------------------------------------
# Initial conditions (1D, lifted to 2D by meshgrid duplication in y)
# ---------------------------------------------------------------------------
def init_test(test_case: int, x: np.ndarray, ny: int, gamma: float,
              mach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Six 1D initial conditions from the original coursework, replicated along the y axis.

    Returns
    -------
    rho, u, v, p : ndarrays of shape (ny, nx)
    X : meshgrid of x for downstream use
    """
    rho_1d = np.ones_like(x)
    u_1d = np.zeros_like(x)
    p_1d = np.ones_like(x)

    if test_case in (1, 2, 4):
        rho_right, p_right = 1.0, 1.0
        sound_right = np.sqrt(gamma * p_right / rho_right)
        u_right = -mach * sound_right
        rho_left = rho_right * (gamma + 1.0) * mach**2 / ((gamma - 1.0) * mach**2 + 2.0)
        p_left = p_right * (2.0 * gamma * mach**2 - (gamma - 1.0)) / (gamma + 1.0)
        u_left = rho_right * u_right / rho_left

        if test_case in (2, 4):
            shift = abs(u_right)
            u_left += shift
            u_right += shift

        left = x < 0.0
        rho_1d[left], u_1d[left], p_1d[left] = rho_left, u_left, p_left
        rho_1d[~left], u_1d[~left], p_1d[~left] = rho_right, u_right, p_right

    elif test_case == 3:  # Sod shock tube
        left = x < 0.0
        rho_1d[left], u_1d[left], p_1d[left] = 1.0, 0.0, 1.0
        rho_1d[~left], u_1d[~left], p_1d[~left] = 0.125, 0.0, 0.1

    elif test_case == 5:  # Explosion between reflecting walls
        hot = (x > 0.0) & (x < 0.2)
        rho_1d[hot], u_1d[hot], p_1d[hot] = 1.0, 0.0, 10.0

    elif test_case == 6:  # Shu-Osher shock/entropy-wave interaction
        left = x < 0.125
        rho_1d[left], u_1d[left], p_1d[left] = 3.857143, 2.629369, 10.3333
        rho_1d[~left] = 1.0 + 0.2 * np.sin(16.0 * np.pi * x[~left])
        u_1d[~left], p_1d[~left] = 0.0, 1.0

    else:
        raise ValueError("test_case must be an integer from 1 to 6")

    # Replicate along the y axis to build the 2D field.
    rho_1d, u_1d, v_1d, p_1d = [np.broadcast_to(arr, (ny, nx)) for arr, nx in
                                  ((rho_1d, x.size), (u_1d, x.size),
                                   (np.zeros_like(x), x.size), (p_1d, x.size))]
    X, _ = np.meshgrid(x, np.zeros(ny), indexing="xy")
    return rho_1d, u_1d, v_1d, p_1d, X


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
def _add_ghost(U_phys: np.ndarray) -> np.ndarray:
    """Wrap a (4, ny, nx) physical-domain array into a (4, ny+2, nx+2) ghosted array."""
    _, ny, nx = U_phys.shape
    out = np.zeros((4, ny + 2, nx + 2), dtype=U_phys.dtype)
    out[:, 1:-1, 1:-1] = U_phys
    return out


def _strip_ghost(U_ghost: np.ndarray) -> np.ndarray:
    """Inverse of _add_ghost."""
    return U_ghost[:, 1:-1, 1:-1].copy()


def apply_ghost_bc(U: np.ndarray, test_case: int) -> np.ndarray:
    kind = boundary_kind(test_case)
    # x = -0.5 wall (column 0)
    if kind == "all":
        U[:, 1:-1, 0] = U[:, 1:-1, 1]
        U[1, 1:-1, 0] = -U[1, 1:-1, 1]
    else:
        # "right" (test 1, 4) and "both" (test 5) keep a zero-gradient
        # ghost on the left so the upstream gas is undisturbed.
        U[:, 1:-1, 0] = U[:, 1:-1, 1]
    # x = +0.5 wall (column -1)
    if kind in ("right", "both", "all"):
        U[:, 1:-1, -1] = U[:, 1:-1, -2]
        U[1, 1:-1, -1] = -U[1, 1:-1, -2]
    else:
        U[:, 1:-1, -1] = U[:, 1:-1, -2]
    # y walls: in 2D we always copy along y because the y direction carries no
    U[:, 0, :] = U[:, 1, :]
    U[:, -1, :] = U[:, -2, :]
    # Corners: copy the nearest physical cell.
    U[:, 0, 0] = U[:, 1, 1]
    U[:, 0, -1] = U[:, 1, -2]
    U[:, -1, 0] = U[:, -2, 1]
    U[:, -1, -1] = U[:, -2, -2]
    return U


# ---------------------------------------------------------------------------
# Physical check
# ---------------------------------------------------------------------------
def is_physical(U: np.ndarray, gamma: float = GAMMA) -> bool:
    """Reject a state before a negative density/pressure corrupts a time step."""
    if not np.all(np.isfinite(U)):
        return False
    rho, _, _, p = cons2prim(U, gamma)
    return bool(np.all(rho > EPS) and np.all(p > EPS))


# ---------------------------------------------------------------------------
# Numerical schemes
# ---------------------------------------------------------------------------
def lax_friedrichs_step(U: np.ndarray, dx: float, dy: float, dt: float,
                        gamma: float = GAMMA, test_case: int = 3) -> np.ndarray:
    """2D Lax-Friedrichs on a ghosted array (4, ny+2, nx+2).

    The interior is U[:, 1:-1, 1:-1]; we update that block in place and the caller's
    apply_ghost_bc refreshes the ghost layer for the next step.

    Memory: the 4-point average, dFdx, dGdy and the RHS are accumulated into
    `out` so the function only ever allocates one extra (4, ny+2, nx+2) array
    beyond U itself.  This matters on 2D grids close to the numpy memory limit.
    """
    out = np.empty_like(U)
    interior = (slice(None), slice(1, -1), slice(1, -1))
    # 4-point average of U as the diffusive part
    out[interior] = 0.25 * (U[:, 1:-1, 2:] + U[:, 1:-1, :-2] +
                             U[:, 2:, 1:-1] + U[:, :-2, 1:-1])
    # x flux: build F only over the interior column stripe, then a single
    # in-place F -= F shifted by one cell before we read it out.
    F = flux_x(U, gamma)
    dFdx = (F[:, 1:-1, 2:] - F[:, 1:-1, :-2])
    np.subtract(out[interior], dt / (2.0 * dx) * dFdx, out=out[interior])
    del F, dFdx
    G = flux_y(U, gamma)
    dGdy = (G[:, 2:, 1:-1] - G[:, :-2, 1:-1])
    np.subtract(out[interior], dt / (2.0 * dy) * dGdy, out=out[interior])
    return out


def maccormack_step(U: np.ndarray, dx: float, dy: float, dt: float,
                    gamma: float = GAMMA, test_case: int = 3,
                    viscosity_b: float = 0.0) -> np.ndarray:
    """2D MacCormack predictor/corrector, equations (38)-(39) of the article.

    With viscosity_b > 0 the explicit smoothing term of equation (41) is added to the
    corrector, using the velocity-difference coefficient of equation (42):

        D_i = b * (dt/dx) * [ nu_{i+1/2} (U_{i+1} - U_i) - nu_{i-1/2} (U_i - U_{i-1}) ],
        nu_{i+1/2} = |u_{i+1} - u_i|.

    Because nu carries a velocity and dt/dx its inverse, b is a dimensionless constant
    of order unity.  In 2D the same rule is applied to v along y.

    Memory: the predictor/corrector buffers are written into a single `scratch`
    ndarray of the same shape as U, so we never hold three full copies of U
    simultaneously (matters on 2D grids near the numpy memory limit).
    """
    interior = (slice(None), slice(1, -1), slice(1, -1))
    lam_x = dt / dx
    lam_y = dt / dy
    F = flux_x(U, gamma)
    G = flux_y(U, gamma)

    # Forward predictor (both directions), written into scratch[:, 1:-1, 1:-1].
    scratch = np.empty_like(U)
    scratch[interior] = (
        U[interior]
        - lam_x * (F[:, 1:-1, 2:] - F[:, 1:-1, 1:-1])
        - lam_y * (G[:, 2:, 1:-1] - G[:, 1:-1, 1:-1])
    )
    apply_ghost_bc(scratch, test_case)

    Fp = flux_x(scratch, gamma)
    Gp = flux_y(scratch, gamma)

    # Backward corrector (both directions), also into scratch.
    scratch[interior] = 0.5 * (
        U[interior] + scratch[interior]
        - lam_x * (Fp[:, 1:-1, 1:-1] - Fp[:, 1:-1, :-2])
        - lam_y * (Gp[:, 1:-1, 1:-1] - Gp[:, :-2, 1:-1])
    )

    if viscosity_b > 0.0:
        # Work in the physical domain to avoid spurious jumps at reflecting walls.
        U_phys = U[:, 1:-1, 1:-1]
        _, u_phys, v_phys, _ = cons2prim(U_phys, gamma)
        # x: |u_{i+1} - u_i| at i+1/2
        nu_x = np.abs(np.diff(u_phys, axis=1))             # (ny, nx-1)
        # y: |v_{j+1} - v_j| at j+1/2
        nu_y = np.abs(np.diff(v_phys, axis=0))             # (ny-1, nx)
        jump_x = U_phys[:, :, 1:] - U_phys[:, :, :-1]     # (4, ny, nx-1)
        jump_y = U_phys[:, 1:, :] - U_phys[:, :-1, :]     # (4, ny-1, nx)
        weighted_x = nu_x[None, :, :] * jump_x              # (4, ny, nx-1)
        weighted_y = nu_y[None, :, :] * jump_y              # (4, ny-1, nx)
        visc_phys = np.zeros_like(U_phys)
        visc_phys[:, :, 1:-1] += viscosity_b * lam_x * (weighted_x[:, :, 1:] - weighted_x[:, :, :-1])
        visc_phys[:, 1:-1, :] += viscosity_b * lam_y * (weighted_y[:, 1:, :] - weighted_y[:, :-1, :])
        scratch[interior] = scratch[interior] + visc_phys

    return scratch


def _limited_differences_2d_ghosted(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return limited interface differences in x and y on a ghosted array.

    field shape: (4, ny+2, nx+2).  The physical domain is [:, 1:-1, 1:-1].
    We compute the limited difference at every physical interface; the limit
    selector of equation (46) is applied on the three-cell stencil of dX and dY.
    """
    dX = field[:, 1:-1, 1:] - field[:, 1:-1, :-1]   # (4, ny, nx+1)
    limX = np.zeros_like(dX)
    if dX.shape[2] >= 3:
        prev = dX[:, :, :-2]
        cur = dX[:, :, 1:-1]
        nxt = dX[:, :, 2:]
        same = (np.sign(prev) == np.sign(cur)) & (np.sign(cur) == np.sign(nxt))
        mag = np.minimum(np.minimum(8.0 * np.abs(prev), np.abs(cur)),
                         8.0 * np.abs(nxt))
        limX[:, :, 1:-1] = np.where(same, mag * np.sign(cur), 0.0)

    dY = field[:, 1:, 1:-1] - field[:, :-1, 1:-1]   # (4, ny+1, nx)
    limY = np.zeros_like(dY)
    if dY.shape[1] >= 3:
        prev = dY[:, :-2, :]
        cur = dY[:, 1:-1, :]
        nxt = dY[:, 2:, :]
        same = (np.sign(prev) == np.sign(cur)) & (np.sign(cur) == np.sign(nxt))
        mag = np.minimum(np.minimum(8.0 * np.abs(prev), np.abs(cur)),
                         8.0 * np.abs(nxt))
        limY[:, 1:-1, :] = np.where(same, mag * np.sign(cur), 0.0)

    return limX, limY


def shasta_step(U: np.ndarray, dx: float, dy: float, dt: float,
                gamma: float = GAMMA, test_case: int = 3,
                use_limiter: bool = True) -> np.ndarray:
    """2D SHASTA / FCT: MacCormack transport + strong diffusion + limited anti-diffusion.

    The diffusion factor is the textbook 0.5*lambda*(1-lambda), clamped from below at
    1/8 to keep the limiter well-conditioned when the time step is small.
    The anti-diffusion uses the same factor in both x and y; this is the simplest
    2D extension of the 1D SHASTA scheme (Boris & Book, 1976, sec. 2).
    """
    transported = maccormack_step(U, dx, dy, dt, gamma, test_case, viscosity_b=0.0)
    transported_phys = _strip_ghost(transported)

    interior_phys = (slice(None), slice(1, -1), slice(1, -1))
    U_phys = U[:, 1:-1, 1:-1]
    lap_x = U_phys[:, :, 2:] - 2.0 * U_phys[:, :, 1:-1] + U_phys[:, :, :-2]
    lap_y = U_phys[:, 2:, :] - 2.0 * U_phys[:, 1:-1, :] + U_phys[:, :-2, :]
    nu_diff = max(0.5 * (dt / dx) * (1.0 - dt / dx), 0.125)
    diffused = transported_phys.copy()
    diffused[interior_phys] = (transported_phys[interior_phys]
                               + nu_diff * lap_x[:, 1:-1, :]
                               + nu_diff * lap_y[:, :, 1:-1])
    diffused_ghost = _add_ghost(diffused)
    apply_ghost_bc(diffused_ghost, test_case)
    diffused = _strip_ghost(diffused_ghost)

    if use_limiter:
        limX, limY = _limited_differences_2d_ghosted(diffused_ghost)
    else:
        limX = diffused_ghost[:, 1:-1, 1:] - diffused_ghost[:, 1:-1, :-1]
        limY = diffused_ghost[:, 1:, 1:-1] - diffused_ghost[:, :-1, 1:-1]

    out = diffused.copy()
    out[interior_phys] = (
        diffused[interior_phys]
        - (limX[:, :, 1:] - limX[:, :, :-1])[interior_phys] * nu_diff
        - (limY[:, 1:, :] - limY[:, :-1, :])[interior_phys] * nu_diff
    )
    return _add_ghost(out)


# ---------------------------------------------------------------------------
# Step dispatch
# ---------------------------------------------------------------------------
METHODS: tuple[str, ...] = (
    "lax_friedrichs",
    "maccormack",
    "maccormack_viscous",
    "shasta",
)


def step_for(method: str, U: np.ndarray, dx: float, dy: float, dt: float,
             gamma: float, config: SimulationConfig) -> np.ndarray:
    """Advance one time step with the selected scheme."""
    test_case = config.test_case
    if method == "lax_friedrichs":
        return lax_friedrichs_step(U, dx, dy, dt, gamma, test_case)
    if method == "maccormack":
        return maccormack_step(U, dx, dy, dt, gamma, test_case, viscosity_b=0.0)
    if method == "maccormack_viscous":
        return maccormack_step(U, dx, dy, dt, gamma, test_case, viscosity_b=config.viscosity_b)
    if method == "shasta":
        return shasta_step(U, dx, dy, dt, gamma, test_case, use_limiter=config.shasta_limiter)
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
def build_overview_figure(output_dir: Path, show: bool = False) -> Path:
    """Re-run each of the 6 tests with Lax-Friedrichs (the most stable scheme) at
    the default parameters and stitch the (rho, u, p) x-profiles into a 3x2 grid.

    The result is written to ``output_dir/all_tests_overview.png`` and the path
    is returned.  Each subplot is annotated with the per-test (nx, ny, cfl, t_end, mach)
    so the picture stands on its own when included in a report.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    test_titles = {
        1: "Test 1: stationary shock",
        2: "Test 2: moving shock",
        3: "Test 3: Sod shock tube",
        4: "Test 4: shock reflection from a wall",
        5: "Test 5: explosion between two walls",
        6: "Test 6: Shu-Osher entropy waves",
    }
    for slot, test_case in enumerate(range(1, 7)):
        nx, ny, cfl, t_end, mach = default_runs(test_case)[0]
        config = SimulationConfig(
            test_case=test_case, nx=nx, ny=ny, cfl=cfl, t_end=t_end, mach=mach
        )
        result = advance(config, "lax_friedrichs")
        rho, u, _, p = cons2prim(result.U, GAMMA)
        rho_x = np.mean(rho, axis=0)
        u_x = np.mean(u, axis=0)
        p_x = np.mean(p, axis=0)
        row, col = divmod(slot, 2)
        ax = axes[row][col]
        ax.plot(result.x, rho_x, color="C0", label=r"$\rho$")
        ax.plot(result.x, u_x, color="C1", label=r"$u$")
        ax.plot(result.x, p_x, color="C2", label=r"$p$")
        ax.set_title(test_titles[test_case])
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        stable_tag = "stable" if result.stable else f"unstable @ step {result.steps}"
        ax.text(
            0.02, 0.98,
            f"nx={nx} ny={ny} cfl={cfl} t={t_end:.4g} mach={mach:g}\nLax-Friedrichs, {stable_tag}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    fig.suptitle("Euler2D — 6 tests, default cfl=0.4, aligned with 1D README", fontsize=14)
    out_path = output_dir / "all_tests_overview.png"
    fig.savefig(out_path, dpi=140)
    if show and not matplotlib.get_backend().lower().startswith("agg"):
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)
    return out_path


def advance(config: SimulationConfig, method: str) -> SimulationResult:
    """Integrate one test problem and return a result instead of crashing on failure."""
    x = domain_for_test(config.test_case, config.nx)
    y = np.linspace(-0.5, 0.5, config.ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    rho_init, u_init, v_init, p_init, _ = init_test(
        config.test_case, x, config.ny, GAMMA, config.mach
    )
    U_phys = prim2cons(rho_init, u_init, v_init, p_init, GAMMA)
    U = _add_ghost(U_phys)
    apply_ghost_bc(U, config.test_case)

    end_time = config.t_end if config.t_end is not None else default_end_time(config.test_case, config.mach)
    time = 0.0
    steps = 0

    while time < end_time - 1.0e-14:
        U_phys_now = _strip_ghost(U)
        if not is_physical(U_phys_now, GAMMA):
            return SimulationResult(method, x, y, U_phys_now, rho_init, time, steps,
                                    False, "negative density or pressure")
        rho, u, v, p = cons2prim(U_phys_now, GAMMA)
        # 2D CFL: dt <= cfl / (|u|/dx + |v|/dy + a*(1/dx+1/dy))
        signal = np.abs(u) / dx + np.abs(v) / dy + np.sqrt(np.maximum(GAMMA * p / rho, EPS)) * (1.0 / dx + 1.0 / dy)
        dt = min(config.cfl / float(np.max(signal)), end_time - time)
        U_next = step_for(method, U, dx, dy, dt, GAMMA, config)
        apply_ghost_bc(U_next, config.test_case)
        U_phys_next = _strip_ghost(U_next)
        if not is_physical(U_phys_next, GAMMA):
            return SimulationResult(method, x, y, U_phys_next, rho_init, time + dt, steps + 1,
                                    False, "negative density or pressure")
        U = U_next
        time += dt
        steps += 1

    return SimulationResult(method, x, y, _strip_ghost(U), rho_init, time, steps, True)


# ---------------------------------------------------------------------------
# Quality scoring and physical diagnostics
# ---------------------------------------------------------------------------
def quality_metrics(result: SimulationResult) -> dict[str, float]:
    """Resolution/oscillation indicators without pretending to have an exact solution."""
    rho, _, _, p = cons2prim(result.U, GAMMA)
    if not result.stable:
        return {"rho_min": float("nan"), "p_min": float("nan"),
                "density_tv": float("nan"), "oscillation": float("inf"),
                "score": float("inf")}
    initial = result.initial_rho
    total_variation = float(np.sum(np.abs(np.diff(rho, axis=1))))
    initial_tv = float(np.sum(np.abs(np.diff(initial, axis=1))))
    oscillation = max(0.0, total_variation - initial_tv)
    max_gradient = float(np.max(np.abs(np.diff(rho, axis=1))))
    score = 10.0 * oscillation + 0.1 / max(max_gradient, EPS)
    return {
        "rho_min": float(np.min(rho)),
        "p_min": float(np.min(p)),
        "density_tv": total_variation,
        "oscillation": oscillation,
        "score": score,
    }


def locate_shock(x: np.ndarray, rho: np.ndarray) -> float:
    """Locate a shock by the steepest density jump along x (averaged over y)."""
    rho_x_avg = np.mean(rho, axis=0)
    index = int(np.argmax(np.abs(np.diff(rho_x_avg))))
    return float(0.5 * (x[index] + x[index + 1]))


def physical_diagnostics(result: SimulationResult, config: SimulationConfig) -> dict[str, float | str]:
    """Per-test physical checks, identical in spirit to the 1D diagnostics.

    The y direction is uniform for all six tests, so the 1D-style x-direction checks
    can be reused by averaging rho/u/p along the y axis.
    """
    if not result.stable:
        return {}
    rho, u, _, p = cons2prim(result.U, GAMMA)
    rho_x = np.mean(rho, axis=0)
    u_x = np.mean(u, axis=0)
    p_x = np.mean(p, axis=0)
    x = result.x

    relations = shock_relations(config.mach)

    if config.test_case == 1:
        mask = x < -0.1
        rho_left = float(np.mean(rho_x[mask]))
        u_left = float(np.mean(u_x[mask]))
        p_left = float(np.mean(p_x[mask]))
        a_left = float(np.sqrt(GAMMA * p_left / rho_left))
        mach_left = abs(u_left) / a_left
        return {
            "M": float(config.mach),
            "a_L": a_left,
            "M_L": mach_left,
            "M_L_theory": relations["mach_downstream"],
            "M_L<1": str(bool(mach_left < 1.0)),
        }

    if config.test_case == 2:
        c_theory = relations["incident_speed"]
        shock_position = locate_shock(x, rho)
        c_numeric = shock_position / result.time if result.time > 0.0 else float("nan")
        return {
            "C_theory": c_theory,
            "C_numeric": c_numeric,
            "C_match": str(bool(np.isclose(c_numeric, c_theory, rtol=0.1))),
        }

    if config.test_case == 4:
        p_1 = float(p_x[0])
        p_wall = float(p_x[-1])
        dp_1 = p_1 - 1.0
        dp_wall = p_wall - 1.0
        dp_theory = (relations["p5"] - 1.0) / (relations["p2"] - 1.0)
        return {
            "P1": p_1,
            "DP1": dp_1,
            "P_w": p_wall,
            "DP_w": dp_wall,
            "DP": dp_wall / dp_1 if abs(dp_1) > EPS else float("nan"),
            "DP_theory": dp_theory,
            "P_w_theory": relations["p5"],
            "rho_w": float(rho_x[-1]),
            "rho_w_theory": relations["rho5"],
            "u_w": float(u_x[-1]),
        }

    return {}


def print_diagnostics_table(test_case: int,
                            per_method: list[tuple[str, dict[str, float | str]]]) -> None:
    """Print one comparison table instead of repeating the block for every scheme."""
    entries = [(name, values) for name, values in per_method if values]
    if not entries:
        return
    first = entries[0][1]
    print()

    if test_case == 1:
        print("===== Test 1: post-shock Mach number =====")
        print(f"M = {first['M']},  M_L (theory) = {first['M_L_theory']:.4f}")
        print(f"{'Scheme':<22}{'a_L':>9}{'M_L':>9}{'M_L < 1':>10}")
        for name, values in entries:
            print(f"{name:<22}{values['a_L']:>9.4f}{values['M_L']:>9.4f}{values['M_L<1']:>10}")

    elif test_case == 2:
        print("===== Test 2: shock speed =====")
        print(f"Theory: C = M*aR = {first['C_theory']:.4f}")
        print(f"{'Scheme':<22}{'C (num.)':>12}{'Match':>13}")
        for name, values in entries:
            print(f"{name:<22}{values['C_numeric']:>12.4f}{values['C_match']:>13}")

    elif test_case == 4:
        print("===== Test 4: wall reflection =====")
        print(
            f"Theory: P_w = {first['P_w_theory']:.4f},  DP = {first['DP_theory']:.4f},  "
            f"rho_w = {first['rho_w_theory']:.4f}"
        )
        print(f"{'Scheme':<22}{'P1':>9}{'P_w':>10}{'DP':>9}{'rho_w':>9}{'u_w':>10}")
        for name, values in entries:
            print(
                f"{name:<22}{values['P1']:>9.4f}{values['P_w']:>10.4f}{values['DP']:>9.4f}"
                f"{values['rho_w']:>9.4f}{values['u_w']:>10.4f}"
            )


def print_diagnostics_single(test_case: int, diagnostics: dict[str, float | str]) -> None:
    """Print the per-test diagnostics for a single scheme."""
    if not diagnostics:
        return
    print()

    if test_case == 1:
        print("===== Test 1: post-shock Mach number =====")
        print(f"M = {diagnostics['M']}")
        print(f"a_L = {diagnostics['a_L']:.4f}")
        print(f"M_L = {diagnostics['M_L']:.4f}")
        print(f"M_L (theory) = {diagnostics['M_L_theory']:.4f}")
        print(f"M_L < 1 ? -> {diagnostics['M_L<1']}")

    elif test_case == 2:
        print("===== Test 2: shock speed =====")
        print(f"Theory: C = M*aR = {diagnostics['C_theory']:.4f}")
        print(f"Numeric: C = {diagnostics['C_numeric']:.4f}")
        print(f"Match: {diagnostics['C_match']}")

    elif test_case == 4:
        print("===== Test 4: wall reflection =====")
        print("Wall state (P_w, rho_w, u_w):")
        print(f"DP1 = {diagnostics['DP1']:.4f}")
        print(f"P1 = {diagnostics['P1']:.4f}")
        print(f"P_w = {diagnostics['P_w']:.4f}")
        print(f"DP_w = {diagnostics['DP_w']:.4f}")
        print(f"DP = {diagnostics['DP']:.4f}")
        print(f"DP (theory) = {diagnostics['DP_theory']:.4f}")
        print(f"P_w (theory) = {diagnostics['P_w_theory']:.4f}")
        print(f"rho_w = {diagnostics['rho_w']:.4f}")
        print(f"rho_w (theory) = {diagnostics['rho_w_theory']:.4f}")
        print(f"u_w = {diagnostics['u_w']:.4f} (should be ~0)")


def scan_viscosity(config: SimulationConfig, values: list[float]) -> tuple[float, list[dict[str, float | str]]]:
    """Evaluate b values on the selected test and choose a reproducible best b."""
    rows: list[dict[str, float | str]] = []
    for b in values:
        run_config = SimulationConfig(**{**config.__dict__, "viscosity_b": b})
        result = advance(run_config, "maccormack_viscous")
        metrics = quality_metrics(result)
        rows.append({"b": b, "stable": str(result.stable), **metrics})
    stable_rows = [row for row in rows if row["stable"] == "True"]
    if not stable_rows:
        raise RuntimeError("No tested viscosity coefficient produced a physical solution")
    best = min(stable_rows, key=lambda row: float(row["score"]))
    return float(best["b"]), rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_results_2d(results: list[SimulationResult], output: Path, test_case: int,
                    show: bool = True) -> None:
    """One figure per end time, 3 columns (density, velocity, pressure), 1 row per scheme.

    The 2D field is averaged along the y axis before plotting so the picture is the same
    1D diagnostic curve as the original 1D code, with a secondary row showing the 2D
    density as a filled-contour sanity check.
    """
    valid = [r for r in results if r.stable]
    if not valid:
        return
    single = len(valid) == 1
    titles = ("Density", "Velocity", "Pressure")
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), constrained_layout=True)
    for result in valid:
        rho, u, v, p = cons2prim(result.U, GAMMA)
        rho_x = np.mean(rho, axis=0)
        u_x = np.mean(u, axis=0)
        p_x = np.mean(p, axis=0)
        for column, data in enumerate((rho_x, u_x, p_x)):
            axis = axes[0][column]
            if single:
                axis.plot(result.x, data, "o", markersize=3, color="C0")
            else:
                axis.plot(result.x, data, linewidth=1.2, label=result.method)
        # 2D density contour for the first method, or the only one if single.
        cf = axes[1][0].contourf(result.x, result.y, rho, levels=30, cmap="viridis")
        axes[1][0].set_title(f"Density 2D — {result.method}")
        axes[1][0].set_aspect("equal")
        fig.colorbar(cf, ax=axes[1][0], shrink=0.8, label=r"$\rho$")
        # y-direction profiles.  For the 2D tests in this project every
        # initial state is uniform in y, so a y-profile is uninformative;
        # we show |u_y| deviation from the y mean instead of a flat line.
        u_y = np.mean(v, axis=1) - np.mean(v)
        axes[1][1].plot(result.y, u_y, "o", markersize=3, color="C1",
                        label=result.method)
        axes[1][1].set_title(r"$v_y$ anomaly along y")
        axes[1][1].set_xlabel("y")
        p_y = np.mean(p, axis=1) - np.mean(p)
        axes[1][2].plot(result.y, p_y, "o", markersize=3, color="C2",
                        label=result.method)
        axes[1][2].set_title("Pressure anomaly along y")
        axes[1][2].set_xlabel("y")
    for column, title in enumerate(titles):
        axes[0][column].set_title(title)
        axes[0][column].set_xlabel("x")
        if not single:
            axes[0][column].legend(loc="best", fontsize=7)
    fig.suptitle(f"Test {test_case}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_time_grid(results_by_time: list[tuple[float, list[SimulationResult]]],
                   output: Path, test_case: int,
                   sweep_labels: list[str] | None = None,
                   show: bool = True) -> None:
    """One block per end time, each block contains the 1D x-curve and a 2D density contour."""
    rows = [(end_time, [r for r in results if r.stable]) for end_time, results in results_by_time]
    rows = [(end_time, results) for end_time, results in rows if results]
    if not rows:
        return

    single = all(len(results) == 1 for _, results in rows)
    titles = ("Density", "Velocity", "Pressure")
    fig, axes = plt.subplots(2 * len(rows), 3, figsize=(15, 4 * len(rows)),
                             squeeze=False, constrained_layout=True)
    for row_index, (end_time, results) in enumerate(rows):
        sweep_label = sweep_labels[row_index] if sweep_labels else f"t_end = {end_time:g}"
        for result in results:
            rho, u, v, p = cons2prim(result.U, GAMMA)
            rho_x = np.mean(rho, axis=0)
            u_x = np.mean(u, axis=0)
            p_x = np.mean(p, axis=0)
            for column, data in enumerate((rho_x, u_x, p_x)):
                axis = axes[2 * row_index][column]
                if single:
                    axis.plot(result.x, data, "o", markersize=3)
                else:
                    axis.plot(result.x, data, linewidth=1.2, label=result.method)
            cf = axes[2 * row_index + 1][0].contourf(
                result.x, result.y, rho, levels=30, cmap="viridis"
            )
            axes[2 * row_index + 1][0].set_title(f"2D density  {sweep_label}  —  {result.method}")
            axes[2 * row_index + 1][0].set_aspect("equal")
            fig.colorbar(cf, ax=axes[2 * row_index + 1][0], shrink=0.8, label=r"$\rho$")
            # Hide the other two cells in the second row when only one scheme is plotted.
            for column in (1, 2):
                axes[2 * row_index + 1][column].axis("off")
        for column, title in enumerate(titles):
            axes[2 * row_index][column].set_title(f"{title},  {sweep_label}")
            axes[2 * row_index][column].set_xlabel("x")
            if not single:
                axes[2 * row_index][column].legend(loc="best", fontsize=7)
    fig.suptitle(f"Test {test_case}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D Euler: Lax-Friedrichs, MacCormack and SHASTA/FCT")
    parser.add_argument(
        "--test", type=int, default=None, choices=range(1, 7),
        help="Original coursework test number (1-6). Prompts interactively when omitted.",
    )
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument(
        "--nx", type=int, default=None,
        help="Number of x grid points; defaults to 300 (200 for test 5, 300 for test 6)",
    )
    parser.add_argument(
        "--ny", type=int, default=None,
        help="Number of y grid points; defaults to max(50, nx//4), or max(20, nx//10) for test 6",
    )
    parser.add_argument(
        "--cfl", type=float, default=None,
        help="CFL number; defaults to 0.4 for every test (2D MacCormack transport needs it)",
    )
    parser.add_argument(
        "--t-end", type=float, nargs="+", default=None,
        help="One or more final times; several values produce one block per time",
    )
    parser.add_argument("--mach", type=float, default=None,
                        help="Mach number for tests 1, 2 and 4 (default: per-test sweep)")
    parser.add_argument("--b", type=float, default=1.0,
                        help="Artificial-viscosity coefficient b of equation (41)")
    parser.add_argument(
        "--scan-viscosity", action="store_true",
        help="Try b = 0, 0.2, 0.5, 1.0, 1.5, 2.0 and select the best physical case",
    )
    parser.add_argument(
        "--no-limiter", action="store_true",
        help="Switch the SHASTA limiter off, reproducing the unlimited case shown in the article",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the plot but do not open the GUI window (useful for batch runs)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Force the interactive TkAgg backend and pop up each figure",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--summary", action="store_true",
        help="After the requested run, regenerate outputs/all_tests_overview.png "
             "from the per-test CSVs already in --output-dir",
    )
    return parser.parse_args()


def prompt_for_test_case() -> int:
    while True:
        raw_value = input("Please enter the test number (1-6, press Enter for default 3): ").strip()
        if not raw_value:
            return 3
        try:
            test_case = int(raw_value)
        except ValueError:
            print("Invalid input: please enter an integer between 1 and 6.")
            continue
        if test_case in range(1, 7):
            return test_case
        print("Invalid input: the test number must be between 1 and 6.")


def main() -> None:
    args = parse_args()
    if args.show:
        # Try TkAgg first; fall back to QtAgg if PyQt is available.  Both are
        # interactive and let the user pan/zoom the saved figure before
        # closing the window.  The default Agg backend cannot show() so it is
        # not an option here.
        for candidate in ("TkAgg", "Qt5Agg"):
            try:
                matplotlib.use(candidate, force=True)
                break
            except Exception:
                continue
    test_case = args.test if args.test is not None else prompt_for_test_case()

    # If the user passes any of --nx / --ny / --cfl / --t-end / --mach on the
    # command line, we run a single simulation with those values.  Otherwise we
    # honour the 1D presentation's per-test parameter sweep so that changing
    # --test alone is enough to switch the picture (e.g. test 1 sweeps M, test
    # 3 sweeps nx, test 5 sweeps t_end).
    user_picked_params = any(
        getattr(args, key) is not None
        for key in ("nx", "ny", "cfl", "t_end", "mach")
    )

    if user_picked_params:
        nx = args.nx if args.nx is not None else default_nx(test_case)
        ny = args.ny if args.ny is not None else max(50, nx // 4)
        cfl = args.cfl if args.cfl is not None else default_cfl(test_case)
        mach = args.mach if args.mach is not None else 2.0
        t_end = args.t_end[0] if args.t_end else None
        if t_end is None or t_end <= 0.0:
            t_end = default_end_time(test_case, mach)
        run_specs: list[tuple[int, int, float, float, float]] = [(nx, ny, cfl, t_end, mach)]
    else:
        # Fill in the default end time for tests that did not specify one in
        # default_runs (test 2 measures the shock speed itself, test 4 reflects
        # off a wall).
        run_specs = []
        for nx, ny, cfl, t_end, mach in default_runs(test_case):
            if t_end <= 0.0:
                t_end = default_end_time(test_case, mach)
            run_specs.append((nx, ny, cfl, t_end, mach))

    methods = list(METHODS) if args.method == "all" else [args.method]

    rows: list[dict[str, float | str]] = []
    results_by_time: list[tuple[float, list[SimulationResult]]] = []
    sweep_labels: list[str] = []

    for spec_index, (nx, ny, cfl, t_end, mach) in enumerate(run_specs):
        if not 0.0 < cfl <= 1.0:
            raise ValueError("Use a CFL number in (0, 1.0]")
        if ny < 4:
            raise ValueError("ny must be at least 4")
        if cfl > 1.0 / np.sqrt(3.0) and any("shasta" in name for name in methods):
            print(
                f"Warning: CFL = {cfl:.2f} exceeds 1/sqrt(3) = 0.577; "
                "the FCT/SHASTA schemes may not stay monotone."
            )

        run_config = SimulationConfig(test_case, nx, ny, cfl, t_end, mach, args.b, not args.no_limiter)

        if len(run_specs) > 1:
            label = f"nx={nx} mach={mach:g} t_end={t_end:g}"
            sweep_labels.append(label)
            print(f"\n--- run {spec_index + 1}/{len(run_specs)}: {label} ---")
        else:
            print(f"\n--- test {test_case}: nx={nx} ny={ny} cfl={cfl} t_end={t_end} mach={mach} ---")

        if args.scan_viscosity and spec_index == 0:
            best_b, scan_rows = scan_viscosity(run_config, [0.0, 0.2, 0.5, 1.0, 1.5, 2.0])
            scan_path = args.output_dir / f"test{test_case}_maccormack_viscosity_scan.csv"
            write_csv(scan_path, scan_rows)
            print(f"Best b for test {test_case}: {best_b:.3f} (criterion: low oscillation + sharp front)")
            print(f"Scan saved to: {scan_path}")
            run_config = SimulationConfig(**{**run_config.__dict__, "viscosity_b": best_b})

        results: list[SimulationResult] = []
        diagnostics_per_method: list[tuple[str, dict[str, float | str]]] = []
        for method in methods:
            result = advance(run_config, method)
            results.append(result)
            metrics = quality_metrics(result)
            row: dict[str, float | str] = {
                "method": method,
                "run": spec_index,
                "nx": nx,
                "mach": mach,
                "t_end": result.time,
                "stable": str(result.stable),
                "steps": result.steps,
                "message": result.message,
                **metrics,
            }
            diagnostics = physical_diagnostics(result, run_config)
            diagnostics_per_method.append((method, diagnostics))
            row.update(diagnostics)
            rows.append(row)
            print(
                f"  {method:22s} stable={result.stable!s:5s} steps={result.steps:5d} "
                f"rho_min={metrics['rho_min']:.4e} p_min={metrics['p_min']:.4e} score={metrics['score']:.4e}"
            )

        if len(diagnostics_per_method) == 1:
            print_diagnostics_single(test_case, diagnostics_per_method[0][1])
        else:
            print_diagnostics_table(test_case, diagnostics_per_method)

        actual_time = max((r.time for r in results), default=0.0)
        results_by_time.append((actual_time, results))

    summary_path = args.output_dir / f"test{test_case}_comparison.csv"
    write_csv(summary_path, rows)
    print(f"\nComparison saved to: {summary_path}")

    plot_path = args.output_dir / f"test{test_case}_comparison.png"
    if len(results_by_time) > 1:
        plot_time_grid(results_by_time, plot_path, test_case, sweep_labels, show=not args.no_show)
    else:
        plot_results_2d(results_by_time[0][1], plot_path, test_case, show=not args.no_show)
    print(f"Plot saved to: {plot_path}")
    if args.summary:
        overview_path = build_overview_figure(args.output_dir, show=not args.no_show)
        print(f"Overview saved to: {overview_path}")
    if not args.no_show and not matplotlib.get_backend().lower().startswith("agg"):
        try:
            plt.show()
        except Exception as exc:
            print(f"(Interactive show unavailable: {exc}; PNG already saved.)")


if __name__ == "__main__":
    main()
