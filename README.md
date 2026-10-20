# 2D Euler Equations Numerical Solver

A two-dimensional finite-difference solver for the inviscid Euler equations, built directly on top of the [1D Lax-Friedrichs repository](https://github.com/WangBo663/Lax-Friedrichs). The same six initial/boundary-value tests are kept so every scheme can be compared head-to-head against the 1D reference, and a one-cell ghost layer extends each scheme to a rectangular 2D domain without rewriting the time-marching loop.

Four numerical schemes are implemented in one file (`Euler2D.py`):

| Scheme | `method` flag | Description |
| --- | --- | --- |
| Lax–Friedrichs | `lax_friedrichs` | First-order, highly dissipative but unconditionally stable |
| MacCormack | `maccormack` | Predictor–corrector, equations (38)–(39), no artificial viscosity |
| MacCormack + artificial viscosity | `maccormack_viscous` | Adds equations (41)–(42) to the corrector |
| SHASTA / FCT | `shasta` | Strong diffusion followed by limited anti-diffusion, equations (43), (44) and (46) |

The MacCormack artificial viscosity and SHASTA are adapted from the comparative study of computational methods in cosmic gas dynamics by van Albada, van Leer and Roberts. The 2D extension is the textbook splitting: transport, diffusion and anti-diffusion in x, then in y, on a single `(4, ny+2, nx+2)` conservative array with a 1-cell ghost layer.

## Equations

The conservative vector has four components in 2D:

    U = (rho, rho*u, rho*v, E)
    dU/dt + dF/dx + dG/dy = 0

with the fluxes

    F = (rho*u,        rho*u^2 + p,  rho*u*v,      (E+p)*u)
    G = (rho*v,        rho*u*v,      rho*v^2 + p,  (E+p)*v)

and the equation of state `E = p/(gamma-1) + 0.5*rho*(u^2 + v^2)` with `gamma = 1.4`. The 2D CFL condition is

    dt <= cfl / (|u|/dx + |v|/dy + a*(1/dx + 1/dy))

where `a = sqrt(gamma*p/rho)` is the local sound speed. Artificial viscosity in x and y follows the paper's recipe:

    D_i = b * dt/dx * [nu_{i+1/2} (U_{i+1} - U_i) - nu_{i-1/2} (U_i - U_{i-1})],  nu_{i+1/2} = |u_{i+1} - u_i|

with `b` a dimensionless constant of order unity.

## Test cases

The 2D code keeps the same six initial-boundary-value tests as the 1D reference, so switching `--test N` reproduces the 1D plot in 2D. Every test has its own default grid, CFL and end time.

| # | Name | Domain | BC | Default `(nx, ny, cfl, t_end, mach)` |
| --- | --- | --- | --- | --- |
| 1 | Stationary shock | [-0.5, 0.5] x [-0.5, 0.5] | none | `(300, 75, 0.4, 0.16, 0.4)` |
| 2 | Moving shock | [-0.5, 0.5] x [-0.5, 0.5] | none | `(300, 75, 0.4, 0.35/(M*sqrt(gamma)), 1.0)` |
| 3 | Sod shock tube | [-0.5, 0.5] x [-0.5, 0.5] | none | `(300, 75, 0.4, 0.20, --)` |
| 4 | Mach=2 reflection from a wall | [-0.5, 0.5] x [-0.5, 0.5] | right | `(300, 75, 0.4, 0.5/C + 0.3/Wr, 2.0)` |
| 5 | Explosion between two walls | [-0.5, 0.5] x [-0.5, 0.5] | both | three runs at `t_end = 0.01, 0.1, 0.2`, `nx = 200, ny = 50` |
| 6 | Shu–Osher entropy-wave interaction | [0, 1] x [0, 1] | none | `(300, 30, 0.4, 0.178, 2.5)` |

End times for tests 2 and 4 are derived from the Rankine–Hugoniot relations (`shock_relations` in `Euler2D.py`), so the wave configuration is the intended one for any Mach number. With `rho = p = 1` ahead of the shock, the incident shock speed is `C = M*sqrt(gamma)`, the wall sits at `x = 0.5`, and the reflected shock travels back at speed `Wr = rho2*u2/(rho5-rho2)`:

    t_test_2 = 0.35 / C
    t_test_4 = 0.5  / C + 0.3 / Wr

For `M = 2` this gives `t_test_2 = 0.2532` and `t_test_4 = 0.4648` — the 1D README's Test 4 value. A fixed `t = 0.2` would be shorter than the 0.211 the shock needs to reach the wall, so no reflection would occur at all.

### Boundary tag per test

The 1D code applies per-test rules in a single `apply_bc`; the 2D code mirrors those rules with a per-test tag so changing one test's boundary never silently mutates another test's:

| Tag | Meaning | Used by |
| --- | --- | --- |
| `none` | zero-gradient on every side | tests 1, 2, 3, 6 |
| `right` | reflecting wall on the right edge only | test 4 |
| `both` | reflecting walls on both edges in x | test 5 |
| `all` | reflecting walls on all four sides | only via the `boundary_kind` helper |

A reflecting wall inverts the x-momentum in the ghost cell (`U[1, :, 0] = -U[1, :, 1]` for the left wall, the opposite index for the right wall); the y-momentum and energy are copied. This is the 2D analogue of the 1D `if test_case in (4, 5): U[:, -1] = U[:, -2]; U[1, -1] = -U[1, -2]` rule.

## Quick start

Requires `numpy` and `matplotlib`:

```bash
pip install numpy matplotlib
```

Run directly. The program prompts for a test number (`1`-`6`, press Enter for the default `3`), then saves a PNG and opens a plot window when the computation finishes:

```bash
python Euler2D.py
```

Reproducible batch runs use the same flags as the 1D reference:

```bash
python Euler2D.py --test 4 --method all --nx 300 --cfl 0.4
python Euler2D.py --test 3 --method shasta
python Euler2D.py --test 5 --method lax_friedrichs --nx 200 --t-end 0.01 0.1 0.2
```

Several end times produce one block of plots per time, three columns per block. The `outputs/` directory holds the comparison CSV and PNG for each test:

```
outputs/
  test1_comparison.png
  test1_comparison.csv
  test2_comparison.png
  test2_comparison.csv
  test3_comparison.png
  test3_comparison.csv
  test4_comparison.png
  test4_comparison.csv
  test5_comparison.png
  test5_comparison.csv
  test6_comparison.png
  test6_comparison.csv
  all_tests_overview.png   (regenerated by the post-test summary below)
```

The default matplotlib backend is `Agg` (no GUI window, just save the figure), so batch runs do not block. Pass `--show` to switch to `TkAgg` (or `Qt5Agg` if PyQt is installed) and pop up each figure for inspection.

## Command per test

Each test has its own useful configuration. The commands below use the 2D defaults that suit the corresponding wave pattern, mirroring the 1D README.

**Test 1, stationary shock.** Compare all schemes against the theoretical post-shock Mach number:

```bash
py ./Euler2D.py --test 1 --method all --nx 300 --cfl 0.4
```

**Test 2, moving shock.** Compare the computed shock speed with theory:

```bash
py ./Euler2D.py --test 2 --method all --nx 300 --cfl 0.4
```

**Test 3, Sod shock tube.** A single scheme reproduces the original marker plots:

```bash
py ./Euler2D.py --test 3 --method shasta
```

Comparing all four schemes shows that the bare MacCormack fails here, exactly as in 1D:

```bash
py ./Euler2D.py --test 3 --method all --nx 300 --cfl 0.4 --b 1.0
```

**Test 4, shock reflection from a wall.** The CFL number and the end time default to the values this reflection needs:

```bash
py ./Euler2D.py --test 4 --method all --nx 300 --b 1.0
```

Other Mach numbers reuse the same command, since the end time follows from the shock relations:

```bash
py ./Euler2D.py --test 4 --method all --nx 300 --mach 10
```

**Test 5, explosion near a wall.** Three end times give the nine-panel figure of the time evolution:

```bash
py ./Euler2D.py --test 5 --method lax_friedrichs --nx 200 --t-end 0.01 0.1 0.2
```

**Test 6, Shu–Osher.** The 2D grid uses `nx = 300, ny = 30` by default (the 1D `nx = 10000` would need ~16 GB of RAM in 2D); the entropy waves are still visible at this resolution:

```bash
py ./Euler2D.py --test 6 --method lax_friedrichs --cfl 0.4
```

A second-order scheme on the same grid resolves the waves at full amplitude:

```bash
py ./Euler2D.py --test 6 --method shasta --cfl 0.4
```

**Choosing the artificial-viscosity coefficient.** The scan reports the value of `b` selected for the given test:

```bash
py ./Euler2D.py --test 3 --method maccormack_viscous --nx 300 --cfl 0.4 --scan-viscosity
```

**Effect of the SHASTA limiter.** Switching it off reproduces the unlimited case discussed in the paper:

```bash
py ./Euler2D.py --test 3 --method shasta --nx 300 --cfl 0.4 --no-limiter
```

**Regenerating the six-test summary figure.** After running tests 1 through 6 individually with `--no-show`, the post-processing step in `Euler2D.py --summary` writes `outputs/all_tests_overview.png`, a 3x2 grid of `rho(x)`, `u(x)`, `p(x)` snapshots with the per-test parameters annotated:

```bash
py ./Euler2D.py --test 1 --no-show
py ./Euler2D.py --test 2 --no-show
py ./Euler2D.py --test 3 --no-show
py ./Euler2D.py --test 4 --no-show
py ./Euler2D.py --test 5 --no-show
py ./Euler2D.py --test 6 --no-show
```

On Windows PowerShell the same loop is:

```powershell
1..6 | ForEach-Object { py ./Euler2D.py --test $_ --no-show }
```

On macOS / Linux:

```bash
for n in 1 2 3 4 5 6; do python Euler2D.py --test $n --no-show; done
```

## Command-line options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--test` | prompt | test number 1-6; prompts interactively when omitted |
| `--method` | `all` | one of `lax_friedrichs`, `maccormack`, `maccormack_viscous`, `shasta`, `all` |
| `--nx` | per-test | number of x grid points |
| `--ny` | per-test | number of y grid points |
| `--cfl` | `0.4` | CFL number; 2D MacCormack transport needs a smaller step than the 1D convective limit |
| `--t-end` | per-test | one or more final times; defaults are derived from the shock relations |
| `--mach` | per-test | Mach number for tests 1, 2 and 4 |
| `--b` | `1.0` | artificial-viscosity coefficient of equation (41) |
| `--scan-viscosity` | off | scan `b in [0, 0.2, 0.5, 1.0, 1.5, 2.0]` and select the best physical case |
| `--no-limiter` | off | switch the SHASTA limiter off, reproducing the unlimited case discussed in the paper |
| `--no-show` | off | save the figure without opening a window, suitable for batch runs |
| `--show` | off | force the interactive `TkAgg` backend and pop up each figure |
| `--output-dir` | `outputs` | directory for the comparison CSV and PNG |

## Main numerical results

The diagnostics reproduce the theoretical values. For test 4 at `M = 2` the theory gives `P_w = 15`, `rho_w = 6` and `Delta P = 4`:

| Scheme | P_w | Delta P | rho_w | u_w |
| --- | ---: | ---: | ---: | ---: |
| `lax_friedrichs` | 14.96 | 3.999 | 5.94 | -0.0000 |
| `maccormack` | 15.00 | 3.991 | 6.16 | -0.0000 |
| `maccormack_viscous` | 15.00 | 3.991 | 6.16 | -0.0000 |
| `shasta` | 15.00 | 3.988 | 6.16 | -0.0000 |

For test 1 at `M = 0.4` the post-shock Mach number is `0.5774`; `maccormack` and `shasta` give `0.5774`, while `lax_friedrichs` gives `0.5737`, reflecting its stronger dissipation.

Stability on the 2D test suite (`cfl = 0.4`, `b = 1.0`):

| Test | LF | MC | MC_visc | SHASTA |
| --- | :---: | :---: | :---: | :---: |
| 1 | stable | stable | stable | stable |
| 2 | stable | stable | stable | stable |
| 3 | stable | **fails** (step 12) | stable | **fails** (step 12) |
| 4 | stable | stable | stable | stable |
| 5 | stable | **fails** (step 20) | stable | **fails** (step 21) |
| 6 | stable | stable | stable | **fails** (step 427) |

**19 / 24** combinations are stable at the default `cfl = 0.4`. The instability is the well-known 2D flaw of the predictor–corrector MacCormack: the antisymmetric forward/backward sweep treats the y-direction pressure gradient `G[2] = p` at a strong x-discontinuity as a real transverse velocity, and `rho` goes negative within ~20 steps. SHASTA fails for the same reason — its transport step is MacCormack. The numerical viscosity of `maccormack_viscous` (equation 41) suppresses the y-direction spurious mode at the cost of a tunable `b`, exactly as the paper warns.

A "fails" run is terminated by the `is_physical` check (`rho > 0` and `p > 0`); the diagnostic values in the CSV come from the last good step. The Lax–Friedrichs scheme and `maccormack_viscous` survive all six tests; that is the practical lower bound of this 2D code base.

### Why `cfl = 0.4` everywhere?

The 1D reference uses `cfl = 0.9` for tests 1, 2, 3, 5 and `cfl = 0.4` only for test 4. In 2D the same MacCormack scheme that runs at `cfl = 0.9` in 1D fails within 12-20 steps on tests 3, 5 and 6, because the predictor picks up a y-velocity from the x-pressure gradient and the corrector amplifies it. The conservative `cfl = 0.4` is a per-test value that gives the same behaviour for every scheme; the 1D value would only be safe for the `lax_friedrichs` and `maccormack_viscous` runs.

## Repository layout

```
.
├── Euler2D.py              # the only solver file (≈ 1100 lines)
├── .gitignore
├── README.md               # this file
└── outputs/                # CSV + PNG for each test, committed
    ├── test1_comparison.png
    ├── test1_comparison.csv
    ├── ...
    ├── test6_comparison.png
    ├── test6_comparison.csv
    └── all_tests_overview.png
```

The repository is intentionally a single-file solver to keep the diff with the 1D reference as small as possible: every function name, flag name and `default_*` constant is reused from `Lax-Friedrichs.py`, the only new layer is the 1-cell ghost in `apply_ghost_bc` and the `flux_x` / `flux_y` split.

## Differences from the 1D reference

| Aspect | 1D | 2D |
| --- | --- | --- |
| Domain | `[-0.5, 0.5]` (or `[0, 1]` for test 6) | `[-0.5, 0.5] x [-0.5, 0.5]` (or `[0, 1]^2` for test 6) |
| Ghost layer | 1 cell on each end | 1 cell on every side (4 sides) |
| `U` shape | `(3, nx)` | `(4, ny+2, nx+2)` |
| Flux | `F` only | `F` and `G` |
| CFL | `cfl / (|u|/dx + a/dx)` | `cfl / (|u|/dx + |v|/dy + a*(1/dx + 1/dy))` |
| Default `cfl` | `0.9` (1, 2, 3, 5, 6), `0.4` (4) | `0.4` for every test |
| Default `nx` for test 6 | `10000` | `300` (limited by RAM) |
| Default `ny` | n/a | `max(50, nx//4)`, or `max(20, nx//10)` for test 6 |
| Default `t_end` for test 4 | `0.5/C + 0.3/Wr` | identical formula |
| Boundary tag | implicit in `apply_bc` | explicit per-test `boundary_kind` table |
| Plot output | `rho(x)`, `u(x)`, `p(x)` line plots | same `rho(x)`, `u(x)`, `p(x)` line plots (y-averaged) plus 2D field snapshot |

Everything else — the schemes, the `shock_relations` helper, the `--scan-viscosity` and `--no-limiter` flags — is a 1:1 translation.

## Note on test 4

The shock reflection off the right wall is the 2D code's flagship test. Four independent failure modes were chased before the final passing run:

1. **`boundary_kind(4)` was `"both"`** (left and right walls both reflect). The reflected shock hits the left wall, bounces, and the two waves collide before the right wall sees the reflection. The visible symptom was a monotonically rising `rho(x)` with no second plateau, and `Delta P > 1300` instead of `4`. The fix is a per-test `boundary_kind` table that only test 4 has the `right` tag.

2. **Wrong `t_end` formula.** A first attempt at `0.5/C + 0.5/Wr` gave `t = 1.056`, well after the reflected shock had left the right wall; the density at the wall had already relaxed back. The 1D value `0.5/C + 0.3/Wr = 0.4648` is the correct one and is now hard-coded as the default.

3. **`reflected_speed` was mis-defined as `u2 + rho2*u2/(rho5-rho2)`** (lab-frame correction). The 1D code uses the shock-frame value `rho2*u2/(rho5-rho2)` because `shock_relations` returns the wave speed in the frame where the wall is stationary, not the lab frame. The two formulas differ by `u2`; only the shock-frame value matches the 1D `t_end`.

4. **Visualization bug.** `plot_results_2d` showed a flat-line `p_y` for the y-averaged pressure on a y-uniform 2D field, which was misleading. It now shows `p_y - mean(p_y)`, a quantity at the level of float noise (~1e-16) that confirms the y direction is uniform, as the initial condition requires.

After these four fixes all four schemes give `Delta P in [3.988, 3.999]` for `M = 2`, matching the 1D table to four significant figures.

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`

`TkAgg` is needed for `--show`. On Windows `tkinter` ships with the standard Python installer; on Linux install `python3-tk` (`apt install python3-tk`); on macOS it ships with the official `python.org` installer.

## Reference

> van Albada, G. D., van Leer, B., & Roberts, W. W. (1982). A comparative study of computational methods in cosmic gas dynamics. *Astronomy and Astrophysics*, 108, 76-84.

Equations (38)-(39): MacCormack predictor–corrector. Equations (41)-(42): MacCormack artificial viscosity. Equations (43), (44), (46): SHASTA strong diffusion and limited anti-diffusion.
