# Spatial curvature, charge, and thermodynamics in McVittie spacetimes

Scripts that produce every figure and every numerical check of the paper
*Spatial curvature, charge, and thermodynamics in McVittie spacetimes*
(T. G. Elaçmaz and Y. Heydarzade). The version of this repository that
corresponds to the submitted manuscript is the release tagged `epjc-revision-2`.

## Requirements

Python 3.10 or later with

```
numpy scipy matplotlib mpmath sympy
```

installed, for example with `pip install numpy scipy matplotlib mpmath sympy`.
The figures and checks in the tagged release were run with Python 3.13, numpy 2.3,
scipy 1.16, matplotlib 3.10, mpmath 1.3 and sympy 1.14. No other software is needed.

## Figures

Run each script from the repository root. It writes the PDF and PNG of the same
name into the current directory. `mcvittie_figure_style.py` holds the common
style and is imported by the figure scripts.

| Script | Figure in the paper |
|---|---|
| `fig1_schematic.py` | Fig. 1, schematic of the marginal spheres |
| `fig2_horizon_merger.py` | Fig. 2, merger of the marginal spheres |
| `fig3_critical_mass.py` | Fig. 3, critical mass against the merger position |
| `fig4_cosmological_horizon.py` | Fig. 4, curvature dependence of the cosmological marginal sphere |
| `fig5_cavity.py` | Fig. 5, cavity crossings |
| `fig5b_crossing_roots.py` | Fig. 6, admitted wall potentials |
| `fig6_horizon_accuracy.py` | Fig. 7, accuracy of the perturbative radius (Appendix F) |

`fig6_horizon_accuracy.py` reads `fig6_cache.npz` when it is present and
recomputes the numerical zeros otherwise, which takes several minutes.

To regenerate all figures:

```
for f in fig*.py; do python3 "$f"; done
```

## Numerical and symbolic checks

Each check is a standalone script in `checks/`, run from the repository root with
`python3 checks/<name>.py`. Each prints what it verifies and ends with an
assertion, so a failed check stops with a traceback.

| Script | What it verifies |
|---|---|
| `checks/verify_field_equations.py` | The Vaidya–Shah metric with the stated density, pressure and Gauss charge satisfies the Einstein–Maxwell equations (Sect. 4) |
| `checks/verify_rho_plus_p.py` | With a central mass or charge, ρ+p vanishes on a whole surface of constant t only for k = 0 and dH/dt = 0 (Sect. 3) |
| `checks/verify_surface_gravity_split.py` | The split κ_A = κ_Λ + 2πR_A(ρ_A+p_A) and ∂_rχ = 2κ_Λ ∂_rR on marginal spheres (Sect. 3) |
| `checks/verify_merger_nondegeneracy.py` | Along the curve of double zeros of Sect. 2.3, the second radial derivative and the M-derivative of Y−HR, so where the double zero is a merger of two simple zeros, and the closed-form merger relations |
| `checks/verify_charged_merger.py` | Explicit charged mergers at fixed Q/M, with the root count, the derivatives at the double zero and Eq. (charged-merger) (Sect. 4.1) |
| `checks/verify_near_extremal.py` | The near-extremal radius at ℓ = 0 and the breakdown of the ε ≪ γ expansion (Sect. 4.1) |
| `checks/verify_euclidean_action.py` | The on-shell Euclidean action of the cavity, Eq. (euclid-action), evaluated term by term against Eq. (action-result) (Sect. 5.1) |
| `checks/verify_cavity_wall_deviations.py` | The deviations from the boundary first law outside the flat stationary case, by direct numerical differentiation (Sect. 5.6, Appendix G) |
| `checks/verify_crossing_algebra.py` | The factorization of the grand potential, both roots, the positivity of ∂T_B/∂X at the crossings and the identities of Appendix I |

The root-level scripts `verify-appendix.py`, `verify-new.py`, `verify-weyl.py`
and `verify_two_roots.py` are earlier checks kept for reference.

To run all checks:

```
for f in checks/verify_*.py; do python3 "$f" || break; done
```
