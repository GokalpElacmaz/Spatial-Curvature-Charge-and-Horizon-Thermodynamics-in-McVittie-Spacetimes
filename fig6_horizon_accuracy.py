import os
import sys
import numpy as np
from scipy.optimize import brentq
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from mcvittie_figure_style import apply_style, GREY, SHADE, BLUE, ORANGE, GREEN, BLACK

apply_style()

NSCAN = 7200
NE, NL = 181, 91                              # same spacing in ell as before, 8.9e-4
eps_grid = np.linspace(0.0, 0.16, NE)
ell_grid = np.linspace(0.0, 0.08, NL)
PANELS = [(0.0, r"$Q/M=0$"), (0.5, r"$Q/M=0.5$"), (0.99, r"$Q/M=0.99$")]
CACHE = "fig6_cache.npz"


# ------------------------------------------------------------ marginal sphere
def residual(x, eps, ell, q):
    x = np.asarray(x, dtype=float)
    u = 1.0 + ell * x**2
    with np.errstate(invalid="ignore", divide="ignore"):
        su = np.sqrt(np.where(u > 0.0, u, np.nan))
        P = (1.0 + su / x)**2 - q**2 * u / x**2
        J = (1.0 - ell * x**2) * P - 2.0 * su / x - 2.0 * (1.0 - q**2) * u / x**2
        out = J - 0.5 * eps * x * P**2
    return np.where(u > 0.0, out, np.nan)


def lapse_numerator(x, ell, q):
    x = np.asarray(x, dtype=float)
    return 1.0 - (1.0 - q**2) * (1.0 + ell * x**2) / x**2


def lapse_zero_x(ell, q):
    d2 = 1.0 - q**2
    denom = 1.0 - d2 * ell
    if denom <= 0.0 or d2 / denom < 0.0:
        return np.nan
    return np.sqrt(d2 / denom)


def areal(x, ell, q):
    x = np.asarray(x, dtype=float)
    u = 1.0 + ell * x**2
    with np.errstate(invalid="ignore", divide="ignore"):
        su = np.sqrt(np.where(u > 0.0, u, np.nan))
        P = (1.0 + su / x)**2 - q**2 * u / x**2
        R = 0.5 * x * P / u
    return np.where(u > 0.0, R, np.nan)


def smaller_root(eps, ell, q, nscan=NSCAN):
    d = np.sqrt(max(0.0, 1.0 - q**2))
    if np.isclose(eps, 0.0, atol=1e-15) and np.isclose(ell, 0.0, atol=1e-15):
        return d
    x_lapse = lapse_zero_x(ell, q)
    if not np.isfinite(x_lapse):
        return np.nan
    xlo = max(np.nextafter(x_lapse, np.inf), 1.0e-10)
    xhi = 6.0
    if ell < 0.0:
        xhi = min(xhi, 0.995 / np.sqrt(-ell))
    if xhi <= xlo:
        return np.nan
    xs = np.linspace(xlo, xhi, int(nscan))
    vals = residual(xs, eps, ell, q)
    N = lapse_numerator(xs, ell, q)
    good = np.isfinite(vals) & np.isfinite(N) & (N > 0.0)
    idx = np.where(good[:-1] & good[1:] & (vals[:-1] < 0.0) & (vals[1:] >= 0.0))[0]
    if idx.size == 0:
        return np.nan
    i = idx[0]
    try:
        root = brentq(lambda z: float(residual(z, eps, ell, q)), xs[i], xs[i + 1],
                      xtol=1.0e-13, rtol=4.0 * np.finfo(float).eps, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan
    if float(lapse_numerator(root, ell, q)) <= 0.0:
        return np.nan
    return root


def R_num(eps, ell, q, nscan=NSCAN):
    x = smaller_root(eps, ell, q, nscan=nscan)
    if not np.isfinite(x):
        return np.nan
    R = float(areal(x, ell, q))
    return R if np.isfinite(R) else np.nan


def R_pert(eps, ell, q):
    """Dimensionless form of eq:bh-charged, R_pert/M, with d = sqrt(1-q^2)."""
    d = np.sqrt(1.0 - q**2)
    return ((1.0 + d) + eps**2 * (1.0 + d)**4 / (2.0 * d)
            - ell * d**2 * (1.0 + d) / 2.0
            - ell**2 * d**3 * (1.0 + d) * (4.0 + 5.0 * d) / 8.0)


def ell_max(eps, q, nscan=NSCAN):
    """Largest ell with a regular smaller root (merger boundary), by bisection."""
    lo, hi, hi_cap = 0.0, 0.30, 5.0
    if not np.isfinite(R_num(eps, lo, q, nscan)):
        return np.nan
    while np.isfinite(R_num(eps, hi, q, nscan)):
        lo = hi
        if hi >= hi_cap:
            return np.nan
        hi = min(2.0 * hi, hi_cap)
    for _ in range(55):
        mid = 0.5 * (lo + hi)
        if np.isfinite(R_num(eps, mid, q, nscan)):
            lo = mid
        else:
            hi = mid
    return lo


def signed_error_map(q, nscan=NSCAN):
    Z = np.full((NL, NE), np.nan)
    for i, ell in enumerate(ell_grid):
        for j, eps in enumerate(eps_grid):
            rn = R_num(eps, ell, q, nscan)
            if np.isfinite(rn) and rn > 0.0:
                Z[i, j] = R_pert(eps, ell, q) / rn - 1.0
    return Z


# fixed Omega_k lines for the bottom row
OMEGAS = [(1.0, BLACK, ":", r"$\Omega_k=+1$"),
          (0.0, BLUE, "-", r"$\Omega_k=0$"),
          (-10.0, ORANGE, "--", r"$\Omega_k=-10$"),
          (-100.0, GREEN, "-.", r"$\Omega_k=-100$")]
EPS_LINE = np.geomspace(0.008, 0.16, 70)


def omega_line(q, om, nscan=NSCAN):
    out = np.full_like(EPS_LINE, np.nan)
    for i, e in enumerate(EPS_LINE):
        ell = -om * e**2 / 16.0
        rn = R_num(e, ell, q, nscan)
        if np.isfinite(rn):
            out[i] = abs(R_pert(e, ell, q) / rn - 1.0)
    return out


def convergence_check():
    """Convergence statements of Appendix F for the grid used in this figure."""
    tot = NE * NL
    for q, _ in PANELS:
        ref = signed_error_map(q, NSCAN)
        bref = np.array([ell_max(e, q, NSCAN) for e in eps_grid])
        for n in (1800, 3600):
            Z = signed_error_map(q, n)
            diff = int(np.sum(np.isfinite(Z) != np.isfinite(ref)))
            b = np.array([ell_max(e, q, n) for e in eps_grid])
            dmax = np.nanmax(np.abs(b - bref))
            both = np.isfinite(Z) & np.isfinite(ref)
            same_1pc = np.nanmax(np.abs(np.abs(Z[both]) - np.abs(ref[both])))
            flips = int(np.sum(((np.abs(Z) > 1e-2) != (np.abs(ref) > 1e-2)) & both))
            print(f"Q/M={q:g} nscan={n}: regular set differs at {diff} of {tot} points; "
                  f"points whose 1% classification flips: {flips}; "
                  f"max boundary shift vs {NSCAN}: {dmax:.2e}; "
                  f"max change of |error| on common points {same_1pc:.2e}", flush=True)
    print(f"grid spacing in ell: {ell_grid[1]-ell_grid[0]:.2e}")


if "--check" in sys.argv:
    convergence_check()
    sys.exit()

# ----------------------------------------------------------------- compute (or load)
if os.path.exists(CACHE) and "--recompute" not in sys.argv:
    dat = np.load(CACHE, allow_pickle=True)
    sgn = {q: dat[f"sgn_{q}"] for q, _ in PANELS}
    bnd = {q: dat[f"bnd_{q}"] for q, _ in PANELS}
    lines = {q: dat[f"lines_{q}"] for q, _ in PANELS}
else:
    sgn, bnd, lines = {}, {}, {}
    for q, _ in PANELS:
        print(f"Q/M = {q:g}: error map, boundary and lines", flush=True)
        sgn[q] = signed_error_map(q)
        bnd[q] = np.array([ell_max(e, q) for e in eps_grid])
        lines[q] = np.array([omega_line(q, om) for om, *_ in OMEGAS])
    np.savez(CACHE, **{f"sgn_{q}": sgn[q] for q in sgn}, **{f"bnd_{q}": bnd[q] for q in bnd},
             **{f"lines_{q}": lines[q] for q in lines})

err = {q: np.abs(sgn[q]) for q in sgn}

# ----------------------------------------------------------------- plot
fig = plt.figure(figsize=(7.1, 4.75))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.78], left=0.085, right=0.88,
                      bottom=0.135, top=0.955, wspace=0.13, hspace=0.36)
axm = [fig.add_subplot(gs[0, j]) for j in range(3)]
axl = [fig.add_subplot(gs[1, j]) for j in range(3)]

norm = mpl.colors.LogNorm(vmin=1.0e-6, vmax=1.0e-1, clip=True)
cmap = mpl.colormaps["YlGnBu"].copy()
halo = [pe.withStroke(linewidth=2.0, foreground="white")]

for n, (ax, (q, title)) in enumerate(zip(axm, PANELS)):
    ax.set_facecolor(SHADE)
    pm = ax.pcolormesh(eps_grid, ell_grid, err[q], norm=norm, cmap=cmap,
                       shading="nearest", rasterized=True)
    # 1% contour
    cs = ax.contour(eps_grid, ell_grid, np.ma.masked_invalid(err[q]), levels=[1.0e-2],
                    colors=[ORANGE], linewidths=1.4)
    ax.clabel(cs, fmt={1.0e-2: "1%"}, fontsize=7, inline=True, inline_spacing=3,
              manual=[(0.136, 0.040)] if q == 0.99 else None)
    # curve on which R_pert = R_num
    ax.contour(eps_grid, ell_grid, np.ma.masked_invalid(sgn[q]), levels=[0.0],
               colors="k", linewidths=0.9, linestyles=[(0, (4, 2))])
    # merger boundary
    ok = np.isfinite(bnd[q])
    ax.plot(eps_grid[ok], bnd[q][ok], color="0.25", lw=1.0)
    # background curvature: ell = -Omega_k eps^2 / 16
    ee = np.linspace(0.0, 0.16, 300)
    # for om, xl, yl, va in ((-10.0, 0.1575, 0.0185, "bottom"), (-100.0, 0.1575, 0.0795, "top")):
    #     ax.plot(ee, -om * ee**2 / 16.0, color="0.15", lw=0.8, ls=(0, (1, 2.0)), zorder=4)
    #     # ax.text(xl, yl, rf"$|\Omega_k|={-om:g}$", fontsize=6.8, color="0.1",
    #     #         rotation=0, ha="right", va=va, path_effects=halo, zorder=5)
    ax.set_title(title, pad=4)
    ax.set_xlim(0.0, 0.16)
    ax.set_ylim(0.0, 0.08)
    ax.set_xticks([0.04, 0.08, 0.12, 0.16])
    ax.set_xlabel(r"$\epsilon=MH$")
    ax.minorticks_on()
    ax.text(0.04, 0.94, f"({chr(ord('a') + n)})", transform=ax.transAxes, ha="left",
            va="top", fontsize=9, color="white",
            bbox=dict(boxstyle="square,pad=0.12", facecolor="black", edgecolor="none",
                      alpha=0.45))
axm[0].set_ylabel(r"$\ell=kM^{2}/(4a^{2})$")
axm[1].tick_params(labelleft=False)
axm[2].tick_params(labelleft=False)
axm[0].text(0.63, 0.86, "no regular smaller\nmarginal sphere", transform=axm[0].transAxes,
            fontsize=7.0, color="0.25", ha="center", va="center")

cax = fig.add_axes([0.90, 0.5395, 0.018, 0.4155])
cb = fig.colorbar(pm, cax=cax)
cb.set_label(r"$|R_{\rm pert}/R_{\rm num}-1|$", fontsize=9, labelpad=3)
cb.ax.tick_params(labelsize=8, width=0.7, direction="in")
cb.outline.set_linewidth(0.7)
cb.set_ticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

# bottom row: error along fixed Omega_k
xg = np.array([0.02, 0.09])
AMP = {0.0: 4.0, 0.5: 4.0, 0.99: 0.5}
for n, (ax, (q, _)) in enumerate(zip(axl, PANELS)):
    for k_, (om, c, ls, lab) in enumerate(OMEGAS):
        ax.plot(EPS_LINE, lines[q][k_], color=c, ls=ls, label=lab)
    ax.plot(xg, AMP[q] * xg**4, color="0.4", lw=0.8, ls=(0, (1, 1.6)))
    ax.text(0.052, AMP[q] * 0.052**4 * 0.30, r"$\propto\epsilon^{4}$", fontsize=7.5, color="0.35",
            rotation=30, ha="center", va="top")
    ax.axhline(1e-2, color=ORANGE, lw=0.8, ls="-", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.008, 0.16)
    ax.set_ylim(1e-9, 3e-1)
    ax.set_xlabel(r"$\epsilon=MH$")
    ax.set_xticks([0.01, 0.02, 0.05, 0.1])
    ax.set_xticklabels(["0.01", "0.02", "0.05", "0.1"])
    ax.text(0.04, 0.94, f"({chr(ord('d') + n)})", transform=ax.transAxes, ha="left",
            va="top", fontsize=9)
axl[0].set_ylabel(r"$|R_{\rm pert}/R_{\rm num}-1|$")
axl[1].tick_params(labelleft=False)
axl[2].tick_params(labelleft=False)
h, l = axl[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, handlelength=2.2, columnspacing=1.8,
           bbox_to_anchor=(0.485, -0.005), fontsize=8)

fig.savefig("fig6_horizon_accuracy.pdf", dpi=600)
fig.savefig("fig6_horizon_accuracy.png", dpi=350)
plt.close(fig)

# ----------------------------------------------------------------- numbers for the text
common = np.all([np.isfinite(err[q]) for q, _ in PANELS], axis=0)
print("\nnumbers for Appendix F")
for q, _ in PANELS:
    Z = err[q]
    print(f"Q/M={q:g}: regular fraction {np.isfinite(Z).mean():.3f}; median error on the common "
          f"regular set {np.median(Z[common]):.2e}; max error {np.nanmax(Z):.2e}; "
          f"ell_max(eps=0) = {bnd[q][0]:.5f}")
print("common regular set:", int(common.sum()), "of", NE * NL, "points")
print("neutral closed form ell_max(0) = 2/27 =", round(2 / 27, 5))
# error at the largest epsilon on the flat line, and 1% crossing
for q, _ in PANELS:
    e_flat = lines[q][1]
    i1 = np.where(e_flat > 1e-2)[0]
    print(f"Q/M={q:g}: error at eps=0.16, Omega_k=0: {e_flat[-1]:.2e}; "
          f"first eps with error>1%: {EPS_LINE[i1[0]] if i1.size else float('nan'):.3f}")
    for k_, (om, *_r) in enumerate(OMEGAS):
        ln = lines[q][k_]
        ok = np.isfinite(ln)
        print(f"     Omega_k={om:+.0f}: error at eps=0.05 -> {np.interp(0.05, EPS_LINE[ok], ln[ok]):.2e},"
              f" at 0.1 -> {np.interp(0.1, EPS_LINE[ok], ln[ok]):.2e}, last finite eps {EPS_LINE[ok][-1]:.3f}")
