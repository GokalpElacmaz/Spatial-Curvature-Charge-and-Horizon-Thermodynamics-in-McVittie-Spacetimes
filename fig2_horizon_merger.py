#!/usr/bin/env python3
import numpy as np
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt

from mcvittie_figure_style import (apply_style, panel_label,
                                   BLUE, ORANGE, BLACK, GREY, SHADE)

apply_style()

K_PANEL_A = 0.0
MASS_RATIOS = (0.60, 1.00, 1.15)
X_STAR = (np.sqrt(6.0) - 2.0) / 2.0


def mu(r, M, k):
    return M * np.sqrt(1.0 + k * r**2) / (2.0 * r)


def areal_radius(r, M, k):
    return r * (1.0 + mu(r, M, k))**2 / (1.0 + k * r**2)


def Y_func(r, M, k):
    """Y = (N - k r^2)/(1 + k r^2), eq:Y-def."""
    m = mu(r, M, k)
    N = (1.0 - m) / (1.0 + m)
    return (N - k * r**2) / (1.0 + k * r**2)


def G(r, M, k):
    """Y - HR with H = 1."""
    return Y_func(r, M, k) - areal_radius(r, M, k)


def r_lapse_zero(M, k):
    return M / np.sqrt(4.0 - k * M * M)


def r_bracket(M, k):
    lo = r_lapse_zero(M, k) * (1.0 + 1e-9)
    hi = 0.999 / np.sqrt(-k) if k < 0 else 60.0
    return lo, hi


def G_max(M, k):
    lo, hi = r_bracket(M, k)
    res = minimize_scalar(lambda r: -G(r, M, k), bounds=(lo, hi),
                          method="bounded")
    return -res.fun


def critical_mass(k):
    return brentq(lambda M: G_max(M, k), 1e-4, 0.45, xtol=1e-13)


def marginal_radii(M, k):
    lo, hi = r_bracket(M, k)
    r_peak = minimize_scalar(lambda r: -G(r, M, k), bounds=(lo, hi),
                             method="bounded").x
    g_peak = G(r_peak, M, k)
    if g_peak < -1e-9:
        return []
    if abs(g_peak) <= 1e-7:
        return [areal_radius(r_peak, M, k)]
    r1 = brentq(G, lo, r_peak, args=(M, k), xtol=1e-13)
    r2 = brentq(G, r_peak, hi, args=(M, k), xtol=1e-13)
    return [areal_radius(r, M, k) for r in (r1, r2)]


def mu_pm(w):
    disc = 3.0 - 12.0 * w - 6.0 * w**2
    root = np.sqrt(np.clip(disc, 0.0, None))
    num = 2.0 - 2.0 * w - w**2
    den = (1.0 + w)**2
    return (num + root) / den, (num - root) / den


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.75))

# Panel (a)
k = K_PANEL_A
Mc = critical_mass(k)
styles = [(BLUE, "-"), (BLACK, "-"), (ORANGE, "--")]
for ratio, (colour, ls) in zip(MASS_RATIOS, styles):
    M = ratio * Mc
    lo, hi = r_bracket(M, k)
    r = np.geomspace(lo, hi, 6000)
    ax1.plot(areal_radius(r, M, k), G(r, M, k), color=colour, ls=ls,
             label=rf"$M/M_c={ratio:g}$")
    for Rm in marginal_radii(M, k):
        ax1.plot(Rm, 0.0, "o", color=colour, ms=3.8, mew=0, zorder=5)

ax1.axhline(0.0, color=GREY, lw=0.6, zorder=0)
# Label the double root reached at M = M_c.
_Rm = marginal_radii(Mc, k)[0]
ax1.annotate("merger", xy=(_Rm, 0.0), xytext=(_Rm + 0.16, 0.135),
             fontsize=8, color=BLACK, ha="left",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=BLACK,
                             shrinkA=1, shrinkB=3))
ax1.set_xlim(0.0, 1.20)
ax1.set_ylim(-0.74, 0.32)
ax1.set_xlabel(r"$H R$")
ax1.set_ylabel(r"$Y-HR$")
# The lower left crosses the M/M_c=0.6 curve. The lower right is empty below
# the ends of the curves.
ax1.legend(loc="lower right", handlelength=1.6, borderpad=0.2)
panel_label(ax1, "(a)")
ax1.minorticks_on()

# Panel (b)
w = np.linspace(-0.40, X_STAR, 2000)
mp, mm = mu_pm(w)
ax2.axhspan(1.0, 5.0, color=SHADE, zorder=0)            # mu > 1, outside N > 0
ax2.axhline(1.0, color=GREY, lw=0.6, zorder=1)
ax2.plot(w, mp, color=ORANGE, ls="--")
ax2.plot(w, mm, color=BLUE, ls="-")
ax2.axvline(X_STAR, color=GREY, lw=0.6, ls=":", zorder=1)
# The merger curve with H > 0 stops at kr^2 = 1/5, where mu_- = 2/3 and Y = 0.
# Between 1/5 and (sqrt6-2)/2 the double roots have Y < 0 and need H < 0.
ax2.axvspan(0.2, X_STAR, ymin=0.0, ymax=1.0, facecolor="none",
            edgecolor=GREY, hatch="/////", lw=0, zorder=1)
ax2.plot(0.2, 2.0 / 3.0, "o", ms=3.8, color=BLUE, zorder=6)
ax2.text(0.2 - 0.008, 3.55, "1/5", color="0.35", fontsize=8.5, ha="right")
ax2.text(X_STAR + 0.008, 3.55, "0.225", color="0.35", fontsize=8.5, ha="left")
# mu = 1 is the limiting point where the lapse vanishes, not a regular merger.
ax2.plot(X_STAR, 1.0, "o", ms=4.2, mfc="white", mec=BLACK, mew=1.0, zorder=5,
         clip_on=False)
ax2.set_xlim(-0.40, 0.30)
ax2.set_ylim(0.0, 4.3)
ax2.set_xlabel(r"$k r_A^{2}$")
ax2.set_ylabel(r"$\mu$")
ax2.set_yticks([0, 1, 2, 3, 4])
ax2.set_xticks([-0.4, -0.2, 0.0, 0.2])
ax2.text(-0.34, 3.55, r"$\mu_{+}$", color=ORANGE)
ax2.text(0.02, 0.42, r"$\mu_{-}$", color=BLUE)
ax2.text(-0.02, 1.25, r"$\mu>1$, outside $N>0$", color="0.35", fontsize=8,
         ha="center", va="bottom")
panel_label(ax2, "(b)")
ax2.minorticks_on()

fig.subplots_adjust(wspace=0.28)
fig.savefig("fig2_horizon_merger.pdf")
fig.savefig("fig2_horizon_merger.png", dpi=400)

print(f"k={k:g}: M_c H={Mc:.8f}; flat value={1/(3*np.sqrt(3)):.8f}")
for ratio in MASS_RATIOS:
    print(ratio, marginal_radii(ratio * Mc, k))