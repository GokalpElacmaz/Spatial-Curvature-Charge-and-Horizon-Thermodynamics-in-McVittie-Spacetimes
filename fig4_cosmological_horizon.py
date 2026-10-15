import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mcvittie_figure_style import (apply_style, panel_label,
                                   BLUE, ORANGE, GREEN, GREY, SHADE)

apply_style()

M_RATIO = 0.15                       # M/R_H in panel (b)


def C_M(w):
    return (5.0 * w - 1.0) / (1.0 + w) ** 2.5


def C_Q(w):
    return (1.0 - 3.0 * w) / (2.0 * (1.0 + w) ** 3)


def Pi(w):
    return 1.0 + 5.0 * w - 15.0 * w ** 2 + 5.0 * w ** 3


def second_order_shift_over_M(w, mratio, q_over_m):
    """(R_CH - R_H)/M truncated at second order, in units R_H = 1."""
    return (C_M(w) - 1.5 * mratio * Pi(w) / (1.0 + w) ** 5
            + q_over_m ** 2 * mratio * C_Q(w))


# ---------------------------------------------------------------- numerical root
# For a given w we set R_H = 1 and r_H = 1, which gives k = w, a = 1 + w and
# H = (1 - w)/(1 + w).
def numerical_cosmological_radius(w, mratio, q_over_m, ngrid=4000):
    """Radius R_CH, in units of R_H, of the root of Y = HR nearest to r_H."""
    k, a = w, 1.0 + w
    H = (1.0 - w) / (1.0 + w)
    M = mratio
    Q = q_over_m * M

    def parts(r):
        u = 1.0 + k * r ** 2
        nu = np.sqrt(u) / (2.0 * a * r)
        mu, lam = M * nu, Q * nu
        Xi = (1.0 + mu) ** 2 - lam ** 2
        N = (1.0 - mu ** 2 + lam ** 2) / Xi
        Y = (N - k * r ** 2) / u
        R = a * r * Xi / u
        return N, Y, R

    def resid(r):
        N, Y, R = parts(r)
        return Y - H * R

    lo = 0.55
    hi = 1.7 if w <= 0 else min(1.7, 0.999 / np.sqrt(w))
    if w < 0:
        hi = min(hi, 0.999 / np.sqrt(-w))
    rg = np.linspace(lo, hi, ngrid)
    N, Y, R = parts(rg)
    vals = np.where(N > 0.0, Y - H * R, np.nan)
    idx = np.where(np.isfinite(vals[:-1]) & np.isfinite(vals[1:])
                   & (vals[:-1] * vals[1:] < 0.0))[0]
    if idx.size == 0:
        return np.nan
    i = idx[-1]
    root = brentq(resid, rg[i], rg[i + 1], xtol=1e-14, rtol=1e-14)
    return float(parts(root)[2])


def omega_of_w(w):
    w = np.asarray(w, dtype=float)
    return -4.0 * w / (1.0 - w) ** 2


def w_of_omega(om):
    om = np.asarray(om, dtype=float)
    return -om / (2.0 - om + 2.0 * np.sqrt(np.clip(1.0 - om, 0.0, None)))


def omega_axis(ax):
    top = ax.secondary_xaxis("top", functions=(omega_of_w, w_of_omega))
    top.set_xticks([0.0, -1.25, -3.0, -5.0])
    top.set_xticklabels(["0", r"$-5/4$", r"$-3$", r"$-5$"])
    top.set_xlabel(r"$\Omega_k$", labelpad=3)
    top.tick_params(direction="in", labelsize=8)
    return top


w = np.linspace(-0.05, 0.45, 1400)
DASH = (0, (4, 2))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.1, 2.65),
                                    gridspec_kw=dict(width_ratios=[1, 1, 0.92]))
fig.subplots_adjust(wspace=0.36, left=0.075, right=0.985, bottom=0.19, top=0.83)

# ------------------------------------------------------------------ panel (a)
ax1.axhline(0.0, color=GREY, lw=0.6)
ax1.axvspan(0.2, 1.0 / 3.0, color=SHADE, zorder=0)
ax1.plot(w, C_M(w), color=BLUE, label=r"of $M/R_{\rm H}$")
ax1.plot(w, C_Q(w), color=ORANGE, ls="--", label=r"of $(Q/R_{\rm H})^{2}$")
for x in (0.2, 1.0 / 3.0):
    ax1.axvline(x, color="0.65", lw=0.6, ls=":")
ax1.set_xlim(-0.05, 0.45)
ax1.set_ylim(-2.1, 1.4)
ax1.set_xlabel(r"$k r_{\rm H}^{2}$")
ax1.set_ylabel("coefficient")
ax1.legend(loc="lower right", handlelength=1.6, borderpad=0.2)
panel_label(ax1, "(a)")
omega_axis(ax1)
ax1.minorticks_on()
ax1.tick_params(top=False, which="both")

# ------------------------------------------------------------------ panel (b)
w_num = np.linspace(-0.05, 0.45, 101)
ax2.axhline(0.0, color=GREY, lw=0.6)
ax2.axvline(0.2, color="0.65", lw=0.6, ls=":")
ax2.plot(w, C_M(w), color="0.55", lw=1.1)
for qom, color in ((0.0, BLUE), (1.0, ORANGE)):
    num = [(numerical_cosmological_radius(wi, M_RATIO, qom) - 1.0) / M_RATIO for wi in w_num]
    ax2.plot(w_num, num, color=color, zorder=2)
    ax2.plot(w, second_order_shift_over_M(w, M_RATIO, qom), color=color,
             ls=DASH, lw=1.0, zorder=3)
ax2.set_xlim(-0.05, 0.45)
ax2.set_ylim(-2.1, 1.4)
ax2.set_xlabel(r"$k r_{\rm H}^{2}$")
ax2.set_ylabel(r"$(R_{\rm cos}-R_{\rm H})/M$")
key = [Line2D([], [], color="0.55", lw=1.1, label=r"$M/R_{\rm H}\to0$"),
       Line2D([], [], color=BLUE, label=r"$Q=0$"),
       Line2D([], [], color=ORANGE, label=r"$Q=M$"),
       Line2D([], [], color="0.3", ls=DASH, lw=1.0, label="second order")]
ax2.legend(handles=key, loc="lower right", handlelength=1.6, borderpad=0.2)
panel_label(ax2, "(b)")
omega_axis(ax2)
ax2.minorticks_on()
ax2.tick_params(top=False, which="both")

# ------------------------------------------------------------------ panel (c)
ms = np.geomspace(0.01, 0.30, 24)
cases = [(-0.05, BLUE, "-", r"$kr_{\rm H}^2=-0.05$"),
         (0.10, ORANGE, "--", r"$0.10$"),
         (0.30, GREEN, "-.", r"$0.30$")]
errs = {}
for wv, c, ls, lab in cases:
    e = []
    for m in ms:
        Rn = numerical_cosmological_radius(wv, m, 0.0)
        R2 = 1.0 + m * second_order_shift_over_M(wv, m, 0.0)
        e.append(abs(R2 - Rn))
    errs[wv] = np.array(e)
    ax3.plot(ms, e, color=c, ls=ls, label=lab)
# guide with slope 3
xg = np.array([0.02, 0.10])
ax3.plot(xg, 0.35 * xg ** 3, color="0.35", lw=0.8, ls=":")
ax3.text(0.045, 0.35 * 0.045 ** 3 * 1.75, r"$\propto (M/R_{\rm H})^{3}$", fontsize=7.5,
         color="0.35", rotation=32, ha="center", va="top")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlim(0.01, 0.30)
ax3.set_ylim(1e-8, 3e-2)
ax3.set_xlabel(r"$M/R_{\rm H}$")
ax3.set_ylabel(r"$|R_{\rm cos}^{(2)}-R_{\rm cos}|/R_{\rm H}$")
ax3.legend(loc="lower right", handlelength=1.6, borderpad=0.2, labelspacing=0.25,
           fontsize=7.5)
panel_label(ax3, "(c)")

fig.savefig("fig4_cosmological_horizon.pdf")
fig.savefig("fig4_cosmological_horizon.png", dpi=400)

# --------------------------------------------------------------- numbers for the text
for qom in (0.0, 1.0):
    z2 = brentq(lambda z: second_order_shift_over_M(z, M_RATIO, qom), 0.20, 0.28)
    zn = brentq(lambda z: numerical_cosmological_radius(z, M_RATIO, qom) - 1.0, 0.20, 0.30,
                xtol=1e-10)
    print(f"M/R_H={M_RATIO}, Q/M={qom:g}: zero of shift at kr_H^2 = {zn:.4f} (numerical), {z2:.4f} (second order)")
print("zero of the shift versus M/R_H (Q=0): numerical, second order")
for m in (0.02, 0.05, 0.10, 0.15):
    zn = brentq(lambda z: numerical_cosmological_radius(z, m, 0.0) - 1.0, 0.19, 0.30, xtol=1e-10)
    z2 = brentq(lambda z: second_order_shift_over_M(z, m, 0.0), 0.19, 0.30)
    print(f"   M/R_H={m}: {zn:.4f}  {z2:.4f}")
for wv in (-0.05, 0.0, 0.2, 0.45):
    line = [f"w={wv:+.2f}"]
    for m in (0.05, 0.15):
        for qom in (0.0, 1.0):
            Rn = numerical_cosmological_radius(wv, m, qom)
            R2 = 1.0 + m * second_order_shift_over_M(wv, m, qom)
            line.append(f"M/R_H={m} Q/M={qom:g}: rel.err R_CH={abs(R2 / Rn - 1.0):.2e}")
    print("  ".join(line))
# fitted slope of the error, all three cases
for wv, *_ in cases:
    sl = np.polyfit(np.log(ms[:12]), np.log(errs[wv][:12]), 1)[0]
    print(f"w={wv:+.2f}: fitted slope of the error at small M/R_H = {sl:.2f}")
