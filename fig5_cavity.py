import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mcvittie_figure_style import (apply_style, panel_label,
                                   BLUE, ORANGE, GREEN, BLACK, GREY, SHADE)

apply_style()

Y_MIN_T = 2.0 / 3.0          # minimum screen temperature, maximum Delta Omega
Y_CROSS = 8.0 / 9.0          # Delta Omega_B = 0
DOM_MAX = 1.0 - np.sqrt(3.0) / 2.0
ETA = 0.05                   # sample de Sitter strength used for the shaded strip


def domega(y):
    return 1.0 - (1.0 - 0.75 * y) / np.sqrt(1.0 - y)


def temp(y, phi):
    return (1.0 - phi ** 2) ** 2 / (4.0 * np.pi * y * np.sqrt(1.0 - y))


def X0(phi):
    return 8.0 / (9.0 * (1.0 - phi ** 2))


def tau_H(phi):
    x = X0(phi)
    return (x + 2.0) * (x - 1.0) / 4.0


PHIS = [0.0, 0.15, 0.25, 1.0 / 3.0]
LABELS = [r"$\Phi_{\rm B}=0$", r"$0.15$", r"$0.25$", r"$1/3$ limit"]
COLORS = [BLACK, BLUE, ORANGE, GREEN]
STYLES = ["-", "--", "-.", ":"]
XTICKS = [0.0, 1.0 / 3.0, 2.0 / 3.0, 8.0 / 9.0]
XLABS = ["0", "1/3", "2/3", "8/9"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.1, 2.35),
                                    gridspec_kw=dict(width_ratios=[1.0, 1.0, 0.86]))
fig.subplots_adjust(wspace=0.40, left=0.075, right=0.985, bottom=0.20, top=0.97)

# ------------------------------------------------------------------ panel (a)
ax1.axvline(Y_MIN_T, color=GREY, lw=0.6, ls=":")
for phi, lab, c, ls in zip(PHIS, LABELS, COLORS, STYLES):
    ymax = 1.0 - phi ** 2 - 1e-4
    yy = np.linspace(1e-4, ymax, 6000)
    ax1.plot(yy, temp(yy, phi), color=c, ls=ls, label=lab)
    ax1.plot(Y_MIN_T, temp(Y_MIN_T, phi), "o", ms=3.4, color=c, zorder=5)
    if np.isclose(phi, 1.0 / 3.0):
        ax1.plot(Y_CROSS, temp(Y_CROSS, phi), "x", ms=5.0, mew=1.1, color=c,
                 zorder=6, clip_on=False)
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.14, 0.50)
ax1.set_yticks([0.2, 0.3, 0.4, 0.5])
ax1.set_xticks(XTICKS)
ax1.set_xticklabels(XLABS)
ax1.set_xlabel(r"$y=(1-\Phi_{\rm B}^{2})R_A/R_{\rm B}$")
ax1.set_ylabel(r"$T_{\rm B}R_{\rm B}$")
ax1.legend(loc="upper center", handlelength=1.6, labelspacing=0.28, borderpad=0.2,
           fontsize=7.5)
panel_label(ax1, "(a)", x=0.80)

# ------------------------------------------------------------------ panel (b)
ax2.axhline(0.0, color=GREY, lw=0.6, ls=":")
for phi, c, ls in zip(PHIS, COLORS, STYLES):
    ymax = 1.0 - phi ** 2 - 1e-4
    ysmall = np.linspace(1e-4, Y_MIN_T, 3000)
    ax2.plot(temp(ysmall, phi), domega(ysmall), color=c, ls=ls, alpha=0.32)
    ylarge = np.linspace(Y_MIN_T, ymax, 3000)
    ax2.plot(temp(ylarge, phi), domega(ylarge), color=c, ls=ls)
    ax2.plot(temp(Y_MIN_T, phi), DOM_MAX, "o", ms=3.4, color=c, zorder=5)
    if phi < 1.0 / 3.0 - 1e-12:
        ax2.plot(temp(Y_CROSS, phi), 0.0, "o", ms=4.2, mfc="white", mec=c, mew=1.0,
                 zorder=5)
    else:
        ax2.plot(temp(Y_CROSS, phi), 0.0, "x", ms=5.0, mew=1.1, color=c, zorder=6)
ax2.set_xlim(0.145, 0.345)
ax2.set_ylim(-0.20, 0.19)
ax2.set_yticks([-0.2, -0.1, 0.0, 0.1])
ax2.set_xticks([0.15, 0.20, 0.25, 0.30])
ax2.set_xlabel(r"$T_{\rm B}R_{\rm B}$")
ax2.set_ylabel(r"$\Delta\Omega_{\rm B}/R_{\rm B}$", labelpad=-2)
branch = [Line2D([], [], color="0.25", lw=1.3, label="large black hole"),
          Line2D([], [], color="0.25", lw=1.3, alpha=0.32, label="small black hole")]
ax2.legend(handles=branch, loc="lower left", handlelength=1.6, labelspacing=0.28,
           borderpad=0.2, fontsize=7.5)
panel_label(ax2, "(b)")

# ------------------------------------------------------------------ panel (c)
phi = np.linspace(0.0, 1.0 / 3.0, 800)
ax3.axhline(0.0, color=GREY, lw=0.6)
ax3.axvspan(1.0 / 3.0 - ETA / 2.0, 1.0 / 3.0, color=SHADE, zorder=0)
ax3.text(1.0 / 3.0 - ETA / 4.0, -0.045, r"excluded, $\eta=0.05$", rotation=90,
         fontsize=6.8, color="0.35", ha="center", va="center")
ax3.plot(phi, tau_H(phi), color=BLUE)
ax3.plot(0.0, -13.0 / 162.0, "o", ms=4.0, color=BLUE, zorder=5)
ax3.plot(1.0 / 3.0, 0.0, "o", ms=4.2, mfc="white", mec=BLUE, mew=1.0, zorder=6,
         clip_on=False)
ax3.annotate(r"$-13/162$", xy=(0.0, -13.0 / 162.0), xytext=(0.03, -0.064), fontsize=8)
ax3.set_xlim(0.0, 1.0 / 3.0)
ax3.set_ylim(-0.09, 0.012)
ax3.set_xticks([0.0, 0.1, 0.2, 1.0 / 3.0])
ax3.set_xticklabels(["0", "0.1", "0.2", "1/3"])
ax3.set_xlabel(r"$\Phi_{\rm B}$")
ax3.set_ylabel(r"$\tau_H$")
panel_label(ax3, "(c)", x=0.05)

for ax in (ax1, ax2, ax3):
    ax.minorticks_on()

fig.savefig("fig5_cavity.pdf")
fig.savefig("fig5_cavity.png", dpi=400)

print(f"Delta Omega_B/R_B at y=2/3: {domega(Y_MIN_T):.6f} (1-sqrt3/2 = {DOM_MAX:.6f})")
print(f"Delta Omega_B at y=8/9: {domega(Y_CROSS):.2e}")
print(f"tau_H(0) = {tau_H(0.0):.6f}   (-13/162 = {-13.0/162.0:.6f})")
for p in PHIS:
    print(f"Phi_B={p:.3f}: T_min R_B={temp(Y_MIN_T, p):.5f}, T_cross R_B={temp(Y_CROSS, p):.5f}, "
          f"ratio={temp(Y_MIN_T, p)/temp(Y_CROSS, p):.6f} (4sqrt3/9={4*np.sqrt(3)/9:.6f})")
