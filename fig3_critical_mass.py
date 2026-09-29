import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from mcvittie_figure_style import apply_style, panel_label, GREY, SHADE

apply_style()


# mu_-(w) = [2 - 2w - w^2 - sqrt(3 - 12w - 6w^2)] / (1+w)^2 is 0/0 at w = -1.
# The stable form is mu_-(w) = (1+w)^2 / [A(w) + sqrt(B(w))].
def mu_minus(w):
    A = 2.0 - 2.0 * w - w * w
    B = 3.0 - 12.0 * w - 6.0 * w * w
    return (1.0 + w) ** 2 / (A + np.sqrt(B))


def McH(w):
    u = 1.0 + w
    mu = mu_minus(w)
    return u ** 1.5 * ((1.0 - w) - u * mu) / (3.0 * (1.0 + mu))


def ell_merger(w):
    mu = mu_minus(w)
    return w * mu * mu / (1.0 + w)


def omega_k(w):
    """Omega_k = -16 ell / eps^2 with eps = M_c H."""
    return -16.0 * ell_merger(w) / McH(w) ** 2


W_RHO = brentq(lambda z: omega_k(z) - 1.0, -0.2, -0.01)   # Omega_k = 1
FLAT = 1.0 / (3.0 * np.sqrt(3.0))
W_END = 0.2
W_MIN = -1.0
CURVE = "#12395f"

w_all = np.linspace(W_MIN, W_END, 4001)
i_max = np.argmax(McH(w_all))
w_pk, y_pk = w_all[i_max], McH(w_all)[i_max]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6),
                               gridspec_kw=dict(width_ratios=[1.0, 1.0]))
fig.subplots_adjust(wspace=0.26)

# ------------------------------------------------------------------ panel (a)
ax1.axvspan(W_MIN, W_RHO, color=SHADE, lw=0, zorder=0)
ax1.text(0.5 * (W_MIN + W_RHO) - 0.02, 0.012, "FLRW density $<0$", ha="center",
         va="bottom", fontsize=7.5, color="0.35")
ax1.axvline(0.0, color="0.85", lw=0.6, zorder=0)
ax1.plot([-0.62, W_END], [FLAT, FLAT], ls=(0, (4, 2.5)), color=GREY, lw=0.8, zorder=1)
ax1.plot(w_all, McH(w_all), color=CURVE, lw=1.6, solid_capstyle="round", zorder=3)
ax1.plot(0.0, FLAT, "o", ms=4.2, color=CURVE, zorder=4)
ax1.plot(w_pk, y_pk, "o", ms=3.6, color=CURVE, zorder=4)
ax1.annotate(rf"peak {y_pk:.3f}", xy=(w_pk, y_pk), xytext=(-4, 6),
             textcoords="offset points", ha="right", va="bottom", fontsize=7.5,
             color="0.25")
ax1.annotate(r"$1/(3\sqrt{3})$", xy=(-0.62, FLAT), xytext=(-3, 0),
             textcoords="offset points", ha="right", va="center", color=GREY, fontsize=8)
for w_end in (W_MIN, W_END):
    ax1.plot(w_end, 0.0, "o", ms=4.2, mfc="white", mec=CURVE, mew=1.1, zorder=5,
             clip_on=False)
ax1.set_xlim(W_MIN, W_END)
ax1.set_ylim(0.0, 0.26)
ax1.set_xlabel(r"$k\,r_A^{2}$")
ax1.set_ylabel(r"$M_c H$")
ax1.set_xticks([-1.0, -0.5, 0.0, 0.2])
ax1.set_xticklabels([r"$-1$", r"$-0.5$", r"$0$", r"$1/5$"])
ax1.minorticks_on()
panel_label(ax1, "(a)")

# ------------------------------------------------------------------ panel (b)
WL = -0.06
wz = np.linspace(WL, W_END, 3001)
ax2.axvspan(WL, W_RHO, color=SHADE, lw=0, zorder=0)
ax2.axvline(0.0, color="0.85", lw=0.6, zorder=0)
ax2.plot([WL, W_END], [FLAT, FLAT], ls=(0, (4, 2.5)), color=GREY, lw=0.8, zorder=1)
ax2.plot(wz, McH(wz), color=CURVE, lw=1.6, solid_capstyle="round", zorder=3)
ax2.plot(0.0, FLAT, "o", ms=4.2, color=CURVE, zorder=4)
ax2.plot(W_END, 0.0, "o", ms=4.2, mfc="white", mec=CURVE, mew=1.1, zorder=5,
         clip_on=False)
ax2.text(0.195, FLAT + 0.004, r"$1/(3\sqrt{3})$", ha="right", va="bottom",
         color=GREY, fontsize=8)

# Omega_k labels: the boundary of non-negative density and three closed universes.
ax2.plot(W_RHO, McH(W_RHO), "o", ms=2.8, color=CURVE, zorder=4)
ax2.annotate(r"$\Omega_k=1$", xy=(W_RHO, McH(W_RHO)), xytext=(4, 5),
             textcoords="offset points", ha="left", va="bottom", fontsize=7, color="0.25")
for om in (-3.0, -10.0, -100.0):
    w_om = brentq(lambda z: omega_k(z) - om, 1e-4, 0.1999)
    ax2.plot(w_om, McH(w_om), "o", ms=2.8, color=CURVE, zorder=4)
    ax2.annotate(rf"$\Omega_k={om:g}$", xy=(w_om, McH(w_om)), xytext=(-10, -4),
                 textcoords="offset points", ha="right", va="center", fontsize=7, color="0.25")
ax2.set_xlim(WL, W_END)
ax2.set_ylim(0.0, 0.26)
ax2.set_xlabel(r"$k\,r_A^{2}$")
ax2.set_xticks([-0.05, 0.0, 0.05, 0.10, 0.15, 0.2])
ax2.set_xticklabels([r"$-0.05$", r"$0$", r"$0.05$", r"$0.10$", r"$0.15$", r"$1/5$"])
ax2.minorticks_on()
panel_label(ax2, "(b)", y=0.14)

fig.savefig("fig3_critical_mass.pdf")
fig.savefig("fig3_critical_mass.png", dpi=400)

print(f"M_c H (k=0)      = {McH(0.0):.8f}   [1/(3 sqrt3) = {FLAT:.8f}]")
print(f"peak M_c H       = {y_pk:.6f} at k r_A^2 = {w_pk:.4f}")
print(f"Omega_k = 1 at k r_A^2 = {W_RHO:.4f}")
