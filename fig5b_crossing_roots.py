"""Wall potentials that admit an equal-grand-potential crossing, against
eta = H^2 R_B^2, for the two roots of the crossing condition (Sec. V D).

First root:   |Phi_B| < Phi_+                       (eta < 1/3)
Second root:  Phi_- < |Phi_B| < 1/sqrt(1-eta)       (eta < 1/3)
              0 <= |Phi_B| < 1/sqrt(1-eta)          (1/3 < eta < 1)
with Phi_pm = sqrt(1-3 eta) / (2 sqrt(1-eta) +- sqrt(1+2 eta)).
"""
import numpy as np
import matplotlib.pyplot as plt
from mcvittie_figure_style import apply_style, BLUE, ORANGE, GREY

apply_style()


def phi_pm(eta, sign):
    return np.sqrt(1 - 3*eta)/(2*np.sqrt(1 - eta) + sign*np.sqrt(1 + 2*eta))


ETA_MAX, PHI_MAX = 0.80, 2.0
e1 = np.linspace(0.0, 1/3, 600)
e2 = np.linspace(0.0, ETA_MAX, 1200)
top = 1/np.sqrt(1 - e2)
low = np.where(e2 < 1/3, phi_pm(np.minimum(e2, 1/3), -1), 0.0)

fig, ax = plt.subplots(figsize=(3.35, 2.55))
ax.fill_between(e1, 0, phi_pm(e1, +1), color=BLUE, alpha=0.22, lw=0)
ax.fill_between(e2, low, top, color=ORANGE, alpha=0.20, lw=0)
ax.plot(e1, phi_pm(e1, +1), color=BLUE, label=r"$\Phi_+$")
ax.plot(e1, phi_pm(e1, -1), color=ORANGE, ls="--", label=r"$\Phi_-$")
ax.plot(e2, top, color=ORANGE, label=r"$1/\sqrt{1-\eta}$")
ax.axvline(1/3, color=GREY, ls=":", lw=0.8)
ax.plot([0, 0], [1/3, 1], ls="none", marker="o", ms=3.6, color="k", clip_on=False, zorder=5)

ax.text(0.045, 0.085, "first root", color=BLUE, fontsize=8)
ax.text(0.47, 0.62, "second root", color=ORANGE, fontsize=8)
ax.text(0.035, 0.62, "no crossing", color=GREY, fontsize=8)
ax.text(0.43, 1.72, "no crossing", color=GREY, fontsize=8)

ax.set_xlim(0, ETA_MAX)
ax.set_ylim(0, PHI_MAX)
ax.set_xticks([0, 1/6, 1/3, 0.5, 2/3])
ax.set_xticklabels(["0", "1/6", "1/3", "1/2", "2/3"])
ax.set_yticks([0, 1/3, 1, 2])
ax.set_yticklabels(["0", "1/3", "1", "2"])
ax.set_xlabel(r"$\eta=H^2R_{\rm B}^2$")
ax.set_ylabel(r"$|\Phi_{\rm B}|$")
ax.legend(loc="upper left", handlelength=1.8, borderaxespad=0.4)

fig.savefig("fig5b_crossing_roots.pdf")
fig.savefig("fig5b_crossing_roots.png", dpi=300)

# checks printed for the record
for eta in (0.1, 0.2, 0.3):
    print(f"eta={eta}: Phi_+={phi_pm(eta, +1):.6f}  Phi_-={phi_pm(eta, -1):.6f}  1/sqrt(1-eta)={1/np.sqrt(1-eta):.6f}")
