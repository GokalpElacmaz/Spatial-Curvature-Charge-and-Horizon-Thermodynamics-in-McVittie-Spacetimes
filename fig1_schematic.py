import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import brentq, minimize_scalar
from mcvittie_figure_style import (apply_style, panel_label,
                                   BLUE, ORANGE, BLACK, GREY, LIGHT_GREY)


apply_style()

# ----------------------------------------------------------------- parameters
M = a = 1.0
H = 0.08
KS = [-0.001, 0.0, 0.001]
RB = 5.0                                   # screen, at fixed areal radius

COL = {-0.001: BLUE, 0.0: BLACK, 0.001: ORANGE}
DASH = {-0.001: (0, (5, 2)), 0.0: "solid", 0.001: (0, (6, 1.6, 1.2, 1.6))}
LBL = {-0.001: r"$k<0$", 0.0: r"$k=0$", 0.001: r"$k>0$"}

MS_SPHERE = 4.6                            # marginal sphere on Y = HR
MS_MINUS = 4.4                             # marginal sphere on Y = -HR
MS_SCREEN = 4.3                            # screen marker
MS_LAPSE = 9.5                             # lapse-zero tick
FS_NOTE = 8.0                              # in-panel annotation size


# ------------------------------------------------------------------- geometry
def funcs(k):
    mu = lambda r: M * np.sqrt(1 + k * r**2) / (2 * a * r)
    N = lambda r: (1 - mu(r)) / (1 + mu(r))
    Y = lambda r: (N(r) - k * r**2) / (1 + k * r**2)
    R = lambda r: a * r * (1 + mu(r)) ** 2 / (1 + k * r**2)
    return mu, N, Y, R


def sign_changes(f, grid):
    g = f(grid)
    idx = np.where(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]
    return [brentq(f, grid[i], grid[i + 1], xtol=1e-14) for i in idx]


def surfaces(k):
    """Lapse-zero surface, roots of Y = HR and Y = -HR, screen, largest R."""
    mu, N, Y, R = funcs(k)
    r_lapse = M / np.sqrt(4 * a**2 - M**2 * k)          # mu = 1
    r_end = 1 / np.sqrt(-k) if k < 0 else np.inf        # end of the chart
    hi = r_end * (1 - 1e-10) if k < 0 else 900.0
    grid = np.linspace(r_lapse * (1 + 1e-10), hi, 700000)
    plus = sign_changes(lambda r: Y(r) - H * R(r), grid)
    minus = sign_changes(lambda r: Y(r) + H * R(r), grid)
    r_s, r_ch = plus[0], plus[-1]
    r_B = brentq(lambda r: R(r) - RB, r_s, r_ch)
    out = dict(r_lapse=r_lapse, r_s=r_s, r_ch=r_ch, r_B=r_B, r_end=r_end,
               R=R, r_minus=minus[0] if minus else None, r_max=None)
    if k > 0:
        out["r_max"] = minimize_scalar(lambda r: -R(r), bounds=(r_ch, 900.0),
                                       method="bounded").x
    return out


S = {k: surfaces(k) for k in KS}
assert S[-0.001]["r_minus"] is None and S[0.0]["r_minus"] is None
assert S[0.001]["r_ch"] < S[0.001]["r_max"] < S[0.001]["r_minus"]

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.00, 3.05),
                               gridspec_kw=dict(width_ratios=[1.34, 1.0]))

# =============================================================== panel (a)
# Schematic ordering. Horizontal positions carry no metric meaning. Only the
# order, and the fact that R_CH moves inward as k increases, are read off here.
POS = {-0.001: dict(lapse=0.06, s=0.16, B=0.42, ch=0.72, end=0.90),
       0.0:    dict(lapse=0.06, s=0.16, B=0.42, ch=0.61, end=0.93),
       0.001:  dict(lapse=0.06, s=0.16, B=0.42, ch=0.53, minus=0.76,
                    end=0.93)}
HB = 0.155                                             # half-height of a row

ax0.set_xlim(-0.12, 1.58)
ax0.set_ylim(-0.98, 3.05)
ax0.axis("off")

# the cosmological sphere moves inward as k increases
ax0.plot([POS[k]["ch"] for k in KS], [2, 1, 0], ls=(0, (1, 2.2)), lw=0.8,
         color=LIGHT_GREY, zorder=1)

for j, k in enumerate(KS):
    y, p, c = 2 - j, POS[k], COL[k]
    ax0.plot([p["lapse"], p["end"]], [y, y], color=c, lw=1.3, zorder=3,
             solid_capstyle="butt")
    ax0.fill_betweenx([y - HB, y + HB], -0.03, p["lapse"],
                      facecolor="white", edgecolor=GREY, hatch="/////",
                      lw=0.6, zorder=1)
    ax0.fill_betweenx([y - HB, y + HB], p["s"], p["ch"], color=c, alpha=0.15,
                      lw=0, zorder=2)
    ax0.plot([p["lapse"]], [y], marker="|", ms=MS_LAPSE, mew=1.5, color=GREY,
             zorder=5)
    ax0.plot([p["s"], p["ch"]], [y, y], "o", ms=MS_SPHERE, color=c,
             mec="white", mew=0.6, zorder=5)
    ax0.plot([p["B"]], [y], "s", ms=MS_SCREEN, mfc="white", mec=c, mew=1.2,
             zorder=5)
    ax0.text(-0.07, y, LBL[k], color=c, ha="right", va="center", fontsize=9)

    if "minus" in p:
        # Y = -HR sphere beyond the largest sphere, untrapped again beyond it.
        ax0.fill_betweenx([y - HB, y + HB], p["minus"], p["end"], color=c,
                          alpha=0.15, lw=0, zorder=2)
        ax0.plot([p["minus"]], [y], "D", ms=MS_MINUS, color=c, mec="white",
                 mew=0.6, zorder=5)
        ax0.annotate(r"$Y=-HR$", (p["minus"], y), xytext=(0, -7),
                     ha="center", va="top", textcoords="offset points",
                     fontsize=FS_NOTE, color=c)

    if k < 0:      # chart closes at finite r, areal radius diverges there
        ax0.plot([p["end"]], [y], marker="|", ms=MS_LAPSE, mew=1.5, color=c,
                 zorder=5)
        ax0.text(p["end"] + 0.05, y, "spatial infinity,\n" + r"$R\to\infty$",
                 color=c, fontsize=FS_NOTE, va="center", linespacing=1.35)
    else:
        ax0.annotate("", xy=(p["end"] + 0.06, y), xytext=(p["end"], y),
                     arrowprops=dict(arrowstyle="-|>", lw=1.3, color=c,
                                     mutation_scale=9))
        txt = ("spatial infinity,\n" + r"$R\to\infty$") if k == 0 else \
              ("regular center,\n" + r"$R\to0$")
        ax0.text(p["end"] + 0.10, y, txt, color=c, fontsize=FS_NOTE,
                 va="center", linespacing=1.35)

# surface labels, hung off the top row
p = POS[-0.001]
for key, lab, dx in (("s", r"$R_{\rm s}$", 9),
                     ("B", r"$R_{\rm B}$", 0),
                     ("ch", r"$R_{\rm CH}$", 0)):
    ax0.annotate(lab, (p[key], 2), xytext=(dx, 13), ha="center",
                 textcoords="offset points", fontsize=FS_NOTE, color=BLACK,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=LIGHT_GREY,
                                 shrinkA=0.5, shrinkB=3))
ax0.annotate(r"$N=0$", (p["lapse"], 2), xytext=(-3, 13), ha="center",
             textcoords="offset points", fontsize=FS_NOTE, color=BLACK)
ax0.annotate("", xy=(0.60, -0.50), xytext=(0.20, -0.50),
             arrowprops=dict(arrowstyle="-|>", lw=0.8, color=GREY,
                             mutation_scale=8))
ax0.text(0.40, -0.68, "radial direction, not to scale", fontsize=FS_NOTE,
         color=GREY, ha="center", va="top")
panel_label(ax0, "(a)")

# =============================================================== panel (b)
XMIN, XMAX = 0.44, 700.0

ax1.fill_betweenx([1e-3, 1e4], XMIN, S[0.0]["r_lapse"],
                  facecolor="white", edgecolor=GREY, hatch="/////", lw=0,
                  zorder=0)
ax1.axhline(RB, color=GREY, lw=0.6, ls=(0, (1, 2)), zorder=1)

for k in KS:
    d, c = S[k], COL[k]
    right = d["r_end"] * (1 - 1e-7) if k < 0 else XMAX
    rr = np.geomspace(d["r_lapse"] * (1 + 1e-12), min(right, XMAX), 6000)
    ax1.plot(rr, d["R"](rr), color=c, ls=DASH[k], label=LBL[k], zorder=3)
    ax1.plot([d["r_s"], d["r_ch"]], [d["R"](d["r_s"]), d["R"](d["r_ch"])],
             "o", ms=MS_SPHERE, color=c, mec="white", mew=0.6, zorder=5)
    ax1.plot([d["r_B"]], [RB], "s", ms=MS_SCREEN, mfc="white", mec=c, mew=1.2,
             zorder=6)
    ax1.plot([d["r_lapse"]], [d["R"](d["r_lapse"])], marker="|", ms=8, mew=1.4,
             color=GREY, zorder=5)
    if d["r_minus"] is not None:
        ax1.plot([d["r_minus"]], [d["R"](d["r_minus"])], "D", ms=MS_MINUS,
                 color=c, mec="white", mew=0.6, zorder=5)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(XMIN, XMAX)
ax1.set_ylim(0.07, 700)
ax1.set_xlabel(r"isotropic radius $r$")
ax1.set_ylabel(r"areal radius $R$")
ax1.legend(loc="lower left", handlelength=2.1, labelspacing=0.3,
           borderaxespad=0.7)
panel_label(ax1, "(b)")

dp = S[0.001]
ax1.annotate(r"$R$ largest", (dp["r_max"], dp["R"](dp["r_max"])),
             xytext=(17, 6), ha="left", textcoords="offset points",
             fontsize=FS_NOTE, color=ORANGE,
             arrowprops=dict(arrowstyle="-", lw=0.6, color=ORANGE,
                             shrinkA=1, shrinkB=3))
ax1.annotate(r"$Y=-HR$", (dp["r_minus"], dp["R"](dp["r_minus"])),
             xytext=(-14, -34), ha="right", textcoords="offset points",
             fontsize=FS_NOTE, color=ORANGE,
             arrowprops=dict(arrowstyle="-", lw=0.6, color=ORANGE,
                             shrinkA=1, shrinkB=3))
ax1.annotate(r"$R\to0$", (330, dp["R"](330)), xytext=(-3, -23),
             ha="center", textcoords="offset points", fontsize=FS_NOTE,
             color=ORANGE,
             arrowprops=dict(arrowstyle="-", lw=0.6, color=ORANGE,
                             shrinkA=1, shrinkB=3))
ax1.annotate(r"$R\to\infty$", (S[-0.001]["r_end"], 330), xytext=(17, -2),
             ha="left", va="center", textcoords="offset points",
             fontsize=FS_NOTE, color=BLUE,
             arrowprops=dict(arrowstyle="-", lw=0.6, color=BLUE,
                             shrinkA=1, shrinkB=2))
ax1.text(0.62, RB * 1.35, r"$R=R_{\rm B}$", fontsize=FS_NOTE, color=GREY,
         va="bottom", ha="left")

# ------------------------------------------------------------------ key
key = [Line2D([], [], ls="", marker="|", ms=MS_LAPSE, mew=1.5, color=GREY,
              label=r"lapse vanishes, $\mu=1$"),
       Line2D([], [], ls="", marker="o", ms=MS_SPHERE, color=GREY,
              label=r"marginal spheres, $Y=HR$"),
       Line2D([], [], ls="", marker="D", ms=MS_MINUS, color=GREY,
              label=r"marginal sphere, $Y=-HR$"),
       Line2D([], [], ls="", marker="s", ms=MS_SCREEN, mfc="white", mec=GREY,
              label="screen"),
       Patch(facecolor=GREY, alpha=0.16, edgecolor="none",
             label=r"untrapped, $\chi>0$"),
       Patch(facecolor="white", edgecolor=LIGHT_GREY, lw=0.6,
             label=r"anti-trapped, $\chi<0$"),
       Patch(facecolor="white", edgecolor=GREY, hatch="/////", lw=0.6,
             label=r"outside the region $N>0$")]
fig.legend(handles=key, loc="lower center", bbox_to_anchor=(0.5, -0.075),
           ncol=4, handletextpad=0.45, columnspacing=1.1, labelspacing=0.5)

fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.25, wspace=0.20)
fig.savefig("./fig1_schematic.pdf")
fig.savefig("./fig1_schematic.png", dpi=220)

for k in KS:
    d = S[k]
    line = (f"k={k:+.4f}  r_lapse={d['r_lapse']:.4f}  "
            f"r_s={d['r_s']:.4f} (R={d['R'](d['r_s']):.4f})  "
            f"r_B={d['r_B']:.4f} (R={RB:.1f})  "
            f"r_CH={d['r_ch']:.4f} (R={d['R'](d['r_ch']):.4f})")
    if d["r_minus"] is not None:
        line += (f"  r_max={d['r_max']:.3f} (R={d['R'](d['r_max']):.3f})"
                 f"  r_(Y=-HR)={d['r_minus']:.3f}"
                 f" (R={d['R'](d['r_minus']):.4f})")
    print(line)
