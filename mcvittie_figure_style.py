
import matplotlib as mpl

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
BLACK = "#000000"
GREY = "0.48"
LIGHT_GREY = "0.61"
SHADE = "0.90"


def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, text, x=0.04, y=0.94, **kw):
    """Panel label in axes coordinates. Extra keywords go to ax.text, for
    example a white bbox when the label sits on top of data."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, **kw)