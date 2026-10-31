"""The paper's figures and tables, built from the runs under logs/.

Usage:
    python scripts/report_paper.py
    python scripts/report_paper.py --tasks image --overwrite
    python scripts/report_paper.py --no-panels
    python scripts/report_paper.py --tasks bound

Writes results/, which holds the paper's figures and tables and nothing else:
    legend.{pdf,pgf}                 the models' legend, shared by every figure,
                                     and super_resolution/legend.{pdf,pgf} the
                                     same with that task's bicubic level
    <task>/convergence.{pdf,pgf}     quality against training time, for FUTON and
                                     the strongest model of each other family
    <task>/training_time.{pdf,pgf}   every model's final quality against its
                                     training time
    <task>/throughput.{pdf,pgf}      the same against its inference throughput
    <task>/training_memory.{pdf,pgf}   the same against its peak training memory
    <task>/inference_memory.{pdf,pgf}  and against its peak inference memory
    <task>/table.{tex,md}            every model's size, training time and peak
                                     memory, inference throughput and peak memory,
                                     and final metrics, the averages with a paired
                                     standard error: occupancy adds every shape's
                                     IoU, and NeRF every scene's metrics under a
                                     super column
    super_resolution/table_held_out.{tex,md}  the table without the three
                                     validation images the protocol was
                                     developed on
    <task>/qualitative_<signal>.pdf  the signal with two regions boxed, and those
                                     regions magnified for the ground truth and
                                     every featured model (for super-resolution,
                                     bicubic interpolation in place of TensoRF)
    <task>/panels/<signal>/          the renders those magnifications are cut from,
                                     rebuilt from the checkpoints and kept, so that
                                     recomposing a figure needs no GPU
    ablation/basis.{tex,md,pdf,pgf}  every basis at the benchmark's size, as a
                                     table and as PSNR against training time
    ablation/bandwidth.{tex,md,pdf,pgf}  the cosine basis's relative bandwidth
                                     kappa = K / N at the benchmark's size, as a
                                     table and as PSNR against kappa
    ablation/rank.{tex,md,pdf,pgf}   its rank at the benchmark's kappa = 1/2,
                                     beyond the budget, likewise against the rank
    ablation/combiner.{tex,md,pdf,pgf}  the tensor-ring and Tucker combiners
                                     against CP, for the cosine basis, as a table
                                     and as PSNR against training time
    ablation/decoder.{tex,md,pdf,pgf}  the MLP decoder's width and the
                                     linear decoder, likewise.
                                     Every ablation table has the benchmark's
                                     columns, size, costs and every metric of
                                     the image task, a row per model
    bound/gap.{pdf,pgf}              the error bound of FUTON with a linear
                                     decoder, from scripts/futon_bound.py: PSNR
                                     above Adam's at K = N/2 per rank, of the
                                     ceiling and the certified FUTON in both
                                     forms, and the region between them where
                                     the best FUTON lies
    bound/decomposition.{pdf,pgf}    Adam's error at K = N/2 split as the bound
                                     splits it, per rank
    bound/cost.{pdf,pgf}             seconds for the bound at every rank of a K,
                                     in either form, against the Adam runs
    bound/table.{tex,md}             the means per setting, the smallest setting
                                     certified for a target PSNR, the SVD form
                                     and the optimizer checks
    tuning/<task>.{tex,md}           every model's score at each learning rate
                                     tried, by which its rate was chosen, the
                                     best in bold

Legends sit outside the data: a row of benchmark panels shares one legend
strip below it, and a panel of its own carries a frameless legend above its
axes, in as many columns as its width takes.

Everything is sized for a NeurIPS page, 5.5 inches of 10pt Times: a panel is a
third of the text width, a strip of magnifications spans it, and the image and
super-resolution tables fit it in small type. Ticks fall on whole or half
numbers, and a band spans one standard error either side, as a table's ± does.

--tasks picks what to build, of the four tasks, ablation, bound and tuning; a
run builds them all by default.

Tables mark the best value in bold and the second underlined. Drawing the panels
needs the signals in data/ and a GPU; --no-panels leaves them out, and
--overwrite draws them again.

EXAMPLES lists the signals a task shows. A scene's view and a signal's magnified
regions are set in VIEWS and CENTERS where given, and found otherwise: the view
at the median of FUTON's gain over the strongest other model shown, and the
regions where the models shown disagree most on detail, whoever is closest
there. SIDES sets how large the regions are. Panels already drawn are reused,
so drawing a figure again costs seconds and no GPU.
"""

import argparse
import json
import math
import re
import statistics
import warnings
from collections.abc import Sequence
from functools import cache
from itertools import groupby
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from matplotlib.axis import Axis
from torch import Tensor
from matplotlib.backends.backend_pgf import LatexError
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import (
    FixedLocator,
    Locator,
    NullFormatter,
    NullLocator,
    StrMethodFormatter,
)
from PIL import Image
from futon_bound import LONGER, STEPS, SWEEPS, of_basis, psnr
from train_common import ROOT, build, fill_rank, merge, resolve
from train_super_resolution import INTERPOLATIONS, low_resolution

import inro

# Header, whether larger is better, the factor the stored value is shown in,
# and the decimals of a table entry.
METRICS = {
    "psnr": ("PSNR (dB)", True, 1, 2),
    "ssim": ("SSIM", True, 1, 3),
    "lpips": ("LPIPS", False, 1, 3),
    "iou": ("IoU (%)", True, 100, 2),
}
SPEED = {  # the unit of each task's inference throughput
    "image": "MPix/s",
    "occupancy": "MVox/s",
    "nerf": "FPS",
    "super_resolution": "MPix/s",
}
SIGNALS = {  # where a task's signal lives
    "image": "data/Kodak/{}.png",
    "occupancy": "data/occupancy/{}.ply",
    "nerf": "data/nerf/blender/{}",
    "super_resolution": "data/DIV2K/DIV2K_valid_HR/{}.png",
}
# Per task, the metric plotted, its axis label, and the metrics tabulated.
TASKS = {
    "image": ("psnr", "PSNR (dB)", ("psnr", "ssim", "lpips")),
    "occupancy": ("iou", "IoU (%)", ("iou",)),
    "nerf": ("psnr", "PSNR (dB)", ("psnr", "ssim", "lpips")),
    "super_resolution": ("psnr", "PSNR (dB)", ("psnr", "ssim", "lpips")),
}
# The validation images super-resolution's protocol was developed on, which a
# second table leaves out.
DEVELOPED_ON = ("0801", "0804", "0882")
EXAMPLES = {  # the signals each task's qualitative figures show
    "image": ("kodim19", "kodim01"),
    "occupancy": ("lucy", "armadillo"),
    "nerf": ("lego", "hotdog"),
    # Test images, none of people, on which both FUTONs lead every other model.
    "super_resolution": ("0880", "0887"),
}
# The test view a scene is drawn from, where its regions are set by hand; any
# other scene takes the view at the median of FUTON's gain (see example_view).
VIEWS = {"lego": 175, "hotdog": 135}
# Centres set by hand, one per magnified region, as fractions of the trimmed
# reference; any other signal's regions are found (see regions). The scenes'
# sit on parts of the object where the models disagree, rather than on the
# plate's rim or the floor, which the search alone would take; the volumes'
# where both FUTONs stay below every other model's error.
CENTERS: dict[str, tuple[tuple[float, float], ...]] = {
    "kodim19": ((0.67, 0.22), (0.25, 0.67)),
    "kodim01": ((0.906, 0.106), (0.606, 0.327)),  # a window, the stone wall
    "lucy": ((0.669, 0.237), (0.555, 0.859)),  # the shoulder under the torch, the hem
    "armadillo": ((0.464, 0.204), (0.279, 0.339)),  # the face, the upper arm
    # The front wheel in its track and the cab with its light.
    "lego": ((0.480, 0.772), (0.801, 0.347)),
    # The bun's top and the mustard on the plate's rim, where the other fields
    # leave blotches, holes and broken edges.
    "hotdog": ((0.505, 0.27), (0.46, 0.85)),
    # Super-resolution's where both FUTONs stay furthest below every other
    # model's error, among the squares of most detail; on the owl's body for
    # 0880, where both lead every baseline in 99 % of the squares: centred on an
    # eye, and on a wing's dark feathers, where the baselines fall furthest behind.
    "0880": ((0.445, 0.234), (0.299, 0.731)),  # the eye on the left, the wing
    "0887": ((0.438, 0.532), (0.214, 0.558)),  # the windows, a balcony over the fields
}
# The tasks whose figures add each model's |error| beneath its magnifications:
# images, volumes and super-resolution, whose best reconstructions differ too
# little to see in the magnifications alone. A figure's maps share one scale,
# so that they compare, saturating at the 90th percentile of every error they
# show, rounded up to a round number; a super-resolution image's errors span a
# decade.
ERRORS = ("image", "occupancy", "super_resolution")
ERROR_MAP = "magma"  # perceptually uniform, black where a model is exact
UP = {"lucy": 2}  # the Stanford Lucy is z up; the other four shapes are y up

# Categorical hues in a fixed order, and a light-to-dark blue ramp for ordered
# levels. Four hues are as many as stay apart under every kind of colour
# blindness, so a hue names a family and the line style names the member.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
RAMP = ("#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b")
# FUTON and the baselines, by family: periodic activations (SIREN and FINER,
# which is SIREN with a variable-periodic one), hash grids (Instant-NGP) and
# grid factorizations (TensoRF). A hue names the family and the line style the
# member, the stronger of a pair solid, and for FUTON the cosine basis.
# Colour, dashes, and a filled marker.
FEATURED = {
    "SIREN": (PALETTE[1], (4, 2), False),
    "FINER": (PALETTE[1], "", True),
    "Instant-NGP": (PALETTE[2], "", True),
    "TensoRF": (PALETTE[3], "", True),
    "FUTON-sinc": (PALETTE[0], (4, 2), False),
    "FUTON-cosine": (PALETTE[0], "", True),
}
OTHERS = "0.7"  # gray of every other model
# The models a task's qualitative strip magnifies, where they differ from the
# featured ones: super-resolution shows the bicubic reference, which every
# model is measured against, in place of TensoRF, the weakest of them there.
STRIP = {
    "super_resolution": (
        "Bicubic",
        "SIREN",
        "FINER",
        "Instant-NGP",
        "FUTON-sinc",
        "FUTON-cosine",
    ),
}
# TensoRF is its default variant for the dimension: CP in 2D, where it is the
# only one, and VM in 3D, the one its authors recommend.
TENSORF = {
    "image": "TensoRF",
    "occupancy": "TensoRF-VM",
    "nerf": "TensoRF-VM",
    "super_resolution": "TensoRF",
}
# Super-resolution alone has untrained baselines: interpolations, drawn as the
# dotted level of the standard one rather than as points, and Deep Image Prior.
REFERENCE = "Bicubic"
ROWS = (
    "Nearest", "Bilinear", "Bicubic", "DIP",
    "RFF", "PE-MLP", "MFN", "SIREN", "FreSh", "Gauss", "WIRE", "FINER",
    "Instant-NGP", "TensoRF", "GA-Planes", "F-INR", "FUTON-sinc", "FUTON-cosine",
    "FUTON-lanczos",
)  # fmt: skip
# How the results name a model where one setting tells it from the other
# variants of its kind, the setting in parentheses: FUTON's basis, and the
# backbone F-INR and FreSh build on. TensoRF keeps its name, its variant being
# fixed by the dimension (see TENSORF). Lanczos, a person's name, is
# capitalized; configs, logs and code keep the short names.
DISPLAY = {
    "FreSh": "FreSh (SIREN)",
    "F-INR": "F-INR (SIREN)",
    "FUTON-sinc": "FUTON (sinc)",
    "FUTON-cosine": "FUTON (cosine)",
    "FUTON-lanczos": "FUTON (Lanczos)",
}
# Metric axes are linear, which needs no explaining; IoU saturates, so its
# curves start here instead, which the ticks show plainly.
FLOOR = {"occupancy": 99.0}
# Where a task's quality axis starts, whatever the models reach: a model below
# it is drawn as a marker on the axis line.
BOTTOM = {"super_resolution": 26.0}
TARGET = 99.8  # IoU (%) whose first time the text quotes
# NeurIPS's page is 5.5 inches wide and set in 10pt Times. Every figure is
# drawn at the size it is printed, so that its text keeps its size there.
WIDTH = 5.5  # inches, the text width, which a strip of magnifications spans
SIZE = (1.75, 1.45)  # inches, a panel: three and their two gutters span the text
# A band spans this many standard errors either side of the mean: one, the
# ± of the tables, which every caption states, as NeurIPS asks.
BAND = 1
# The metrics whose errors are taken within signal: PSNR, SSIM and LPIPS shift
# with the image or scene, alike for every model, and that shift is removed.
# IoU saturates near 100% on every shape, so the shapes share no level, and
# removing one would only lend the strongest models the weakest ones' spread.
PAIRED = {"psnr", "ssim", "lpips"}
# The type size and column gap of every table: NeurIPS's small, 9pt, at which
# the image and super-resolution tables, nine columns, fit the text.
TABLE_SIZE, TABLE_GAP = r"\small", "3pt"

# One colour per magnified region, so that a box and the row it magnifies are
# read together: the red of the super-resolution literature, and a cyan that
# keeps its distance from it under every kind of colour blindness.
COLORS = ("#e02020", "#00a2c7")
# The side of a magnified region, in fractions of the reference's geometric
# mean, which sizes it alike whatever the aspect ratio. The regions themselves
# are found, not set: see :func:`regions`. A NeRF view is 200 pixels across, so
# its regions hold fewer of them and stay wider.
SIDES = {"image": 0.075, "occupancy": 0.14, "nerf": 0.20, "super_resolution": 0.1}
# The magnifications' titles and scores: the largest size they are set at, and
# the inches kept above the panels for the titles and below for the scores. A
# talk raises both before drawing; the paper keeps them.
LABEL_SIZE = 7.0
LABEL_ROOM = (0.17, 0.19)
# The smallest a title or score should be set at, NeurIPS's tiny. The reference
# is never cut, so a strip whose labels would fall below it is reported instead.
LABEL_FLOOR = 6.0
# The widest a reference is drawn, as an aspect ratio: a wider one is scaled
# down and centred on the rows, so that a panorama does not squeeze the
# magnifications. None draws every reference at the full height of the rows,
# its top and bottom on theirs, as the paper does; the deck caps it.
REFERENCE_ASPECT: float | None = None


def display_name(model: str) -> str:
    """A model's name as the results print it (see DISPLAY)."""
    return DISPLAY.get(model, model.replace("lanczos", "Lanczos"))


def text_width(text: str, weight: str = "normal") -> float:
    """The width of ``text`` at a font size of 1pt, in points, in the style's font."""
    path = TextPath((0, 0), text, size=1, prop=FontProperties(weight=weight))
    return path.get_extents().width


UNITS = {"psnr": "{:.2f} dB", "iou": "{:.2f}%"}

# The FUTON ablations, all on the image task and, but for the bases themselves,
# with the cosine basis: the bases from the smoothest to the most local; the
# relative bandwidth kappa = K / N at the benchmark's size, the rank filling
# the budget; the rank at the benchmark's K, the size growing with it; and the
# variants of the two stages after the basis.
STUDIES = ("basis", "bandwidth", "rank", "tensor_net", "decoder")
# A study's files are named for what it compares, where that differs from
# the name of its config and logs.
OUTPUTS = {"tensor_net": "combiner"}
FRACTIONS = {0.125: "1/8", 0.25: "1/4", 0.5: "1/2", 1.0: "1", 2.0: "2"}
BASES = ("Cosine", "Chebyshev", "Legendre", "Triangle", "Lanczos", "Sinc")
# Per study, its table's rows and the model each stands for; the benchmark's
# model heads both.
COMBINERS = {
    "CP": "FUTON-cosine",
    "TR": "FUTON-cosine-TR",
    "Tucker": "FUTON-cosine-Tucker",
}
DECODERS = {
    "MLP, 64 wide (R = 280)": "FUTON-cosine",
    "MLP, 128 wide (R = 255)": "FUTON-cosine-w128",
    "MLP, 256 wide (R = 220)": "FUTON-cosine-w256",
    "Linear (R = 305)": "FUTON-cosine-linear",
}
# The room an ablation figure adds above its plot per row of its legend, in
# inches, so that its plot is as large as a benchmark panel's.
LEGEND_ROW = 0.15
# The column every table opens with. Thousands take the SI prefix k, since
# K is the spectral resolution everywhere else.
PARAMS = ("", "#Params (k)", None, 1)
DASHES = ("", (4, 2), (1, 1.5))  # solid, dashed, dotted
# The convergence of the three studies that get a curve, as label, colour and
# dashes per model, in the order the legend takes them, a column at a time: a
# hue and dashes per variant, and for the bases a hue per family, the
# interpolation kernels in the first column and the orthogonal families in the
# second, the dashes pairing them across a row.
CURVES = {
    "basis": {
        "FUTON-triangle": ("Triangle", PALETTE[0], DASHES[2]),
        "FUTON-lanczos": ("Lanczos", PALETTE[0], DASHES[1]),
        "FUTON-sinc": ("Sinc", PALETTE[0], DASHES[0]),
        "FUTON-legendre": ("Legendre", PALETTE[1], DASHES[2]),
        "FUTON-chebyshev": ("Chebyshev", PALETTE[1], DASHES[1]),
        "FUTON-cosine": ("Cosine", PALETTE[1], DASHES[0]),
    },
    "tensor_net": {
        "FUTON-cosine": ("CP", PALETTE[0], DASHES[0]),
        "FUTON-cosine-TR": ("TR", PALETTE[2], DASHES[1]),
        "FUTON-cosine-Tucker": ("Tucker", PALETTE[3], DASHES[2]),
    },
    "decoder": {
        "FUTON-cosine": ("MLP 64", PALETTE[0], DASHES[0]),
        "FUTON-cosine-w128": ("MLP 128", PALETTE[2], DASHES[1]),
        "FUTON-cosine-w256": ("MLP 256", PALETTE[3], DASHES[2]),
        "FUTON-cosine-linear": ("Linear", PALETTE[1], DASHES[0]),
    },
}
# The two sweeps, by the quantity on their x axis and its label there: the
# relative bandwidth at the budget, and the rank at the benchmark's bandwidth.
SWEPT = {
    "bandwidth": ("kappa", r"Relative bandwidth $\kappa$"),
    "rank": ("rank", r"Rank $R$"),
}

# The bottom of a study's PSNR axis where the range the converged values set
# would cut a slow model's rise: the Tucker combiners pass 32 dB only after
# about 10 s, and 34 dB after 14 s.
FLOORS = {"tensor_net": 32}
# The columns of a study's legend where two would not suit it: the three
# combiners fit one row.
LEGEND_COLUMNS = {"tensor_net": 3}
# The error bound of FUTON with a linear decoder against FUTONs trained by Adam
# (scripts/futon_bound.py): the datasets its figures and tables report, and the
# PSNRs (dB) its design table finds the smallest setting for.
BOUND_DATASETS = {"image": "Kodak", "occupancy": "Occupancy"}
BOUND_TARGETS = {"image": (22, 24, 26, 28, 30), "occupancy": (19, 20, 21, 22, 23)}
# What a run builds: the four tasks, the FUTON ablations and the error bound.
PARTS = (*TASKS, "ablation", "bound", "tuning")
# Every learning rate comes from this grid, and what a task's rates were chosen
# by: its metric, on which signals.
LEARNING_RATES = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
TUNED_BY = {
    "image": "PSNR (dB), the mean over the 24 Kodak images",
    "occupancy": "IoU (%), the mean over the five shapes",
    "nerf": "PSNR (dB) on 25 validation views of lego",
    "super_resolution": "PSNR gain over bicubic (dB), the mean over the DIV2K "
    "training images 0395, 0431 and 0777",
}


def style() -> None:
    """One look for every figure, at the sizes it is printed at.

    A PGF figure is included as it is, and with ``pgf.rcfonts`` off its text
    takes the document's Times. The PDFs set STIX, the Times that ships with
    matplotlib, so that every machine draws them alike, embedded as TrueType,
    since NeurIPS takes no Type 3 fonts. Text runs from the page's scriptsize
    to its footnotesize, under the 10pt of the body: 7pt for tick labels and
    legends, 8pt for axis labels. Lines and ticks thin to match.
    """
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "figure.figsize": SIZE,
            "figure.constrained_layout.use": True,
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "legend.handlelength": 1.6,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.labelpad": 2.0,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "savefig.pad_inches": 0.02,
        }
    )


def detach_axes(figure: plt.Figure, offset: float = 3.0) -> None:
    """Move the left and bottom spines ``offset`` points outward.

    The axes then no longer meet in a corner, so an axis can start on a round
    tick without its label crowding the other axis's first one, and no marker
    on the edge of the data is cut by a spine.
    """
    for plot in figure.axes:
        for side in ("left", "bottom"):
            plot.spines[side].set_position(("outward", offset))


def save(figure: plt.Figure, path: Path) -> None:
    """Write a figure, with its axes detached, as PDF, and as PGF to include
    in a LaTeX document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    detach_axes(figure)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    try:
        figure.savefig(path.with_suffix(".pgf"), bbox_inches="tight")
    except (RuntimeError, FileNotFoundError, LatexError) as error:  # PGF runs LaTeX
        warnings.warn(f"no {path.name}.pgf: {error}")
    plt.close(figure)


def read(log_dir: Path, task: str) -> list[dict[str, Any]]:
    """Load the ``task`` runs under ``log_dir``, each tagged with its directory."""
    runs = []
    for path in sorted(log_dir.rglob("results.json")):
        run = json.loads(path.read_text())
        if run["task"] == task:
            runs.append(run | {"directory": path.parent})
    return runs


def scale(metric: str, value: float) -> float:
    """A stored value as it is reported, e.g. IoU as a percentage."""
    return value * METRICS[metric][2]


def results(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per run: its signal, model, size, time and final metrics."""
    rows = []
    for run in runs:
        row = {
            "signal": run["data"],
            "model": run["model"],
            "parameters": run["num_params"] / 1000,
            "time": run["train_time"],
        }
        row |= {
            metric: scale(metric, value)
            for metric, value in run["metrics"].items()
            if metric in METRICS
        }
        rows.append(row)
    return pd.DataFrame(rows)


def curves(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per evaluation: the metrics at that iteration and elapsed time."""
    rows = []
    for run in runs:
        for entry in run["history"]:
            if "eval" not in entry:
                continue
            row = {
                "signal": run["data"],
                "model": run["model"],
                "iteration": entry.get("epoch", entry.get("step")),
                "time": entry["elapsed"],
            }
            row |= {
                metric: scale(metric, value)
                for metric, value in entry["eval"].items()
                if metric in METRICS
            }
            rows.append(row)
    return pd.DataFrame(rows)


def signal_path(task: str, signal: str) -> Path:
    """Where one of a task's signals lives; super-resolution's tuning images
    are DIV2K training images."""
    path = ROOT / SIGNALS[task].format(signal)
    if task == "super_resolution" and not path.exists():
        path = ROOT / SIGNALS[task].replace("valid", "train").format(signal)
    return path


@cache
def points(task: str, signal: str) -> int:
    """The pixels or voxels of one of the task's signals."""
    path = signal_path(task, signal)
    if task == "occupancy":
        data = yaml.safe_load((ROOT / "configs" / "occupancy.yaml").read_text())
        return math.prod(
            inro.OccupancyCoordinateDataset(path, **data["data"]).grid_size
        )
    width, height = Image.open(path).size
    return width * height


def costs(runs: list[dict[str, Any]], task: str) -> pd.DataFrame:
    """Each run's inference throughput and peak memories.

    Every evaluation records how long its reconstruction took and the most
    memory it allocated, and every training step its own peak. The median over
    a run's evaluations, for a radiance field over its 200 test views, gives
    millions of pixels or voxels per second, and for a radiance field rendered
    views per second (FPS); a first evaluation that compiles does not move it.
    A run evaluated fewer than three times has no throughput: its first
    evaluation compiles.
    The interpolations run on the CPU and train nothing, so they have no memory.
    """
    rows = []
    for run in runs:
        evaluations = (
            run["test_views"]
            if task == "nerf"
            else [entry["eval"] for entry in run["history"] if "eval" in entry]
        )
        durations = [entry["duration"] for entry in evaluations]
        seconds = statistics.median(durations) if len(durations) >= 3 else math.nan
        peaks = [
            entry["peak_memory"] for entry in evaluations if "peak_memory" in entry
        ]
        rows.append({
            "signal": run["data"],
            "model": run["model"],
            "speed": 1 / seconds if task == "nerf" else points(task, run["data"]) / seconds / 1e6,
            "train_memory": run["peak_memory"] / 1e9 if peaks else math.nan,
            "infer_memory": statistics.median(peaks) / 1e9 if peaks else math.nan,
        })  # fmt: skip
    return pd.DataFrame(rows)


def memory_unit(values: pd.Series) -> tuple[str, float, int]:
    """The unit a memory column is shown in, its factor from GB, and its decimals.

    Gigabytes, unless every value stays under 1.5 GB: an axis that short has
    too few whole or half gigabytes to tick, so it counts megabytes, and the
    table follows its axis.
    """
    return ("GB", 1.0, 2) if values.max() >= 1.5 else ("MB", 1e3, 0)


def load(task: str) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """A task's runs, its final metrics and its curves, with one TensoRF."""
    others = {"TensoRF", "TensoRF-CP", "TensoRF-VM"} - {TENSORF[task]}
    if not (runs := read(ROOT / "logs" / task, task)):
        raise SystemExit(f"no {task} runs under logs/{task}")
    runs = [
        run | {"model": "TensoRF" if run["model"] == TENSORF[task] else run["model"]}
        for run in runs
        if run["model"] not in others
    ]
    final = results(runs).merge(costs(runs, task), on=["signal", "model"])
    return runs, final, curves(runs)


def model_config(run: dict[str, Any]) -> dict[str, Any]:
    """The config a run was trained with, for rebuilding its model."""
    recorded = run.get("config") or [f"configs/{run['task']}.yaml"]
    config: dict[str, Any] = {}
    for path in (Path(path) for path in recorded if "=" not in path):  # drop --set
        # Runs from another checkout recorded its absolute path, so fall back
        # to the same file here.
        local = ROOT / path
        if not local.exists():
            local = ROOT / "configs" / path.name
        merge(config, yaml.safe_load(local.read_text()))
    return config


def rebuild(run: dict[str, Any]) -> tuple[type[torch.nn.Module], dict[str, Any]]:
    """A run's model class and constructor arguments, as it was trained: its
    config entry, and for FreSh the first-layer frequency its search chose,
    which the run records beside its setup rather than in it."""
    model_class, kwargs = build(run["setup"])
    if run.get("fresh"):
        kwargs["first_omega"] = run["fresh"]["omega"]
    return model_class, kwargs


def load_model(run: dict[str, Any], in_features: int, out_features: int, device):
    """Rebuild a run's model and load its checkpoint."""
    model_class, kwargs = rebuild(run)
    model = model_class(in_features=in_features, out_features=out_features, **kwargs)
    state = torch.load(run["directory"] / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(state)
    return model.to(device)


def within_signal(data: pd.DataFrame, metric: str) -> pd.Series:
    """A metric with each signal's own level removed, keeping the model means.

    Signals differ far more in difficulty than models do, so a standard error
    over signals mostly measures the signals. Subtracting each signal's mean
    over models at every evaluation, and adding back the overall mean, leaves
    each model's mean curve unchanged while its spread shows only how
    consistently it scores relative to the others: the within-subject standard
    error of Cousineau (2005), with the correction of Morey (2008). A signal's
    level is its mean over every model evaluated there, not only the ones a
    figure draws, so that a band pools the models its table pools.
    """
    step = ["signal", "iteration"]
    cell = [data["model"], data["iteration"]]
    centered = (
        data[metric]
        - data.groupby(step)[metric].transform("mean")
        + data.groupby("iteration")[metric].transform("mean")
    )
    mean = centered.groupby(cell).transform("mean")
    models = data.groupby("iteration")["model"].transform("nunique")
    correction = np.sqrt(models / (models - 1)).where(models > 1, 1.0)
    return mean + (centered - mean) * correction


def comparable(data: pd.DataFrame, metric: str) -> pd.Series:
    """A metric as its errors are taken: within signal if it is PAIRED, else as is."""
    return within_signal(data, metric) if metric in PAIRED else data[metric]


def y_range(means: pd.DataFrame, metric: str) -> tuple[float, float]:
    """A range on which the models separate, from their mean curves.

    Early transients would squeeze the rest together, so the weak end is
    trimmed to the tenth percentile of the second half of the run, then
    widened to every final value but a collapsed one.
    """
    converged = means[means["iteration"] >= means["iteration"].median()][metric]
    finals = means.groupby("model")[metric].last()
    finals = finals[finals >= 0.5 * finals.max()]
    low, high = min(converged.quantile(0.10), finals.min()), means[metric].max()
    margin = 0.05 * (high - low)
    return low - margin, high + margin


def paired_error(final: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Each model's standard error over signals, of the metric as compared.

    Where signals differ far more than models do, the plain deviation over
    them would say how hard the signals are, so their levels are removed
    first; see :func:`comparable`.
    """
    final = final.assign(iteration=0)  # one evaluation, the last
    return pd.DataFrame(
        {
            ("mean", metric): final.assign(**{metric: comparable(final, metric)})
            .groupby("model")[metric]
            .sem()
            for metric in metrics
        }
    )


class Halves(Locator):
    """At most ``count`` ticks, every one a whole or a half number.

    Of the steps 1/2, 1, 2, 2.5 and 5 times a power of ten, the finest that
    keeps to ``count`` ticks within the axis' limits; ``whole`` leaves out the
    steps that put a tick on a half.
    """

    def __init__(self, count: int = 5, whole: bool = False):
        self.count, self.whole = count, whole

    def __call__(self) -> list[float]:
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin: float, vmax: float) -> list[float]:
        low, high = sorted((vmin, vmax))
        steps = {0.5} | {m * 10.0**k for k in range(12) for m in (1, 2, 2.5, 5)}
        for step in sorted(steps):
            if self.whole and step % 1:
                continue
            first, last = math.ceil(low / step), math.floor(high / step)
            if last - first < self.count:
                return [k * step for k in range(first, last + 1)]
        return []


def is_half(value: float) -> bool:
    """Whether a number is whole or a half, as every tick is."""
    return math.isclose(2 * value, round(2 * value), abs_tol=1e-9)


def log_ticks(axis: Axis) -> None:
    """Label a log axis 1, 2, 5, 10, ... rather than 10^0, at most five ticks.

    Of the 1-2-3-5, 1-2-5 and 1 ticks per decade, it takes the densest that
    keeps to five within the axis' limits; below one, only a half is ticked.
    """
    low, high = axis.get_view_interval()
    decades = range(int(np.log10(low)) - 1, int(np.log10(high)) + 2)
    for subs in ((1, 2, 3, 5), (1, 2, 5), (1,)):
        ticks = [
            tick
            for tick in (m * 10.0**k for k in decades for m in subs)
            if low <= tick <= high and is_half(tick)
        ]
        if len(ticks) <= 5:
            break
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(StrMethodFormatter("{x:g}"))
    axis.set_minor_formatter(NullFormatter())


def quality_ticks(plot: plt.Axes, metric: str, low: float, high: float) -> None:
    """Set the quality axis: whole decibels for PSNR, half points for IoU.

    A PSNR tick with decimals reads as a precision the plots do not have, so
    the limits widen to whole numbers and the ticks fall on them; IoU, on
    which the models separate within a point, keeps halves.
    """
    if metric == "psnr":
        low, high = math.floor(low), math.ceil(high)
    else:  # IoU, which cannot pass 100
        low, high = math.floor(2 * low) / 2, min(math.ceil(2 * high) / 2, 100.0)
    plot.yaxis.set_major_locator(Halves(whole=metric == "psnr"))
    plot.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    plot.set_ylim(low, high)


def padded(values: pd.Series, fraction: float = 0.08) -> tuple[float, float]:
    """Limits that leave a margin, so that no marker is clipped by a spine."""
    low, high = values.min(), values.max()
    margin = fraction * (high - low) or abs(low) * fraction or 1.0
    return low - margin, high + margin


def legend_handles(reference: bool = False) -> list[Line2D]:
    """The featured models' lines and markers and the other models' gray dot;
    ``reference`` adds the dotted level of the untrained reference."""
    handles = [
        Line2D([], [], color=color, dashes=dashes or (None, None), marker="o",
               markerfacecolor=color if filled else "white", label=display_name(model))
        for model, (color, dashes, filled) in FEATURED.items()
    ] + [Line2D([], [], color=OTHERS, marker="o", ls="", label="Other models")]  # fmt: skip
    if reference:
        handles.append(Line2D([], [], color="0.4", ls=":", lw=0.8, label=REFERENCE))
    return handles


def legend(reference: bool = False) -> plt.Figure:
    """The featured models' lines and markers, and the other models' gray dot.

    One row, as wide as the text, to sit under a row of panels; ``reference``
    adds the dotted level of the untrained reference, which super-resolution
    draws.
    """
    handles = legend_handles(reference)
    figure = plt.figure(figsize=(WIDTH, 0.2))
    figure.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        columnspacing=0.7 if reference else 1.2,
        handlelength=1.2 if reference else 1.6,
        handletextpad=0.4 if reference else 0.5,
    )
    return figure


def convergence(curve: pd.DataFrame, task: str) -> plt.Figure:
    """The task's metric against training time, for the featured models.

    Time is logarithmic, since the curves span more than a decade of it.
    """
    metric, label, _ = TASKS[task]
    reference = curve[curve["model"] == REFERENCE][metric].mean()
    # Where a signal's level is removed, every model evaluated there sets it.
    curve = curve.assign(**{metric: comparable(curve, metric)})
    curve = curve[curve["model"].isin(FEATURED)]
    curve = curve.assign(
        time=curve.groupby(["model", "iteration"])["time"].transform("median")
    )
    figure, plot = plt.subplots(figsize=SIZE)
    sns.lineplot(
        data=curve,
        x="time",
        y=metric,
        hue="model",
        palette={model: color for model, (color, _, _) in FEATURED.items()},
        style="model",
        dashes={model: dashes for model, (_, dashes, _) in FEATURED.items()},
        errorbar=("se", BAND),
        err_kws={"alpha": 0.2, "linewidth": 0},
        legend=False,
        ax=plot,
    )
    # A range on which the models separate, from the first evaluation inside
    # it to the last.
    means = curve.groupby(["model", "iteration"], as_index=False)[
        [metric, "time"]
    ].mean()
    low, high = y_range(means, metric)
    low = BOTTOM.get(task, max(low, FLOOR.get(task, low)))
    if metric == "psnr":
        low, high = math.floor(low), math.ceil(high)
    shown = means[means[metric].between(low, high)]["time"]
    # The marker the legend carries, at the end of each curve.
    for model, (color, _, filled) in FEATURED.items():
        if model not in means["model"].values:  # a sweep still running
            continue
        last = means[means["model"] == model].iloc[-1]
        plot.plot(last["time"], last[metric], marker="o", markersize=3.5,
                  markeredgewidth=0.9, color=color,
                  markerfacecolor=color if filled else "white",
                  clip_on=False, zorder=4)  # fmt: skip
    plot.set(
        xscale="log",
        xlim=(shown.min() / 1.1, means["time"].max() * 1.1),
        xlabel="Training time (s)",
        ylabel=label,
    )
    log_ticks(plot.xaxis)
    quality_ticks(plot, metric, low, high)
    if not math.isnan(reference):
        plot.axhline(reference, color="0.4", linestyle=":", linewidth=0.8, zorder=1)
    return figure


def tradeoff(final: pd.DataFrame, task: str, cost: str = "time") -> plt.Figure:
    """Each model's final quality against a cost: training time, inference rate,
    or peak training or inference memory.

    Both axes are linear: the models lie within a decade of each other, where
    a log axis would only cost the reader the plain reading of the distances.
    """
    metric, label, _ = TASKS[task]
    # The cost axis carries its direction, as the table headers do: a reader
    # pauses over whether more throughput is better, never over PSNR.
    speed = "Rendering speed" if task == "nerf" else "Throughput"
    means = final.groupby("model").agg({cost: "median", metric: "mean"})
    # The interpolations take no training: they set the reference level, not a point.
    trained = means[final.groupby("model")["time"].median() > 0]
    unit = "GB"
    if cost.endswith("memory"):
        unit, factor, _ = memory_unit(trained[cost])
        means[cost] *= factor
        trained = means.loc[trained.index]
    axis = {
        "time": r"Training time (s)$\,\downarrow$",
        "speed": rf"{speed} ({SPEED[task]})$\,\uparrow$",
        "train_memory": rf"Training memory ({unit})$\,\downarrow$",
        "infer_memory": rf"Inference memory ({unit})$\,\downarrow$",
    }[cost]
    others = trained[~trained.index.isin(FEATURED)]
    figure, plot = plt.subplots(figsize=SIZE)
    if REFERENCE in means.index:
        level = means.loc[REFERENCE, metric]
        plot.axhline(level, color="0.4", linestyle=":", linewidth=0.8, zorder=1)
    plot.scatter(
        others[cost], others[metric], s=10, color=OTHERS, linewidth=0, zorder=2
    )
    for model, (color, _, filled) in FEATURED.items():
        if model not in means.index:  # a sweep still running
            continue
        x, y = means.loc[model, [cost, metric]]
        plot.scatter(
            x,
            y,
            s=20,
            facecolor=color if filled else "white",
            edgecolor=color,
            linewidth=0.9,
            zorder=3,
        )
    # Explicit limits, so that the pair of figures shares a quality axis and
    # no marker touches a spine.
    plot.set(xlim=padded(trained[cost]), xlabel=axis, ylabel=label)
    plot.xaxis.set_major_locator(Halves())
    plot.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    low, high = padded(trained[metric])
    if task in BOTTOM:
        low = BOTTOM[task]
        under = others[others[metric] < low]
        plot.scatter(under[cost], [low] * len(under), s=10, marker="v",
                     color=OTHERS, linewidth=0, clip_on=False, zorder=2)  # fmt: skip
    quality_ticks(plot, metric, low, high)
    return figure


def spanning(groups: Sequence[str]) -> list[str]:
    """A LaTeX header row with one centred cell over each run of equal groups,
    and a rule under every named one."""
    spans, rules, first = [], [], 1
    for group, columns in groupby(groups):
        width = len(list(columns))
        spans.append(rf"\multicolumn{{{width}}}{{c}}{{{group}}}")
        if group:
            rules.append(rf"\cmidrule(lr){{{first}-{first + width - 1}}}")
        first += width
    return [" & ".join(spans) + r" \\", "".join(rules)]


def table(
    values: pd.DataFrame,
    spec: dict,
    errors: pd.DataFrame | None = None,
    order: Sequence[str] = ROWS,
    index: str | Sequence[str] = "Model",
    highlight: bool = False,
    labels: dict[str, Sequence[str]] | None = None,
) -> dict[str, str]:
    """LaTeX and Markdown of models by columns, best bold and second underlined.

    ``spec`` maps each column of ``values`` to its group (empty for none), its
    header, whether larger is better (``None`` not to rank it) and its
    decimals; values tied at those decimals share a rank. A column of
    ``errors`` follows its value as ``mean+-error``. Consecutive columns of one
    group share a spanning header. Rows follow ``order``, which the ablations
    give their own, and ``index`` heads them, each named as
    :func:`display_name` names it; ``labels`` names each row instead by its
    cells, one under each header of ``index``. ``highlight`` tints the two
    best cells of every ranked column, which reads from the back of a room
    but would be noise in a printed table.
    """
    models = [model for model in order if model in values.index]
    heads = [index] if isinstance(index, str) else list(index)
    names_of = {
        model: list(labels[model]) if labels else [display_name(model)]
        for model in models
    }
    cells = {}
    for name, (_, _, larger_is_better, decimals) in spec.items():
        shown = values[name].reindex(models).round(decimals)
        error = None if errors is None or name not in errors else errors[name]
        ranks = shown.rank(ascending=not larger_is_better, method="min")
        for model in models:
            text = "–" if pd.isna(shown[model]) else f"{shown[model]:.{decimals}f}"
            spread = (
                f"{error[model]:.{decimals}f}"
                if error is not None and not pd.isna(shown[model])
                else ""
            )
            rank = ranks[model] if larger_is_better is not None else np.nan
            cells[model, name] = text, spread, rank

    def line(model: str, mark: dict[int, str], spread: str) -> list[str]:
        # The mean carries the emphasis; the error follows it, set smaller, so
        # that a column of values reads down the page before it reads across.
        return names_of[model] + [
            mark.get(rank, "{}").format(text) + (spread.format(error) if error else "")
            for text, error, rank in (cells[model, name] for name in spec)
        ]

    groups = [""] * len(heads) + [group for group, *_ in spec.values()]
    # A title under a group is a lowercase phrase, "time (s)" under
    # "Training", so that the flat header reads "Training time (s)"; LaTeX
    # capitalizes it on its own row.
    titles = heads + [title for _, title, _, _ in spec.values()]
    arrows = [""] * len(heads) + [
        "" if larger_is_better is None else "↑" if larger_is_better else "↓"
        for _, _, larger_is_better, _ in spec.values()
    ]
    flat = [
        " ".join(filter(None, (group, title, arrow)))
        for group, title, arrow in zip(groups, titles, arrows)
    ]
    # Numbers right against a fixed number of decimals, which lines their
    # points up; the headers stay centred over them, a name over its unit and
    # direction, so that a column is as wide as its numbers rather than its
    # header. No padding outside the first and last columns, so that the
    # rules end with the text.
    columns = "l" * len(heads) + "r" * (len(titles) - len(heads))
    latex = [rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule"]
    if any(groups):  # a spanning header over each group's columns
        latex += spanning(groups)
    names, units = zip(*(title.partition(" (")[::2] for title in titles))
    names = [name[:1].upper() + name[1:] for name in names]
    units = [
        " ".join(filter(None, (f"({unit}" if unit else "", arrow)))
        for unit, arrow in zip(units, arrows)
    ]
    for row in (names, units) if any(units) else (names,):
        header = list(row[: len(heads)]) + [
            rf"\multicolumn{{1}}{{c}}{{{cell}}}" for cell in row[len(heads) :]
        ]
        latex.append(" & ".join(header) + r" \\")
    latex.append(r"\midrule")
    markdown = ["| " + " | ".join(flat) + " |", "| --- " * len(flat) + "|"]
    tex_mark = (
        {1: r"\cellcolor{{best}}\textbf{{{}}}", 2: r"\cellcolor{{second}}{}"}
        if highlight
        else {1: r"\textbf{{{}}}", 2: r"\underline{{{}}}"}
    )
    for model in models:
        if model == "FUTON-sinc" and not labels:
            latex.append(r"\midrule")
        latex.append(
            " & ".join(line(model, tex_mark, r"{{\scriptsize$\pm${}}}")) + r" \\"
        )
        markdown.append(
            "| " + " | ".join(line(model, {1: "**{}**", 2: "_{}_"}, " ± {}")) + " |"
        )
    latex += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(latex) + "\n"
    # None of the commands above holds one of these, so escaping is safe.
    for symbol, command in (
        ("↑", r"$\uparrow$"),
        ("↓", r"$\downarrow$"),
        ("κ", r"$\kappa$"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
    ):
        tex = tex.replace(symbol, command)
    return {"tex": tex, "md": "\n".join(markdown) + "\n"}


def per_signal(
    final: pd.DataFrame, metrics: Sequence[str]
) -> tuple[pd.DataFrame, dict]:
    """Each signal's metrics and their mean, as values and columns.

    Several metrics group the columns by signal and name the metrics inside
    them; one metric is itself the group, over a column per signal.
    """
    signals = sorted(final["signal"].unique())
    per = {
        metric: final.pivot_table(index="model", columns="signal", values=metric)
        for metric in metrics
    }
    values, spec = {}, {}
    for signal in [*signals, "mean"]:  # a signal's metrics stay side by side
        title = "Mean" if signal == "mean" else signal.replace("_", " ").title()
        for metric in metrics:
            name, larger_is_better, _, decimals = METRICS[metric]
            column = (
                per[metric][signals].mean(axis=1)
                if signal == "mean"
                else per[metric][signal]
            )
            group, header = (
                (title, name.split(" (")[0]) if len(metrics) > 1 else (name, title)
            )
            values[signal, metric] = column
            spec[signal, metric] = (group, header, larger_is_better, decimals)
    frame = pd.DataFrame(values)
    # Plain tuples, not a column MultiIndex, so that they join the flat
    # columns beside them and key the spec.
    frame.columns = pd.Index(list(values), tupleize_cols=False)
    return frame, spec


def cost_columns(final: pd.DataFrame, task: str) -> tuple[pd.DataFrame, dict]:
    """Each model's size and the cost of each phase, as a table's values and
    spec: training time, as TensoRF, K-Planes and NeuRBF report it, inference
    throughput, and the peak memory of both, each memory in its axis' unit."""
    throughput = SPEED[task] if task == "nerf" else f"throughput ({SPEED[task]})"
    spec = {
        "parameters": PARAMS,
        "time": ("Training", "time (s)", False, 1),
        "train_memory": ("Training", "memory", False, 2),
        "speed": ("Inference", throughput, True, 1),
        "infer_memory": ("Inference", "memory", False, 2),
    }
    # Costs are medians over the signals: the runs share their nodes with other
    # jobs, which can only add time, and a median sets aside the runs they slow.
    values = final.groupby("model")[list(spec)].median()
    # Interpolations have no parameters and train nothing, so neither column
    # applies to them, nor should they top its ranking.
    values[values["time"] <= 0] = np.nan
    for column in ("train_memory", "infer_memory"):  # in the unit of its axis
        unit, factor, decimals = memory_unit(values[column])
        values[column] *= factor
        spec[column] = (spec[column][0], f"memory ({unit})", False, decimals)
    return values, spec


def metric_columns(
    final: pd.DataFrame, metrics: Sequence[str]
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Each model's mean metrics over its signals as a table's values, spec
    and errors, one paired standard error per mean."""
    spec = {}
    for metric in metrics:
        name, larger_is_better, _, decimals = METRICS[metric]
        spec[metric] = ("", name, larger_is_better, decimals)
    errors = paired_error(final, metrics)
    errors.columns = list(metrics)
    return final.groupby("model")[list(metrics)].mean(), spec, errors


def tables(final: pd.DataFrame, task: str) -> dict[str, str]:
    """The task's table, with its metrics per signal where the signals are few.

    Every model's size, the cost of each phase and its mean metrics over the
    signals, each mean with one paired standard error. The shapes and the
    scenes are few, so their table gives each one's scores as well, under a
    super column per scene for NeRF's three metrics: wider than the text, it
    takes a page turned sideways.
    """
    metrics = TASKS[task][2]
    values, spec = cost_columns(final, task)
    if task in ("image", "super_resolution"):  # many images, so their mean alone
        means, mean_spec, errors = metric_columns(final, metrics)
        return table(values.join(means), spec | mean_spec, errors)
    per, per_spec = per_signal(final, metrics)
    return table(values.join(per), spec | per_spec, paired_error(final, metrics))


def time_to_target(curve: pd.DataFrame) -> pd.Series:
    """Mean time each model first reaches the target IoU, if on every shape."""
    reached = curve[curve["iou"] >= TARGET].groupby(["model", "signal"])["time"]
    first = reached.min().groupby("model")
    return first.mean().where(first.size() == curve["signal"].nunique())


def render_image(signal: str, runs: list[dict], directory: Path, device) -> None:
    """The image itself, and each model's reconstruction of it."""
    dataset = inro.ImageCoordinateDataset(ROOT / SIGNALS["image"].format(signal))
    dataset.save(dataset.original, directory / "truth")
    for run in runs:
        model = load_model(run, 2, 3, device)
        output = inro.chunked_inference(model, dataset.input, device=device)
        dataset.save(dataset.postprocess(output.cpu()), directory / run["model"])


def render_occupancy(signal: str, runs: list[dict], directory: Path, device) -> None:
    """The shape's mesh, and each model's, drawn from one camera.

    Each run is checked against the shape it sees here, since a mesh file that
    differs from the one it was trained on would rebuild silently wrong.
    """
    resolution = model_config(runs[0])["data"]["resolution"]
    dataset = inro.OccupancyCoordinateDataset(
        ROOT / SIGNALS["occupancy"].format(signal),
        resolution=resolution,
        item_id=signal,
    )
    up = UP.get(signal, 1)
    dataset.save(dataset.original, directory / "truth", up=up, mesh=False)
    for run in runs:
        # Sizes such as a basis's grid_size, [D, H, W], follow the cropped
        # volume, and so does the rank that fills FUTON's budget.
        sizes = dict(zip("DHW", dataset.grid_size))
        setup = fill_rank(resolve(run["setup"], **sizes), outputs=1)
        model = load_model({**run, "setup": setup}, 3, 1, device)
        output = inro.chunked_inference(
            model, dataset.input, chunk_size=512**2, device=device
        )
        volume = dataset.postprocess(output.cpu())
        rebuilt = scale("iou", inro.iou(volume, dataset.original))
        recorded = scale("iou", run["metrics"]["iou"])
        if abs(rebuilt - recorded) > 0.5:
            raise SystemExit(
                f"{run['model']} rebuilds to {rebuilt:.2f}% IoU, not the "
                f"{recorded:.2f}% of its run: {signal} is not the shape it saw."
            )
        dataset.save(volume, directory / run["model"], up=up, mesh=False)


def field_and_renderer(run: dict, config: dict, dataset, device):
    """Rebuild a run's radiance field and its renderer, occupancy grid included.

    The grid comes from the run's ``renderer.pt``. Runs from before it was
    saved checkpointed only the field, so their grid is rebuilt from the field
    as training's warmup did: a fresh renderer starts fully occupied, and the
    faint density a model leaves in empty space, which the grid pruned when
    the run was scored, would be sampled everywhere and rendered as fog.
    """
    field = inro.nerf.RadianceField(
        density_net=rebuild(run), aabb=dataset.aabb, **config["field"]
    )
    state = torch.load(run["directory"] / "checkpoint.pt", map_location="cpu")
    field.load_state_dict(state)
    renderer = inro.nerf.create_renderer(
        backend="nerfacc" if inro.nerf.is_nerfacc_available() else "pytorch",
        aabb=dataset.aabb,
        near=dataset.near,
        far=dataset.far,
    )
    field, renderer = field.to(device), renderer.to(device)
    grid = run["directory"] / "renderer.pt"
    if grid.exists():
        renderer.load_state_dict(torch.load(grid, map_location=device))
        return field, renderer
    with torch.no_grad():
        for step in range(renderer.warmup_steps + renderer.update_interval):
            renderer.update_occupancy(field, step)
    return field, renderer


def render_nerf(signal: str, runs: list[dict], directory: Path, device) -> None:
    """One test view of the scene, the one :func:`panels` chose, and each
    model's render of it."""
    config = model_config(runs[0])
    dataset = inro.nerf.BlenderDataset(
        ROOT / SIGNALS["nerf"].format(signal),
        "test",
        downsample=config["data"]["downsample"],
    )
    batch = dataset[int((directory / "view.txt").read_text())]
    target = (dataset.composite(batch["rgba"]) * 255).round().byte().numpy()
    Image.fromarray(target).save(directory / "truth.png")
    for run in runs:
        field, renderer = field_and_renderer(run, config, dataset, device)
        rendered = inro.nerf.render_image(
            field,
            renderer,
            batch["rays_o"].to(device),
            batch["rays_d"].to(device),
            background=dataset.background.to(device),
        )
        image = (rendered["rgb"].clamp(0, 1) * 255).round().byte().cpu().numpy()
        Image.fromarray(image).save(directory / f"{run['model']}.png")


def render_orbit(
    signal: str,
    runs: list[dict],
    directory: Path,
    device,
    frames: int,
    overwrite: bool = False,
) -> None:
    """An orbit of each model as a GIF, at the elevation the dataset uses.

    Nothing in the paper takes these; they are for a talk, and cost a minute
    a scene, so ``--orbit`` asks for them.
    """
    config = model_config(runs[0])
    dataset = inro.nerf.BlenderDataset(
        ROOT / SIGNALS["nerf"].format(signal),
        "test",
        downsample=config["data"]["downsample"],
    )
    poses = [
        inro.nerf.pose_spherical(theta, -30.0, dataset.radius)
        for theta in np.linspace(-180, 180, frames, endpoint=False)
    ]
    for run in runs:
        path = directory / f"{run['model']}.gif"
        if path.exists() and not overwrite:
            continue
        print(f"  orbiting {run['model']} on {signal}", flush=True)
        field, renderer = field_and_renderer(run, config, dataset, device)
        orbit = [
            Image.fromarray(frame)
            for frame in inro.nerf.render_views(field, renderer, dataset, poses)
        ]
        orbit[0].save(
            path,
            save_all=True,
            append_images=orbit[1:],
            duration=1000 // 15,
            loop=0,
        )


def load_dip(run: dict[str, Any], path: Path, device) -> tuple[torch.nn.Module, Tensor]:
    """Rebuild a DIP run's network and the fixed noise it renders the image from."""
    model_class, kwargs = build(run["setup"])
    dataset = inro.DIPImageDataset(
        path, noise_channels=kwargs["in_channels"], reg_noise_std=0
    )
    model = model_class(**kwargs)
    state = torch.load(run["directory"] / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(state)
    return model.to(device), dataset.input[None]


def render_super_resolution(
    signal: str, runs: list[dict], directory: Path, device
) -> None:
    """The high-resolution image, and each model's recovery of it from the low one."""
    path = ROOT / SIGNALS["super_resolution"].format(signal)
    dataset = inro.ImageCoordinateDataset(path)
    dataset.save(dataset.original, directory / "truth")
    scale = model_config(runs[0])["data"]["scale"]
    for run in runs:
        setup = run["setup"]
        if "interpolation" in setup:
            with Image.open(low_resolution(path, scale)) as image:
                size = (dataset.original.shape[2], dataset.original.shape[1])
                upsampled = image.convert("RGB").resize(
                    size, INTERPOLATIONS[setup["interpolation"]]
                )
            output = torch.from_numpy(np.asarray(upsampled)).permute(2, 0, 1)
        elif setup["class"] == "DIPSkip":
            model, noise = load_dip(run, path, device)
            rendered = inro.chunked_inference(model, noise, device=device)[0]
            output = inro.DIPImageDataset.postprocess(rendered.cpu())
        else:
            model = load_model(run, 2, 3, device)
            rendered = inro.chunked_inference(
                model, dataset.input, chunk_size=2**18, device=device
            )
            output = dataset.postprocess(rendered.cpu())
        dataset.save(output, directory / run["model"])


RENDER = {
    "image": render_image,
    "occupancy": render_occupancy,
    "nerf": render_nerf,
    "super_resolution": render_super_resolution,
}


def margin(scores: pd.DataFrame, task: str) -> pd.Series:
    """Per row, the better FUTON's score less the strongest other model's, of
    those in the task's strip, ``scores`` holding a column per model."""
    shown = STRIP.get(task, tuple(FEATURED))
    ours = [model for model in shown if model.startswith("FUTON")]
    others = [model for model in shown if model not in ours]
    return scores[ours].max(axis=1) - scores[others].max(axis=1)


def typical(gain: pd.Series, count: int) -> list:
    """The ``count`` labels whose gain lies closest to the median gain."""
    distance = (gain - gain.median()).abs()
    return list(distance.sort_values(kind="stable").index[:count])


def example_view(signal: str, runs: list[dict]) -> int:
    """The test view a scene is drawn from, unless VIEWS sets it: of the views
    that show the scene whole, its silhouette clear of the frame by 2% of the
    side so that a magnification never meets the image's edge, the one whose
    gain lies closest to their median gain."""
    if signal in VIEWS:
        return VIEWS[signal]
    config = model_config(runs[0])
    dataset = inro.nerf.BlenderDataset(
        ROOT / SIGNALS["nerf"].format(signal),
        "test",
        downsample=config["data"]["downsample"],
    )
    alpha = dataset.images[..., 3] > 0.05  # (views, height, width)
    band = max(1, round(0.02 * min(alpha.shape[1:])))
    frame = torch.ones_like(alpha[0])
    frame[band:-band, band:-band] = False
    clear = ~(alpha & frame).flatten(1).any(1)
    whole = [int(view) for view in clear.nonzero().flatten()] or list(range(len(alpha)))
    psnr = pd.DataFrame(
        {
            run["model"]: {view["view"]: view["psnr"] for view in run["test_views"]}
            for run in runs
        }
    )
    return int(typical(margin(psnr.loc[whole], "nerf"), 1)[0])


def panels(
    task: str, signal: str, runs: list[dict], device: str, overwrite: bool
) -> Path:
    """The signal and every strip model's reconstruction of it, as PNGs.

    Rendering rebuilds each model from its checkpoint, which needs the signal
    in data/ and a GPU, so the panels are kept and reused. A scene's are kept
    with the view they show, and drawn again when the rule picks another.
    """
    directory = ROOT / "results" / task / "panels" / signal
    if task == "nerf":
        shown = STRIP.get(task, tuple(FEATURED))
        scene = [run for run in runs if run["data"] == signal and run["model"] in shown]
        view = example_view(signal, scene)
        stored = directory / "view.txt"
        if not stored.exists() or stored.read_text().strip() != str(view):
            directory.mkdir(parents=True, exist_ok=True)
            for old in directory.glob("*.png"):  # another view's panels
                old.unlink()
            stored.write_text(f"{view}\n")
    missing = {
        model
        for model in STRIP.get(task, tuple(FEATURED))
        if overwrite or not (directory / f"{model}.png").exists()
    }
    if not missing and (directory / "truth.png").exists():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    chosen = [run for run in runs if run["data"] == signal and run["model"] in missing]
    if len(chosen) != len(missing):
        raise SystemExit(f"{len(chosen)} of {len(missing)} missing runs for {signal}")
    print(f"  rendering {len(chosen)} panels of {signal}", flush=True)
    RENDER[task](signal, chosen, directory, device)
    return directory


def trim(image: np.ndarray, margin: float = 0.02) -> tuple[slice, slice]:
    """The part of a render the signal occupies, plus a margin.

    Mesh and NeRF renders sit in a wide blank field that would leave the
    reference panel tiny. The box comes from the ground truth alone, so that
    every panel of a signal is cropped alike and the magnifications align.
    """
    ink = np.abs(image[..., :3] - image[0, 0, :3]).sum(-1) > 0.05
    rows, columns = (np.flatnonzero(ink.any(axis=axis)) for axis in (1, 0))
    if not rows.size or not columns.size:  # a photograph fills its frame
        return slice(None), slice(None)
    pad = int(margin * max(image.shape[:2]))
    return (
        slice(max(rows[0] - pad, 0), rows[-1] + pad),
        slice(max(columns[0] - pad, 0), columns[-1] + pad),
    )


def box_sums(field: np.ndarray, side: int) -> np.ndarray:
    """The sum over every ``side`` by ``side`` square, from an integral image."""
    total = np.pad(field, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (
        total[side:, side:]
        - total[:-side, side:]
        - total[side:, :-side]
        + total[:-side, :-side]
    )


def detail(gray: np.ndarray, window: int = 5) -> np.ndarray:
    """How much fine detail the reference carries around each pixel.

    The energy left when a local mean is taken away, which is high on a fence,
    a lettering or a row of studs, and low on the long smooth edges whose blur
    costs the most error but tells a reader least.
    """
    pad = ((window // 2, window - 1 - window // 2),) * 2
    mean = np.pad(box_sums(gray, window) / window**2, pad, mode="edge")
    return (gray - mean) ** 2


def regions(panel: dict[str, np.ndarray], side: int) -> list[tuple[int, int]]:
    """The squares where the models shown disagree most on detail.

    A magnification earns its space only where the models disagree, so the
    regions are found rather than set: the spread of the panels about their
    mean, weighted by the detail the reference carries there and summed over
    every square of the given side. Spread favours no model, and the squares
    are taken whoever is closest in them, so that the figure shows where the
    models differ rather than where FUTON wins; the detail keeps them off the
    long smooth edges whose blur costs error but tells a reader little.

    A render sits on a blank field, where a silhouette a voxel out of place
    costs more error than any surface detail, so there the squares are kept
    inside the signal; a photograph fills its frame and keeps every square. The
    second square stands twice its side clear of the first where the signal
    leaves room for it, and merely clear of it where it does not.
    """
    truth = panel["truth"]
    shown = [name for name in panel if name != "truth"]
    spread = np.var([panel[name].mean(-1) for name in shown], axis=0)
    scores = box_sums(spread * detail(truth.mean(-1)), side)
    blank = np.abs(truth - truth[0, 0]).sum(-1) <= 0.05
    if blank.mean() > 0.15:  # a render on a blank field, not a photograph
        inside = box_sums((~blank).astype(float), side) >= 0.75 * side**2
        scores = np.where(inside, scores, -np.inf)

    top, left = np.unravel_index(np.argmax(scores), scores.shape)
    found = [(int(left), int(top))]
    for apart in (2.0, 1.0):  # room to breathe, or merely no overlap
        clear = scores.copy()
        keep = round(apart * side)
        clear[max(top - keep, 0) : top + keep, max(left - keep, 0) : left + keep] = (
            -np.inf
        )
        if np.isfinite(clear).any():
            row, column = np.unravel_index(np.argmax(clear), clear.shape)
            found.append((int(column), int(row)))
            break
    return found


def round_up(value: float) -> float:
    """The least of 1, 2, 2.5 and 5 times a power of ten not below ``value``."""
    power = 10.0 ** math.floor(math.log10(value))
    return next(step * power for step in (1, 2, 2.5, 5, 10) if step * power >= value)


def error_scale(plot: plt.Axes, top: float, size: float) -> None:
    """The error maps' colour scale, drawn in the cell the ground truth leaves."""
    plot.set_axis_off()
    plot.imshow(np.linspace(0, 1, 256)[None], cmap=ERROR_MAP, aspect="auto",
                extent=(0.1, 0.9, 0.44, 0.56))  # fmt: skip
    plot.set(xlim=(0, 1), ylim=(0, 1))
    plot.text(0.5, 0.64, "|error|", ha="center", va="bottom", fontsize=size)
    plot.text(0.1, 0.36, "0", ha="left", va="top", fontsize=size)
    plot.text(0.9, 0.36, f"{top:g}", ha="right", va="top", fontsize=size)


def qualitative(task: str, signal: str, final: pd.DataFrame, directory: Path):
    """The reference with its regions boxed, and every panel's magnification.

    One row of magnifications per region, framed in the colour of its box, and
    for the tasks in ERRORS a row of each model's |error| beneath it; the
    reference beside them spans the rows, as wide as its aspect ratio makes it,
    so that no panel is padded with white.
    """
    metric, _, _ = TASKS[task]
    models = STRIP.get(task, tuple(FEATURED))
    scores = final[final["signal"] == signal].set_index("model")[metric]
    panel = {
        name: plt.imread(directory / f"{name}.png")[..., :3]
        for name in ["truth", *models]
    }
    box = trim(panel["truth"])
    panel = {name: image[box] for name, image in panel.items()}
    height, width = panel["truth"].shape[:2]

    side = round(SIDES[task] * np.sqrt(width * height))
    if signal in CENTERS:  # set by hand, in fractions of the trimmed reference
        found = [
            (
                min(max(round(x * width - side / 2), 0), width - side),
                min(max(round(y * height - side / 2), 0), height - side),
            )
            for x, y in CENTERS[signal]
        ]
    else:
        found = regions(panel, side)

    # The reference drops the blank rows a render keeps above and below its
    # signal, so that the signal itself, not white space, spans the rows; the
    # magnifications keep the regions as found.
    ink = np.abs(panel["truth"] - panel["truth"][0, 0]).sum(-1) > 0.05
    lines = np.flatnonzero(ink.any(axis=1))
    first = min(lines[0], *(top for _, top in found)) if lines.size else 0
    last = (
        max(lines[-1] + 1, *(top + side for _, top in found)) if lines.size else height
    )
    reference = panel["truth"][first:last]
    aspect = width / (last - first)

    # A row per region, stacked as their boxes are, the highest first, and
    # beneath each, for the tasks in ERRORS, a row of each model's |error|
    # there, every map on the figure's one scale. A box keeps its colour.
    errors = None
    if task in ERRORS:
        crops = [
            np.abs(panel[name] - panel["truth"]).mean(-1)[
                top : top + side, left : left + side
            ]
            for name in models
            for left, top in found
        ]
        errors = round_up(float(np.percentile(np.stack(crops), 90)))
    stacked = sorted(range(len(found)), key=lambda region: found[region][1])
    lines_of = [(region, kind) for region in stacked
                for kind in (("crop", "error") if errors else ("crop",))]  # fmt: skip
    rows, columns = len(lines_of), len(models) + 1
    # Laid out in inches rather than by a gridspec, so that the reference
    # spans the rows exactly, its top and bottom edges on theirs, whatever its
    # aspect ratio. A magnification is square; the reference is whole, as wide
    # as its aspect ratio makes it at that height, and a gutter sets it apart
    # from the magnifications it indexes.
    gap, gutter = 0.06, 0.3  # both in units of one magnification's side
    span = rows + (rows - 1) * gap
    drawn = aspect if REFERENCE_ASPECT is None else min(aspect, REFERENCE_ASPECT)
    winner = scores[list(models)].idxmax()  # both plotted metrics are larger-better
    # The titles read as the tables' header does: a model with a setting is
    # named once, over its columns and a rule, as a spanning header, and each
    # column carries its setting beneath; any other column carries its name
    # there.
    shown = [("", "Ground truth", "truth", "")]
    for model in models:
        group, _, setting = display_name(model).partition(" (")
        score = UNITS[metric].format(scores[model])
        shown.append(
            (group, setting.removesuffix(")"), model, score)
            if setting
            else ("", group, model, score)
        )
    spans = []  # (group, first column, last column)
    for group, run in groupby(enumerate(shown), key=lambda item: item[1][0]):
        columns_run = [column for column, _ in run]
        if group:
            spans.append((group, columns_run[0], columns_run[-1]))

    # Every label keeps to one line and to the width of its magnification, so
    # that the columns, and with them the whole reference, need not make room
    # for the longest names. All of a figure's text is set at one size: the
    # largest, at most LABEL_SIZE, at which the widest title or score spans 98%
    # of its magnification, which keeps it clear of the edges whatever the side
    # bearings of its glyphs; the scores measured in bold, as the best one is.
    unit = WIDTH / (drawn * span + gutter + columns + (columns - 1) * gap)
    room = 72 * unit  # a magnification's width, in points
    widest = max(
        text_width(text, weight)
        for _, title, _, score in shown
        for text, weight in ((title, "normal"), (score, "bold"))
        if text
    )
    size = min(LABEL_SIZE, 0.98 * room / widest)
    if size < LABEL_FLOOR:
        print(f"    labels of {signal} at {size:.2f}pt, under {LABEL_FLOOR}pt")
    # Above the panels, in points: the titles' baseline, the rule a cap height
    # and a hair above them, and the spanning headers' baseline over the rule.
    cap = 0.7 * size
    title_base, rule, header_base = 3.0, 3.0 + cap + 1.6, 3.0 + cap + 4.0
    above, below = LABEL_ROOM  # inches kept for the titles and the scores
    if spans:
        above = max(above, (header_base + cap + 3.0) / 72)
    tall = span * unit + above + below
    figure = plt.figure(figsize=(WIDTH, tall))
    figure.set_layout_engine("none")

    def place(left: float, bottom: float, wide: float, high: float):
        """An axes at a position given in inches from the lower left."""
        return figure.add_axes([left / WIDTH, bottom / tall, wide / WIDTH, high / tall])

    reference_wide = drawn * span * unit
    reference_tall = reference_wide / aspect
    plot = place(
        0, below + (span * unit - reference_tall) / 2, reference_wide, reference_tall
    )
    plot.imshow(reference)
    plot.set_axis_off()
    for (left, top), color in zip(found, COLORS):
        plot.add_patch(
            Rectangle(
                (left, top - first),
                side,
                side,
                fill=False,
                color=color,
                linewidth=0.9,
            )
        )

    def column_left(column: int) -> float:
        """The left edge of a column of magnifications, in inches."""
        return (drawn * span + gutter + column * (1 + gap)) * unit

    def tint(name: str) -> str:
        """FUTON's labels in its colour, every other one in black."""
        return PALETTE[0] if name.startswith("FUTON") else "black"

    top_edge = below + span * unit  # the top of the first row, in inches
    for group, first_column, last_column in spans:
        left, right = column_left(first_column), column_left(last_column) + unit
        trimmed = 0.08 * unit  # the rule stops short of the columns' edges
        height = top_edge + rule / 72
        figure.add_artist(
            Line2D(
                [(left + trimmed) / WIDTH, (right - trimmed) / WIDTH],
                [height / tall] * 2,
                transform=figure.transFigure,
                color=tint(group),
                linewidth=0.5,
            )
        )
        figure.text(
            (left + right) / 2 / WIDTH,
            (top_edge + header_base / 72) / tall,
            group,
            ha="center",
            va="baseline",
            fontsize=size,
            color=tint(group),
        )

    for column, (_, title, name, score) in enumerate(shown):
        for row, (region, kind) in enumerate(lines_of):
            left, top = found[region]
            plot = place(
                column_left(column),
                below + (rows - 1 - row) * (1 + gap) * unit,
                unit,
                unit,
            )
            crop = (slice(top, top + side), slice(left, left + side))
            if kind == "error" and name == "truth":
                error_scale(plot, errors, size)
                continue
            if kind == "error":
                difference = np.abs(panel[name][crop] - panel["truth"][crop])
                plot.imshow(difference.mean(-1), cmap=ERROR_MAP, vmin=0, vmax=errors,
                            interpolation="nearest")  # fmt: skip
            else:
                # Nearest keeps a NeRF render's few pixels crisp, not blurred.
                plot.imshow(panel[name][crop], interpolation="nearest")
            plot.set(xticks=[], yticks=[])
            for spine in plot.spines.values():
                spine.set(visible=True, color=COLORS[region], linewidth=0.9)
            if row == 0:
                plot.set_title(title, fontsize=size, pad=title_base, color=tint(name))
            if row == rows - 1:  # the best score is set bold, as in the tables
                plot.set_xlabel(
                    score,
                    fontsize=size,
                    labelpad=3,
                    fontweight="bold" if name == winner else "normal",
                )
    return figure


def study(name: str) -> list[dict]:
    """The study's runs whose setup matches its config."""
    log_dir = ROOT / "logs" / "ablation-futon" / name
    if not log_dir.exists():
        return []
    config = ROOT / "configs" / "ablation-futon" / f"{name}.yaml"
    setups = yaml.safe_load(config.read_text())["models"]
    runs = read(log_dir, "image")
    return [
        run
        for run in runs
        if resolve_setup(setups.get(run["model"]), run) == run["setup"]
    ]


def resolve_setup(setup: dict | None, run: dict) -> dict | None:
    """A config entry with the image's size filled in, as the run recorded it."""
    if setup is None:
        return None
    path = ROOT / SIGNALS["image"].format(run["data"])
    width, height = Image.open(path).size
    return resolve(setup, H=height, W=width)


def study_frame(runs: list[dict], task: str = "image") -> pd.DataFrame:
    """One row per run of a study: its size, costs and final metrics."""
    return results(runs).merge(costs(runs, task), on=["signal", "model"])


def labelled_table(
    runs: list[dict], labels: dict[str, tuple[str, ...]], index: tuple[str, ...]
) -> dict[str, str]:
    """A study's table with the benchmark's columns, a row per model of
    ``labels`` in its order, named by its cells under the headers of
    ``index``; the errors are taken among its rows."""
    final = study_frame(runs)
    final = final[final["model"].isin(labels)]
    values, spec = cost_columns(final, "image")
    means, mean_spec, errors = metric_columns(final, TASKS["image"][2])
    return table(values.join(means), spec | mean_spec, errors, order=list(labels),
                 index=index, labels=labels)  # fmt: skip


def basis_table(runs: list[dict]) -> dict[str, str]:
    """Every basis at the benchmark's size."""
    final = study_frame(runs)
    final["model"] = final["model"].str.removeprefix("FUTON-").str.title()
    values, spec = cost_columns(final, "image")
    means, mean_spec, errors = metric_columns(final, TASKS["image"][2])
    return table(values.join(means), spec | mean_spec, errors, order=BASES,
                 index="Basis")  # fmt: skip


def variant_table(
    runs: list[dict], variants: dict[str, str], index: str
) -> dict[str, str]:
    """A study's variants against the benchmark's model. ``variants`` maps each
    row to the model it stands for, and ``index`` heads the rows."""
    present = {run["model"] for run in runs}
    labels = {model: (label,) for label, model in variants.items() if model in present}
    return labelled_table(runs, labels, (index,)) if labels else {}


def sweep(runs: list[dict]) -> pd.DataFrame:
    """One row per run of a sweep: its size, costs and final metrics, and the
    relative bandwidth kappa = K / N and the rank its setup records."""
    final = study_frame(runs)
    kwargs = {run["model"]: run["setup"]["kwargs"] for run in runs}
    basis = {model: settings["basis"][1] for model, settings in kwargs.items()}
    final["kappa"] = final["model"].map(
        lambda model: basis[model]["num_components"][0] / basis[model]["grid_size"][0]
    )
    final["rank"] = final["model"].map(
        lambda model: kwargs[model]["combiner"][1]["rank"]
    )
    return final


def sweep_figure(final: pd.DataFrame, x: str, label: str) -> plt.Figure:
    """PSNR against a swept quantity, its levels doubling along the axis, with a
    band of one within-signal standard error and a ring on the benchmark's
    model."""
    data = final.assign(iteration=0)  # one evaluation, the last
    data["psnr"] = within_signal(data, "psnr")  # the study's every model
    figure, plot = plt.subplots(figsize=SIZE)
    sns.lineplot(data=data, x=x, y="psnr", color=PALETTE[0], marker="o",
                 errorbar=("se", BAND), err_kws={"alpha": 0.2, "linewidth": 0},
                 markersize=3.5, markeredgecolor="white", markeredgewidth=0.4,
                 ax=plot)  # fmt: skip
    benchmark = data[data["model"] == "FUTON-cosine"]
    plot.scatter(benchmark[x].iloc[:1], [benchmark["psnr"].mean()], s=40,
                 facecolor="none", edgecolor="0.15", lw=0.8, zorder=3,
                 clip_on=False)  # fmt: skip
    levels = sorted(data[x].unique())
    # Past five levels the labels crowd the panel: every other one is labelled,
    # the benchmark's among them, and the rest keep their ticks.
    labelled = levels.index(benchmark[x].iloc[0]) % 2 if len(levels) > 5 else None
    names = [
        FRACTIONS.get(level, f"{level:g}") if labelled in (None, i % 2) else ""
        for i, level in enumerate(levels)
    ]
    plot.set(xscale="log", xlabel=label, ylabel="PSNR (dB)")
    plot.set_xticks(levels, names)
    plot.xaxis.set_minor_locator(NullLocator())
    plot.yaxis.set_major_locator(Halves(whole=True))
    return figure


def sweep_table(final: pd.DataFrame, runs: list[dict], name: str) -> dict[str, str]:
    """A sweep's table, a row per level: the bandwidth and the rank filling the
    budget there, or the rank."""
    levels = (
        final.groupby("model")[["kappa", "rank"]].first().sort_values(SWEPT[name][0])
    )
    labels = {
        model: (FRACTIONS[row["kappa"]], f"{row['rank']:g}") if name == "bandwidth"
        else (f"{row['rank']:g}",)
        for model, row in levels.iterrows()
    }  # fmt: skip
    index = ("κ", "R") if name == "bandwidth" else ("R",)
    return labelled_table(runs, labels, index)


def study_curves(
    curve: pd.DataFrame,
    styles: dict[str, tuple[str, str, tuple]],
    path: Path,
    floor: float | None = None,
    columns: int = 2,
) -> None:
    """PSNR against training time for a study's models, with its own legend.

    ``styles`` gives each model its label, colour and dashes, in the order the
    legend takes them, over the plot in ``columns`` columns. The band is one
    within-signal standard error, as in the task figures. ``floor`` lowers the
    bottom of the PSNR axis to show more of a slow model's rise.
    """
    curve = curve.assign(psnr=within_signal(curve, "psnr"))  # the study's every model
    curve = curve[curve["model"].isin(styles)]
    curve = curve.assign(
        time=curve.groupby(["model", "iteration"])["time"].transform("median")
    )
    # The legend above the plot, filled a column at a time.
    rows = math.ceil(len(styles) / columns)
    figure, plot = plt.subplots(figsize=(SIZE[0], SIZE[1] + rows * LEGEND_ROW))
    sns.lineplot(
        data=curve,
        x="time",
        y="psnr",
        hue="model",
        hue_order=list(styles),
        palette={model: color for model, (_, color, _) in styles.items()},
        style="model",
        dashes={model: dashes for model, (_, _, dashes) in styles.items()},
        errorbar=("se", BAND),
        err_kws={"alpha": 0.2, "linewidth": 0},
        legend=False,
        ax=plot,
    )
    means = curve.groupby(["model", "iteration"], as_index=False)[
        ["psnr", "time"]
    ].mean()
    low, high = y_range(means, "psnr")
    low, high = math.floor(min(low, floor or low)), math.ceil(high)
    shown = means[means["psnr"].between(low, high)]["time"]
    plot.set(
        xscale="log",
        xlim=(shown.min() / 1.1, means["time"].max() * 1.1),
        xlabel="Training time (s)",
        ylabel="PSNR (dB)",
    )
    log_ticks(plot.xaxis)
    quality_ticks(plot, "psnr", low, high)
    handles = [
        Line2D([], [], color=color, dashes=dashes or (None, None), label=label)
        for label, color, dashes in styles.values()
    ]
    figure.legend(
        handles=handles,
        loc="outside upper center",
        ncol=columns,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.4,
    )
    save(figure, path)


def write(rendered: dict[str, str], path: Path) -> None:
    """Save a table's renderings as <path>.tex and <path>.md.

    The LaTeX comes set in TABLE_SIZE with TABLE_GAP between columns, in a
    group of its own, so that it fits the text width wherever it is input.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix, text in rendered.items():
        if suffix == "tex":
            text = (
                rf"{{{TABLE_SIZE}\setlength{{\tabcolsep}}{{{TABLE_GAP}}}%"
                + f"\n{text}}}\n"
            )
        path.with_suffix(f".{suffix}").write_text(text)


def ablation(out_dir: Path) -> None:
    """The paper's tables and figures of the FUTON ablations."""
    studies = {name: study(name) for name in STUDIES}
    counts = ", ".join(
        f"{name} {len({run['model'] for run in runs})}"
        for name, runs in studies.items()
    )
    print(f"ablation: {counts} models")
    if studies["basis"]:
        write(basis_table(studies["basis"]), out_dir / "basis")
    for name, (x, label) in SWEPT.items():
        if studies[name]:
            final = sweep(studies[name])
            save(sweep_figure(final, x, label), out_dir / name)
            write(sweep_table(final, studies[name], name), out_dir / name)
    tabled = {"tensor_net": (COMBINERS, "Combiner"), "decoder": (DECODERS, "Decoder")}
    for name, (variants, index) in tabled.items():
        if studies[name]:
            table_of = variant_table(studies[name], variants, index)
            write(table_of, out_dir / OUTPUTS.get(name, name))
    for name, styles in CURVES.items():
        if studies[name]:
            study_curves(
                curves(studies[name]),
                styles,
                out_dir / OUTPUTS.get(name, name),
                FLOORS.get(name),
                LEGEND_COLUMNS.get(name, 2),
            )


def resolution(kappa: float) -> str:
    """A spectral resolution K = kappa N as text: N/8, N/4, N/2, N or 2N."""
    return FRACTIONS[kappa] + "N" if kappa > 1 else "N" + FRACTIONS[kappa][1:]


def rank_range(ranks: pd.Series) -> str:
    """Ranks as text: one, or the smallest to the largest."""
    low, high = ranks.min(), ranks.max()
    return f"{low}" if low == high else f"{low}–{high}"


def bound_records(task: str) -> pd.DataFrame:
    """Every recorded setting of the error bound on a task, in the study's basis,
    one row each, with PSNRs."""
    rows = []
    for path in sorted((ROOT / "logs" / "bound" / task).glob("*.json")):
        record = json.loads(path.read_text())
        rows += [{"signal": record["signal"], **c} for c in of_basis(record["configs"])]
    frame = pd.DataFrame(rows).drop(columns="basis", errors="ignore")
    for key in ("truncation", "lower", "upper", "certified", "adam", "closed"):
        if key in frame:
            frame[f"{key}_psnr"] = psnr(frame[key])
    return frame


def bound_means(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Means of the PSNRs ``keys`` per setting, each with one within-signal
    standard error over the signals."""
    long = pd.concat(
        frame[["signal", "kappa", "R", key]]
        .rename(columns={key: "psnr"})
        .assign(model=key)
        for key in keys
    )
    long["iteration"] = long["kappa"].astype(str) + long["R"].astype(str)
    long["psnr"] = within_signal(long, "psnr")
    return long.groupby(["kappa", "R", "model"])["psnr"].agg(["mean", "sem"])


def bound_gap(frames: dict[str, pd.DataFrame]) -> plt.Figure:
    """Each quantity's PSNR above Adam's at K = N/2, averaged over a dataset per
    rank with one standard error of the difference, one panel per dataset: the
    ceiling and the certified FUTON in its ALS and SVD forms, and between the
    ALS form and the ceiling, the region where the best FUTON lies."""
    lines = {
        "lower": {"label": "Ceiling", "color": RAMP[-1], "ls": (0, (3.5, 1.5))},
        "upper": {"label": "Certified (ALS)", "color": RAMP[2], "marker": "o"},
        "closed": {"label": "Certified (SVD)", "color": RAMP[2], "marker": "o",
                   "markerfacecolor": "white"},
    }  # fmt: skip
    region = {"color": RAMP[0], "alpha": 0.45, "linewidth": 0,
              "label": "Best FUTON lies here"}  # fmt: skip
    adam = {"color": "0.35", "lw": 0.8, "label": "Adam"}
    width = {"lw": 1.0, "markersize": 3}
    bar = {"capsize": 1.5, "elinewidth": 0.6, "capthick": 0.6, "zorder": 3}
    figure, axes = plt.subplots(
        1,
        len(frames),
        figsize=(len(frames) * SIZE[0], SIZE[1] + 2 * LEGEND_ROW),
    )
    for plot, (task, frame) in zip(axes, frames.items()):
        g = frame[frame["kappa"] == 0.5]
        above = {
            key: (g[f"{key}_psnr"] - g["adam_psnr"])
            .groupby(g["R"])
            .agg(["mean", "sem"])
            for key in lines
        }
        x = np.log2(above["upper"].index)
        plot.fill_between(x, above["upper"]["mean"], above["lower"]["mean"], zorder=1,
                          **region)  # fmt: skip
        plot.axhline(0, zorder=2, **adam)
        for key, style in lines.items():
            plot.errorbar(x, above[key]["mean"], yerr=above[key]["sem"], **width,
                          **bar, **style)  # fmt: skip
        low = (above["closed"]["mean"] - above["closed"]["sem"]).min()
        high = (above["lower"]["mean"] + above["lower"]["sem"]).max()
        plot.set_xticks(x, [f"{rank}" for rank in above["upper"].index])
        plot.set_ylim(low - 0.1 * (high - low), high + 0.1 * (high - low))
        plot.yaxis.set_major_locator(Halves(count=6))
        plot.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        plot.set_xlabel("Rank $R$")
        plot.set_title(BOUND_DATASETS[task], fontsize=8, pad=2)
    axes[0].set_ylabel("PSNR above Adam (dB)")
    # filled by column: the ceiling, the region and Adam, then the certified FUTON
    handles = [
        Line2D([], [], **width, **lines["lower"]),
        Line2D([], [], **width, **lines["upper"]),
        Patch(**region),
        Line2D([], [], **width, **lines["closed"]),
        Line2D([], [], **adam),
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False,
                  columnspacing=0.8, handlelength=1.8, handletextpad=0.4)  # fmt: skip
    return figure


def bound_decomposition(frames: dict[str, pd.DataFrame]) -> plt.Figure:
    """Adam's error at K = N/2 split as the theorem splits it, in percent of
    it averaged over a dataset per rank, one panel per dataset: the
    truncation, which K forces; the rank's part up to the ceiling, which R
    forces; the part from the ceiling to the certified FUTON, which the bound
    leaves open; and the rest, which training leaves."""
    parts = {"Truncation": RAMP[-1], "Rank, certain": RAMP[2],
             "Rank, open": RAMP[0], "Left by training": "0.85"}  # fmt: skip
    figure, axes = plt.subplots(
        1,
        len(frames),
        figsize=(len(frames) * SIZE[0], SIZE[1] + LEGEND_ROW),
    )
    for plot, (task, frame) in zip(axes, frames.items()):
        g = frame[frame["kappa"] == 0.5]
        share = {
            key: (100 * g[key] / g["adam"]).groupby(g["R"]).mean()
            for key in ("truncation", "lower", "upper")
        }
        x = np.arange(len(share["upper"]))
        levels = [0, share["truncation"], share["lower"], share["upper"], 100]
        for color, base, level in zip(parts.values(), levels, levels[1:]):
            plot.bar(x, level - base, bottom=base, width=0.6, color=color)
        plot.set_xticks(x, [f"{rank}" for rank in share["upper"].index])
        plot.set_ylim(0, 100)
        plot.yaxis.set_major_locator(Halves())
        plot.set_xlabel("Rank $R$")
        plot.set_title(BOUND_DATASETS[task], fontsize=8, pad=2)
    axes[0].set_ylabel("Adam's error (%)")
    handles = [Patch(color=color, label=label) for label, color in parts.items()]
    figure.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False,
                  columnspacing=0.8, handlelength=1.2, handletextpad=0.4)  # fmt: skip
    return figure


def bound_cost(frames: dict[str, pd.DataFrame]) -> plt.Figure:
    """Seconds per signal for every rank of each K: the bound with the
    certified FUTON in its SVD form and in its ALS form, which do every rank
    at once, against the Adam runs, one per rank, all averaged over the
    signals."""
    bars = {"closed": (RAMP[0], "SVD"), "bound": (RAMP[3], "ALS"),
            "adam": ("0.7", "Adam")}  # fmt: skip
    figure, plot = plt.subplots(figsize=(SIZE[0], SIZE[1] + 2 * LEGEND_ROW))
    labels, positions, centres, position = [], [], {}, 0.0
    for task, frame in frames.items():
        per_kappa = frame.groupby(["signal", "kappa"]).agg(
            **{
                key: (f"{key}_seconds", "sum" if key == "adam" else "first")
                for key in bars
            }
        )
        first = position
        # Past four levels the two datasets' labels crowd the panel: every other
        # one is labelled, the benchmark's 1/2 among them.
        kappas = sorted(per_kappa.index.unique("kappa"))
        labelled = kappas.index(0.5) % 2 if len(kappas) > 4 else None
        for level, (kappa, group) in enumerate(per_kappa.groupby(level="kappa")):
            width = 0.8 / len(bars)
            for i, (key, (color, _)) in enumerate(bars.items()):
                offset = (i - (len(bars) - 1) / 2) * width
                plot.bar(position + offset, group[key].mean(), width, color=color)
            labels.append(FRACTIONS[kappa] if labelled in (None, level % 2) else "")
            positions.append(position)
            position += 1
        centres[task] = (first + position - 1) / 2
        position += 0.5
    plot.yaxis.set_major_locator(Halves(whole=True))
    plot.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    plot.set_xticks(positions, labels)
    plot.tick_params(axis="x", length=0)
    plot.set(xlabel=r"Relative bandwidth $\kappa$", ylabel="Time (s)")
    for task, centre in centres.items():
        plot.text(centre, 1.0, BOUND_DATASETS[task], ha="center", va="bottom",
                  transform=plot.get_xaxis_transform(), fontsize=7)  # fmt: skip
    handles = [
        Rectangle((0, 0), 1, 1, color=color, label=label)
        for color, label in bars.values()
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False,
                  columnspacing=1.0, handlelength=1.0, handletextpad=0.4)  # fmt: skip
    return figure


def bound_design(frame: pd.DataFrame, targets) -> pd.DataFrame:
    """For each target PSNR, the smallest setting the bound certifies for each
    signal against the smallest Adam reaches, by parameter count."""
    rows = []
    for target in targets:
        picks = []
        for signal, group in frame.groupby("signal"):
            certified = group[group["upper_psnr"] >= target]["parameters"]
            trained = group[group["adam_psnr"] >= target]["parameters"]
            possible = group[group["lower_psnr"] >= target]["parameters"]
            picks.append((certified.min(), trained.min(), possible.min()))
        picks = pd.DataFrame(picks, columns=["bound", "adam", "ceiling"])
        both = picks.dropna(subset=["bound", "adam"])
        rows.append({
            "target": target,
            "signals": len(picks),
            "reached_adam": int(picks["adam"].notna().sum()),
            "reached_bound": int(picks["bound"].notna().sum()),
            "possible": int(picks["ceiling"].notna().sum()),
            "same": float((both["bound"] == both["adam"]).mean()),
            "smaller": float((both["bound"] < both["adam"]).mean()),
            "larger": float((both["bound"] > both["adam"]).mean()),
            "parameters_bound": picks["bound"].median(),
            "parameters_adam": picks["adam"].median(),
        })  # fmt: skip
    return pd.DataFrame(rows)


def bound_cell(mean: float, sem: float) -> str:
    return f"{mean:.2f} ± {sem:.2f}"


def bound_tables(
    frames: dict[str, pd.DataFrame], every: dict[str, pd.DataFrame]
) -> list[dict]:
    """The report's tables: means over each task's signals per setting, the
    design for a target PSNR and the optimizer checks, from ``frames``, and
    the SVD form's cost, from ``every``, which adds K = N. Each is a title, a
    note, a header of (group, title) pairs and rows of text."""
    built = []
    columns = ["truncation_psnr", "upper_psnr", "lower_psnr", "adam_psnr"]
    for task, frame in frames.items():
        stats = bound_means(frame, columns)
        seconds = frame.groupby(["kappa", "R"])[["bound_seconds", "adam_seconds"]]
        rows = [
            [resolution(kappa), str(rank),
             *(bound_cell(*stats.loc[(kappa, rank, key)]) for key in columns),
             f"{row.bound_seconds:.1f}", f"{row.adam_seconds:.1f}"]
            for (kappa, rank), row in seconds.mean().iterrows()
        ]  # fmt: skip
        built.append({
            "title": f"{task} ({frame['signal'].nunique()} signals)",
            "note": "Mean PSNR (dB) over the signals, ± one within-signal standard "
                    "error; seconds for the bound at all ranks of a K and for one "
                    "Adam run.",
            "header": [("", "K"), ("", "R"), ("", "Truncation"),
                       ("", "Certified (ALS)"), ("", "Ceiling"), ("", "Adam"),
                       ("Seconds", "bound"), ("Seconds", "Adam")],
            "rows": rows,
        })  # fmt: skip
    for task, frame in frames.items():
        picks = bound_design(frame, BOUND_TARGETS[task])
        bound = frame.groupby(["signal", "kappa"])["bound_seconds"].first()
        bound = bound.groupby("signal").sum().mean()
        adam = frame.groupby("signal")["adam_seconds"].sum().mean()
        built.append({
            "title": f"{task}: the smallest setting for a target PSNR",
            "note": f"The bound takes {bound:.1f} s per signal for every setting, "
                    f"Adam {adam:.0f} s.",
            "header": [("", "Target (dB)"), ("Reached", "Adam"), ("Reached", "bound"),
                       ("Reached", "possible"), ("Pick vs. oracle", "same"),
                       ("Pick vs. oracle", "smaller"), ("Pick vs. oracle", "larger"),
                       ("#Params (k)", "bound"), ("#Params (k)", "Adam")],
            "rows": [[
                f"{row.target:g}", f"{row.reached_adam:.0f}/{row.signals:.0f}",
                f"{row.reached_bound:.0f}/{row.signals:.0f}",
                f"{row.possible:.0f}/{row.signals:.0f}", f"{row.same:.0%}",
                f"{row.smaller:.0%}", f"{row.larger:.0%}",
                f"{row.parameters_bound / 1e3:.1f}", f"{row.parameters_adam / 1e3:.1f}",
            ] for _, row in picks.iterrows()],
        })  # fmt: skip
    for task, frame in every.items():
        rows = []
        for kappa, group in frame.groupby("kappa"):
            per_signal = group.groupby("signal")
            gap = group["upper_psnr"] - group["closed_psnr"]
            rows.append([
                resolution(kappa),
                f"{per_signal['closed_seconds'].first().mean():.2f}",
                f"{per_signal['bound_seconds'].first().mean():.1f}",
                f"{per_signal['adam_seconds'].sum().mean():.0f}",
                f"{gap.median():.2f}", f"{gap.max():.2f}",
            ])  # fmt: skip
        built.append({
            "title": f"{task}: the SVD form",
            "note": "Seconds per signal for every rank of a K, averaged over the "
                    "signals: the bound with the certified FUTON in its SVD form (no "
                    "sweeps), in its ALS form, and the Adam runs; and how far below the "
                    "ALS form the SVD form certifies, in dB, over the signals and "
                    "ranks.",
            "header": [("", "K"), ("Seconds", "SVD"), ("Seconds", "ALS"),
                       ("Seconds", "Adam"), ("SVD, dB below ALS", "median"),
                       ("SVD, dB below ALS", "max")],
            "rows": rows,
        })  # fmt: skip
    methods = ["upper", "lower", "adam", "adam_longer", *(f"als_{s}" for s in SWEEPS)]
    rows = []
    for task in frames:
        path = ROOT / "logs" / "bound" / f"{task}_checks.json"
        if not path.exists():
            continue
        runs = pd.DataFrame(of_basis(json.loads(path.read_text())))
        long = runs.melt(id_vars="signal", value_vars=methods, var_name="model")
        long = long.assign(psnr=psnr(long["value"]), iteration=0)
        long["psnr"] = within_signal(long, "psnr")
        stats = long.groupby("model")["psnr"].agg(["mean", "sem"])
        rows.append([BOUND_DATASETS[task], rank_range(runs["R"]),
                     *(bound_cell(*stats.loc[key]) for key in methods)])  # fmt: skip
    if rows:
        built.append({
            "title": "optimizer checks, K = N/2 and the benchmark's rank",
            "note": "Mean PSNR (dB) over the signals, ± one within-signal standard "
                    "error: the certified FUTON in its ALS form and the ceiling, Adam "
                    f"at the benchmark's {STEPS} steps and at {LONGER * STEPS:,}, "
                    "and alternating least squares on the whole grid from a random "
                    f"start, after {SWEEPS[0]} and {SWEEPS[1]} sweeps.",
            "header": [("", "Dataset"), ("", "R"), ("", "Certified (ALS)"),
                       ("", "Ceiling"), ("Adam", f"{STEPS} steps"),
                       ("Adam", f"{LONGER * STEPS:,} steps"),
                       ("ALS, random start", f"{SWEEPS[0]} sweeps"),
                       ("ALS, random start", f"{SWEEPS[1]} sweeps")],
            "rows": rows,
        })  # fmt: skip
    return built


def bound_markdown(built: list[dict]) -> str:
    """The tables in Markdown, one section each."""
    lines = []
    for table in built:
        header = [", ".join(filter(None, pair)) for pair in table["header"]]
        lines += [f"## {table['title']}", "", table["note"], "",
                  "| " + " | ".join(header) + " |",
                  "| --- |" + " ---: |" * (len(header) - 1)]  # fmt: skip
        lines += ["| " + " | ".join(row) + " |" for row in table["rows"]]
        lines.append("")
    return "\n".join(lines)


def bound_latex(built: list[dict]) -> str:
    """The tables as booktabs tabulars, set as the benchmarks' tables are: a
    spanning header over each group, titles centred over right-aligned
    numbers, and the standard error in a smaller size after the mean."""

    def tex(text: str) -> str:
        text = re.sub(r" ± (\S+)", r"{\\scriptsize$\\pm$\1}", text)
        text = re.sub(r"\bN/(\d)", r"$N/\1$", text)
        text = re.sub(r"^(\d*)N$", r"$\1N$", text)  # N, 2N
        for symbol, command in (
            ("×", r"$\times$"),
            ("–", "--"),
            ("#", r"\#"),
            ("%", r"\%"),
        ):
            text = text.replace(symbol, command)
        return {"K": "$K$", "R": "$R$"}.get(text, text)

    blocks = []
    for table in built:
        groups, titles = zip(*table["header"])
        lines = [f"% {table['title']}",
                 rf"{{{TABLE_SIZE}\setlength{{\tabcolsep}}{{{TABLE_GAP}}}%",
                 rf"\begin{{tabular}}{{@{{}}l{'r' * (len(titles) - 1)}@{{}}}}",
                 r"\toprule"]  # fmt: skip
        if any(groups):
            lines += spanning([tex(group) for group in groups])
        titles = [tex(title) for title in titles]
        names = [titles[0]] + [
            rf"\multicolumn{{1}}{{c}}{{{title[:1].upper() + title[1:]}}}"
            for title in titles[1:]
        ]
        lines += [" & ".join(names) + r" \\", r"\midrule"]
        lines += [" & ".join(map(tex, row)) + r" \\" for row in table["rows"]]
        lines += [r"\bottomrule", r"\end{tabular}}", ""]
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def rate(lr: float) -> str:
    """A learning rate as the grid writes it, 3e-4 for 0.0003."""
    return f"{lr:.0e}".replace("e-0", "e-")


def tuning_scores(task: str) -> pd.DataFrame:
    """One row per tuning run under logs/tune-lr/: its model, the setting
    tuned beside the rate, the rate, the signal and the score the rate was
    chosen by.

    Each rate tried, the configured one too, has its own run there, so that
    a model's rates are compared on one GPU. RFF's frequency scale is tuned
    with its rate on images, and each radiance field's colour network gets a
    rate of its own, "coupled" when it shares the density network's.
    """
    metric = TASKS[task][0]
    runs = read(ROOT / "logs" / "tune-lr", task)
    bicubic = {
        run["data"]: run["metrics"]["psnr"] for run in runs if run["model"] == REFERENCE
    }
    rows = []
    for run in runs:
        setup = run["setup"]
        if "interpolation" in setup:
            continue
        setting = ""
        if task == "nerf":
            score = run["history"][-1]["eval"]["psnr"]  # the validation views
            color = run.get("color_lr")
            setting = "coupled" if color is None else f"colour {rate(color)}"
        else:
            score = scale(metric, run["metrics"][metric])
            if task == "super_resolution":
                score -= bicubic[run["data"]]
            if task == "image" and run["model"] == "RFF":
                setting = f"σ {setup['kwargs']['sigma']:g}"
        rows.append(
            {
                "model": run["model"],
                "setting": setting,
                "lr": setup["train"]["lr"],
                "signal": run["data"],
                "score": score,
            }
        )
    return pd.DataFrame(rows)


def tuning_table(scores: pd.DataFrame, task: str) -> dict[str, str]:
    """LaTeX and Markdown of every model's mean score at each rate it was run
    at, its best in bold; a cell over fewer signals than the task's is starred."""
    grouped = scores.groupby(["model", "setting", "lr"])["score"]
    means, counts = grouped.mean().unstack("lr"), grouped.count().unstack("lr")
    rates = [lr for lr in LEARNING_RATES if lr in means.columns]
    full = scores["signal"].nunique()
    order = {model: rank for rank, model in enumerate(ROWS)}
    keys = sorted(means.index, key=lambda key: (order.get(key[0], len(ROWS)), key[1]))
    header = ["Model", *map(rate, rates)]
    lines = {"md": [], "tex": []}
    for model, setting in keys:
        row = means.loc[(model, setting), rates]
        name = display_name(model) + (f", {setting}" if setting else "")
        cells = {"md": [name], "tex": [name]}
        for lr in rates:
            if pd.isna(row[lr]):
                cells["md"].append("–")
                cells["tex"].append("–")
                continue
            text = f"{row[lr]:.2f}" + (
                "*" if counts.loc[(model, setting), lr] < full else ""
            )
            best = row[lr] == row.max()
            cells["md"].append(f"**{text}**" if best else text)
            cells["tex"].append(rf"\textbf{{{text}}}" if best else text)
        lines["md"].append("| " + " | ".join(cells["md"]) + " |")
        lines["tex"].append(" & ".join(cells["tex"]) + r" \\")
    markdown = "\n".join(
        [
            f"Learning rates of the {task.replace('_', ' ')} task, by {TUNED_BY[task]}.",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header),
            *lines["md"],
        ]
    )
    latex = "\n".join(
        [
            r"\begin{tabular}{l" + "r" * len(rates) + "}",
            r"\toprule",
            " & ".join(header) + r" \\",
            r"\midrule",
            *lines["tex"],
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return {"md": markdown + "\n", "tex": latex + "\n"}


def tuning(out_dir: Path) -> None:
    """The learning-rate tables of every task that has tuning runs."""
    written = False
    for task in TASKS:
        scores = tuning_scores(task)
        if scores.empty:
            continue
        write(tuning_table(scores, task), out_dir / task)
        print(f"tuning: {task} written to {out_dir / task}")
        written = True
    if not written:
        print("tuning: no runs under logs/tune-lr, skipped")


def bound(out_dir: Path) -> None:
    """The figures and tables of the error bound, from the records of
    scripts/futon_bound.py."""
    every = {
        task: bound_records(task)
        for task in BOUND_DATASETS
        if (ROOT / "logs" / "bound" / task).exists()
    }
    if not every:
        print("bound: no records under logs/bound, skipped")
        return
    # K = N and 2N enter the cost alone, where they show how the time grows with K.
    frames = {task: frame[frame["kappa"] < 1] for task, frame in every.items()}
    save(bound_gap(frames), out_dir / "gap")
    save(bound_decomposition(frames), out_dir / "decomposition")
    save(bound_cost(every), out_dir / "cost")
    built = bound_tables(frames, every)
    (out_dir / "table.md").write_text(bound_markdown(built))
    (out_dir / "table.tex").write_text(bound_latex(built))
    print(f"bound: written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(PARTS),
        choices=PARTS,
        help="what to build: tasks, the ablations, the error bound (default: all)",
    )
    parser.add_argument(
        "--signals", nargs="+", help="signals to draw (default: the task's examples)"
    )
    parser.add_argument(
        "--no-panels", action="store_true", help="skip the renders, which need a GPU"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="render panels that are already there"
    )
    parser.add_argument(
        "--orbit",
        type=int,
        default=0,
        metavar="FRAMES",
        help="also save an orbit GIF of each NeRF model, for a talk",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    style()
    out_dir = ROOT / "results"
    save(legend(), out_dir / "legend")
    for task in (task for task in args.tasks if task in TASKS):
        if not (ROOT / "logs" / task).exists():
            print(f"{task}: no runs under logs/{task}, skipped")
            continue
        runs, final, curve = load(task)
        write(tables(final, task), out_dir / task / "table")
        if task == "super_resolution":
            held_out = final[~final["signal"].isin(DEVELOPED_ON)]
            write(tables(held_out, task), out_dir / task / "table_held_out")
        if not final["model"].isin(FEATURED).any():  # a sweep still running
            print(f"{task}: no featured model has finished, so no figures yet")
            continue
        if REFERENCE in final["model"].values:
            save(legend(reference=True), out_dir / task / "legend")
        save(convergence(curve, task), out_dir / task / "convergence")
        save(tradeoff(final, task), out_dir / task / "training_time")
        save(tradeoff(final, task, "speed"), out_dir / task / "throughput")
        save(tradeoff(final, task, "train_memory"), out_dir / task / "training_memory")
        save(tradeoff(final, task, "infer_memory"), out_dir / task / "inference_memory")
        for signal in [] if args.no_panels else args.signals or EXAMPLES[task]:
            directory = panels(task, signal, runs, args.device, args.overwrite)
            if args.orbit and task == "nerf":
                chosen = [
                    run
                    for run in runs
                    if run["data"] == signal and run["model"] in FEATURED
                ]
                render_orbit(
                    signal, chosen, directory, args.device, args.orbit, args.overwrite
                )
            figure = qualitative(task, signal, final, directory)
            # PDF only: a PGF of raster panels writes each one out beside it.
            # Laid out in inches to WIDTH, so saved whole rather than trimmed.
            figure.savefig(out_dir / task / f"qualitative_{signal}.pdf", dpi=600)
            plt.close(figure)
        print(f"{task}: written to {out_dir / task}")
    if "ablation" in args.tasks:
        ablation(out_dir / "ablation")
    if "bound" in args.tasks:
        bound(out_dir / "bound")
    if "tuning" in args.tasks:
        tuning(out_dir / "tuning")
    if "occupancy" in args.tasks and (ROOT / "logs" / "occupancy").exists():
        # For the text: who reaches a high IoU, and how soon.
        print(f"\ntime to {TARGET:g}% IoU on every shape (s):")
        print(time_to_target(load("occupancy")[2]).dropna().sort_values().to_string())


if __name__ == "__main__":
    main()
