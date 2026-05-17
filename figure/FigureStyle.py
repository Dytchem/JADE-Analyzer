"""
FigureStyle: Defines publication-quality figure styles and parameters.
"""

import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

# Load custom fonts from project directory
_font_dir = Path(__file__).resolve().parent.parent / "font"
if _font_dir.exists():
    for font_file in _font_dir.glob("*.otf"):
        try:
            fm.fontManager.addfont(str(font_file))
        except Exception:
            pass


class FigureStyle:
    """
    Manages figure styling parameters for publication-quality plots.

    Provides predefined styles for different journals and custom styling
    options for axes, labels, legends, and other figure elements.
    """

    # Publication standards for different journals
    JOURNAL_STYLES = {
        "nature": {
            "fig_width": 3.5,
            "fig_height": 3.5,
            "font_family": "Arial",
            "font_size": 8,
            "label_size": 8,
            "legend_size": 7,
            "line_width": 1.0,
            "marker_size": 3,
            "dpi": 300,
        },
        "science": {
            "fig_width": 3.5,
            "fig_height": 2.5,
            "font_family": "Arial",
            "font_size": 9,
            "label_size": 9,
            "legend_size": 8,
            "line_width": 1.0,
            "marker_size": 3,
            "dpi": 300,
        },
        "acs": {
            "fig_width": 3.25,
            "fig_height": 2.25,
            "font_family": "Arial",
            "font_size": 8,
            "label_size": 8,
            "legend_size": 7,
            "line_width": 0.75,
            "marker_size": 2.5,
            "dpi": 300,
        },
        "rsc": {
            "fig_width": 3.25,
            "fig_height": 2.5,
            "font_family": "Arial",
            "font_size": 8,
            "label_size": 8,
            "legend_size": 7,
            "line_width": 0.75,
            "marker_size": 2.5,
            "dpi": 300,
        },
        "default": {
            "fig_width": 4,
            "fig_height": 3,
            "font_family": "sans-serif",
            "font_size": 10,
            "label_size": 10,
            "legend_size": 9,
            "line_width": 1.0,
            "marker_size": 4,
            "dpi": 150,
        },
    }

    # Color palettes
    COLOR_PALETTES = {
        "default": [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ],
        "colorblind": [
            "#0173b2",
            "#de8f05",
            "#029e73",
            "#d55e00",
            "#cc78bc",
            "#ca9161",
            "#fbaed2",
            "#949494",
            "#ece133",
            "#56b4e9",
        ],
        "grayscale": ["#000000", "#404040", "#808080", "#bfbfbf", "#e0e0e0"],
        "vibrant": [
            "#0077bb",
            "#33bbee",
            "#ee7733",
            "#cc3311",
            "#009988",
            "#ee3377",
            "#bbbbbb",
        ],
    }

    def __init__(self, journal: str = "default"):
        """
        Initialize FigureStyle with journal-specific settings.

        Args:
            journal: Journal style preset ('nature', 'science', 'acs', 'rsc', 'default')
        """
        self.journal = journal
        self.params = self.JOURNAL_STYLES.get(journal, self.JOURNAL_STYLES["default"])
        self.colors = self.COLOR_PALETTES["default"]
        self.apply_style()

    def apply_style(self):
        """Apply the current style to matplotlib."""
        mpl.rcParams["figure.figsize"] = (
            self.params["fig_width"],
            self.params["fig_height"],
        )
        mpl.rcParams["font.family"] = [
            "Source Han Sans SC",
            "Arial",
            "DejaVu Sans",
            self.params["font_family"],
        ]
        mpl.rcParams["font.size"] = self.params["font_size"]
        mpl.rcParams["axes.labelsize"] = self.params["label_size"]
        mpl.rcParams["axes.titlesize"] = self.params["label_size"]
        mpl.rcParams["xtick.labelsize"] = self.params["label_size"]
        mpl.rcParams["ytick.labelsize"] = self.params["label_size"]
        mpl.rcParams["legend.fontsize"] = self.params["legend_size"]
        mpl.rcParams["lines.linewidth"] = self.params["line_width"]
        mpl.rcParams["lines.markersize"] = self.params["marker_size"]
        mpl.rcParams["figure.dpi"] = self.params["dpi"]
        mpl.rcParams["savefig.dpi"] = self.params["dpi"]
        mpl.rcParams["font.sans-serif"] = ["Source Han Sans SC", "Arial", "DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False

    def set_color_palette(self, palette: str):
        """
        Set the color palette for plots.

        Args:
            palette: Palette name ('default', 'colorblind', 'grayscale', 'vibrant')
        """
        self.colors = self.COLOR_PALETTES.get(palette, self.COLOR_PALETTES["default"])

    def get_color(self, index: int) -> str:
        """
        Get color from the current palette.

        Args:
            index: Index of the color in the palette

        Returns:
            Hex color code
        """
        return self.colors[index % len(self.colors)]

    def update_params(self, **kwargs):
        """
        Update specific style parameters.

        Args:
            **kwargs: Parameters to update (e.g., fig_width=5, font_size=12)
        """
        self.params.update(kwargs)
        self.apply_style()

    @staticmethod
    def reset_style():
        """Reset matplotlib to default style."""
        mpl.rcdefaults()

    @staticmethod
    def get_available_journals():
        """Get list of available journal styles."""
        return list(FigureStyle.JOURNAL_STYLES.keys())

    @staticmethod
    def get_available_palettes():
        """Get list of available color palettes."""
        return list(FigureStyle.COLOR_PALETTES.keys())


def configure_publication_style(
    chinese_family: str = "SimSun",
    english_family: str = "Times New Roman",
    base_font_size: float = 10.5,
    legend_font_size: float = 8,
    prefer_latex: bool = True,
):
    """Configure publication-style rcParams and return reusable font properties.

    Args:
        chinese_family: Font family for Chinese labels.
        english_family: Font family for English text and numbers.
        base_font_size: Base font size in points.
        legend_font_size: Legend font size in points.
        prefer_latex: Enable LaTeX rendering when TeX engine is available.

    Returns:
        A dictionary containing font properties and latex availability.
    """
    latex_available = False
    if prefer_latex:
        latex_available = any(
            shutil.which(cmd) for cmd in ("xelatex", "pdflatex", "latex")
        )

    mpl.rcParams["text.usetex"] = latex_available
    if latex_available:
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    mpl.rcParams["font.family"] = [english_family, chinese_family, "serif"]
    mpl.rcParams["font.size"] = base_font_size
    mpl.rcParams["axes.unicode_minus"] = True

    # Keep math symbols and subscripts consistent with paper typography.
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = english_family
    mpl.rcParams["mathtext.it"] = f"{english_family}:italic"
    mpl.rcParams["mathtext.bf"] = f"{english_family}:bold"

    # Publication-like visual details.
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["lines.linewidth"] = 1.1
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.major.size"] = 3.5
    mpl.rcParams["ytick.major.size"] = 3.5
    mpl.rcParams["xtick.major.width"] = 0.8
    mpl.rcParams["ytick.major.width"] = 0.8
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["savefig.bbox"] = "tight"

    return {
        "latex_available": latex_available,
        "chinese_font": fm.FontProperties(family=[chinese_family], size=base_font_size),
        "english_font": fm.FontProperties(
            family=[english_family, "DejaVu Serif"], size=base_font_size
        ),
        "legend_font": fm.FontProperties(
            family=[english_family, "DejaVu Serif"], size=legend_font_size
        ),
    }
