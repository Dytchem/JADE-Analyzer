"""
JADE-Analyzer Figure Module for Publication-Ready Plots.

This module provides publication-quality figure formatting and styling
for academic papers, ensuring consistent and professional visualizations.
"""

from .PaperFigure import PaperFigure
from .FigureStyle import FigureStyle, configure_publication_style

__all__ = ["PaperFigure", "FigureStyle", "configure_publication_style"]
