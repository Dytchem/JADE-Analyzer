"""
PaperFigure: High-level interface for creating publication-quality figures.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any

from .FigureStyle import FigureStyle


def _find_inkscape():
    """Find Inkscape executable path."""
    import os
    candidates = [
        r'C:\Program Files\Inkscape\bin\inkscape.exe',
        r'C:\Program Files\Inkscape\inkscape.exe',
        r'C:\Program Files (x86)\Inkscape\bin\inkscape.exe',
        r'C:\Program Files (x86)\Inkscape\inkscape.exe',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    import shutil
    if shutil.which('inkscape'):
        return shutil.which('inkscape')
    return None


def _find_pstoedit():
    """Find pstoedit executable path."""
    import os, shutil
    candidates = [
        r'C:\Program Files\pstoedit\pstoedit.exe',
        r'C:\Program Files (x86)\pstoedit\pstoedit.exe',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return shutil.which('pstoedit')


class PaperFigure:
    """
    A wrapper class for creating publication-ready matplotlib figures.

    Provides convenient methods for common plot types and ensures
    consistent styling across all figures for a publication.

    Example:
        >>> fig = PaperFigure(journal='nature')
        >>> fig.plot(x, y, label='data')
        >>> fig.xlabel('Time (fs)')
        >>> fig.ylabel('Energy (eV)')
        >>> fig.legend()
        >>> fig.save('output/figure1.png')
    """

    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        journal: str = 'default',
        sharex: bool = False,
        sharey: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs
    ):
        """
        Initialize PaperFigure with subplot grid.

        Args:
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            journal: Journal style preset ('nature', 'science', 'acs', 'rsc', 'default')
            sharex: Whether to share x-axis across subplots
            sharey: Whether to share y-axis across subplots
            figsize: Custom figure size (width, height) in inches
            **kwargs: Additional arguments passed to plt.subplots
        """
        self.style = FigureStyle(journal)
        if figsize is not None:
            self.style.update_params(fig_width=figsize[0], fig_height=figsize[1])

        self.fig, self.axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            figsize=(self.style.params['fig_width'], self.style.params['fig_height']),
            **kwargs
        )

        if nrows == 1 and ncols == 1:
            self.axes = np.array([self.axes])
        self.axes = np.atleast_2d(self.axes)

        self.current_subplot = (0, 0)
        self._labels_added = set()

    def _get_ax(self, ax_index: Optional[Tuple[int, int]] = None):
        """Get the axes object for the specified subplot."""
        if ax_index is None:
            ax_index = self.current_subplot
        row, col = ax_index
        if row < self.axes.shape[0] and col < self.axes.shape[1]:
            return self.axes[row, col]
        return self.axes.flat[0]

    def set_subplot(self, row: int, col: int):
        """Set the current active subplot by row and column."""
        self.current_subplot = (row, col)
        return self

    def plot(
        self,
        x,
        y,
        ax_index: Optional[Tuple[int, int]] = None,
        color: Optional[str] = None,
        label: Optional[str] = None,
        linestyle: str = '-',
        linewidth: Optional[float] = None,
        marker: Optional[str] = None,
        markersize: Optional[float] = None,
        **kwargs
    ):
        """
        Plot data on the current or specified subplot.

        Args:
            x: x-axis data
            y: y-axis data
            ax_index: Subplot index (row, col), None for current
            color: Line color
            label: Legend label
            linestyle: Line style ('-', '--', ':', '-.')
            linewidth: Line width
            marker: Marker style ('o', 's', '^', etc.)
            markersize: Marker size
            **kwargs: Additional arguments passed to ax.plot

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)

        if color is None:
            color = self.style.get_color(len(self._labels_added))

        lw = linewidth if linewidth is not None else self.style.params['line_width']
        ms = markersize if markersize is not None else self.style.params['marker_size']

        line, = ax.plot(x, y, color=color, label=label, linestyle=linestyle,
                        linewidth=lw, marker=marker, markersize=ms, **kwargs)

        if label is not None:
            self._labels_added.add(label)

        return self

    def scatter(
        self,
        x,
        y,
        ax_index: Optional[Tuple[int, int]] = None,
        color: Optional[str] = None,
        label: Optional[str] = None,
        s: float = 20,
        marker: str = 'o',
        **kwargs
    ):
        """
        Create a scatter plot.

        Args:
            x: x-axis data
            y: y-axis data
            ax_index: Subplot index
            color: Point color
            label: Legend label
            s: Marker size
            marker: Marker style
            **kwargs: Additional arguments passed to ax.scatter

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)

        if color is None:
            color = self.style.get_color(len(self._labels_added))

        ax.scatter(x, y, c=color, s=s, label=label, marker=marker, **kwargs)

        if label is not None:
            self._labels_added.add(label)

        return self

    def errorbar(
        self,
        x,
        y,
        yerr: Optional[np.ndarray] = None,
        xerr: Optional[np.ndarray] = None,
        ax_index: Optional[Tuple[int, int]] = None,
        color: Optional[str] = None,
        label: Optional[str] = None,
        linestyle: str = '-',
        capsize: float = 2,
        **kwargs
    ):
        """
        Create an error bar plot.

        Args:
            x: x-axis data
            y: y-axis data
            yerr: Error in y values
            xerr: Error in x values
            ax_index: Subplot index
            color: Line color
            label: Legend label
            linestyle: Line style
            capsize: Error bar cap size
            **kwargs: Additional arguments passed to ax.errorbar

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)

        if color is None:
            color = self.style.get_color(len(self._labels_added))

        ax.errorbar(x, y, yerr=yerr, xerr=xerr, color=color, label=label,
                    linestyle=linestyle, capsize=capsize, **kwargs)

        if label is not None:
            self._labels_added.add(label)

        return self

    def fill_between(
        self,
        x,
        y1,
        y2,
        ax_index: Optional[Tuple[int, int]] = None,
        color: Optional[str] = None,
        alpha: float = 0.3,
        label: Optional[str] = None,
        **kwargs
    ):
        """
        Fill the area between two curves.

        Args:
            x: x-axis data
            y1: First y values
            y2: Second y values
            ax_index: Subplot index
            color: Fill color
            alpha: Transparency (0-1)
            label: Legend label
            **kwargs: Additional arguments passed to ax.fill_between

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)

        if color is None:
            color = self.style.get_color(len(self._labels_added))

        ax.fill_between(x, y1, y2, color=color, alpha=alpha, label=label, **kwargs)

        if label is not None:
            self._labels_added.add(label)

        return self

    def histogram(
        self,
        data,
        ax_index: Optional[Tuple[int, int]] = None,
        bins: int = 30,
        color: Optional[str] = None,
        label: Optional[str] = None,
        alpha: float = 0.7,
        density: bool = False,
        **kwargs
    ):
        """
        Create a histogram.

        Args:
            data: Data to histogram
            ax_index: Subplot index
            bins: Number of bins
            color: Bar color
            label: Legend label
            alpha: Transparency
            density: Normalize to density
            **kwargs: Additional arguments passed to ax.hist

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)

        if color is None:
            color = self.style.get_color(len(self._labels_added))

        ax.hist(data, bins=bins, color=color, label=label, alpha=alpha,
                density=density, **kwargs)

        if label is not None:
            self._labels_added.add(label)

        return self

    def imshow(
        self,
        data,
        ax_index: Optional[Tuple[int, int]] = None,
        cmap: str = 'viridis',
        aspect: Optional[str] = None,
        origin: str = 'lower',
        **kwargs
    ):
        """
        Display an image or 2D array.

        Args:
            data: 2D array to display
            ax_index: Subplot index
            cmap: Colormap name
            aspect: Aspect ratio ('auto', 'equal', or a number)
            origin: Image origin ('upper', 'lower')
            **kwargs: Additional arguments passed to ax.imshow

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)
        im = ax.imshow(data, cmap=cmap, aspect=aspect, origin=origin, **kwargs)
        return im, self

    def xlabel(
        self,
        label: str,
        ax_index: Optional[Tuple[int, int]] = None,
        fontsize: Optional[float] = None,
        **kwargs
    ):
        """
        Set the x-axis label.

        Args:
            label: Label text
            ax_index: Subplot index
            fontsize: Font size
            **kwargs: Additional arguments passed to ax.set_xlabel

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)
        fs = fontsize if fontsize is not None else self.style.params['label_size']
        ax.set_xlabel(label, fontsize=fs, **kwargs)
        return self

    def ylabel(
        self,
        label: str,
        ax_index: Optional[Tuple[int, int]] = None,
        fontsize: Optional[float] = None,
        **kwargs
    ):
        """
        Set the y-axis label.

        Args:
            label: Label text
            ax_index: Subplot index
            fontsize: Font size
            **kwargs: Additional arguments passed to ax.set_ylabel

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)
        fs = fontsize if fontsize is not None else self.style.params['label_size']
        ax.set_ylabel(label, fontsize=fs, **kwargs)
        return self

    def title(
        self,
        title: str,
        ax_index: Optional[Tuple[int, int]] = None,
        fontsize: Optional[float] = None,
        **kwargs
    ):
        """
        Set the subplot title.

        Args:
            title: Title text
            ax_index: Subplot index
            fontsize: Font size
            **kwargs: Additional arguments passed to ax.set_title

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)
        fs = fontsize if fontsize is not None else self.style.params['label_size']
        ax.set_title(title, fontsize=fs, **kwargs)
        return self

    def legend(
        self,
        ax_index: Optional[Tuple[int, int]] = None,
        loc: str = 'best',
        frameon: bool = True,
        fontsize: Optional[float] = None,
        **kwargs
    ):
        """
        Add or configure the legend.

        Args:
            ax_index: Subplot index
            loc: Legend location ('best', 'upper right', etc.)
            frameon: Whether to draw a frame around legend
            fontsize: Font size
            **kwargs: Additional arguments passed to ax.legend

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)
        fs = fontsize if fontsize is not None else self.style.params['legend_size']
        ax.legend(loc=loc, frameon=frameon, fontsize=fs, **kwargs)
        return self

    def xlim(
        self,
        left: Optional[float] = None,
        right: Optional[float] = None,
        ax_index: Optional[Tuple[int, int]] = None
    ):
        """
        Set the x-axis limits.

        Args:
            left: Left limit
            right: Right limit
            ax_index: Subplot index

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)
        ax.set_xlim(left=left, right=right)
        return self

    def ylim(
        self,
        bottom: Optional[float] = None,
        top: Optional[float] = None,
        ax_index: Optional[Tuple[int, int]] = None
    ):
        """
        Set the y-axis limits.

        Args:
            bottom: Bottom limit
            top: Top limit
            ax_index: Subplot index

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)
        ax.set_ylim(bottom=bottom, top=top)
        return self

    def grid(
        self,
        b: bool = True,
        which: str = 'major',
        axis: str = 'both',
        linestyle: str = '--',
        linewidth: float = 0.5,
        alpha: float = 0.3,
        ax_index: Optional[Tuple[int, int]] = None
    ):
        """
        Configure grid lines.

        Args:
            b: Whether to show grid
            which: Which grid lines ('major', 'minor', 'both')
            axis: Which axis ('x', 'y', 'both')
            linestyle: Line style
            linewidth: Line width
            alpha: Transparency
            ax_index: Subplot index

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)
        ax.grid(b=b, which=which, axis=axis, linestyle=linestyle,
                linewidth=linewidth, alpha=alpha)
        return self

    def tick_params(
        self,
        axis: str = 'both',
        which: str = 'major',
        direction: str = 'in',
        length: float = 3,
        width: float = 0.5,
        pad: float = 2,
        ax_index: Optional[Tuple[int, int]] = None,
        **kwargs
    ):
        """
        Configure tick parameters.

        Args:
            axis: Which axis ('x', 'y', 'both')
            which: Which ticks ('major', 'minor', 'both')
            direction: Tick direction ('in', 'out', 'inout')
            length: Tick length
            width: Tick width
            pad: Distance to label
            ax_index: Subplot index
            **kwargs: Additional arguments passed to ax.tick_params

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)
        ax.tick_params(axis=axis, which=which, direction=direction,
                       length=length, width=width, pad=pad, **kwargs)
        return self

    def text(
        self,
        x: float,
        y: float,
        s: str,
        ax_index: Optional[Tuple[int, int]] = None,
        fontsize: Optional[float] = None,
        **kwargs
    ):
        """
        Add text to the subplot.

        Args:
            x: x position
            y: y position
            s: Text string
            ax_index: Subplot index
            fontsize: Font size
            **kwargs: Additional arguments passed to ax.text

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)
        fs = fontsize if fontsize is not None else self.style.params['font_size']
        ax.text(x, y, s, fontsize=fs, **kwargs)
        return self

    def annotate(
        self,
        s: str,
        xy: Tuple[float, float],
        xytext: Optional[Tuple[float, float]] = None,
        ax_index: Optional[Tuple[int, int]] = None,
        fontsize: Optional[float] = None,
        arrowprops: Optional[Dict] = None,
        **kwargs
    ):
        """
        Add an annotation with an arrow.

        Args:
            s: Text string
            xy: Point to annotate (x, y)
            xytext: Text position (x, y)
            ax_index: Subplot index
            fontsize: Font size
            arrowprops: Arrow properties dict
            **kwargs: Additional arguments passed to ax.annotate

        Returns:
            self for method chaining
        """
        ax = self._get_ax(ax_index)
        fs = fontsize if fontsize is not None else self.style.params['font_size']
        ax.annotate(s, xy=xy, xytext=xytext, fontsize=fs,
                    arrowprops=arrowprops, **kwargs)
        return self

    def tight_layout(self, pad: float = 1.0, h_pad: Optional[float] = None,
                     w_pad: Optional[float] = None):
        """
        Adjust subplot parameters to avoid overlap.

        Args:
            pad: Padding between figure edge and subplots
            h_pad: Height padding
            w_pad: Width padding

        Returns:
            self for method chaining
        """
        self.fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        return self

    def save(
        self,
        filename: Union[str, Path],
        dpi: Optional[int] = None,
        bbox_inches: str = 'tight',
        pad_inches: float = 0.1,
        transparent: bool = False,
        **kwargs
    ):
        """
        Save the figure to a file.

        Args:
            filename: Output filename (supports png, pdf, svg, eps, emf)
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box mode ('tight' recommended for publications)
            pad_inches: Padding around the figure
            transparent: Make background transparent
            **kwargs: Additional arguments passed to fig.savefig
        """
        if dpi is None:
            dpi = self.style.params['dpi']

        filename = Path(filename)
        suffix = filename.suffix.lower()

        if suffix == '.emf':
            self._save_as_emf(filename, dpi, bbox_inches, pad_inches, transparent)
        else:
            self.fig.savefig(
                filename,
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                transparent=transparent,
                **kwargs
            )

    def _save_as_emf(
        self,
        filename: Path,
        dpi: int,
        bbox_inches: str,
        pad_inches: float,
        transparent: bool
    ):
        """
        Save figure as EMF (Enhanced Metafile) vector format.

        Strategy: convert via true vector toolchains only.
        Prefer Inkscape SVG->EMF, then PDF->EMF via pstoedit.
        """
        import os, subprocess

        svg_path = filename.with_suffix('.svg')
        pdf_path = filename.with_suffix('.pdf')
        self.fig.savefig(
            str(svg_path), dpi=dpi,
            bbox_inches=bbox_inches, pad_inches=pad_inches,
            transparent=transparent
        )

        inkscape = _find_inkscape()
        if inkscape:
            result = subprocess.run(
                [inkscape, str(svg_path), '--export-type=emf',
                 f'--export-filename={str(filename)}'],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0 and os.path.exists(str(filename)):
                os.unlink(str(svg_path))
                if os.path.exists(str(pdf_path)):
                    os.unlink(str(pdf_path))
                return

        self.fig.savefig(
            str(pdf_path), dpi=dpi,
            bbox_inches=bbox_inches, pad_inches=pad_inches,
            transparent=transparent
        )

        pstoedit = _find_pstoedit()
        if pstoedit:
            result = subprocess.run(
                [pstoedit, '-f', 'emf', str(pdf_path), str(filename)],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0 and os.path.exists(str(filename)):
                os.unlink(str(svg_path))
                os.unlink(str(pdf_path))
                return

        os.unlink(str(svg_path))
        os.unlink(str(pdf_path))
        raise OSError('EMF export requires Inkscape or pstoedit for true vector output')

    def save_svg(
        self,
        filename: Union[str, Path],
        bbox_inches: str = 'tight',
        pad_inches: float = 0.1,
        transparent: bool = False
    ):
        """
        Save figure as SVG (Scalable Vector Graphics).

        Args:
            filename: Output SVG filename
            bbox_inches: Bounding box mode
            pad_inches: Padding around the figure
            transparent: Make background transparent
        """
        filename = Path(filename)

        self.fig.savefig(
            filename,
            format='svg',
            dpi=self.style.params['dpi'],
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            transparent=transparent
        )

    def show(self):
        """Display the figure."""
        plt.show()

    def close(self):
        """Close the figure to free memory."""
        plt.close(self.fig)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
