"""Plotting utilities and configuration for Shadow CI scripts.

This module provides consistent matplotlib configuration across all scripts,
with LaTeX rendering, serif fonts, and seaborn styling.
"""

import matplotlib.pyplot as plt


def setup_plotting_style():

    tex_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rc('text.latex', preamble=R'\usepackage{amsmath} \usepackage{bbold}')
    plt.rcParams.update(tex_fonts)
    plt.rcParams['axes.axisbelow'] = True


def save_figure(filename, dpi=300, bbox_inches='tight', **kwargs):
    """Save figure as both PNG and SVG.

    Args:
        filename: Output filename without extension (e.g., 'plot').
            If an extension is provided it will be stripped.
        dpi: Resolution in dots per inch for PNG (default: 300)
        bbox_inches: Bounding box setting (default: 'tight')
        **kwargs: Additional arguments passed to plt.savefig()

    Example:
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> save_figure('my_plot')
    """
    import os
    stem, _ = os.path.splitext(filename)
    for ext in (".png", ".svg"):
        path = stem + ext
        plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        print(f"Figure saved to: {path}")
