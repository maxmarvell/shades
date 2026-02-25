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
    """Save figure with consistent high-quality settings.

    Args:
        filename: Output filename (with extension, e.g., 'plot.pdf')
        dpi: Resolution in dots per inch (default: 300)
        bbox_inches: Bounding box setting (default: 'tight')
        **kwargs: Additional arguments passed to plt.savefig()

    Example:
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> save_figure('my_plot.pdf')
    """
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    print(f"Figure saved to: {filename}")
