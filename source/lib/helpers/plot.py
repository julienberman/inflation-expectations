
import unicodedata
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from textwrap import wrap
from source.lib.remove_eps_info import remove_eps_info

def plot_time_series(
    df,
    plot_vars=None,
    date_var="year",
    out_file='output/analysis/time_series',
    title=None,
    labels=None,
    ylabel=None,
    xlabel=None,
    figsize=None,
    legend=True,
    start=None,
    end=None,
    facet_var=None,
    facets=None,
    facet_titles=None,
    dimensions=None,
    sharey=True,
    sharex=True,
    output_format="pdf"
):
    
    if not plot_vars or not all(v in df.columns for v in plot_vars):
        raise ValueError("Some plot variables not found in dataframe.")
    if date_var not in df.columns:
        raise KeyError(f"'{date_var}' not found in dataframe.")
    if labels is not None and len(labels) != len(plot_vars):
        raise ValueError("Labels must match length of plot_vars.")

    if start:
        df = df.query(f"{date_var} >= @start")
    if end:
        df = df.query(f"{date_var} <= @end")
    
    output_format = [output_format] if isinstance(output_format, str) else output_format
    faceting = facet_var is not None
    if faceting and facet_var not in df.columns:
        raise KeyError(f"'{facet_var}' not found in dataframe.")
    if faceting:
        groups = df[facet_var].dropna().unique()
        if facets is not None and not all(facet in groups for facet in facets):
            raise ValueError("Some facets not found in dataframe.")
        facets = list(groups) if facets is None else list(facets)
        if not facets:
            raise ValueError("No facets found.")
        facet_titles = list(facets) if facet_titles is None else list(facet_titles)
        if len(facet_titles) != len(facets):
            raise ValueError("Must provide the same number of facet titles as facets.")
    else:
        facets = [None]
        facet_titles = [None]
    
    n_panels = len(facets)
    if dimensions is not None:
        n_rows, n_cols = dimensions[0], dimensions[1]
    else:
        n_cols = min(3, n_panels)
        n_rows = math.ceil(n_panels / n_cols)
    
    labels = labels if labels is not None else plot_vars
    fig, axes = plot_setup(n_rows, n_cols, sharex=sharex, sharey=sharey, figsize=figsize)

    used_axes = 0
    for idx, facet in enumerate(facets):
        ax = axes[idx]
        data = df if facet is None else df.query(f"{facet_var} == '{facet}'")
        
        for i, var in enumerate(plot_vars):
            ax.plot(data[date_var], data[var], label=labels[i], marker='o', alpha=0.8)
        
        y_min = min(data[plot_vars].min().min(), 0)
        y_max = data[plot_vars].max().max()
        set_axis_labels(ax, title=facet_titles[idx], xlabel=xlabel, ylabel=ylabel, y_min=y_min, y_max=y_max, fontsize=16)
        used_axes += 1
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=20, y=0.98)
    
    first_ax = next((a for a in axes if a.get_visible()), axes[0])
    h, l = first_ax.get_legend_handles_labels()
    l = ['\n'.join(wrap(lbl, 50)) for lbl in l]
    if h and legend:

        fig.legend(
            h, l,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.01),
            ncol=min(3, len(l)),
            title=title,
            fancybox=True, shadow=True,
            fontsize=16,
            title_fontsize=20,
            handlelength=1.5, columnspacing=0.8, labelspacing=0.6, borderpad=0.6
        )
    plt.tight_layout()
    
    for fmt in output_format:
        plt.savefig(f'{out_file}.{fmt}', bbox_inches='tight', dpi=300)
        if fmt == 'eps':
            remove_eps_info(f'{out_file}.eps')
    plt.close()
    
def plot_setup(n_rows, n_cols, figsize=None, sharex=True, sharey=True):
    if figsize is None:
        base_w, base_h = 8, 6
        figsize = (base_w * n_cols, base_h * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=sharex, sharey=sharey)
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = np.array([axes])
    return fig, axes

def set_axis_labels(ax, title=None, xlabel=None, ylabel=None, y_min=None, y_max=None, fontsize=None):
    ax.set_xlabel(xlabel if xlabel else 'Year', fontsize=fontsize)
    ax.set_ylabel(ylabel if ylabel else 'Value', fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    if y_min is not None and y_max is not None:
        ax.set_ylim(bottom=y_min * 1.5, top = y_max * 1.5)
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)
    if len(ax.get_xticklabels()) > 15:
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    return ax

