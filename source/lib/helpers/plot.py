
import unicodedata
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from textwrap import wrap
from scipy import stats
from scipy.stats import t
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
    output_format="png"
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
    
def plot_scatter(
    df,
    x_var=None,
    y_var=None,
    color_var=None,
    size_var=None,
    out_file='output/analysis/scatter',
    title=None,
    xlabel=None,
    ylabel=None,
    figsize=None,
    legend=True,
    colormap='viridis',
    size_range=(20, 200),
    fit_line=False,
    confidence_interval=0.95,
    show_equation=False,
    output_format="png"
):
    if x_var not in df.columns:
        raise KeyError(f"'{x_var}' not found in dataframe.")
    if y_var not in df.columns:
        raise KeyError(f"'{y_var}' not found in dataframe.")
    if color_var is not None and color_var not in df.columns:
        raise KeyError(f"'{color_var}' not found in dataframe.")
    if size_var is not None and size_var not in df.columns:
        raise KeyError(f"'{size_var}' not found in dataframe.")
    output_format = [output_format] if isinstance(output_format, str) else output_format
    fig, axes = plot_setup(1, 1, figsize=figsize, sharex=False, sharey=False)
    ax = axes[0]
    
    plot_df = df[[x_var, y_var]].dropna()
    x_data = plot_df[x_var]
    y_data = plot_df[y_var]
    
    sizes = None
    if size_var is not None:
        size_data = df[size_var]
        size_min, size_max = size_data.min(), size_data.max()
        if size_max > size_min:
            sizes = size_range[0] + (size_data - size_min) / (size_max - size_min) * (size_range[1] - size_range[0])
        else:
            sizes = np.full(len(size_data), np.mean(size_range))
    else:
        sizes = 50
    
    if color_var is not None:
        color_data = df[color_var]
        if color_data.dtype == 'object' or color_data.nunique() < 10:
            categories = color_data.unique()
            colors = plt.cm.get_cmap(colormap, len(categories))
            for idx, category in enumerate(categories):
                mask = color_data == category
                cat_sizes = sizes[mask] if isinstance(sizes, np.ndarray) else sizes
                ax.scatter(df.loc[mask, x_var], df.loc[mask, y_var], s=cat_sizes, c=[colors(idx)], label=category, alpha=0.8, edgecolors='white', linewidth=0.5)
        else:
            scatter = ax.scatter(df[x_var], df[y_var], s=sizes, c=color_data, cmap=colormap, alpha=0.8, edgecolors='white', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_var, fontsize=14)
            cbar.ax.tick_params(labelsize=12)
    else:
        ax.scatter(df[x_var], df[y_var], s=sizes, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    if fit_line:
        reg = get_regression(x_data, y_data, confidence_interval=confidence_interval if confidence_interval else None)
        if reg:
            ax.plot(reg['x_line'], reg['y_line'], 'r-', linewidth=2, label='Linear fit', zorder=10)
            if reg['ci_lower'] is not None and reg['ci_upper'] is not None:
                ax.fill_between(reg['x_line'], reg['ci_lower'], reg['ci_upper'], alpha=0.2, color='red', label=f'{int(confidence_interval*100)}% CI')
            if show_equation:
                ax.text(0.05, 0.95, reg['equation_text'], transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    y_min = min(df[y_var].min(), 0)
    y_max = df[y_var].max()
    
    set_axis_labels(ax, title=title, xlabel=xlabel if xlabel else x_var, ylabel=ylabel if ylabel else y_var, y_min=y_min, y_max=y_max, fontsize=16)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', rotation=0)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles and legend:
        labels = ['\n'.join(wrap(lbl, 30)) for lbl in labels]
        ax.legend(
            handles, 
            labels,
            loc='best',
            fontsize=12,
            framealpha=0.9,
            edgecolor='gray'
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

def get_regression(x_data, y_data, confidence_interval=0.95, n_points=100):
    """
    Calculate linear regression with confidence intervals.
    
    Parameters:
    -----------
    x_data : array-like
        X values
    y_data : array-like
        Y values
    confidence_interval : float
        Confidence level (e.g., 0.95 for 95% CI)
    n_points : int
        Number of points for smooth line
    
    Returns:
    --------
    dict with keys:
        - slope, intercept, r_squared, p_value, std_err
        - x_line, y_line: arrays for plotting the fitted line
        - ci_lower, ci_upper: confidence interval bounds
        - equation_text: formatted string of equation
    """
    if len(x_data) < 3:
        return None
    
    # Fit linear model
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    r_squared = r_value ** 2
    
    # Create prediction line
    x_line = np.linspace(x_data.min(), x_data.max(), n_points)
    y_line = slope * x_line + intercept
    
    # Calculate confidence interval
    ci_lower, ci_upper = None, None
    if confidence_interval:
        predict_y = slope * x_data + intercept
        residuals = y_data - predict_y
        residual_std = np.sqrt(np.sum(residuals**2) / (len(x_data) - 2))
        
        # Standard error of prediction
        x_mean = np.mean(x_data)
        sxx = np.sum((x_data - x_mean)**2)
        se_line = residual_std * np.sqrt(1/len(x_data) + (x_line - x_mean)**2 / sxx)
        
        # Confidence interval
        t_val = t.ppf((1 + confidence_interval) / 2, len(x_data) - 2)
        ci = t_val * se_line
        ci_lower = y_line - ci
        ci_upper = y_line + ci
    
    equation_text = f'y = {slope:.3f}x + {intercept:.3f}\n$R^2$ = {r_squared:.3f}'
    if p_value < 0.001:
        equation_text += '\np < 0.001'
    else:
        equation_text += f'\np = {p_value:.3f}'
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err,
        'x_line': x_line,
        'y_line': y_line,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'equation_text': equation_text
    }




