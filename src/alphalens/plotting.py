#
# Copyright 2017 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
from scipy import stats
import logging
import warnings

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Union

from functools import wraps

logger = logging.getLogger("alphalens")

from . import utils
from . import performance as perf

DECIMAL_TO_BPS = 10000


def create_visibility_buttons(labels: List, traces_per_label: int, total_traces: int) -> List[Dict]:
    """
    Creates a list of dropdown buttons that control trace visibility.
    
    Implements period-based toggle buttons following the VertexLens pattern.
    
    Parameters
    ----------
    labels : List
        List of button labels (e.g., ['5D', '20D', '60D'])
    traces_per_label : int
        Number of traces per label
    total_traces : int
        Total number of traces
    
    Returns
    -------
    List[Dict]
        List of buttons for Plotly updatemenus
    """
    buttons = []
    for i, label in enumerate(labels):
        visibility_mask = [False] * total_traces
        start, end = i * traces_per_label, (i + 1) * traces_per_label
        for j in range(start, end):
            if j < total_traces:
                visibility_mask[j] = True
        buttons.append(dict(label=str(label), method="restyle", args=[{"visible": visibility_mask}]))
    return buttons


def apply_standard_layout(fig: go.Figure, title_text: str, updatemenus: Optional[List[Dict]] = None, **kwargs) -> go.Figure:
    """
    Applies standard layout to a Plotly figure.
    
    Implemented in the same way as VertexLens:
    - updatemenus can be omitted for simple charts (defaults to empty list)
    - kwargs are only updated (no pop used)

    Parameters
    ----------
    fig : go.Figure
        Plotly figure object
    title_text : str
        Chart title
    updatemenus : List[Dict], optional
        Update menus (dropdown, buttons, etc.). If None, defaults to empty list.
        This allows simple charts to call apply_standard_layout(fig, "Title") without
        explicitly passing updatemenus=[].
    **kwargs
        Additional layout options (xaxis_title, yaxis_title, barmode, legend_title, etc.)

    Returns
    -------
    go.Figure
        Figure with layout applied
    """
    if title_text is None or title_text == "":
        title_text = "Chart"
    
    # Default to empty list if updatemenus is not provided
    if updatemenus is None:
        updatemenus = []
    
    layout_options = {
        "title_text": f"<b>{title_text}</b>",
        "title_x": 0.5,
        "updatemenus": updatemenus,
        "margin": dict(t=120, b=80, l=80, r=120),
        "legend": dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    }
    
    layout_options.update(kwargs)
    fig.update_layout(**layout_options)
    
    return fig


def customize(func):
    """
    Decorator to set plotting context and axes style during function call.

    The previous version applied seaborn/matplotlib styles, but after
    transitioning to Plotly, functions mainly return Plotly figures,
    so this simply wraps the function.
    """

    @wraps(func)
    def call_w_context(*args, **kwargs):
        kwargs.pop("set_context", None)
        return func(*args, **kwargs)

    return call_w_context


def _period_to_days(period):
    """
    Convert period string to number of days for sorting.
    
    Parameters
    ----------
    period : str, int, or float
        Period string (e.g., '5D', '20D') or number
    
    Returns
    -------
    int
        Number of days
    """
    if isinstance(period, str):
        # Extract number from string like '5D', '20D', etc.
        try:
            return int(period.replace('D', '').replace('d', ''))
        except ValueError:
            return 0
    elif isinstance(period, (int, float)):
        return int(period)
    else:
        return 0


def _sort_periods(periods):
    """
    Sort periods by numeric value (5D, 10D, 20D order).
    
    Parameters
    ----------
    periods : list or iterable
        List of period strings or numbers
    
    Returns
    -------
    list
        Sorted list of periods
    """
    return sorted(periods, key=_period_to_days)


def merge_track_a_metrics(base_table, track_a_metrics, periods=None):
    """
    Merges Track A metrics into the base table.

    Parameters
    ----------
    base_table : pd.DataFrame
        Base table
    track_a_metrics : dict
        Track A metrics dictionary {period: {"metric_name": value, ...}, ...}
    periods : list, optional
        List of periods (default: uses keys from track_a_metrics)

    Returns
    -------
    pd.DataFrame
        Table with metrics merged
    """
    if not track_a_metrics:
        return base_table
    
    result_table = base_table.copy()
    
    if periods is None:
        periods = list(track_a_metrics.keys())
    
    for period in periods:
        period_str = utils.format_period(period)
        
        if period_str in track_a_metrics:
            metric_dict = track_a_metrics[period_str]
            for metric_name, metric_value in metric_dict.items():
                result_table.loc[metric_name, period_str] = metric_value
    
    return result_table


def plot_returns_table(
    alpha_beta,
    mean_ret_quantile,
    mean_ret_spread_quantile,
    track_a_metrics=None,
    return_df=False,
):
    """
    Plots a returns analysis table with alpha, beta, and quantile returns.
    
    Parameters
    ----------
    alpha_beta : pd.DataFrame
        DataFrame containing annualized alpha and beta values.
        Index should contain "Ann. alpha" and "beta".
    mean_ret_quantile : pd.DataFrame
        DataFrame with mean period-wise returns by quantile.
        Rows represent quantiles, columns represent periods.
    mean_ret_spread_quantile : pd.Series or pd.DataFrame
        Mean period-wise spread between top and bottom quantiles.
    track_a_metrics : dict, optional
        Track A metrics dictionary {period: {"metric_name": value, ...}, ...}
        to merge into the returns table.
    return_df : bool, optional
        If True, returns the DataFrame instead of printing it.
        Default is False.
    
    Returns
    -------
    pd.DataFrame or None
        Returns table as DataFrame if return_df=True, otherwise None (prints table).
    """
    if not alpha_beta.empty and len(alpha_beta.columns) > 0:
        returns_table = pd.DataFrame(index=[], columns=alpha_beta.columns)
    else:
        if not mean_ret_quantile.empty:
            returns_table = pd.DataFrame(index=[], columns=mean_ret_quantile.columns)
        elif not mean_ret_spread_quantile.empty:
            returns_table = pd.DataFrame(index=[], columns=mean_ret_spread_quantile.columns)
        else:
            returns_table = pd.DataFrame()
    
    if "Ann. alpha" in alpha_beta.index and not alpha_beta.empty:
        if len(returns_table.columns) == 0:
            returns_table = pd.DataFrame(index=[], columns=alpha_beta.columns)
        returns_table.loc["Ann. alpha"] = alpha_beta.loc["Ann. alpha"]
    
    if "beta" in alpha_beta.index and not alpha_beta.empty:
        if len(returns_table.columns) == 0:
            returns_table = pd.DataFrame(index=[], columns=alpha_beta.columns)
        returns_table.loc["beta"] = alpha_beta.loc["beta"]
    
    if not mean_ret_quantile.empty:
        if len(returns_table.columns) == 0:
            returns_table = pd.DataFrame(index=[], columns=mean_ret_quantile.columns)
    returns_table.loc["Mean Period Wise Return Top Quantile (bps)"] = (
        mean_ret_quantile.iloc[-1] * DECIMAL_TO_BPS
    )
    returns_table.loc["Mean Period Wise Return Bottom Quantile (bps)"] = (
        mean_ret_quantile.iloc[0] * DECIMAL_TO_BPS
    )
    
    if not mean_ret_spread_quantile.empty:
        if len(returns_table.columns) == 0:
            returns_table = pd.DataFrame(index=[], columns=mean_ret_spread_quantile.columns)
    returns_table.loc["Mean Period Wise Spread (bps)"] = (
        mean_ret_spread_quantile.mean() * DECIMAL_TO_BPS
    )
    returns_table = merge_track_a_metrics(returns_table, track_a_metrics)
    
    # Sort columns by period numeric order (5D, 10D, 20D, etc.)
    if not returns_table.empty and len(returns_table.columns) > 0:
        sorted_cols = _sort_periods(returns_table.columns)
        returns_table = returns_table.reindex(columns=sorted_cols)
    
    # Round to 3 decimal places for consistent formatting
    returns_table = returns_table.round(3)

    if return_df:
        return returns_table
    else:
        print("Returns Analysis")
        print("Note: Beta should be close to 0 for Long-Short Market Neutral portfolios.")
        print("      If Beta > 0.2, the factor may be capturing market exposure rather than alpha.\n")
        utils.print_table(returns_table)


def plot_turnover_table(
    autocorrelation_data, quantile_turnover, track_a_metrics=None, return_df=False
):
    """
    Plots a turnover analysis table with quantile turnover and factor autocorrelation.
    
    Parameters
    ----------
    autocorrelation_data : dict
        Dictionary of period-wise factor rank autocorrelation Series.
        Keys are periods, values are pd.Series.
    quantile_turnover : dict
        Dictionary of period-wise quantile turnover DataFrames.
        Structure: {period: {quantile: Series}}
    track_a_metrics : dict, optional
        Track A metrics dictionary {period: {"metric_name": value, ...}, ...}
        to merge into the turnover table.
    return_df : bool, optional
        If True, returns the DataFrame instead of printing it.
        Default is False.
    
    Returns
    -------
    pd.DataFrame or tuple
        If return_df=True, returns (combined_table, combined_table).
        Otherwise None (prints table).
    """
    turnover_table = pd.DataFrame()
    # Sort periods by numeric order (5D, 10D, 20D, etc.)
    for period in _sort_periods(quantile_turnover.keys()):
        for quantile, p_data in quantile_turnover[period].items():
            turnover_table.loc[
                "Quantile {} Mean Turnover".format(quantile),
                "{}D".format(period),
            ] = p_data.mean()
    
    auto_corr = pd.DataFrame()
    for period, p_data in autocorrelation_data.items():
        period_str = utils.format_period(period)
        auto_corr.loc[
            "Mean Factor Rank Autocorrelation", period_str
        ] = p_data.mean()

    auto_corr = merge_track_a_metrics(auto_corr, track_a_metrics)

    combined_table = pd.concat([turnover_table, auto_corr], axis=0)
    # Sort columns by period numeric order (5D, 10D, 20D, etc.)
    combined_table = combined_table.reindex(columns=_sort_periods(combined_table.columns))
    
    # Round to 3 decimal places for consistent formatting
    combined_table = combined_table.round(3)

    if return_df:
        return combined_table, combined_table
    else:
        print("Turnover Analysis")
        utils.print_table(combined_table)


def plot_information_table(ic_data, return_df=False, yearly_win_rate=None):
    """
    Plots an information coefficient (IC) analysis table.
    
    Computes and displays IC statistics including risk-adjusted IC, t-statistic,
    p-value, mean, standard deviation, skewness, and kurtosis.
    
    Parameters
    ----------
    ic_data : pd.Series or pd.DataFrame
        Information coefficient time series data.
        If DataFrame, each column represents a different period.
    return_df : bool, optional
        If True, returns the DataFrame instead of printing it.
        Default is False.
    yearly_win_rate : pd.Series, optional
        Yearly win rate Series to include in the table.
        Should be indexed by period (matching ic_data index/columns).
    
    Returns
    -------
    pd.DataFrame or None
        IC summary table as DataFrame if return_df=True, otherwise None (prints table).
    """
    ic_summary_table = pd.DataFrame()
    
    ic_summary_table["Risk-Adjusted IC"] = ic_data.mean() / ic_data.std()
    # Calculate t-stat using entire dataset: tests whether overall IC mean is significantly different from 0
    # This provides a single summary statistic for the entire time period
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)

    if yearly_win_rate is not None:
        try:
            aligned = yearly_win_rate.reindex(ic_summary_table.index)
            ic_summary_table["Yearly Win Rate"] = aligned
        except Exception:
            pass
    
    # Sort index/columns by period numeric order (5D, 10D, 20D, etc.)
    if isinstance(ic_data, pd.DataFrame):
        # If DataFrame, sort columns
        if not ic_summary_table.empty and len(ic_summary_table.index) > 0:
            sorted_idx = _sort_periods(ic_summary_table.index)
            ic_summary_table = ic_summary_table.reindex(index=sorted_idx)
    else:
        # If Series, index is already sorted (single period)
        pass
    
    # Round to 3 decimal places for consistent formatting
    ic_summary_table = ic_summary_table.round(3)

    if return_df:
        return ic_summary_table
    else:
        print("Information Analysis")
        print("Note: IC Mean alone can be misleading. Focus on Risk-Adjusted IC (IR) and t-stat(IC).")
        print("      IC Mean=0.05 with IC Std=0.10 is worse than IC Mean=0.02 with IC Std=0.005.\n")
        utils.print_table(ic_summary_table.T)


def plot_quantile_statistics_table(factor_data, return_df=False):
    """
    Plots a quantile statistics table showing factor distribution by quantile.
    
    Computes min, max, mean, standard deviation, count, and count percentage
    for each factor quantile.
    
    Parameters
    ----------
    factor_data : pd.DataFrame
        Factor data with 'factor_quantile' and 'factor' columns.
    return_df : bool, optional
        If True, returns the DataFrame instead of printing it.
        Default is False.
    
    Returns
    -------
    pd.DataFrame or None
        Quantile statistics table as DataFrame if return_df=True,
        otherwise None (prints table).
    """
    quantile_stats = factor_data.groupby("factor_quantile", sort=False)["factor"].agg(
        ["min", "max", "mean", "std", "count"]
    )
    if quantile_stats.empty or len(quantile_stats.columns) == 0:
        if return_df:
            return pd.DataFrame()
        else:
            print("Quantiles Statistics")
            print("No data available.")
            return

    quantile_stats["count %"] = (
        quantile_stats["count"] / quantile_stats["count"].sum() * 100.0
    )

    if return_df:
        # Round to 3 decimal places for consistency
        return quantile_stats.round(3)
    else:
        print("Quantiles Statistics")
        utils.print_table(quantile_stats.round(3))


def plot_ic_ts(ic, threshold=None):
    """
    Plots Spearman Rank Information Coefficient and IC moving
    average for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    threshold : float, optional
        t-statistic threshold value. If provided, threshold lines and annotations
        will be displayed to show statistical significance.
        Default is None (no threshold lines).

    Returns
    -------
    go.Figure
        Plotly Figure object with subplots:
        - Row 1: IC and IC moving average
        - Row 2 (if threshold provided): t-stat MA and threshold lines
    """
    # Convert index to datetime for time series charts
    ic = ic.copy()
    if not isinstance(ic.index, pd.DatetimeIndex):
        ic.index = pd.to_datetime(ic.index)
    
    periods = list(ic.columns)
    actual_periods = []
    
    # Create single figure (no subplots)
    fig = go.Figure()
    
    for period in periods:
        ic_series = ic[period]
        
        valid_data = ic_series.dropna()
        
        if ic_series.empty:
            logger.warning(f"Period {period}: IC series is completely empty. Skipping.")
            continue
        
        if valid_data.empty:
            logger.warning(f"Period {period}: IC series has no valid (non-NaN) data points. "
                         f"Total length: {len(ic_series)}, NaN count: {ic_series.isna().sum()}. Skipping.")
            continue
        if len(valid_data) < 2:
            logger.warning(f"Period {period}: IC series has only {len(valid_data)} valid data point(s). "
                         f"Need at least 2 points to plot. Skipping.")
            continue
        
        actual_periods.append(period)
        ma = ic_series.rolling(window=22).mean()
        
        is_visible = (period == actual_periods[0] if actual_periods else False)
        
        # IC line
        fig.add_trace(go.Scatter(
            x=ic_series.index,
            y=ic_series.values,
            mode='lines',
            name='IC',
            line=dict(color='steelblue', width=1.5 if threshold else 1),
            visible=is_visible,
            legendgroup='ic',
            showlegend=True
        ))
        
        # Moving average line
        fig.add_trace(go.Scatter(
            x=ma.index,
            y=ma.values,
            mode='lines',
            name='1 month moving avg',
            line=dict(color='forestgreen', width=2.5 if threshold else 2),
            visible=is_visible,
            legendgroup='ma',
            showlegend=True
        ))
        
        # Zero line
        x_range = [ic_series.index[0], ic_series.index[-1]]
        if threshold is not None:
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[0.0, 0.0],
                mode='lines',
                name='Zero line',
                line=dict(color='black', width=1, dash='solid'),
                visible=is_visible,
                legendgroup='zero',
                showlegend=False,
                hoverinfo='skip'
            ))
        else:
            fig.add_hline(y=0, line_dash="solid", line_color="black", 
                         line_width=1, opacity=0.8, visible=is_visible)
        
        # Calculate t-stat time series for each IC value (on second y-axis)
        if threshold is not None:
            # Calculate t-stat using rolling window: tests whether rolling mean IC is significantly different from 0 at each point
            # Unlike the table which uses entire dataset, rolling window captures time-varying statistical significance
            # This allows tracking how IC significance changes over time, revealing periods of stronger/weaker factor performance
            # t-stat = (rolling_mean - 0) / (rolling_std / sqrt(rolling_n))
            rolling_mean = ic_series.rolling(window=22, min_periods=2).mean()
            rolling_std = ic_series.rolling(window=22, min_periods=2).std()
            rolling_n = ic_series.rolling(window=22, min_periods=2).count()
            
            # Calculate t-stat for each point
            # Avoid division by zero
            t_stat_series = pd.Series(index=ic_series.index, dtype=float)
            valid_mask = (rolling_std > 0) & (rolling_n > 1)
            t_stat_series[valid_mask] = rolling_mean[valid_mask] / (rolling_std[valid_mask] / np.sqrt(rolling_n[valid_mask]))
            t_stat_series[~valid_mask] = np.nan
            
            # Apply moving average to t-stat series for smoother line visualization
            # Use same window as IC moving average (22 days = 1 month)
            # min_periods=1 allows calculation even with fewer than 22 points (for early periods)
            t_stat_ma = t_stat_series.rolling(window=22, min_periods=1).mean()
            
            # T-stat moving average line (on second y-axis) - Only show MA, not raw t-stat
            fig.add_trace(go.Scatter(
                x=t_stat_ma.index,
                y=t_stat_ma.values,
                mode='lines',
                name='t-stat(IC) MA',
                line=dict(
                    color='orange',
                    width=2.5,
                    dash='solid'
                ),
                connectgaps=True,  # Connect across NaN gaps to create continuous line
                visible=is_visible,
                legendgroup='tstat',
                showlegend=True,
                yaxis='y2',
                hovertemplate='Date: %{x}<br>t-stat MA: %{y:.2f}<extra></extra>'
            ))
            
            # Upper threshold line (on second y-axis) - Red dashed line at +threshold
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[threshold, threshold],
                mode='lines',
                name=f't-stat = +{threshold}',
                line=dict(
                    color='red',  # Bright red for better visibility
                    width=2.0,  # Thick line
                    dash='dash'  # Dashed line style
                ),
                visible=is_visible,
                legendgroup='threshold',
                showlegend=True,  # Show in legend for clarity
                yaxis='y2',
                hoverinfo='skip',
                hovertemplate=f't-stat = +{threshold} (99% confidence)<extra></extra>'
            ))
            
            # Lower threshold line (on second y-axis) - Red dashed line at -threshold
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[-threshold, -threshold],
                mode='lines',
                name=f't-stat = -{threshold}',
                line=dict(
                    color='red',  # Bright red for better visibility
                    width=2.0,  # Thick line
                    dash='dash'  # Dashed line style
                ),
                visible=is_visible,
                legendgroup='threshold',
                showlegend=True,  # Show in legend for clarity
                yaxis='y2',
                hoverinfo='skip',
                hovertemplate=f't-stat = -{threshold} (99% confidence)<extra></extra>'
            ))
            
            # Annotations on t-stat axis (yaxis2)
            upper_text = f"t-stat = +{threshold}<br>(99% confidence)"
            lower_text = f"t-stat = -{threshold}<br>(99% confidence)"
            fig.add_annotation(
                xref="paper", yref="y2",
                x=0.98, y=threshold,
                text=upper_text,
                showarrow=True,
                arrowhead=2,
                arrowcolor="crimson",
                bgcolor="rgba(220, 20, 60, 0.9)",
                bordercolor="darkred",
                borderwidth=2,
                font=dict(size=10, color="white", family="Arial Black"),
                visible=is_visible,
                xanchor="right",
                yanchor="bottom"
            )
            
            fig.add_annotation(
                xref="paper", yref="y2",
                x=0.98, y=-threshold,
                text=lower_text,
                showarrow=True,
                arrowhead=2,
                arrowcolor="crimson",
                bgcolor="rgba(220, 20, 60, 0.9)",
                bordercolor="darkred",
                borderwidth=2,
                font=dict(size=10, color="white", family="Arial Black"),
                visible=is_visible,
                xanchor="right",
                yanchor="top"
            )
    
    # Return empty figure if no valid periods
    if not actual_periods:
        fig = go.Figure()
        fig = apply_standard_layout(
            fig,
            "Information Coefficient (IC) Time Series",
            [],
            xaxis_title="Date",
            yaxis_title="IC"
        )
        return fig
    
    # Create period toggle buttons
    if threshold is not None:
        # IC: IC, MA, Zero (3 traces)
        # t-stat: t-stat MA, Upper threshold, Lower threshold (3 traces)
        traces_per_period = 6
    else:
        # IC only: IC, MA, Zero (3 traces)
        traces_per_period = 3
    
    buttons = create_visibility_buttons(actual_periods if actual_periods else periods, traces_per_period, len(fig.data))
    
    # Apply layout
    title_text = "Information Coefficient (IC) Time Series"
    if threshold is not None:
        title_text = f"{title_text}<br><sub>t-stat threshold = ±{threshold} (99% confidence)</sub>"
    
    layout_updates = {}
    
    if threshold is not None:
        # Add second y-axis for t-stat scale
        # Use fixed range based on threshold
        t_stat_range = max(abs(threshold) * 1.5, 5.0)
        layout_updates["yaxis2"] = dict(
            title="t-statistic",
            overlaying="y",
            side="right",
            range=[-t_stat_range, t_stat_range],
            showgrid=False
        )
        layout_updates["height"] = 600
        layout_updates["margin"] = dict(r=140, l=80, t=100, b=60)
        layout_updates["legend"] = dict(
            x=1.05,  # Move legend further to the right
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    else:
        layout_updates["legend"] = dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
        layout_updates["margin"] = dict(r=120, l=80, t=100, b=60)
    
    fig = apply_standard_layout(
        fig,
        title_text,
        [dict(active=0, buttons=buttons, direction="down", x=0.01, xanchor="left", y=1.15, yanchor="top")] if buttons else [],
        xaxis_title="Date",
        yaxis_title="IC",
        **layout_updates
    )
    
    # Set x-axis type to 'date' to prevent exponential notation
    fig.update_xaxes(type='date')
    
    return fig


def plot_ic_hist(ic, ax=None):
    """
    Plots Spearman Rank Information Coefficient histogram for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Deprecated. Kept for backward compatibility only. Not used.

    Returns
    -------
    go.Figure
        Plotly Figure object with subplots for each period.
    """
    if ax is not None:
        warnings.warn(
            "ax parameter is deprecated and will be ignored. "
            "This function now returns a Plotly figure.",
            DeprecationWarning,
            stacklevel=2
        )

    num_plots = len(ic.columns)
    n_cols = 3
    n_rows = ((num_plots - 1) // n_cols) + 1

    subplot_titles = [f"{period} Period IC" for period in ic.columns]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    for idx, (period_num, ic_series) in enumerate(ic.items()):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        
        ic_clean = ic_series.replace(np.nan, 0.0).dropna()
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=ic_clean.values,
                nbinsx=50,
                name=f'{period_num} Period',
                showlegend=False,
                marker_color='steelblue',
                opacity=0.7
            ),
            row=row,
            col=col
        )
        
        # KDE (Kernel Density Estimation)
        if len(ic_clean) > 1:
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(ic_clean.values)
                x_kde = np.linspace(ic_clean.min(), ic_clean.max(), 200)
                y_kde = kde(x_kde)
                # Normalize to match histogram scale
                hist_max = np.histogram(ic_clean.values, bins=50)[0].max()
                y_kde_scaled = y_kde * hist_max / y_kde.max()
                
                fig.add_trace(
                    go.Scatter(
                        x=x_kde,
                        y=y_kde_scaled,
                        mode='lines',
                        name='KDE',
                        showlegend=False,
                        line=dict(color='red', width=2),
                        hoverinfo='skip'
                    ),
                    row=row,
                    col=col
                )
            except:
                pass
        
        # Mean line
        mean_val = ic_clean.mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="white",
            line_width=2,
            row=row,
            col=col
        )
        
        std_val = ic_clean.std()
        fig.add_annotation(
            x=-0.95,
            y=0.95,
            xref=f"x{idx+1}",
            yref=f"y{idx+1}",
            text=f"Mean {mean_val:.3f}<br>Std. {std_val:.3f}",
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            xanchor="left",
            yanchor="top"
        )
        
        fig.update_xaxes(range=[-1, 1], row=row, col=col)
        fig.update_xaxes(title_text="IC", row=row, col=col)

    for idx in range(num_plots, n_rows * n_cols):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, row=row, col=col)

    fig.update_layout(
        height=n_rows * 400,
        showlegend=False,
        title_text="IC Distribution by Period",
        margin=dict(t=120, b=80, l=80, r=120)
    )
    

    return fig


def plot_ic_qq(ic, theoretical_dist=stats.norm, ax=None):
    """
    Plots Spearman Rank Information Coefficient "Q-Q" plot relative to
    a theoretical distribution.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    theoretical_dist : scipy.stats._continuous_distns
        Continuous distribution generator. scipy.stats.norm and
        scipy.stats.t are popular options.
    ax : matplotlib.Axes, optional
        Deprecated. Kept for backward compatibility only. Not used.

    Returns
    -------
    go.Figure
        Plotly Figure object with Q-Q plots for each period.
    """
    if ax is not None:
        warnings.warn(
            "ax parameter is deprecated and will be ignored. "
            "This function now returns a Plotly figure.",
            DeprecationWarning,
            stacklevel=2
        )

    num_plots = len(ic.columns)
    n_cols = 3
    n_rows = ((num_plots - 1) // n_cols) + 1

    if isinstance(theoretical_dist, stats.norm.__class__):
        dist_name = "Normal"
    elif isinstance(theoretical_dist, stats.t.__class__):
        dist_name = "T"
    else:
        dist_name = "Theoretical"

    subplot_titles = [f"{period} Period IC {dist_name} Dist. Q-Q" for period in ic.columns]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    for idx, (period_num, ic_series) in enumerate(ic.items()):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        
        ic_clean = ic_series.replace(np.nan, 0.0).dropna()
        
        if len(ic_clean) == 0:
            continue
        
        # Observed quantiles (sorted data)
        observed_quantiles = np.sort(ic_clean.values)
        
        # Theoretical quantiles
        n = len(observed_quantiles)
        # Probability points (0.5/n, 1.5/n, ..., (n-0.5)/n)
        prob_points = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = theoretical_dist.ppf(prob_points)
        
        # Fit line (45-degree line through percentiles)
        # Use percentiles for fitting
        percentiles = [25, 75]
        obs_percentiles = np.percentile(observed_quantiles, percentiles)
        theo_percentiles = theoretical_dist.ppf([p/100 for p in percentiles])
        
        # Linear fit: y = a + b*x
        if len(theo_percentiles) == 2 and theo_percentiles[1] != theo_percentiles[0]:
            slope = (obs_percentiles[1] - obs_percentiles[0]) / (theo_percentiles[1] - theo_percentiles[0])
            intercept = obs_percentiles[0] - slope * theo_percentiles[0]
        else:
            slope = 1.0
            intercept = 0.0
        
        fit_line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
        fit_line_y = intercept + slope * fit_line_x
        
        # Q-Q scatter plot
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=observed_quantiles,
                mode='markers',
                name=f'{period_num} Period',
                showlegend=False,
                marker=dict(color='steelblue', size=4, opacity=0.6)
            ),
            row=row,
            col=col
        )
        
        # 45-degree reference line
        fig.add_trace(
            go.Scatter(
                x=fit_line_x,
                y=fit_line_y,
                mode='lines',
                name='Reference Line',
                showlegend=False,
                line=dict(color='red', width=2, dash='dash')
            ),
            row=row,
            col=col
        )
        
        # Axis labels
        fig.update_xaxes(title_text=f"{dist_name} Distribution Quantile", row=row, col=col)
        fig.update_yaxes(title_text="Observed Quantile", row=row, col=col)

    for idx in range(num_plots, n_rows * n_cols):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, row=row, col=col)

    fig.update_layout(
        height=n_rows * 400,
        showlegend=False,
        title_text=f"IC Q-Q Plot ({dist_name} Distribution)",
        margin=dict(t=120, b=80, l=80, r=120)
    )
    

    return fig


def plot_quantile_returns_bar(
    mean_ret_by_q, by_group=False, ylim_percentiles=None
):
    """
    Plots mean period wise returns for factor quantiles.

    Parameters
    ----------
    mean_ret_by_q : pd.DataFrame
        DataFrame with quantile, (group) and mean period wise return values.
        Columns are forward return periods (e.g., '5D', '20D').
    by_group : bool
        Disaggregated figures by group.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """

    if by_group:
        groups = mean_ret_by_q.index.get_level_values("group").unique()
        num_groups = len(groups)
        n_cols = 2
        n_rows = (num_groups + n_cols - 1) // n_cols
        
        subplot_titles = [str(g) for g in groups]
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        periods = list(mean_ret_by_q.columns)
        quantiles = mean_ret_by_q.index.get_level_values("factor_quantile").unique()
        
        for idx, group in enumerate(groups):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1
            
            group_data = mean_ret_by_q.xs(group, level="group")
            
            for period in periods:
                period_data = group_data[period] * DECIMAL_TO_BPS
                fig.add_trace(
                    go.Bar(
                        x=[f"Q{q}" for q in quantiles],
                        y=period_data.values,
                        name=period,
                        text=[f"{val:.1f}" for val in period_data.values],
                        textposition="outside",
                        textfont=dict(size=10),
                        showlegend=(idx == 0)
                    ),
                    row=row,
                    col=col
                )
        
        fig.update_layout(
            title=dict(
                text="Mean Period Wise Return By Factor Quantile",
                x=0.5,
                xanchor="center",
                font=dict(size=16)
            ),
            height=400 * n_rows,
            template="plotly_white"
        )
        
        return fig
    else:
        fig = go.Figure()
        periods = list(mean_ret_by_q.columns)
        quantiles = list(mean_ret_by_q.index)
        
        # Create bar traces for each period (initially only first period is visible)
        # Calculate y-axis range first to prevent text clipping
        all_values = []
        for period in periods:
            period_data = mean_ret_by_q[period] * DECIMAL_TO_BPS
            all_values.extend(period_data.values)
        
        if ylim_percentiles is not None:
            ymin = np.nanpercentile(mean_ret_by_q.values, ylim_percentiles[0]) * DECIMAL_TO_BPS
            ymax = np.nanpercentile(mean_ret_by_q.values, ylim_percentiles[1]) * DECIMAL_TO_BPS
        else:
            ymin = min(all_values) if all_values else None
            ymax = max(all_values) if all_values else None
        
        # Add padding to y-axis range to prevent text clipping (15% of range)
        if ymin is not None and ymax is not None:
            y_range = ymax - ymin
            ymin_adjusted = ymin - y_range * 0.15  # 15% padding below
            ymax_adjusted = ymax + y_range * 0.15  # 15% padding above
        else:
            ymin_adjusted = ymin
            ymax_adjusted = ymax
        
        for period in periods:
            period_data = mean_ret_by_q[period] * DECIMAL_TO_BPS
            is_visible = (period == periods[0])
            
            fig.add_trace(go.Bar(
                x=[f"Q{q}" for q in quantiles],
                y=period_data.values,
                name=period,
                visible=is_visible,
                text=[f"{val:.1f}" for val in period_data.values],
                textposition="outside",
                textfont=dict(size=11)
            ))
        
        # Create period toggle buttons
        buttons = create_visibility_buttons(periods, 1, len(fig.data))
        
        # Apply layout
        fig = apply_standard_layout(
            fig,
            "Mean Period Wise Return By Factor Quantile",
            [dict(active=0, buttons=buttons, direction="down", x=0.01, xanchor="left", y=1.15, yanchor="top")],
            xaxis_title="Factor Quantile",
            yaxis_title="Mean Return (bps)",
            showlegend=False
        )
        
        # Set y-axis range to prevent text clipping
        if ymin_adjusted is not None and ymax_adjusted is not None:
            fig.update_yaxes(range=[ymin_adjusted, ymax_adjusted])
        
        return fig


def plot_quantile_returns_violin(return_by_q, ylim_percentiles=None):
    """
    Plots a violin box plot of period wise returns for factor quantiles.

    Parameters
    ----------
    return_by_q : pd.DataFrame - MultiIndex
        DataFrame with date and quantile as rows MultiIndex,
        forward return windows as columns, returns as values.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """

    if ylim_percentiles is not None:
        ymin = (
            np.nanpercentile(return_by_q.values, ylim_percentiles[0]) * DECIMAL_TO_BPS
        )
        ymax = (
            np.nanpercentile(return_by_q.values, ylim_percentiles[1]) * DECIMAL_TO_BPS
        )
    else:
        ymin = None
        ymax = None

    unstacked_dr = return_by_q.multiply(DECIMAL_TO_BPS)
    unstacked_dr.columns = unstacked_dr.columns.set_names("forward_periods")
    unstacked_dr = unstacked_dr.stack()
    unstacked_dr.name = "return"
    unstacked_dr = unstacked_dr.reset_index()

    fig = go.Figure()
    
    # Generate colors for each period
    periods = sorted(unstacked_dr["forward_periods"].unique())
    quantiles = sorted(unstacked_dr["factor_quantile"].unique())
    colors = px.colors.qualitative.Set3[:len(periods)] if len(periods) <= 12 else px.colors.qualitative.Set3 * (len(periods) // 12 + 1)
    
    # Create violin plots for each period
    for idx, period in enumerate(periods):
        period_data = unstacked_dr[unstacked_dr["forward_periods"] == period]
        
        # Extract data for each quantile
        for quantile in quantiles:
            quantile_data = period_data[period_data["factor_quantile"] == quantile]["return"].dropna()
            
            if not quantile_data.empty:
                # Add violin plot
                fig.add_trace(go.Violin(
                    x=[quantile] * len(quantile_data),
                    y=quantile_data.values,
                    name=str(period),
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=colors[idx],
                    line_color=colors[idx],
                    opacity=0.7,
                    showlegend=(quantile == quantiles[0]),  # Show legend only for first quantile
                    legendgroup=str(period)
                ))
    
    # Zero horizontal line
    fig.add_hline(
        y=0.0,
        line_dash="solid",
        line_color="black",
        line_width=0.7,
        opacity=0.6
    )
    
    # Apply layout
    fig = apply_standard_layout(
        fig,
        "Period Wise Return By Factor Quantile",
        [],
        xaxis_title="",
        yaxis_title="Return (bps)"
    )
    
    # Configure x-axis (quantile labels)
    fig.update_xaxes(
        tickmode='array',
        tickvals=quantiles,
        ticktext=[f"Q{q}" for q in quantiles]
    )
    
    # Set y-axis range
    if ymin is not None and ymax is not None:
        fig.update_yaxes(range=[ymin, ymax])
    
    return fig


def plot_mean_quantile_returns_spread_time_series(
    mean_returns_spread, std_err=None, bandwidth=1
):
    """
    Plots mean period wise returns for factor quantiles.

    Parameters
    ----------
    mean_returns_spread : pd.Series or pd.DataFrame
        Series or DataFrame with difference between quantile mean returns by period.
        If DataFrame, columns are forward return periods.
    std_err : pd.Series or pd.DataFrame, optional
        Series or DataFrame with standard error of difference between quantile
        mean returns each period.
    bandwidth : float
        Width of displayed error bands in standard deviations.
    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    # Convert index to datetime for time series charts
    mean_returns_spread = mean_returns_spread.copy()
    if isinstance(mean_returns_spread, pd.DataFrame):
        if not isinstance(mean_returns_spread.index, pd.DatetimeIndex):
            mean_returns_spread.index = pd.to_datetime(mean_returns_spread.index)
    else:
        if not isinstance(mean_returns_spread.index, pd.DatetimeIndex):
            mean_returns_spread.index = pd.to_datetime(mean_returns_spread.index)
    
    # Convert std_err to datetime as well
    if std_err is not None:
        std_err = std_err.copy()
        if isinstance(std_err, pd.DataFrame):
            if not isinstance(std_err.index, pd.DatetimeIndex):
                std_err.index = pd.to_datetime(std_err.index)
        else:
            if not isinstance(std_err.index, pd.DatetimeIndex):
                std_err.index = pd.to_datetime(std_err.index)

    if isinstance(mean_returns_spread, pd.DataFrame):
        # DataFrame case: display multiple periods in a single Plotly graph with toggle
        fig = go.Figure()
        periods = list(mean_returns_spread.columns)
        
        # Track periods that actually have traces added
        actual_periods = []
        
        # Create traces for each period (initially only first period is visible)
        for period in periods:
            spread_series = mean_returns_spread[period]
            if spread_series.isnull().all():
                continue
            
            actual_periods.append(period)
            spread_bps = spread_series * DECIMAL_TO_BPS
            ma = spread_bps.rolling(window=22).mean()
            
            is_visible = (period == actual_periods[0] if actual_periods else False)
            
            # Mean returns spread line (left y-axis)
            # Increased transparency to make moving average more visible
            fig.add_trace(go.Scatter(
                x=spread_bps.index,
                y=spread_bps.values,
                mode='lines',
                name=f'mean returns spread ({period})',
                line=dict(color='forestgreen', width=0.5),
                opacity=0.2,  # Increased transparency (was 0.4) to make moving average more visible
                visible=is_visible,
                legendgroup=f'spread_{period}',
                showlegend=True,
                yaxis='y'  # Use left y-axis
            ))
            
            # 1 month moving average line (right y-axis)
            fig.add_trace(go.Scatter(
                x=ma.index,
                y=ma.values,
                mode='lines',
                name=f'1 month moving avg ({period})',
                line=dict(color='orangered', width=2.5),
                opacity=1.0,  # Full opacity for moving average to make it stand out
                visible=is_visible,
                legendgroup=f'ma_{period}',
                showlegend=True,
                yaxis='y2'  # Use right y-axis
            ))
            
            # Error bands (if std_err is provided)
            if std_err is not None and isinstance(std_err, pd.DataFrame) and period in std_err.columns:
                std_err_series = std_err[period]
                std_err_bps = std_err_series * DECIMAL_TO_BPS
                upper = spread_bps.values + (std_err_bps.values * bandwidth)
                lower = spread_bps.values - (std_err_bps.values * bandwidth)
                
                # Upper bound (left y-axis)
                fig.add_trace(go.Scatter(
                    x=spread_bps.index,
                    y=upper,
                    mode='lines',
                    name='upper bound',
                    line=dict(width=0),
                    showlegend=False,
                    visible=is_visible,
                    legendgroup=period,
                    hoverinfo='skip',
                    yaxis='y'  # Use left y-axis
                ))
                
                # Lower bound (left y-axis)
                fig.add_trace(go.Scatter(
                    x=spread_bps.index,
                    y=lower,
                    mode='lines',
                    name='lower bound',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(70, 130, 180, 0.3)',
                    showlegend=False,
                    visible=is_visible,
                    legendgroup=period,
                    hoverinfo='skip',
                    yaxis='y'  # Use left y-axis
                ))
            
            # Zero line (displayed on left y-axis)
            fig.add_trace(go.Scatter(
                x=[spread_bps.index[0], spread_bps.index[-1]],
                y=[0.0, 0.0],
                mode='lines',
                name='zero line',
                line=dict(color='black', width=0.7, dash='solid'),
                showlegend=False,
                visible=is_visible,
                legendgroup=period,
                hoverinfo='skip',
                yaxis='y'  # Use left y-axis
            ))
        
        # Return empty figure if actual_periods is empty
        if not actual_periods:
            fig = apply_standard_layout(
                fig,
                "Top Minus Bottom Quantile Mean Return",
                [],
                xaxis_title="Date",
                yaxis_title="Difference In Quantile Mean Return (bps)"
            )
            return fig
        
        # Create period toggle buttons
        # Calculate traces_per_period based on actual number of traces added
        total_traces = len(fig.data)
        num_periods = len(actual_periods)
        traces_per_period = total_traces // num_periods if num_periods > 0 else 3
            
        buttons = []
        for i, period in enumerate(actual_periods):
            visibility_mask = [False] * total_traces
            start_idx = i * traces_per_period
            end_idx = start_idx + traces_per_period
            for j in range(start_idx, min(end_idx, total_traces)):
                visibility_mask[j] = True
            
            buttons.append(dict(
                label=str(period),
                method="restyle",
                args=[{"visible": visibility_mask}]
            ))
        
        # Calculate y-axis ranges (left y-axis: spread, right y-axis: moving average)
        all_spread_values = []
        all_ma_values = []
        for period in actual_periods:
            spread_series = mean_returns_spread[period]
            if not spread_series.isnull().all():
                spread_bps = spread_series * DECIMAL_TO_BPS
                all_spread_values.extend(spread_bps.dropna().values.tolist())
                # Moving average values
                ma = spread_bps.rolling(window=22).mean()
                all_ma_values.extend(ma.dropna().values.tolist())
        
        # Left y-axis range (spread values) - use percentiles to remove outliers
        if all_spread_values:
            # Use 5th and 95th percentiles to exclude outliers
            spread_min = np.nanpercentile(all_spread_values, 5)
            spread_max = np.nanpercentile(all_spread_values, 95)
            spread_range = spread_max - spread_min
            padding = spread_range * 0.1 if spread_range > 0 else 10
            y_min = spread_min - padding
            y_max = spread_max + padding
        else:
            y_min, y_max = -100, 100
        
        # Right y-axis range (moving average values)
        if all_ma_values:
            ma_min = min(all_ma_values)
            ma_max = max(all_ma_values)
            ma_range = ma_max - ma_min
            padding = ma_range * 0.05 if ma_range > 0 else 10
            y2_min = ma_min - padding
            y2_max = ma_max + padding
        else:
            y2_min, y2_max = -100, 100
        
        # Apply layout
        fig = apply_standard_layout(
            fig,
            "Top Minus Bottom Quantile Mean Return",
            [dict(active=0, buttons=buttons, direction="down", x=0.01, xanchor="left", y=1.15, yanchor="top")],
            xaxis_title="Date",
            yaxis_title="Difference In Quantile Mean Return (bps)",
            showlegend=True  # Show legend
        )
        
        # Set left y-axis range (spread values)
        fig.update_yaxes(range=[y_min, y_max], title="Spread (bps)")
        
        # Add right y-axis (moving average values)
        # Adjust margin and chart size to prevent overlap with legend
        fig.update_layout(
            height=600,  # Increase chart height
            margin=dict(r=140, l=80, t=100, b=60),  # Increase right margin (space for legend)
            legend=dict(
                x=1.05,  # Move legend further right (avoid overlap with y-axis)
                y=1.0,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            yaxis2=dict(
                title="Moving Average (bps)",
                overlaying="y",
                side="right",
                range=[y2_min, y2_max],
                showgrid=False  # Remove gridlines for right y-axis (avoid duplication)
            )
        )
        
        # Set x-axis type to 'date' to prevent exponential notation
        fig.update_xaxes(type='date')
        
        return fig

    # Series case (single period)
    if mean_returns_spread.isnull().all():
        fig = go.Figure()
        fig = apply_standard_layout(
            fig,
            "Top Minus Bottom Quantile Mean Return",
            [],
            xaxis_title="Date",
            yaxis_title="Difference In Quantile Mean Return (bps)"
        )
        return fig

    periods = mean_returns_spread.name
    if periods is not None:
        title = (
            "Top Minus Bottom Quantile Mean Return "
            "({} Period Forward Return)".format(periods)
        )
    else:
        title = "Top Minus Bottom Quantile Mean Return"
    
    # Plotly mode (single period)
    fig = go.Figure()
    spread_bps = mean_returns_spread * DECIMAL_TO_BPS
    ma = spread_bps.rolling(window=22).mean()
    
    # Mean returns spread line
    fig.add_trace(go.Scatter(
        x=spread_bps.index,
        y=spread_bps.values,
        mode='lines',
        name='mean returns spread',
        line=dict(color='forestgreen', width=0.5),
        opacity=0.4  # Opacity set at trace level
    ))
    
    # 1 month moving average line
    fig.add_trace(go.Scatter(
        x=ma.index,
        y=ma.values,
        mode='lines',
        name='1 month moving avg',
        line=dict(color='orangered', width=2.5),
        opacity=0.7  # Opacity set at trace level
    ))
    
    # Error bands
    if std_err is not None:
        std_err_bps = std_err * DECIMAL_TO_BPS
        upper = spread_bps.values + (std_err_bps.values * bandwidth)
        lower = spread_bps.values - (std_err_bps.values * bandwidth)
        
        fig.add_trace(go.Scatter(
            x=spread_bps.index,
            y=upper,
            mode='lines',
            name='upper bound',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=spread_bps.index,
            y=lower,
            mode='lines',
            name='lower bound',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(70, 130, 180, 0.3)',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Zero line
    fig.add_trace(go.Scatter(
        x=[spread_bps.index[0], spread_bps.index[-1]],
        y=[0.0, 0.0],
        mode='lines',
        name='zero line',
        line=dict(color='black', width=0.7, dash='solid'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # y-axis range
    ylim = np.nanpercentile(np.abs(spread_bps.values), 95)
    
    fig = apply_standard_layout(
        fig,
        title,
        [],
        xaxis_title="Date",
        yaxis_title="Difference In Quantile Mean Return (bps)"
    )
    
    fig.update_yaxes(range=[-ylim, ylim])
    
    # Set x-axis type to 'date' to prevent exponential notation
    fig.update_xaxes(type='date')
    
    return fig


def plot_ic_by_group(ic_group):
    """
    Plots Spearman Rank Information Coefficient for a given factor over
    provided forward returns. Separates by group.

    Parameters
    ----------
    ic_group : pd.DataFrame
        group-wise mean period wise returns.
        Expected structure: index contains 'group' (or is MultiIndex with 'group'),
        columns are forward return periods.

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    fig = go.Figure()
    
    # Handle empty or invalid data
    if ic_group is None or ic_group.empty:
        fig.add_annotation(
            text="No group data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig = apply_standard_layout(
            fig,
            "Information Coefficient By Group",
            [],
            xaxis_title="",
            yaxis_title="IC"
        )
        return fig
    
    # Find group column
    ic_group_df = ic_group.reset_index()
    
    if 'group' in ic_group_df.columns:
        group_col = 'group'
    elif ic_group_df.index.names and 'group' in ic_group_df.index.names:
        ic_group_df = ic_group.reset_index()
        group_col = 'group' if 'group' in ic_group_df.columns else ic_group_df.columns[0]
    else:
        group_col = ic_group_df.columns[0]
        if len(ic_group_df.columns) > len(utils.get_forward_returns_columns(ic_group.columns)):
            group_col = ic_group_df.columns[0]
    
    # Select only forward return columns
    forward_cols = utils.get_forward_returns_columns(ic_group.columns)
    if len(forward_cols) == 0:
        fig.add_annotation(
            text="No forward return periods found",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig = apply_standard_layout(
            fig,
            "Information Coefficient By Group",
            [],
            xaxis_title="",
            yaxis_title="IC"
        )
        return fig
    
    # Use only forward return columns when melting
    ic_group_melted = ic_group_df.melt(
        id_vars=[group_col],
        value_vars=forward_cols,
        var_name='Period',
        value_name='IC'
    )
    
    # Remove NaN values
    ic_group_melted = ic_group_melted.dropna(subset=['IC'])
    
    if ic_group_melted.empty:
        fig.add_annotation(
            text="No valid IC data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig = apply_standard_layout(
            fig,
            "Information Coefficient By Group",
            [],
            xaxis_title="",
            yaxis_title="IC"
        )
        return fig
    
    # Create Bar traces for each Period
    periods = sorted(ic_group_melted['Period'].unique())
    groups = sorted(ic_group_melted[group_col].unique())
    
    # Calculate x-axis positions (arranged by group)
    x_positions = list(range(len(groups)))
    
    # Generate colors for each period
    colors = px.colors.qualitative.Set3[:len(periods)] if len(periods) <= 12 else px.colors.qualitative.Set3 * (len(periods) // 12 + 1)
    
    # Calculate bar width (adjusted based on number of periods)
    bar_width = 0.8 / len(periods) if len(periods) > 1 else 0.8
    
    for idx, period in enumerate(periods):
        period_data = ic_group_melted[ic_group_melted['Period'] == period]
        
        # Extract IC values for each group (maintain order)
        ic_values = []
        for group in groups:
            group_data = period_data[period_data[group_col] == group]
            if not group_data.empty:
                ic_values.append(group_data['IC'].iloc[0])
            else:
                ic_values.append(0)
        
        # x-axis position offset (slightly shift for each period)
        x_offset = (idx - (len(periods) - 1) / 2) * bar_width
        
        fig.add_trace(go.Bar(
            x=[x + x_offset for x in x_positions],
            y=ic_values,
            name=str(period),
            marker_color=colors[idx],
            width=bar_width,
            text=[f"{val:.3f}" for val in ic_values],
            textposition="outside",
            textfont=dict(size=10)
        ))
    
    # Apply layout
    fig = apply_standard_layout(
        fig,
        "Information Coefficient By Group",
        None,
        xaxis_title="",
        yaxis_title="IC"
    )
    
    # Configure x-axis
    fig.update_xaxes(
        tickmode='array',
        tickvals=x_positions,
        ticktext=[str(g) for g in groups],
        tickangle=-45
    )
    
    return fig


def plot_factor_rank_auto_correlation(factor_autocorrelation, period=1, factor_autocorrelation_dict=None):
    """
    Plots factor rank autocorrelation over time.
    See factor_rank_autocorrelation for more details.

    Parameters
    ----------
    factor_autocorrelation : pd.Series or dict
        Rolling 1 period (defined by time_rule) autocorrelation
        of factor values.
        If dict, keys are periods and values are Series.
    period: int, optional
        Period over which the autocorrelation is calculated (only used in single Series mode).
    factor_autocorrelation_dict : dict, optional
        Autocorrelation data for multiple periods {period: Series}.
        If provided, multiple periods can be toggled.

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    # Convert index to datetime for time series charts
    if factor_autocorrelation_dict is not None:
        # Convert each Series index in dict to datetime
        factor_autocorrelation_dict = {
            k: v.copy() if not isinstance(v.index, pd.DatetimeIndex) else v
            for k, v in factor_autocorrelation_dict.items()
        }
        for k, v in factor_autocorrelation_dict.items():
            if not isinstance(v.index, pd.DatetimeIndex):
                v.index = pd.to_datetime(v.index)
    elif factor_autocorrelation is not None:
        factor_autocorrelation = factor_autocorrelation.copy()
        if not isinstance(factor_autocorrelation.index, pd.DatetimeIndex):
            factor_autocorrelation.index = pd.to_datetime(factor_autocorrelation.index)
    
    fig = go.Figure()
    
    # Handle multiple periods
    if factor_autocorrelation_dict is not None:
            periods = sorted(factor_autocorrelation_dict.keys())
            # Convert period labels to "5D", "20D" format
            period_labels = [f"{int(p)}D" for p in periods]
            
            for period_key in periods:
                fac = factor_autocorrelation_dict[period_key]
                is_visible = (period_key == periods[0])
                
                fig.add_trace(go.Scatter(
                    x=fac.index,
                    y=fac.values,
                    mode='lines',
                    name=f'{int(period_key)}D Period',
                    line=dict(color='steelblue', width=1.5),
                    visible=is_visible
                ))
            
            # Create period toggle buttons
            # Use period_labels to display in "5D", "20D" format
            buttons = create_visibility_buttons(period_labels, 1, len(fig.data))
            
            fig = apply_standard_layout(
                fig,
                "Factor Rank Autocorrelation",
                [dict(active=0, buttons=buttons, direction="down", x=0.01, xanchor="left", y=1.15, yanchor="top")],
                xaxis_title="Date",
                yaxis_title="Autocorrelation Coefficient",
                showlegend=False
            )
            
            # Add zero line
            fig.add_hline(y=0.0, line_dash="solid", line_color="black", line_width=1)
    else:
        # Handle single period
        fig.add_trace(go.Scatter(
            x=factor_autocorrelation.index,
            y=factor_autocorrelation.values,
            mode='lines',
            name=f'{period}D Period',
            line=dict(color='steelblue', width=1.5)
        ))
        
        fig = apply_standard_layout(
            fig,
            f"{period}D Period Factor Rank Autocorrelation",
            [],
            xaxis_title="Date",
            yaxis_title="Autocorrelation Coefficient",
            showlegend=False
        )
        
        # Zero line 추가
        fig.add_hline(y=0.0, line_dash="solid", line_color="black", line_width=1)
        
        # Mean annotation 추가
        mean_val = factor_autocorrelation.mean()
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            text=f"Mean {mean_val:.3f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=12),
            align="left"
        )
    
    # Set x-axis type to 'date' to prevent exponential notation
    fig.update_xaxes(type='date')
    
    return fig


def plot_top_bottom_quantile_turnover(quantile_turnover, period=1, quantile_turnover_dict=None):
    """
    Plots period wise top and bottom quantile factor turnover.

    Parameters
    ----------
    quantile_turnover: pd.DataFrame or dict
        Quantile turnover (each DataFrame column a quantile).
        If dict, keys are periods and values are DataFrames.
    period: int, optional
        Period over which to calculate the turnover (only used in single DataFrame mode).
    quantile_turnover_dict : dict, optional
        Turnover data for multiple periods {period: DataFrame}.
        If provided, multiple periods can be toggled.

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    # Convert index to datetime for time series charts
    if quantile_turnover_dict is not None:
        # Convert each DataFrame index in dict to datetime
        quantile_turnover_dict = {
            k: v.copy() if not isinstance(v.index, pd.DatetimeIndex) else v
            for k, v in quantile_turnover_dict.items()
        }
        for k, v in quantile_turnover_dict.items():
            if not isinstance(v.index, pd.DatetimeIndex):
                v.index = pd.to_datetime(v.index)
    else:
        quantile_turnover = quantile_turnover.copy()
        if not isinstance(quantile_turnover.index, pd.DatetimeIndex):
            quantile_turnover.index = pd.to_datetime(quantile_turnover.index)
    
    fig = go.Figure()
    
    # Handle multiple periods
    if quantile_turnover_dict is not None:
            periods = sorted(quantile_turnover_dict.keys())
            # Convert period labels to "5D", "20D" format
            period_labels = [f"{int(p)}D" for p in periods]
            
            for period_key in periods:
                qt = quantile_turnover_dict[period_key]
                max_quantile = qt.columns.max()
                min_quantile = qt.columns.min()
                
                is_visible = (period_key == periods[0])
                
                # Top quantile
                fig.add_trace(go.Scatter(
                    x=qt.index,
                    y=qt[max_quantile].values,
                    mode='lines',
                    name='Top quantile turnover',
                    line=dict(color='blue', width=1),
                    visible=is_visible,
                    legendgroup=period_key,
                    showlegend=True  # Show legend for all periods
                ))
                
                # Bottom quantile
                fig.add_trace(go.Scatter(
                    x=qt.index,
                    y=qt[min_quantile].values,
                    mode='lines',
                    name='Bottom quantile turnover',
                    line=dict(color='red', width=1),
                    visible=is_visible,
                    legendgroup=period_key,
                    showlegend=True  # Show legend for all periods
                ))
            
            # Create period toggle buttons (top + bottom = 2 traces per period)
            # Use period_labels to display in "5D", "20D" format
            buttons = create_visibility_buttons(period_labels, 2, len(fig.data))
            
            fig = apply_standard_layout(
                fig,
                "Top and Bottom Quantile Turnover",
                [dict(active=0, buttons=buttons, direction="down", x=0.01, xanchor="left", y=1.15, yanchor="top")],
                xaxis_title="Date",
                yaxis_title="Proportion Of Names New To Quantile"
            )
    else:
        # Handle single period
        max_quantile = quantile_turnover.columns.max()
        min_quantile = quantile_turnover.columns.min()
        
        fig.add_trace(go.Scatter(
            x=quantile_turnover.index,
            y=quantile_turnover[max_quantile].values,
            mode='lines',
            name='Top quantile turnover',
            line=dict(color='blue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=quantile_turnover.index,
            y=quantile_turnover[min_quantile].values,
            mode='lines',
            name='Bottom quantile turnover',
            line=dict(color='red', width=1)
        ))
        
        fig = apply_standard_layout(
            fig,
            f"{period}D Period Top and Bottom Quantile Turnover",
            [],
            xaxis_title="Date",
            yaxis_title="Proportion Of Names New To Quantile"
        )
    
    # Set x-axis type to 'date' to prevent exponential notation
    fig.update_xaxes(type='date')
    
    return fig


def plot_monthly_ic_heatmap(mean_monthly_ic):
    """
    Plots a heatmap of the information coefficient or returns by month.
    Uses explicit integer mapping to prevent Plotly axis type confusion.

    Important: Coordinate system interpretation issue resolution (Integer Mapping technique)
    
    Problem:
    - Text displays correctly when using fig.show() in Jupyter Notebook,
      but disappears or appears in wrong position when saved as HTML file
    - Cause: Coordinate system interpretation mismatch between Python and browser (JS)
      * Python: Interprets "2000" as string (Category)
      * Browser: Interprets "2000" as number (Linear), causing coordinate mismatch
    
    Solution (Integer Mapping):
    1. Force coordinate unification: Convert data to integer indices (0, 1, 2...)
       - Heatmap: x=[0, 1, 2...], y=[0, 1, 2...]
       - Annotation: x=0, y=0 (use integer coordinates)
    2. Tick text overlay: Display original labels ("2000", "2001") instead of integers on axes
       - tickmode='array', tickvals=[0,1,2...], ticktext=["2000","2001"...]
    
    This approach works stably regardless of Plotly version or environment (HTML/Notebook).

    Parameters
    ----------
    mean_monthly_ic : pd.DataFrame
        The mean monthly IC for N periods forward.
        Columns are forward return periods (e.g., '5D', '20D').

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    mean_monthly_ic = mean_monthly_ic.copy()
    periods = list(mean_monthly_ic.columns)
    
    # Convert index to year-month
    new_index_year = []
    new_index_month = []
    for date in mean_monthly_ic.index:
        new_index_year.append(date.year)
        new_index_month.append(date.month)

    mean_monthly_ic.index = pd.MultiIndex.from_arrays(
        [new_index_year, new_index_month], names=["year", "month"]
    )
    
    # 1. Subplot configuration
    n_cols = 2 if len(periods) <= 4 else 3
    n_rows = (len(periods) + n_cols - 1) // n_cols
    
    subplot_titles = [f"{period} Period" for period in periods]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        # Reduced vertical spacing to bring subplot titles closer together
        # Still maintains enough space to prevent overlap with chart areas
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 2. Graph generation loop
    for idx, period in enumerate(periods):
        ic = mean_monthly_ic[period]
        heatmap_data = ic.unstack()
        
        # (1) Define labels (strings)
        # x-axis: Jan, Feb... / y-axis: 2000, 2001... (order guaranteed)
        x_labels = [f"{m}월" for m in heatmap_data.columns]
        y_labels = [str(y) for y in heatmap_data.index]
        
        # (2) Generate integer mapping (0, 1, 2...)
        # Key: Force coordinates to integers so Plotly doesn't confuse strings/numbers.
        # Using strings like "2000", "2001" as coordinates causes browser to interpret
        # them as numbers, leading to coordinate mismatch. Convert to integer indices for stability.
        x_indices = list(range(len(x_labels)))
        y_indices = list(range(len(y_labels)))
        
        # (3) Generate hover text (manual creation since hover breaks with integer coordinates)
        hover_text = []
        for y_val in heatmap_data.index:
            row_hover = []
            for x_val in heatmap_data.columns:
                z_val = heatmap_data.loc[y_val, x_val]
                row_hover.append(f"{x_val}월 {y_val}<br>IC: {z_val:.3f}")
            hover_text.append(row_hover)

        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        
        # (4) Draw heatmap (use integer coordinates!)
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=x_indices,  # [0, 1, 2...]
                y=y_indices,  # [0, 1, 2...]
                hovertext=hover_text,  # Connect manually created hover text
                hoverinfo="text",       # Ignore default format, only show above text
                colorscale="RdBu",
                zmid=0.0,
                colorbar=dict(title="IC", len=0.6) if idx == 0 else None,
                showscale=(idx == 0)
            ),
            row=row,
            col=col
        )
        
        # (5) Annotations (numbers in boxes)
        # Also use integer coordinates (x_idx, y_idx) here to guarantee 100% positioning
        for y_idx, _ in enumerate(y_labels):
            for x_idx, _ in enumerate(x_labels):
                val = heatmap_data.values[y_idx, x_idx]
                if np.isnan(val):
                    # Display "NaN" text for NaN values
                    fig.add_annotation(
                        text="NaN",
                        x=x_idx,  # Integer coordinate
                        y=y_idx,  # Integer coordinate
                        showarrow=False,
                        font=dict(size=9, color="gray"),  # Display NaN in gray
                        row=row,
                        col=col
                    )
                else:
                    # Display numeric values with 3 decimal places
                    text_color = "white" if abs(val) > 0.15 else "black"
                    fig.add_annotation(
                        text=f"{val:.3f}",
                        x=x_idx,  # Integer coordinate
                        y=y_idx,  # Integer coordinate
                        showarrow=False,
                        font=dict(size=9, color=text_color),
                        row=row,
                        col=col
                    )
        
        # (6) Axis configuration (overlay string labels on integer coordinates)
        # Key: Put integers in tickvals and original strings in ticktext for "trick".
        # Internally uses integer coordinates (0, 1, 2...) but displays
        # original labels ("2000", "2001"...) to users for natural UI
        fig.update_xaxes(
            tickmode='array',
            tickvals=x_indices,
            ticktext=x_labels,
            row=row, col=col
        )
        fig.update_yaxes(
            tickmode='array',
            tickvals=y_indices,
            ticktext=y_labels,
            row=row, col=col
        )

    # 3. Layout
    fig.update_layout(
        title=dict(
            text="Monthly Mean Information Coefficient (IC) Heatmap",
            x=0.5,
            xanchor="center",
            font=dict(size=16)
        ),
        # Increase height to 450px per row for better spacing
        height=450 * n_rows,
        template="plotly_white",
        # Ensure top margin to prevent title from being cut off
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    # 4. Adjust subplot title positions (prevent chart area overlap)
    # Push all 'paper' reference Annotations (i.e., subplot titles) up by 20px.
    # Use selector to avoid affecting numbers inside charts (data Annotations).
    # This ensures titles don't overlap with the chart's top border (x-axis labels)
    fig.update_annotations(
        yshift=20,                                  # Move up 20 pixels
        selector=dict(xref="paper", yref="paper")   # Apply only to Subplot Titles
    )
    
    return fig


def plot_cumulative_returns(factor_returns, period, freq=None, title=None):
    """
    Plots the cumulative returns of the returns series passed in.

    Parameters
    ----------
    factor_returns : pd.Series
        Period wise returns of dollar neutral portfolio weighted by factor
        value.
    period : pandas.Timedelta or string
        Length of period for which the returns are computed (e.g. 1 day)
        if 'period' is a string it must follow pandas.Timedelta constructor
        format (e.g. '1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset, optional
        Deprecated. Not used in the current implementation.
        Used to specify a particular trading calendar e.g. BusinessDay or Day
        Usually this is inferred from utils.infer_trading_calendar, which is
        called by either get_clean_factor_and_forward_returns or
        compute_forward_returns
    title: string, optional
        Custom title

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    # Convert index to datetime for time series charts
    factor_returns = factor_returns.copy()
    if not isinstance(factor_returns.index, pd.DatetimeIndex):
        factor_returns.index = pd.to_datetime(factor_returns.index)
    
    factor_returns = perf.cumulative_returns(factor_returns)

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=factor_returns.index,
        y=factor_returns.values,
        mode='lines',
        name='Cumulative Returns',
        line=dict(color='forestgreen', width=3),
        opacity=0.6
    ))
    
    # Zero line
    fig.add_hline(
        y=1.0,
        line_dash="solid",
        line_color="black",
        line_width=1
    )
    
    # Apply layout
    title_text = (
        f"Portfolio Cumulative Return ({period} Fwd Period)"
            if title is None
            else title
    )
    fig = apply_standard_layout(
        fig,
        title_text,
        None,
        xaxis_title="",
        yaxis_title="Cumulative Returns"
    )
    
    # Set x-axis type to 'date' to prevent exponential notation
    fig.update_xaxes(type='date')
    
    return fig


def plot_cumulative_returns_by_quantile(quantile_returns, period, freq=None, quantile_returns_dict=None):
    """
    Plots the cumulative returns of various factor quantiles.

    Parameters
    ----------
    quantile_returns : pd.DataFrame or dict
        Returns by factor quantile.
        If dict, keys are periods and values are DataFrames.
    period : pandas.Timedelta or string
        Length of period for which the returns are computed (only used in single DataFrame mode).
    freq : pandas DateOffset, optional
        Deprecated. Not used in the current implementation.
        Used to specify a particular trading calendar e.g. BusinessDay or Day
    quantile_returns_dict : dict, optional
        Quantile returns data for multiple periods {period: DataFrame}.
        If provided, multiple periods can be toggled.

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    # Convert index to datetime for time series charts
    if quantile_returns_dict is not None:
        # Convert each DataFrame index in dict to datetime
        quantile_returns_dict = {
            k: v.copy() if not isinstance(v.index, pd.DatetimeIndex) else v
            for k, v in quantile_returns_dict.items()
        }
        for k, v in quantile_returns_dict.items():
            if not isinstance(v.index, pd.DatetimeIndex):
                v.index = pd.to_datetime(v.index)
    else:
        quantile_returns = quantile_returns.copy()
        if not isinstance(quantile_returns.index, pd.DatetimeIndex):
            quantile_returns.index = pd.to_datetime(quantile_returns.index)

    fig = go.Figure()

    # Handle multiple periods
    if quantile_returns_dict is not None:
        periods = sorted(quantile_returns_dict.keys())
        quantile_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for period_key in periods:
            qr = quantile_returns_dict[period_key]
            ret_wide = qr.unstack("factor_quantile")
            cum_ret = ret_wide.apply(perf.cumulative_returns)
            cum_ret = cum_ret.loc[:, ::-1]  # negative quantiles as 'red'
            
            is_visible = (period_key == periods[0])
            
            # Add traces for each quantile
            for i, quantile in enumerate(cum_ret.columns):
                color = quantile_colors[i % len(quantile_colors)]
                fig.add_trace(go.Scatter(
                    x=cum_ret.index,
                    y=cum_ret[quantile].values,
                    mode='lines',
                    name=f'Quantile {quantile}',
                    line=dict(color=color, width=2),
                    visible=is_visible,
                    legendgroup=period_key,
                    showlegend=is_visible
                ))
        
        # Create period toggle buttons (number of quantiles traces per period)
        num_quantiles = len(cum_ret.columns)
        buttons = create_visibility_buttons(periods, num_quantiles, len(fig.data))
        
        fig = apply_standard_layout(
            fig,
            "Cumulative Return by Quantile",
            [dict(active=0, buttons=buttons, direction="down", x=0.01, xanchor="left", y=1.15, yanchor="top")],
            xaxis_title="Date",
            yaxis_title="Cumulative Returns"
        )
        
        # Add zero line
        fig.add_hline(y=1.0, line_dash="solid", line_color="black", line_width=1)
    else:
        # Handle single period
        ret_wide = quantile_returns.unstack("factor_quantile")
        cum_ret = ret_wide.apply(perf.cumulative_returns)
        cum_ret = cum_ret.loc[:, ::-1]
        
        quantile_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, quantile in enumerate(cum_ret.columns):
            color = quantile_colors[i % len(quantile_colors)]
            fig.add_trace(go.Scatter(
                x=cum_ret.index,
                y=cum_ret[quantile].values,
                mode='lines',
                name=f'Quantile {quantile}',
                line=dict(color=color, width=2)
            ))
        
        fig = apply_standard_layout(
            fig,
            f"Cumulative Return by Quantile ({period} Period Forward Return)",
            [],
            xaxis_title="Date",
            yaxis_title="Cumulative Returns"
        )
        
        fig.add_hline(y=1.0, line_dash="solid", line_color="black", line_width=1)
        
        # Set y-axis range
        all_cum_values = cum_ret.values.flatten()
        if len(all_cum_values) > 0:
            y_min = min(all_cum_values)
            y_max = max(all_cum_values)
            y_range = y_max - y_min
            padding = max(y_range * 0.1, 0.1)  # Minimum 0.1 padding
            fig.update_yaxes(range=[y_min - padding, y_max + padding])
    
    # Set x-axis type to 'date' to prevent exponential notation
    fig.update_xaxes(type='date')
    
    return fig


def plot_quantile_average_cumulative_return(
    avg_cumulative_returns,
    by_quantile=False,
    std_bar=False,
    title=None,
):
    """
    Plots sector-wise mean daily returns for factor quantiles
    across provided forward price movement columns.

    Parameters
    ----------
    avg_cumulative_returns: pd.Dataframe
        The format is the one returned by
        performance.average_cumulative_return_by_quantile
    by_quantile : boolean, optional
        Disaggregated figures by quantile (useful to clearly see std dev bars)
    std_bar : boolean, optional
        Plot standard deviation plot
    title: string, optional
        Custom title

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """

    avg_cumulative_returns = avg_cumulative_returns.multiply(DECIMAL_TO_BPS)
    # Use get_level_values() as recommended
    quantiles = len(avg_cumulative_returns.index.get_level_values(0).unique())
    
    # Generate color palette (coolwarm style, negative quantiles as 'red')
    # Use Plotly colors
    quantile_colors = px.colors.sequential.RdYlBu_r[:quantiles] if quantiles <= 10 else px.colors.qualitative.Set3[:quantiles]
    quantile_colors = quantile_colors[::-1]  # negative quantiles as 'red'

    if by_quantile:
        # Create subplots for each quantile
        quantile_list = sorted(avg_cumulative_returns.index.get_level_values("factor_quantile").unique())
        n_cols = 2
        n_rows = (len(quantile_list) + n_cols - 1) // n_cols
        
        subplot_titles = [f"Quantile {q}" for q in quantile_list]
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Performance optimization: disable sorting with sort=False
        for idx, (quantile, q_ret) in enumerate(
            avg_cumulative_returns.groupby(level="factor_quantile", sort=False)
        ):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1

            mean = q_ret.loc[(quantile, "mean")]
            
            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=mean.index,
                    y=mean.values,
                    mode='lines',
                    name=f'Quantile {quantile}',
                    line=dict(color=quantile_colors[idx], width=2),
                    showlegend=False,
                ),
                row=row,
                col=col
            )
            
            # Error bars (when std_bar is True)
            if std_bar:
                std = q_ret.loc[(quantile, "std")]
                fig.add_trace(
                    go.Scatter(
                        x=std.index,
                        y=mean.values,
                        mode='markers',
                        marker=dict(size=0),
                        error_y=dict(
                            type='data',
                            array=std.values,
                            visible=True,
                            color=quantile_colors[idx]
                        ),
                        showlegend=False,
                    ),
                    row=row,
                    col=col
                )
            
            # Zero vertical line
            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="black",
                line_width=1,
                row=row,
                col=col
            )
            
            # Y-axis label
            fig.update_yaxes(title_text="Mean Return (bps)", row=row, col=col)
        
        fig.update_layout(
            title=dict(
                text=title if title else "Average Cumulative Returns by Quantile",
                x=0.5,
                xanchor="center",
                font=dict(size=16)
            ),
            height=400 * n_rows,
            template="plotly_white"
        )
        
        return fig

    else:
        # Display all quantiles in a single chart
        fig = go.Figure()
        
        # Performance optimization: disable sorting with sort=False
        for i, (quantile, q_ret) in enumerate(
            avg_cumulative_returns.groupby(level="factor_quantile", sort=False)
        ):
            mean = q_ret.loc[(quantile, "mean")]
            
            # Mean line
            fig.add_trace(go.Scatter(
                x=mean.index,
                y=mean.values,
                mode='lines',
                name=f'Quantile {quantile}',
                line=dict(color=quantile_colors[i], width=2)
            ))
            
            # Error bars (when std_bar is True)
            if std_bar:
                std = q_ret.loc[(quantile, "std")]
                fig.add_trace(go.Scatter(
                    x=std.index,
                    y=mean.values,
                    mode='markers',
                    marker=dict(size=0),
                    error_y=dict(
                        type='data',
                        array=std.values,
                        visible=True,
                        color=quantile_colors[i]
                    ),
                    name=f'Quantile {quantile} (std)',
                    showlegend=False
                ))
        
        # Zero vertical line
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="black",
            line_width=1
        )
        
        # Apply layout
        fig = apply_standard_layout(
            fig,
            title if title else "Average Cumulative Returns by Quantile",
            None,
            xaxis_title="Periods",
            yaxis_title="Mean Return (bps)"
        )
        
        return fig


def plot_events_distribution(events, num_bars=50):
    """
    Plots the distribution of events in time.

    Parameters
    ----------
    events : pd.Series
        A pd.Series whose index contains at least 'date' level.
    num_bars : integer, optional
        Number of bars to plot

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """

    start = events.index.get_level_values("date").min()
    end = events.index.get_level_values("date").max()
    group_interval = (end - start) / num_bars
    grouper = pd.Grouper(level="date", freq=group_interval)
    # Performance optimization: disable sorting with sort=False
    event_counts = events.groupby(grouper, sort=False).count()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=event_counts.index,
        y=event_counts.values,
        marker_color='steelblue',
        text=[f"{val}" for val in event_counts.values],
        textposition="outside",
        textfont=dict(size=10)
    ))
    
    # Apply layout
    fig = apply_standard_layout(
        fig,
        "Distribution of events in time",
        None,
        xaxis_title="Date",
        yaxis_title="Number of events"
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=-45)
    
    return fig


def plot_underwater_drawdown(drawdown_series, period, mdd=None, mdd_date=None, drawdown_dict=None):
    """
    Plots an underwater plot (drawdown chart) for Long-Short Spread portfolio.
    
    Underwater plot shows the drawdown from the peak cumulative return.
    This helps visualize both the depth (how much) and duration (how long) of drawdowns.
    
    Parameters
    ----------
    drawdown_series : pd.Series
        Drawdown time series (negative values showing drawdown from peak).
        Typically from cumulative_spread_drawdown() function.
        Used when drawdown_dict is None (single period mode).
    period : str
        Forward return period (e.g., '20D')
        Used when drawdown_dict is None (single period mode).
    mdd : float, optional
        Maximum drawdown value (for annotation)
    mdd_date : pd.Timestamp, optional
        Date when MDD occurred (for annotation)
    drawdown_dict : dict, optional
        Dictionary of {period: (drawdown_series, mdd, mdd_date)} for multiple periods.
        If provided, enables period toggle functionality.
    
    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    fig = go.Figure()
    
    # Handle multiple periods
    if drawdown_dict is not None:
        # Convert period to number and sort ascending (5D, 20D, 60D order)
        def period_to_days(p):
            """Convert period to days (e.g., '5D' -> 5, '20D' -> 20)"""
            if isinstance(p, str):
                return int(p.replace('D', ''))
            elif isinstance(p, (int, float)):
                return int(p)
            else:
                return 0
        
        periods = sorted(drawdown_dict.keys(), key=period_to_days, reverse=False)
        period_labels = [utils.format_period(p) for p in periods]
        
        for period_key in periods:
            drawdown_data = drawdown_dict[period_key]
            if isinstance(drawdown_data, tuple):
                dd_series, dd_mdd, dd_mdd_date = drawdown_data
            else:
                dd_series = drawdown_data
                dd_mdd = None
                dd_mdd_date = None
            
            is_visible = (period_key == periods[0])
            drawdown_pct = dd_series * 100
            
            # Underwater plot (fill negative area)
            fig.add_trace(go.Scatter(
                x=drawdown_pct.index,
                y=drawdown_pct.values,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(220, 20, 60, 0.3)',
                line=dict(color='crimson', width=2),
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>',
                visible=is_visible,
                legendgroup=period_key,
                showlegend=False
            ))
            
            # MDD annotation (if provided)
            if dd_mdd is not None and dd_mdd_date is not None and dd_mdd_date in drawdown_pct.index:
                mdd_pct = dd_mdd * 100
                fig.add_annotation(
                    x=dd_mdd_date,
                    y=mdd_pct,
                    text=f"MDD: {mdd_pct:.2f}%<br>{dd_mdd_date.strftime('%Y-%m-%d')}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="darkred",
                    bgcolor="rgba(220, 20, 60, 0.9)",
                    bordercolor="darkred",
                    borderwidth=2,
                    font=dict(size=11, color="white", family="Arial Black"),
                    ax=0,
                    ay=-40,
                    visible=is_visible
                )
        
        # Period toggle 버튼 생성
        buttons = create_visibility_buttons(period_labels, 1, len(fig.data))
        
        # Zero line
        fig.add_hline(
            y=0.0,
            line_dash="solid",
            line_color="black",
            line_width=1,
            opacity=0.8
        )
        
        # 레이아웃 적용 (toggle 버튼이 plot을 침범하지 않도록 margin 추가)
        title_text = "Underwater Plot - Long-Short Spread Drawdown"
        fig = apply_standard_layout(
            fig,
            title_text,
            [dict(active=0, buttons=buttons, direction="down", x=0.01, xanchor="left", y=1.02, yanchor="bottom")],
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            margin=dict(t=100, b=50, l=50, r=50)  # 상단 여백 증가 (toggle 버튼 공간 확보)
        )
        
        # y축 범위 설정 (모든 period의 최소값 사용)
        all_drawdowns = []
        for period_key in periods:
            drawdown_data = drawdown_dict[period_key]
            if isinstance(drawdown_data, tuple):
                dd_series = drawdown_data[0]
            else:
                dd_series = drawdown_data
            all_drawdowns.extend((dd_series * 100).values)
        
        y_min = min(min(all_drawdowns) * 1.1, -5) if all_drawdowns else -5
        y_max = 2
        fig.update_yaxes(range=[y_min, y_max])
        
        return fig
    
    # Handle single period (existing logic)
    # Convert period to string if it's a scalar
    if not isinstance(period, str):
        period = str(period)
    
    # Convert drawdown to percentage (negative values)
    drawdown_pct = drawdown_series * 100
    
    # Underwater plot (fill negative area)
    fig.add_trace(go.Scatter(
        x=drawdown_pct.index,
        y=drawdown_pct.values,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        fillcolor='rgba(220, 20, 60, 0.3)',
        line=dict(color='crimson', width=2),
        hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    # Zero line
    fig.add_hline(
        y=0.0,
        line_dash="solid",
        line_color="black",
        line_width=1,
        opacity=0.8
    )
    
    # MDD annotation (if provided)
    if mdd is not None and mdd_date is not None and mdd_date in drawdown_pct.index:
        mdd_pct = mdd * 100
        fig.add_annotation(
            x=mdd_date,
            y=mdd_pct,
            text=f"MDD: {mdd_pct:.2f}%<br>{mdd_date.strftime('%Y-%m-%d')}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="darkred",
            bgcolor="rgba(220, 20, 60, 0.9)",
            bordercolor="darkred",
            borderwidth=2,
            font=dict(size=11, color="white", family="Arial Black"),
            ax=0,
            ay=-40
        )
    
    # Apply layout
    title_text = f"Underwater Plot - Long-Short Spread Drawdown ({period})"
    fig = apply_standard_layout(
        fig,
        title_text,
        None,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)"
    )
    
    # Set y-axis range (extend in negative direction)
    y_min = min(drawdown_pct.min() * 1.1, -5)  # 10% margin from minimum or -5%
    y_max = 2  # Top margin space
    fig.update_yaxes(range=[y_min, y_max])
    
    return fig


def plot_long_short_contribution(
    mean_quant_ret_bydate, period=None, long_short=True, group_neutral=False
):
    """
    Plots Long Leg, Short Leg, and Spread cumulative returns separately.
    
    This helps identify whether spread returns come from Long outperformance,
    Short outperformance, or both.
    
    **Calculation Process:**
    
    1. **Long Leg**: 
       - Daily average returns of the quantile with highest factor values (e.g., Q5)
       - Returns of portfolio that buys (long) stocks in this quantile
       - Example: If factor is "momentum", returns of top 20% stocks with strongest momentum
    
    2. **Short Leg**:
       - Returns of portfolio that shorts stocks in the quantile with lowest factor values (e.g., Q1)
       - Shorting reverses returns: if bottom_returns is -5%, shorting yields +5%
       - Calculation: `cumulative_returns(-bottom_returns)`
       - Example: If factor is "momentum", returns from shorting bottom 20% stocks with weakest momentum
    
    3. **Spread**:
       - Difference between Long Leg and Short Leg
       - Calculation: `cumulative_returns(top_returns - bottom_returns)`
       - Pure returns of Long/Short strategy
       - Spread increases when Long Leg rises and Short Leg falls
       - Mathematically: Spread = Long Leg - Short Leg = top_cum - short_cum
    
    4. **Cumulative Returns Calculation**:
       - Calculate cumulative returns by multiplying daily returns of each Leg
       - `cumulative_returns = (1 + r1) * (1 + r2) * ... * (1 + rn)`
    
    **Data Structure:**
    - `mean_quant_ret_bydate`: MultiIndex DataFrame
      - Index: (factor_quantile, date) - average returns by quantile and date
      - Columns: forward return periods (e.g., '5D', '20D', '60D', '120D')
      - Example: `mean_quant_ret_bydate.loc[(5, '2020-01-01'), '20D']` = 20-day forward return for Q5 quantile on 2020-01-01
    
    Parameters
    ----------
    mean_quant_ret_bydate : pd.DataFrame
        Mean returns by quantile and date.
        Index: factor_quantile (and date if MultiIndex), Columns: forward return periods
        If multiple periods are present in columns, all will be plotted with toggle.
    period : str, optional
        Forward return period (e.g., '20D')
        If None and mean_quant_ret_bydate has multiple columns, all periods will be plotted.
        If provided, only that period will be plotted (backward compatibility).
    long_short : bool, optional
        Deprecated. Not used in the current implementation.
        Whether to use long-short portfolio
    group_neutral : bool, optional
        Deprecated. Not used in the current implementation.
        Whether to use group-neutral portfolio
    
    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    fig = go.Figure()
    
    # If period is None or multiple periods exist: create toggle for all periods
    if period is None or len(mean_quant_ret_bydate.columns) > 1:
        # Convert period to number and sort ascending (5D, 20D, 60D order)
        def period_to_days(p):
            """Convert period to days (e.g., '5D' -> 5, '20D' -> 20)"""
            if isinstance(p, str):
                return int(p.replace('D', ''))
            elif isinstance(p, (int, float)):
                return int(p)
            else:
                return 0
        
        # If period is specified, use only that period; otherwise use all periods
        if period is not None:
            periods = [period] if period in mean_quant_ret_bydate.columns else []
        else:
            periods = sorted(mean_quant_ret_bydate.columns, key=period_to_days, reverse=False)
        
        if not periods:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid periods found",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig = apply_standard_layout(
                fig,
                "Long/Short Contribution Analysis",
                None,
                xaxis_title="Date",
                yaxis_title="Cumulative Returns"
            )
            return fig
        
        period_labels = [utils.format_period(p) for p in periods]
        
        for period_str in periods:
            if period_str not in mean_quant_ret_bydate.columns:
                continue
            
            is_visible = (period_str == periods[0])
            
            # Top and Bottom quantile returns
            if isinstance(mean_quant_ret_bydate.index, pd.MultiIndex):
                top_quantile = mean_quant_ret_bydate.index.get_level_values(0).max()
                bottom_quantile = mean_quant_ret_bydate.index.get_level_values(0).min()
                
                top_returns = mean_quant_ret_bydate.xs(top_quantile, level=0)[period_str]
                bottom_returns = mean_quant_ret_bydate.xs(bottom_quantile, level=0)[period_str]
            else:
                top_quantile = mean_quant_ret_bydate.index.max()
                bottom_quantile = mean_quant_ret_bydate.index.min()
                
                top_returns = mean_quant_ret_bydate.loc[top_quantile, period_str]
                bottom_returns = mean_quant_ret_bydate.loc[bottom_quantile, period_str]
                
                if not isinstance(top_returns, pd.Series):
                    if period_str in mean_quant_ret_bydate.columns:
                        top_returns = mean_quant_ret_bydate[period_str]
                    else:
                        continue
                if not isinstance(bottom_returns, pd.Series):
                    if period_str in mean_quant_ret_bydate.columns:
                        bottom_returns = mean_quant_ret_bydate[period_str]
                    else:
                        continue
            
            # Calculate cumulative returns
            # Long Leg: Cumulative returns of Top quantile
            top_cum = perf.cumulative_returns(top_returns)
            
            # Short Leg: Cumulative returns from shorting Bottom quantile
            # Shorting reverses returns, so cumulative returns of -bottom_returns
            # Mathematically: (1 + (-r1)) * (1 + (-r2)) * ... = (1 - r1) * (1 - r2) * ...
            short_cum = perf.cumulative_returns(-bottom_returns)
            
            # Spread: Cumulative returns of Long - Short
            # Cumulative of top_returns - bottom_returns = Long Leg - Short Leg
            spread_cum = perf.cumulative_returns(top_returns - bottom_returns)
            
            # Include period label in name to distinguish each period
            period_label = utils.format_period(period_str)
            
            # Long Leg (Top Quantile)
            fig.add_trace(go.Scatter(
                x=top_cum.index,
                y=top_cum.values,
                mode='lines',
                name=f'Long Leg (Q{top_quantile})',
                line=dict(color='forestgreen', width=2.5),
                hovertemplate=f'Period: {period_label}<br>Date: %{{x}}<br>Long: %{{y:.4f}}<extra></extra>',
                visible=is_visible,
                legendgroup='long_leg',  # Same type of traces in same group
                showlegend=True  # Show legend for all periods
            ))
            
            # Short Leg (Bottom Quantile) - shorting returns
            # If bottom_returns is -5%, shorting yields +5%, so short_cum becomes 1.05
            fig.add_trace(go.Scatter(
                x=short_cum.index,
                y=short_cum.values,
                mode='lines',
                name=f'Short Leg (Q{bottom_quantile})',
                line=dict(color='crimson', width=2.5),
                hovertemplate=f'Period: {period_label}<br>Date: %{{x}}<br>Short: %{{y:.4f}}<extra></extra>',
                visible=is_visible,
                legendgroup='short_leg',  # Same type of traces in same group
                showlegend=True  # Show legend for all periods
            ))
            
            # Spread (Long - Short)
            # Spread = Long Leg - Short Leg = top_cum - short_cum
            # This should equal cumulative of top_returns - bottom_returns
            fig.add_trace(go.Scatter(
                x=spread_cum.index,
                y=spread_cum.values,
                mode='lines',
                name='Spread (Long - Short)',
                line=dict(color='steelblue', width=3, dash='dash'),
                hovertemplate=f'Period: {period_label}<br>Date: %{{x}}<br>Spread: %{{y:.4f}}<extra></extra>',
                visible=is_visible,
                legendgroup='spread',  # Same type of traces in same group
                showlegend=True  # Show legend for all periods
            ))
        
        # Create period toggle buttons (Long + Short + Spread = 3 traces per period)
        if len(periods) > 1:
            buttons = create_visibility_buttons(period_labels, 3, len(fig.data))
            
            # Apply layout (with toggle buttons)
            title_text = "Long/Short Contribution Analysis"
            fig = apply_standard_layout(
                fig,
                title_text,
                [dict(active=0, buttons=buttons, direction="down", x=0.01, xanchor="left", y=1.02, yanchor="bottom")],
                xaxis_title="Date",
                yaxis_title="Cumulative Returns",
                margin=dict(t=100, b=50, l=50, r=50)  # Increase top margin (space for toggle buttons)
            )
        else:
            # No toggle buttons for single period
            title_text = f"Long/Short Contribution Analysis ({periods[0]})"
            fig = apply_standard_layout(
                fig,
                title_text,
                None,
                xaxis_title="Date",
                yaxis_title="Cumulative Returns"
            )
        
        # Zero line
        fig.add_hline(
            y=1.0,
            line_dash="solid",
            line_color="black",
            line_width=1,
            opacity=0.8
        )
        
        return fig
    
    # Handle single period (backward compatibility)
    # Convert period to string if it's a scalar
    if not isinstance(period, str):
        period = str(period)
    
    if period not in mean_quant_ret_bydate.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Period '{period}' not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig = apply_standard_layout(
            fig,
            f"Long/Short Contribution Analysis ({period})",
            None,
            xaxis_title="Date",
            yaxis_title="Cumulative Returns"
        )
        return fig
    
    # Top and Bottom quantile returns
    # mean_quant_ret_bydate has MultiIndex structure (factor_quantile, date)
    # Index: MultiIndex (factor_quantile, date)
    # Columns: forward return periods (e.g., '5D', '20D', '60D', '120D')
    
    if isinstance(mean_quant_ret_bydate.index, pd.MultiIndex):
        # MultiIndex case: (factor_quantile, date)
        top_quantile = mean_quant_ret_bydate.index.get_level_values(0).max()
        bottom_quantile = mean_quant_ret_bydate.index.get_level_values(0).min()
        
        # Select period column for specific quantile (returns date-wise Series)
        top_returns = mean_quant_ret_bydate.xs(top_quantile, level=0)[period]
        bottom_returns = mean_quant_ret_bydate.xs(bottom_quantile, level=0)[period]
    else:
        # Single Index case (exception)
        top_quantile = mean_quant_ret_bydate.index.max()
        bottom_quantile = mean_quant_ret_bydate.index.min()
        
        # Check if period is in columns
        if period in mean_quant_ret_bydate.columns:
            top_returns = mean_quant_ret_bydate.loc[top_quantile, period]
            bottom_returns = mean_quant_ret_bydate.loc[bottom_quantile, period]
            
            # Handle non-Series case (scalar)
            if not isinstance(top_returns, pd.Series):
                # Use same value for all dates (temporary solution)
                top_returns = pd.Series([top_returns] * len(mean_quant_ret_bydate.index), 
                                      index=mean_quant_ret_bydate.index)
            if not isinstance(bottom_returns, pd.Series):
                bottom_returns = pd.Series([bottom_returns] * len(mean_quant_ret_bydate.index),
                                          index=mean_quant_ret_bydate.index)
        else:
            # Raise error if period not in columns
            raise ValueError(f"Period '{period}' not found in mean_quant_ret_bydate columns")
    
    # Calculate cumulative returns
    # Long Leg: Cumulative returns of Top quantile
    top_cum = perf.cumulative_returns(top_returns)
    
    # Short Leg: Cumulative returns from shorting Bottom quantile
    # Shorting reverses returns, so cumulative returns of -bottom_returns
    short_cum = perf.cumulative_returns(-bottom_returns)
    
    # Spread: Cumulative returns of Long - Short
    spread_cum = perf.cumulative_returns(top_returns - bottom_returns)
    
    fig = go.Figure()
    
    # Long Leg (Top Quantile)
    fig.add_trace(go.Scatter(
        x=top_cum.index,
        y=top_cum.values,
        mode='lines',
        name=f'Long Leg (Q{top_quantile})',
        line=dict(color='forestgreen', width=2.5),
        hovertemplate='Date: %{x}<br>Long: %{y:.4f}<extra></extra>'
    ))
    
    # Short Leg (Bottom Quantile) - shorting returns
    fig.add_trace(go.Scatter(
        x=short_cum.index,
        y=short_cum.values,
        mode='lines',
        name=f'Short Leg (Q{bottom_quantile})',
        line=dict(color='crimson', width=2.5),
        hovertemplate='Date: %{x}<br>Short: %{y:.4f}<extra></extra>'
    ))
    
    # Spread (Long - Short)
    fig.add_trace(go.Scatter(
        x=spread_cum.index,
        y=spread_cum.values,
        mode='lines',
        name='Spread (Long - Short)',
        line=dict(color='steelblue', width=3, dash='dash'),
        hovertemplate='Date: %{x}<br>Spread: %{y:.4f}<extra></extra>'
    ))
    
    # Zero line
    fig.add_hline(
        y=1.0,
        line_dash="solid",
        line_color="black",
        line_width=1,
        opacity=0.8
    )
    
    # Apply layout
    title_text = f"Long/Short Contribution Analysis ({period})"
    fig = apply_standard_layout(
        fig,
        title_text,
        None,
        xaxis_title="Date",
        yaxis_title="Cumulative Returns"
    )
    
    return fig


def plot_alpha_decay(ic_data=None, spread_returns=None, periods=None):
    """
    Plots alpha decay: how IC or Spread Return changes with forward period.
    
    This helps identify whether the factor's predictive power decays quickly
    (needs fast execution) or slowly (can trade more leisurely).
    
    Parameters
    ----------
    ic_data : pd.DataFrame, optional
        IC time series data (date × forward_periods).
        If provided, plots IC decay.
    spread_returns : pd.Series or pd.DataFrame, optional
        Spread returns by period.
        If provided, plots Spread Return decay.
    periods : list, optional
        List of periods to plot (e.g., [1, 5, 10, 20]).
        If not provided, inferred from data.
    
    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    fig = go.Figure()
    
    if ic_data is not None:
        # IC Decay Plot
        if periods is None:
            # Extract periods from ic_data.columns
            periods = [int(str(p).replace('D', '').replace('d', '')) for p in ic_data.columns if str(p).replace('D', '').replace('d', '').isdigit()]
        else:
            # If periods provided (string list or Index)
            # Convert string format ('5D', '20D') to integers
            periods_int = []
            for p in periods:
                if isinstance(p, str):
                    # '5D' -> 5
                    p_clean = p.replace('D', '').replace('d', '')
                    if p_clean.isdigit():
                        periods_int.append(int(p_clean))
                elif isinstance(p, (int, float)):
                    periods_int.append(int(p))
            periods = periods_int
        
        if not periods:
            # Return empty figure if periods cannot be found
            fig = apply_standard_layout(
                fig,
                "Alpha Decay: IC by Forward Period",
                None,
                xaxis_title="Forward Period (Days)",
                yaxis_title="Information Coefficient (IC)"
            )
            return fig
        
        # Calculate average IC for each period
        ic_means = []
        valid_periods = []
        for period in sorted(periods):
            period_str = f"{period}D"
            # Check if period exists in ic_data.columns (case-insensitive)
            matching_cols = [col for col in ic_data.columns if str(col).upper() == period_str.upper()]
            if matching_cols:
                # Use matching column if found
                col_name = matching_cols[0]
                ic_mean = ic_data[col_name].mean()
                if not np.isnan(ic_mean):
                    ic_means.append(ic_mean)
                    valid_periods.append(period)
            else:
                # Check directly with period_str format
                if period_str in ic_data.columns:
                    ic_mean = ic_data[period_str].mean()
                    if not np.isnan(ic_mean):
                        ic_means.append(ic_mean)
                        valid_periods.append(period)
        
        if not valid_periods:
            # Return empty figure if no valid periods
            fig = apply_standard_layout(
                fig,
                "Alpha Decay: IC by Forward Period",
                None,
                xaxis_title="Forward Period (Days)",
                yaxis_title="Information Coefficient (IC)"
            )
            return fig
        
        fig.add_trace(go.Scatter(
            x=valid_periods,
            y=ic_means,
            mode='lines+markers',
            name='IC Mean',
            line=dict(color='steelblue', width=3),
            marker=dict(size=10, color='steelblue'),
            hovertemplate='Period: %{x}D<br>IC: %{y:.4f}<extra></extra>'
        ))
        
        yaxis_title = "Information Coefficient (IC)"
        title_text = "Alpha Decay: IC by Forward Period"
        
    elif spread_returns is not None:
        # Spread Return Decay Plot
        if isinstance(spread_returns, pd.DataFrame):
            # DataFrame case: average spread return for each period
            if periods is None:
                periods = [int(str(p).replace('D', '')) for p in spread_returns.columns if str(p).replace('D', '').isdigit()]
            
            spread_means = []
            for period in sorted(periods):
                period_str = f"{period}D"
                if period_str in spread_returns.columns:
                    spread_means.append(spread_returns[period_str].mean() * DECIMAL_TO_BPS)
                else:
                    spread_means.append(np.nan)
            
            valid_periods = [p for p, s in zip(sorted(periods), spread_means) if not np.isnan(s)]
            valid_spread_means = [s for s in spread_means if not np.isnan(s)]
            
            fig.add_trace(go.Scatter(
                x=valid_periods,
                y=valid_spread_means,
                mode='lines+markers',
                name='Spread Return',
                line=dict(color='forestgreen', width=3),
                marker=dict(size=10, color='forestgreen'),
                hovertemplate='Period: %{x}D<br>Spread: %{y:.2f} bps<extra></extra>'
            ))
            
            yaxis_title = "Mean Spread Return (bps)"
            title_text = "Alpha Decay: Spread Return by Forward Period"
        else:
            # Series case: single period
            fig.add_annotation(
                text="Multiple periods required for decay plot",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig = apply_standard_layout(
                fig,
                "Alpha Decay: Spread Return by Forward Period",
                None,
                xaxis_title="Forward Period (Days)",
                yaxis_title="Mean Spread Return (bps)"
            )
            return fig
    else:
        # No data provided
        fig.add_annotation(
            text="No data provided. Provide either ic_data or spread_returns.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig = apply_standard_layout(
            fig,
            "Alpha Decay",
            None,
            xaxis_title="Forward Period (Days)",
            yaxis_title="Value"
        )
        return fig
    
    # Zero line (for IC case)
    if ic_data is not None:
        fig.add_hline(
            y=0.0,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            opacity=0.6
        )
    
    # Apply layout
    fig = apply_standard_layout(
        fig,
        title_text,
        None,
        xaxis_title="Forward Period (Days)",
        yaxis_title=yaxis_title
    )
    
    return fig


def plot_worst_periods(spread_returns, period, top_n=5):
    """
    Identifies and displays the worst N periods for spread returns.
    
    This helps identify when the strategy performed poorly and allows
    investigation of what happened during those periods (e.g., market crashes,
    regime changes).
    
    Parameters
    ----------
    spread_returns : pd.Series
        Spread returns time series for a specific period.
    period : str
        Forward return period (e.g., '20D')
    top_n : int
        Number of worst periods to display (default: 5)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with worst periods, sorted by return (worst first).
        Columns: Date, Return (bps), Cumulative Return
    """
    if spread_returns.empty or spread_returns.isnull().all():
        return pd.DataFrame(columns=['Date', 'Return (bps)', 'Cumulative Return'])
    
    # Calculate cumulative returns
    cum_returns = perf.cumulative_returns(spread_returns)
    
    # Calculate daily returns (rate of change of cumulative returns)
    daily_returns = spread_returns
    
    # Find worst days (lowest returns)
    worst_days = daily_returns.nsmallest(top_n)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'Date': worst_days.index,
        'Return (bps)': worst_days.values * DECIMAL_TO_BPS,
        'Cumulative Return': cum_returns.loc[worst_days.index].values
    })
    
    # Sort by Return (worst first)
    result = result.sort_values('Return (bps)')
    result = result.reset_index(drop=True)
    
    return result


