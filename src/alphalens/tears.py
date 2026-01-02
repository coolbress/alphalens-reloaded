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

import warnings
import logging
from pathlib import Path
from typing import Dict, Optional, List, Union, Any
from functools import wraps
import numpy as np
import pandas as pd
from datetime import datetime

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Environment, FileSystemLoader

from . import plotting
from . import performance as perf
from . import utils

logger = logging.getLogger("alphalens")

# ---------------------------------------
# Tear Sheet Result Cache (LRU)
# ---------------------------------------
# Cache tear sheet function results for reuse when creating full tear sheet
_TEAR_SHEET_CACHE_SIZE = 16  # LRU cache size (minimize memory burden)

# Global cache dictionary (LRU implementation)
_tear_sheet_cache = {}
_tear_sheet_cache_order = []  # Track LRU order


def _get_tear_sheet_cache_key(factor_data, func_name, **kwargs):
    """
    Generate cache key for tear sheet function.
    
    Parameters
    ----------
    factor_data : pd.DataFrame
        Factor data
    func_name : str
        Tear sheet function name
    **kwargs : dict
        Function keyword arguments
    
    Returns
    -------
    str
        Cache key
    """
    # Convert factor_data to hashable key
    factor_key = utils.make_factor_data_hashable(factor_data, **kwargs)
    # Combine function name and arguments to create unique key
    kwargs_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return f"{func_name}_{factor_key}_{kwargs_str}"


def _get_cached_tear_sheet(cache_key):
    """
    Get tear sheet result from cache (with LRU update).
    
    Parameters
    ----------
    cache_key : str
        Cache key
    
    Returns
    -------
    result : dict or None
        Cached result (None if not found)
    """
    if cache_key in _tear_sheet_cache:
        # LRU: move used item to the end
        _tear_sheet_cache_order.remove(cache_key)
        _tear_sheet_cache_order.append(cache_key)
        return _tear_sheet_cache[cache_key]
    return None


def _set_cached_tear_sheet(cache_key, result):
    """
    Store tear sheet result in cache (LRU implementation).
    
    Parameters
    ----------
    cache_key : str
        Cache key
    result : dict
        Result to store
    """
    # Cache size limit (LRU implementation)
    if len(_tear_sheet_cache) >= _TEAR_SHEET_CACHE_SIZE:
        # Remove oldest item
        oldest_key = _tear_sheet_cache_order.pop(0)
        del _tear_sheet_cache[oldest_key]
    
    _tear_sheet_cache[cache_key] = result
    _tear_sheet_cache_order.append(cache_key)


# ---------------------------------------
# Jinja2 Template Environment (Singleton)
# ---------------------------------------
# Global variable for Jinja2 environment (cached for performance)
_JINJA_ENV = None


def _get_jinja_env():
    """
    Get or create Jinja2 environment using singleton pattern.
    
    This caches the template environment to avoid reloading the template
    file from disk on every HTML generation call, improving performance
    when generating multiple reports.
    
    Returns
    -------
    jinja2.Environment
        Cached Jinja2 environment instance
    """
    global _JINJA_ENV
    if _JINJA_ENV is None:
        base_dir = Path(__file__).parent
        template_dir = base_dir  # Template is in the same directory as tears.py
        template_path = template_dir / 'base_template.html'
        
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template file not found at '{template_path}'. "
                "Please ensure base_template.html exists in the alphalens package directory."
            )
        
        try:
            # Convert pathlib Path to string (FileSystemLoader requires string path)
            _JINJA_ENV = Environment(loader=FileSystemLoader(str(template_dir)))
        except Exception as e:
            raise RuntimeError(
                f"Error loading Jinja2 template: {e}. "
                "Please ensure jinja2 is installed and the template file is accessible."
            ) from e
    
    return _JINJA_ENV


# ---------------------------------------
# User Functions 
# ---------------------------------------

@plotting.customize
def create_summary_tear_sheet(factor_data, long_short=True, group_neutral=False, display_output=True):
    """
    Creates a small summary tear sheet with returns, information, and turnover
    analysis.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned across the factor universe.
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
    display_output : bool, default True
        If True, immediately display plots and tables.
        If False, return all figures and tables as a dictionary without displaying.

    Returns
    -------
    dict or None
        If display_output=False, returns a dictionary containing:
        - 'figures': dict of plotly figure objects
        - 'table': pandas DataFrame (combined summary table)
        If display_output=True, returns None (displays output immediately).
    """

    # Returns Analysis
    mean_quant_ret, _ = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(
        factor_data, demeaned=long_short, group_adjust=group_neutral
    )

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    plotting.plot_quantile_statistics_table(factor_data)

    # Track A: Pre-calculate additional metrics to include in table
    periods_str = utils.get_forward_returns_columns(factor_data.columns)
    track_a_metrics = {}
    mono_scores = perf.quantile_monotonicity_score(mean_quant_rateret)

    for period_str in periods_str:
        mdd = np.nan
        score_val = np.nan
        calmar_ratio = np.nan
        cagr = np.nan
        try:
            mdd, mdd_date, _ = perf.cumulative_spread_drawdown(
                factor_data, period_str, long_short, group_neutral
            )
        except Exception:
            pass

        try:
            score_val = mono_scores.get(period_str, np.nan)
        except Exception:
            pass

        try:
            calmar_ratio, cagr, mdd_calmar = perf.spread_calmar_ratio(
                factor_data, period_str, long_short, group_neutral
            )
        except Exception:
            pass

        track_a_metrics[period_str] = {
            "Cumulative Spread MDD": mdd,
            "Monotonicity Score (Spearman)": score_val,
            "Spread Calmar Ratio": calmar_ratio,
            "Spread CAGR": cagr,
        }

    # Returns Analysis table
    returns_table = plotting.plot_returns_table(
        alpha_beta, mean_quant_rateret, mean_ret_spread_quant, track_a_metrics, return_df=True
    )
    if display_output:
        plotting.plot_returns_table(
            alpha_beta, mean_quant_rateret, mean_ret_spread_quant, track_a_metrics
        )

    # Quantile Returns Bar Chart (plotly)
    fig_quantile_returns = plotting.plot_quantile_returns_bar(
        mean_quant_rateret,
        by_group=False,
        ylim_percentiles=None,
    )
    if display_output:
        fig_quantile_returns.show()

    # Top Minus Bottom Quantile Mean Return (integrated with plotly)
    fig_spread_ts = plotting.plot_mean_quantile_returns_spread_time_series(
        mean_ret_spread_quant,
        std_err=std_spread_quant,
        bandwidth=0.5,
    )
    if display_output:
        fig_spread_ts.show()

    # Information Analysis
    ic = perf.factor_information_coefficient(factor_data)
    ic_table = plotting.plot_information_table(ic, return_df=True)
    if display_output:
        plotting.plot_information_table(ic)

    # Turnover analysis
    quantile_factor = factor_data["factor_quantile"]
    periods_turnover = utils.get_forward_returns_columns(factor_data.columns, require_exact_day_multiple=True).to_numpy()
    turnover_periods_int = utils.timedelta_strings_to_integers(periods_turnover)

    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for p in turnover_periods_int
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in turnover_periods_int
        ],
        axis=1,
    )

    # Track A: Calculate additional metrics (Autocorr + Breakeven Transaction Cost)
    track_a_metrics = {}
    for period in turnover_periods_int:
        # Convert period to standard format (using common function)
        period_str = utils.format_period(period)
        try:
            breakeven_cost, spread_bps, avg_turnover = perf.breakeven_transaction_cost(
                factor_data, period_str, long_short=True, group_neutral=False
            )
        except Exception:
            breakeven_cost = np.nan
            spread_bps = np.nan
            avg_turnover = np.nan

        track_a_metrics[period_str] = {
            "Breakeven Transaction Cost (bps)": breakeven_cost,
            "Avg Turnover (%)": avg_turnover * 100 if pd.notnull(avg_turnover) else np.nan,
            "Spread (bps)": spread_bps,
        }

    # Turnover table (now returns combined table)
    combined_turnover_table, _ = plotting.plot_turnover_table(
        autocorrelation, quantile_turnover, track_a_metrics, return_df=True
    )
    if display_output:
        plotting.plot_turnover_table(
            autocorrelation, quantile_turnover, track_a_metrics
        )

    # Turnover graph (plotly)
    valid_turnover = {p: quantile_turnover[p] for p in turnover_periods_int 
                      if not quantile_turnover[p].isnull().all().all()}
    fig_turnover = None
    if valid_turnover:
        fig_turnover = plotting.plot_top_bottom_quantile_turnover(
            quantile_turnover[list(valid_turnover.keys())[0]], 
            period=list(valid_turnover.keys())[0],
            quantile_turnover_dict=valid_turnover
        )
        if display_output:
            fig_turnover.show()
    
    # Factor Rank Autocorrelation graph (plotly)
    valid_autocorr = {p: autocorrelation[p] for p in turnover_periods_int 
                      if p in autocorrelation.columns and not autocorrelation[p].isnull().all()}
    fig_autocorr = None
    if valid_autocorr:
        # Convert autocorrelation DataFrame to dict (period: Series)
        autocorr_dict = {p: autocorrelation[p] for p in valid_autocorr.keys()}
        fig_autocorr = plotting.plot_factor_rank_auto_correlation(
            autocorrelation[list(valid_autocorr.keys())[0]] if len(valid_autocorr) > 0 else None,
            period=list(valid_autocorr.keys())[0] if len(valid_autocorr) > 0 else 1,
            factor_autocorrelation_dict=autocorr_dict
        )
        if display_output:
            fig_autocorr.show()
    
    # Return value composition
    if not display_output:
        # Combine multiple tables into one (since this is a summary)
        # Combine main tables into a single DataFrame
        combined_table = pd.concat([
            returns_table.T,
            ic_table.T,
            combined_turnover_table.T
        ], axis=0)
        
        figures_dict = {
            'quantile_returns': fig_quantile_returns,
            'spread_time_series': fig_spread_ts,
        }
        # Add only if turnover exists (remove None)
        if fig_turnover is not None:
            figures_dict['turnover'] = fig_turnover
        # Add only if autocorrelation exists (remove None)
        if 'fig_autocorr' in locals() and fig_autocorr is not None:
            figures_dict['autocorrelation'] = fig_autocorr
        
        return {
            'figures': figures_dict,
            'table': combined_table
        }


@plotting.customize
def create_returns_tear_sheet(
    factor_data, long_short=True, group_neutral=False, by_group=False, display_output=True
):
    """
    Creates a tear sheet for returns analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned across the factor universe.
        Additionally factor values will be demeaned across the factor universe
        when factor weighting the portfolio for cumulative returns plots
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
        Additionally each group will weight the same in cumulative returns
        plots
    by_group : bool
        If True, display graphs separately for each group.
    display_output : bool
        If True, displays plots and tables immediately. If False, returns results as dict.
    """
    # Generate cache key
    cache_key = _get_tear_sheet_cache_key(
        factor_data, 
        'create_returns_tear_sheet',
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=by_group
    )
    
    # Check cache (always use cache)
    cached_result = _get_cached_tear_sheet(cache_key)
    if cached_result is not None:
        logger.debug(f"Cache hit for create_returns_tear_sheet: {cache_key[:20]}...")
        # Use cached result if available
        if display_output:
            # Display cached result when display_output=True
            if 'table' in cached_result and cached_result['table'] is not None:
                utils.print_table(cached_result['table'])
            if 'worst_periods' in cached_result and cached_result['worst_periods'] is not None:
                utils.print_table(cached_result['worst_periods'])
            if 'figures' in cached_result:
                for fig_name, fig in cached_result['figures'].items():
                    if fig is not None:
                        fig.show()
            return None
        else:
            return cached_result

    factor_returns = perf.factor_returns(factor_data, long_short, group_neutral)

    mean_quant_ret, _ = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(
        factor_data, factor_returns, long_short, group_neutral
    )

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    # Track A: Pre-calculate additional metrics to include in table
    periods_str = utils.get_forward_returns_columns(factor_data.columns)
    track_a_metrics = {}
    mono_scores = perf.quantile_monotonicity_score(mean_quant_rateret)

    for period_str in periods_str:
        mdd = np.nan
        score_val = np.nan
        calmar_ratio = np.nan
        cagr = np.nan
        try:
            mdd, mdd_date, _ = perf.cumulative_spread_drawdown(
                factor_data, period_str, long_short, group_neutral
            )
        except Exception:
            pass

        try:
            score_val = mono_scores.get(period_str, np.nan)
        except Exception:
            pass

        try:
            calmar_ratio, cagr, mdd_calmar = perf.spread_calmar_ratio(
                factor_data, period_str, long_short, group_neutral
            )
        except Exception:
            pass

        track_a_metrics[period_str] = {
            "Cumulative Spread MDD": mdd,
            "Monotonicity Score (Spearman)": score_val,
            "Spread Calmar Ratio": calmar_ratio,
            "Spread CAGR": cagr,
        }

    # Returns Analysis table
    returns_table = plotting.plot_returns_table(
        alpha_beta, mean_quant_rateret, mean_ret_spread_quant, track_a_metrics, return_df=True
    )
    if display_output:
        plotting.plot_returns_table(
            alpha_beta, mean_quant_rateret, mean_ret_spread_quant, track_a_metrics
        )

    # Quantile Returns Bar Chart (plotly)
    fig_quantile_returns = plotting.plot_quantile_returns_bar(
        mean_quant_rateret,
        by_group=False,
        ylim_percentiles=None,
    )
    if display_output:
        fig_quantile_returns.show()

    # Track A: Violin plot removed (Bar chart is sufficient)
    # plotting.plot_quantile_returns_violin(
    #     mean_quant_ret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row()
    # )
    
    # Compute cumulative returns from daily simple returns, if '1D'
    # returns are provided.
    fig_cumulative = None
    if "1D" in factor_returns:
        # Convert to plotly (single period processing since only 1D is supported)
        # Pass as DataFrame with MultiIndex (factor_quantile, date) to match expected format
        fig_cumulative = plotting.plot_cumulative_returns_by_quantile(
            mean_quant_ret_bydate[["1D"]], period="1D"
        )
        if display_output:
            fig_cumulative.show()

    # Top Minus Bottom Quantile Mean Return (integrated with plotly)
    fig_spread_ts = plotting.plot_mean_quantile_returns_spread_time_series(
        mean_ret_spread_quant,
        std_err=std_spread_quant,
        bandwidth=0.5,
    )
    if display_output:
        fig_spread_ts.show()
    
    # 1. Underwater Plot (drawdown visualization) - toggle support for all periods
    fig_underwater = None
    if len(periods_str) > 0:
        try:
            # Collect drawdown data for all periods
            drawdown_dict = {}
            for period_str in periods_str:
                try:
                    mdd, _, drawdown_series = perf.cumulative_spread_drawdown(
                        factor_data, period_str, long_short, group_neutral
                    )
                    drawdown_dict[period_str] = (drawdown_series, mdd, _)
                except Exception:
                    continue  # Skip if individual period fails
            
            if drawdown_dict:
                fig_underwater = plotting.plot_underwater_drawdown(
                    drawdown_series=None,  # Not needed when using drawdown_dict
                    period=None,  # Not needed when using drawdown_dict
                    drawdown_dict=drawdown_dict
                )
                if display_output:
                    fig_underwater.show()
        except Exception as e:
            logger.warning(f"Failed to create underwater plot: {e}")
    
    # Note: Long/Short Contribution (Spread) is now included in cumulative returns by quantile plot
    # No separate plot needed since Q5 (Long) and Q1 (Short) are already shown,
    # and Spread (Q5 - Q1) is added to the same graph
    
    # 3. Worst 5 Periods Analysis - for first period
    worst_periods_table = None
    if len(periods_str) > 0:
        try:
            first_period = periods_str[0]
            # Calculate spread returns
            spread_returns = mean_ret_spread_quant[first_period] if first_period in mean_ret_spread_quant.columns else None
            if spread_returns is not None:
                worst_periods_table = plotting.plot_worst_periods(spread_returns, first_period, top_n=5)
                if display_output:
                    print(f"\nWorst 5 Periods Analysis ({first_period}):")
                    print("=" * 80)
                    utils.print_table(worst_periods_table)
        except Exception as e:
            logger.warning(f"Failed to create worst periods analysis: {e}")
    
    # matplotlib axes no longer needed (replaced with plotly)
    # ax_mean_quantile_returns_spread_ts = None  # Removed: unnecessary variable

    # by_group processing (additional processing when by_group=True)
    fig_quantile_returns_by_group = None
    if by_group:
        (
            mean_return_quantile_group,
            _,
        ) = perf.mean_return_by_quantile(
            factor_data,
            by_date=False,
            by_group=True,
            demeaned=long_short,
            group_adjust=group_neutral,
        )

        mean_quant_rateret_group = mean_return_quantile_group.apply(
            utils.rate_of_return,
            axis=0,
            base_period=mean_return_quantile_group.columns[0],
        )

        # Convert to Plotly (plot_quantile_returns_bar returns Plotly figure when by_group=True)
        fig_quantile_returns_by_group = plotting.plot_quantile_returns_bar(
            mean_quant_rateret_group,
            by_group=True,
            ylim_percentiles=(5, 95),
        )

    # Always create result as dict (for cache storage and reuse)
    figures_dict = {
        'quantile_returns': fig_quantile_returns,
        'spread_time_series': fig_spread_ts,
    }
    # Add only if cumulative_returns exists (remove None)
    if fig_cumulative is not None:
        figures_dict['cumulative_returns'] = fig_cumulative
    # Add new analyses
    if fig_underwater is not None:
        figures_dict['underwater_drawdown'] = fig_underwater
    # Long/Short Contribution (Spread) is now included in cumulative returns by quantile plot
    # Add by_group figure if exists
    if fig_quantile_returns_by_group is not None:
        figures_dict['quantile_returns_by_group'] = fig_quantile_returns_by_group
    
    return_dict = {
        'figures': figures_dict,
        'table': returns_table
    }
    # Add if worst_periods_table exists
    if worst_periods_table is not None:
        return_dict['worst_periods'] = worst_periods_table
    
    # Store in cache (always)
    _set_cached_tear_sheet(cache_key, return_dict)
    
    # Return or display based on display_output
    if display_output:
        # Display to screen
        if 'table' in return_dict and return_dict['table'] is not None:
            utils.print_table(return_dict['table'])
        if 'worst_periods' in return_dict and return_dict['worst_periods'] is not None:
            utils.print_table(return_dict['worst_periods'])
        if 'figures' in return_dict:
            for fig_name, fig in return_dict['figures'].items():
                if fig is not None:
                    fig.show()
        return None
    else:
        return return_dict


@plotting.customize
def create_information_tear_sheet(factor_data, group_neutral=False, by_group=False, display_output=True):
    """
    Creates a tear sheet for information analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    group_neutral : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, display graphs separately for each group.
    display_output : bool, default True
        If True, displays tables and plots immediately (default behavior).
        If False, returns a dict containing tables and figure objects for later display.
    
    Returns
    -------
    dict or None
        If display_output=False, returns a dict with keys:
        - 'figures': dict of plotly figure objects (keys: 'ic_ts', 'monthly_ic')
        - 'table': DataFrame containing IC summary table
        If display_output=True, returns None (displays output immediately).
    """
    # Generate cache key
    cache_key = _get_tear_sheet_cache_key(
        factor_data,
        'create_information_tear_sheet',
        group_neutral=group_neutral,
        by_group=by_group
    )
    
    # Check cache (always use cache)
    cached_result = _get_cached_tear_sheet(cache_key)
    if cached_result is not None:
        logger.debug(f"Cache hit for create_information_tear_sheet: {cache_key[:20]}...")
        # Use cached result if available
        if display_output:
            # Display cached result when display_output=True
            if 'table' in cached_result and cached_result['table'] is not None:
                utils.print_table(cached_result['table'])
            if 'figures' in cached_result:
                for fig_name, fig in cached_result['figures'].items():
                    if fig is not None:
                        fig.show()
            return None
        else:
            return cached_result

    ic = perf.factor_information_coefficient(factor_data, group_neutral)

    # Track A: Include Yearly Win Rate in summary table
    win_rate, yearly_ic = perf.yearly_win_rate(
        factor_data, group_adjust=group_neutral
    )
    
    # Create table
    ic_summary_table = plotting.plot_information_table(ic, yearly_win_rate=win_rate, return_df=True)
    
    # Track A: IC Time Series with t-stat threshold (integrated with plotly)
    fig_ic_ts = plotting.plot_ic_ts(ic, threshold=3.0)
    
    # Track A: Additional Information Analysis metrics already included in table, so output removed
    
    fig_monthly_ic = None
    
    if not by_group:
        mean_monthly_ic = perf.mean_information_coefficient(
            factor_data,
            group_adjust=group_neutral,
            by_group=False,
            by_time="ME",  # Changed 'M' → 'ME' (remove FutureWarning)
        )
        # Convert to plotly (includes period toggle)
        fig_monthly_ic = plotting.plot_monthly_ic_heatmap(mean_monthly_ic)

    fig_ic_by_group = None
    if by_group:
        # Convert to Plotly
        mean_group_ic = perf.mean_information_coefficient(
            factor_data, group_adjust=group_neutral, by_group=True
        )
        fig_ic_by_group = plotting.plot_ic_by_group(mean_group_ic)
        if display_output:
            fig_ic_by_group.show()

    # Alpha Decay Plot (IC Decay)
    fig_alpha_decay = None
    try:
        periods_str = utils.get_forward_returns_columns(factor_data.columns)
        fig_alpha_decay = plotting.plot_alpha_decay(ic_data=ic, periods=periods_str)
        if display_output:
            fig_alpha_decay.show()
    except Exception as e:
        logger.warning(f"Failed to create alpha decay plot: {e}")

    # Always create result as dict (for cache storage and reuse)
    figures_dict = {
        'ic_ts': fig_ic_ts,
    }
    # Add only if monthly_ic exists (remove None)
    if fig_monthly_ic is not None:
        figures_dict['monthly_ic'] = fig_monthly_ic
    # Add only if ic_by_group exists (remove None)
    if fig_ic_by_group is not None:
        figures_dict['ic_by_group'] = fig_ic_by_group
    # Add only if alpha_decay exists (remove None)
    if fig_alpha_decay is not None:
        figures_dict['alpha_decay'] = fig_alpha_decay
    
    result = {
        'figures': figures_dict,
        'table': ic_summary_table
    }
    
    # Store in cache (always)
    _set_cached_tear_sheet(cache_key, result)
    
    # Return or display based on display_output
    if display_output:
        # Display to screen
        plotting.plot_information_table(ic, yearly_win_rate=win_rate)
        if 'figures' in result:
            for fig_name, fig in result['figures'].items():
                if fig is not None:
                    fig.show()
        return None
    else:
        return result


@plotting.customize
def create_turnover_tear_sheet(factor_data, turnover_periods=None, display_output=True):
    """
    Creates a tear sheet for analyzing the turnover properties of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    turnover_periods : sequence[string], optional
        Periods to compute turnover analysis on. By default periods in
        'factor_data' are used but custom periods can provided instead. This
        can be useful when periods in 'factor_data' are not multiples of the
        frequency at which factor values are computed i.e. the periods
        are 2h and 4h and the factor is computed daily and so values like
        ['1D', '2D'] could be used instead
    display_output : bool, default True
        If True, immediately display plots and tables.
        If False, return all figures and tables as a dictionary without displaying.

    Returns
    -------
    dict or None
        If display_output=False, returns a dictionary containing:
        - 'figures': dict of plotly figure objects
        - 'table': pandas DataFrame
        If display_output=True, returns None (displays output immediately).
    """
    # turnover_periods를 해시 가능한 형태로 변환
    turnover_periods_hash = hash(tuple(turnover_periods)) if turnover_periods is not None else None
    
    # Generate cache key
    cache_key = _get_tear_sheet_cache_key(
        factor_data,
        'create_turnover_tear_sheet',
        turnover_periods=turnover_periods_hash
    )
    
    # Check cache (always use cache)
    cached_result = _get_cached_tear_sheet(cache_key)
    if cached_result is not None:
        logger.debug(f"Cache hit for create_turnover_tear_sheet: {cache_key[:20]}...")
        # Use cached result if available
        if display_output:
            # Display cached result when display_output=True
            if 'table' in cached_result and cached_result['table'] is not None:
                utils.print_table(cached_result['table'])
            if 'figures' in cached_result:
                for fig_name, fig in cached_result['figures'].items():
                    if fig is not None:
                        fig.show()
            return None
        else:
            return cached_result

    if turnover_periods is None:
        input_periods = utils.get_forward_returns_columns(
            factor_data.columns, require_exact_day_multiple=True
        ).to_numpy()
        turnover_periods = utils.timedelta_strings_to_integers(input_periods)
    else:
        turnover_periods = utils.timedelta_strings_to_integers(turnover_periods)

    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for p in turnover_periods
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in turnover_periods
        ],
        axis=1,
    )

    # Track A: Calculate additional metrics (Autocorr + Breakeven Transaction Cost) → for table merging
    track_a_metrics = {}
    for period in turnover_periods:
        # Convert period to standard format (using common function)
        period_str = utils.format_period(period)
        try:
            breakeven_cost, spread_bps, avg_turnover = perf.breakeven_transaction_cost(
                factor_data, period_str, long_short=True, group_neutral=False
            )
        except Exception:
            breakeven_cost = np.nan
            spread_bps = np.nan
            avg_turnover = np.nan

        track_a_metrics[period_str] = {
            # "Factor Rank Autocorrelation (Track A)" removed - duplicates "Mean Factor Rank Autocorrelation"
            "Breakeven Transaction Cost (bps)": breakeven_cost,
            "Avg Turnover (%)": avg_turnover * 100 if pd.notnull(avg_turnover) else np.nan,
            "Spread (bps)": spread_bps,
        }

    # Turnover table (now returns combined table)
    combined_turnover_table, _ = plotting.plot_turnover_table(
        autocorrelation, quantile_turnover, track_a_metrics, return_df=True
    )
    if display_output:
        plotting.plot_turnover_table(autocorrelation, quantile_turnover, track_a_metrics)

    # Turnover graph (plotly)
    valid_turnover = {p: quantile_turnover[p] for p in turnover_periods 
                      if not quantile_turnover[p].isnull().all().all()}
    fig_turnover = None
    if valid_turnover:
        fig_turnover = plotting.plot_top_bottom_quantile_turnover(
            quantile_turnover[list(valid_turnover.keys())[0]], 
            period=list(valid_turnover.keys())[0],
            quantile_turnover_dict=valid_turnover
        )
        if display_output:
            fig_turnover.show()
    
    # Factor Rank Autocorrelation graph (plotly)
    valid_autocorr = {p: autocorrelation[p] for p in turnover_periods 
                      if p in autocorrelation.columns and not autocorrelation[p].isnull().all()}
    fig_autocorr = None
    if valid_autocorr:
        # Convert autocorrelation DataFrame to dict (period: Series)
        autocorr_dict = {p: autocorrelation[p] for p in valid_autocorr.keys()}
        fig_autocorr = plotting.plot_factor_rank_auto_correlation(
            autocorrelation[list(valid_autocorr.keys())[0]] if len(valid_autocorr) > 0 else None,
            period=list(valid_autocorr.keys())[0] if len(valid_autocorr) > 0 else 1,
            factor_autocorrelation_dict=autocorr_dict
        )
        if display_output:
            fig_autocorr.show()
    
    # Always create result as dict (for cache storage and reuse)
    combined_table = combined_turnover_table.T
    
    figures_dict = {}
    if fig_turnover is not None:
        figures_dict['turnover'] = fig_turnover
    if fig_autocorr is not None:
        figures_dict['autocorrelation'] = fig_autocorr
    
    result = {
        'figures': figures_dict,
        'table': combined_table
    }
    
    # Store in cache (always)
    _set_cached_tear_sheet(cache_key, result)
    
    # Return or display based on display_output
    if display_output:
        # Display to screen (already displayed above, but for consistency)
        return None
    else:
        return result


@plotting.customize
def create_full_tear_sheet(
    factor_data, long_short=True, group_neutral=False, by_group=False, save_html=False, display_output=True
):
    """
    Creates a full tear sheet for analysis and evaluating single
    return predicting (alpha) factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
    group_neutral : bool
        Should this computation happen on a group neutral portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
        - See tears.create_information_tear_sheet for details on how this
        flag affects information analysis
    by_group : bool
        If True, display graphs separately for each group.
    save_html : bool, optional
        If True, saves all plots and tables to an HTML file with auto-generated filename.
        The filename will be generated using today's date and factor name.
        Default is False.
        Example: save_html=True
    display_output : bool, optional
        If True, displays plots and tables immediately. If False, returns results as dict.
        Default is True.
    """
    # Compute phase: perform calculations once and reuse results for both display and HTML saving
    # This prevents duplicate calculations when save_html=True and display_output=True
    # Check cached results first, compute if not available
    quantile_stats = plotting.plot_quantile_statistics_table(factor_data, return_df=True)
    
    # Returns tear sheet: check cache, compute if not available
    returns_cache_key = _get_tear_sheet_cache_key(
        factor_data,
        'create_returns_tear_sheet',
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=by_group
    )
    returns_result = _get_cached_tear_sheet(returns_cache_key)
    if returns_result is None:
        logger.debug("Cache miss for returns_tear_sheet, computing...")
        returns_result = create_returns_tear_sheet(
            factor_data, long_short, group_neutral, by_group, display_output=False
        )
    else:
        logger.debug("Cache hit for returns_tear_sheet, reusing cached result")
    
    # Information tear sheet: check cache, compute if not available
    information_cache_key = _get_tear_sheet_cache_key(
        factor_data,
        'create_information_tear_sheet',
        group_neutral=group_neutral,
        by_group=by_group
    )
    information_result = _get_cached_tear_sheet(information_cache_key)
    if information_result is None:
        logger.debug("Cache miss for information_tear_sheet, computing...")
        information_result = create_information_tear_sheet(
            factor_data, group_neutral, by_group, display_output=False
        )
    else:
        logger.debug("Cache hit for information_tear_sheet, reusing cached result")
    
    # Turnover tear sheet: check cache, compute if not available
    turnover_cache_key = _get_tear_sheet_cache_key(
        factor_data,
        'create_turnover_tear_sheet',
        turnover_periods=None
    )
    turnover_result = _get_cached_tear_sheet(turnover_cache_key)
    if turnover_result is None:
        logger.debug("Cache miss for turnover_tear_sheet, computing...")
        turnover_result = create_turnover_tear_sheet(
            factor_data, display_output=False
        )
    else:
        logger.debug("Cache hit for turnover_tear_sheet, reusing cached result")
    
    # View phase: screen output (when display_output=True)
    if display_output:
        _display_results(
            quantile_stats=quantile_stats,
            returns_result=returns_result,
            information_result=information_result,
            turnover_result=turnover_result,
            factor_data=factor_data
        )
    
    # View phase: HTML saving (when save_html=True)
    if save_html:
        factor_name = 'Factor'
        if 'factor' in factor_data.columns:
            factor_col = factor_data['factor']
            if hasattr(factor_col, 'name') and factor_col.name:
                factor_name = factor_col.name
        
        today = datetime.now().strftime('%Y%m%d')
        safe_factor_name = "".join(c for c in factor_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_factor_name = safe_factor_name.replace(' ', '_')
        html_filename = f"Alphalens_Full_TearSheet_{safe_factor_name}_{today}.html"
        html_content = _render_html(
            quantile_stats=quantile_stats,
            returns_result=returns_result,
            information_result=information_result,
            turnover_result=turnover_result,
            factor_name=factor_name
        )
        
        output_path = Path(html_filename)
        output_path.write_text(html_content, encoding='utf-8')
        
        return
    
    # Return results when display_output=False and save_html=False
    if not display_output:
        # Collect all tables and figures (order guaranteed: Quantile -> Returns -> Information -> Turnover)
        from collections import OrderedDict
        all_tables = OrderedDict()
        all_figures = OrderedDict()
        
        # 1. Quantile Statistics
        if quantile_stats is not None and not quantile_stats.empty:
            all_tables['1. Quantile Statistics'] = quantile_stats
        
        # 2. Returns
        if returns_result and 'table' in returns_result and returns_result['table'] is not None:
            all_tables['2. Returns'] = returns_result['table']
        if returns_result and 'worst_periods' in returns_result and returns_result['worst_periods'] is not None:
            all_tables['2.1 Worst 5 Periods'] = returns_result['worst_periods']
        if returns_result and 'figures' in returns_result:
            for fig_name, fig in returns_result['figures'].items():
                if fig is not None:
                    # Sort Returns figures as 2
                    all_figures[f'2_returns_{fig_name}'] = fig
        
        # 3. Information
        if information_result and 'table' in information_result and information_result['table'] is not None:
            all_tables['3. Information'] = information_result['table']
        if information_result and 'figures' in information_result:
            for fig_name, fig in information_result['figures'].items():
                if fig is not None:
                    # Sort Information figures as 3
                    all_figures[f'3_information_{fig_name}'] = fig
        
        # 4. Turnover
        if turnover_result and 'table' in turnover_result and turnover_result['table'] is not None:
            all_tables['4. Turnover'] = turnover_result['table']
        if turnover_result and 'figures' in turnover_result:
            for fig_name, fig in turnover_result['figures'].items():
                if fig is not None:
                    # Sort Turnover figures as 4
                    all_figures[f'4_turnover_{fig_name}'] = fig
        
        return {
            'tables': all_tables,
            'figures': all_figures
        }
        
@plotting.customize
def create_event_returns_tear_sheet(
    factor_data,
    returns,
    avgretplot=(5, 15),
    long_short=True,
    group_neutral=False,
    std_bar=True,
    by_group=False,
    display_output=True,
):
    """
    Creates a tear sheet to view the average cumulative returns for a
    factor within a window (pre and post event).

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, the factor
        quantile/bin that factor value belongs to and (optionally) the group
        the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    returns : pd.DataFrame
        A DataFrame indexed by date with assets in the columns containing daily
        returns.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after)
        If not None, plot quantile average cumulative returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so then
        factor returns will be demeaned across the factor universe
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
    std_bar : boolean, optional
        Show plots with standard deviation bars, one for each quantile
    by_group : bool
        If True, display graphs separately for each group.
    """

    before, after = avgretplot

    avg_cumulative_returns = perf.average_cumulative_return_by_quantile(
        factor_data,
        returns,
        periods_before=before,
        periods_after=after,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    # Conversion to Plotly completed
    fig_avg_cumulative = plotting.plot_quantile_average_cumulative_return(
        avg_cumulative_returns,
        by_quantile=False,
        std_bar=False,
    )
    if display_output:
        fig_avg_cumulative.show()
    
    if std_bar:
        fig_avg_cumulative_by_q = plotting.plot_quantile_average_cumulative_return(
            avg_cumulative_returns,
            by_quantile=True,
            std_bar=True,
        )
        if display_output:
            fig_avg_cumulative_by_q.show()

    if by_group:
        # by_group feature: create subplot for each group
        groups = factor_data["group"].unique()

        avg_cumret_by_group = perf.average_cumulative_return_by_quantile(
            factor_data,
            returns,
            periods_before=before,
            periods_after=after,
            demeaned=long_short,
            group_adjust=group_neutral,
            by_group=True,
        )

        for group, avg_cumret in avg_cumret_by_group.groupby(level="group"):
            avg_cumret.index = avg_cumret.index.droplevel("group")
            fig_group = plotting.plot_quantile_average_cumulative_return(
                avg_cumret,
                by_quantile=False,
                std_bar=False,
                title=str(group),
            )
            if display_output:
                fig_group.show()


@plotting.customize
def create_event_study_tear_sheet(
    factor_data, returns, avgretplot=(5, 15), rate_of_ret=True, n_bars=50
):
    """
    Creates an event study tear sheet for analysis of a specific event.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single event, forward returns for each
        period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
    returns : pd.DataFrame, required only if 'avgretplot' is provided
        A DataFrame indexed by date with assets in the columns containing daily
        returns.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after), optional
        If not None, plot event style average cumulative returns within a
        window (pre and post event).
    rate_of_ret : bool, optional
        Display rate of return instead of simple return in 'Mean Period Wise
        Return By Factor Quantile' and 'Period Wise Return By Factor Quantile'
        plots
    n_bars : int, optional
        Number of bars in event distribution plot
    """

    long_short = False

    plotting.plot_quantile_statistics_table(factor_data)

    # Conversion to Plotly completed
    fig_events_dist = plotting.plot_events_distribution(
        events=factor_data["factor"], num_bars=n_bars
    )
    fig_events_dist.show()

    if returns is not None and avgretplot is not None:
        create_event_returns_tear_sheet(
            factor_data=factor_data,
            returns=returns,
            avgretplot=avgretplot,
            long_short=long_short,
            group_neutral=False,
            std_bar=True,
            by_group=False,
        )

    factor_returns = perf.factor_returns(factor_data, demeaned=False, equal_weight=True)

    mean_quant_ret, _ = perf.mean_return_by_quantile(
        factor_data, by_group=False, demeaned=long_short
    )
    if rate_of_ret:
        mean_quant_ret = mean_quant_ret.apply(
            utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
        )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data, by_date=True, by_group=False, demeaned=long_short
    )
    if rate_of_ret:
        mean_quant_ret_bydate = mean_quant_ret_bydate.apply(
            utils.rate_of_return,
            axis=0,
            base_period=mean_quant_ret_bydate.columns[0],
        )

    # Conversion to Plotly completed
    fig_quantile_returns = plotting.plot_quantile_returns_bar(
        mean_quant_ret, by_group=False, ylim_percentiles=None
    )
    fig_quantile_returns.show()
    
    # Conversion to Plotly completed
    fig_violin = plotting.plot_quantile_returns_violin(
        mean_quant_ret_bydate, ylim_percentiles=(1, 99)
    )
    fig_violin.show()

    # Output Plotly figure
    if fig_quantile_returns is not None:
        fig_quantile_returns.show()

# ---------------------------------------
# Helper Functions
# ---------------------------------------

def _rebuild_figure_without_compression(fig):
    """
    Extract data directly from original figure object and create new figure without compression.
    
    [Fix] Now correctly handles datetime64[ns] arrays by converting them to strings
    instead of integers, preventing the 'exponential x-axis' issue.
    
    ⚠️ Why is rebuild necessary? (The Safety Net)
    
    **Problem: HTML rendering failure due to Plotly's bdata compression**
    
    Plotly automatically performs Binary compression (bdata) to efficiently store large data.
    This compressed data may not be properly decoded in certain environments (local HTML files, some browsers),
    causing charts to break.
    
    **VertexLens vs Our Code Differences:**
    
    1. **VertexLens Pattern** (no compression issues):
       ```
       figure creation → immediate cache storage → immediate HTML conversion
       ```
       - figure object is converted to HTML in "fresh" state
       - pio.to_html() internally calls to_dict() but no compression occurs
       - Reason: figure object is converted before being stored in dictionary
    
    2. **Our Code Pattern** (compression issues may occur):
       ```
       figure creation → dictionary storage → pass through multiple functions → HTML conversion
       ```
       - During the process of storing figure object in dictionary and passing through multiple functions,
         to_dict() may be called internally
       - Plotly may automatically apply bdata compression during this process
       - Compressed data: `"y": {"dtype": "f8", "bdata": "xN+RPeVv..."}`
       - Result: Charts break in HTML (Bars appear as thin lines, Heatmap appears empty)
    
    **Solution (Integer Mapping Technique):**
    
    This function disassembles the figure object, converts it to pure Python List, then reassembles it.
    
    1. **Data Extraction**: Extract data directly from original figure's traces (numpy.ndarray state)
    2. **List Conversion**: Convert numpy.ndarray to list to prevent compression
    3. **Reassembly**: Create new figure with converted data
    4. **Attribute Preservation**: Preserve all visualization attributes like text, texttemplate, textfont
       - ⚠️ Important: text field maintains already formatted string array from plot_monthly_ic_heatmap
       - Do not regenerate text from z (preserve formatting)
    
    **Efficiency Analysis:**
    
    - CPU computation cost: Yes (data disassembly and reassembly)
    - Data integrity: Very high (0% compression issues)
    - HTML rendering stability: 100% guaranteed
    
    This function serves as a "Safety Net" to ensure stability of HTML report generation.
    It is essential in structures where figures cannot be converted immediately like VertexLens.
    
    Parameters
    ----------
    fig : go.Figure
        Original Plotly Figure object
    
    Returns
    -------
    go.Figure
        Newly regenerated Figure object without compression
    """
    # 1. Check if subplot exists
    has_subplots = hasattr(fig, '_grid_ref') and fig._grid_ref is not None
    
    # Initialize subplot information
    n_rows = 1
    n_cols = 1
    
    if has_subplots:
        # Understand subplot structure
        try:
            grid_ref = fig._grid_ref
            n_rows = len(grid_ref) if grid_ref else 1
            n_cols = len(grid_ref[0]) if grid_ref and len(grid_ref) > 0 else 1
        except:
            # If _grid_ref doesn't exist, estimate from trace's row/col information
            max_row = 0
            max_col = 0
            for trace in fig.data:
                if hasattr(trace, 'row') and trace.row:
                    max_row = max(max_row, trace.row)
                if hasattr(trace, 'col') and trace.col:
                    max_col = max(max_col, trace.col)
            n_rows = max_row if max_row > 0 else 1
            n_cols = max_col if max_col > 0 else 1
        
        # Extract subplot_titles
        subplot_titles = []
        if hasattr(fig.layout, 'annotations') and fig.layout.annotations:
            # Extract subplot titles from annotations
            for ann in fig.layout.annotations:
                if hasattr(ann, 'text') and ann.text:
                    subplot_titles.append(ann.text)
        
        fig_clean = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles if subplot_titles else None,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
    else:
        fig_clean = go.Figure()
    
    # 2. Extract trace data and convert to List (core of compression prevention)
    for trace in fig.data:
        trace_dict = {}
        row = trace.row if hasattr(trace, 'row') else None
        col = trace.col if hasattr(trace, 'col') else None
        
        # Directly access trace's main fields to convert numpy array to list
        # Plotly trace objects are not dictionaries, so .items() cannot be used
        # Directly access required fields
        field_list = ['x', 'y', 'z', 'text', 'customdata', 'name', 'visible', 
                     'showlegend', 'legendgroup', 'mode', 'marker', 'line', 
                     'fill', 'fillcolor', 'hovertemplate', 'hoverinfo',
                     'colorscale', 'zmid', 'texttemplate', 'textfont', 
                     'colorbar', 'showscale', 'yaxis', 'xaxis', 'type']
        
        for field in field_list:
            if hasattr(trace, field):
                value = getattr(trace, field)
                
                # Skip if None or empty (except text field which is processed even if None)
                if value is None and field != 'text':
                    continue
                
                if isinstance(value, np.ndarray):
                    # [Critical Fix] If data is datetime, convert to string (ISO)
                    # This prevents dates from becoming huge integers (nanoseconds)
                    if np.issubdtype(value.dtype, np.datetime64):
                        trace_dict[field] = value.astype(str).tolist()
                    else:
                        trace_dict[field] = value.tolist()
                elif isinstance(value, pd.Index):
                    # Handle pandas Index (DatetimeIndex etc.)
                    if isinstance(value, pd.DatetimeIndex):
                        # Convert DatetimeIndex to string list (ISO format)
                        trace_dict[field] = value.astype(str).tolist()
                    else:
                        trace_dict[field] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    if field == 'text':
                        # text field: keep string arrays as is, convert NaN to None for numeric arrays
                        if len(value) > 0:
                            if isinstance(value[0], (list, tuple)):
                                # 2D array
                                if len(value[0]) > 0:
                                    # Check if first element is string
                                    if isinstance(value[0][0], str):
                                        # Keep string arrays as is (NaN check unnecessary)
                                        trace_dict[field] = [list(row_data) for row_data in value]
                                    else:
                                        # Convert NaN to None for numeric arrays
                                        trace_dict[field] = [[v if not (isinstance(v, float) and np.isnan(v)) 
                                                              else None for v in row_data] for row_data in value]
                                else:
                                    trace_dict[field] = [list(row_data) for row_data in value]
                            else:
                                # 1D array
                                if isinstance(value[0], str):
                                    # Keep string arrays as is
                                    trace_dict[field] = list(value)
                                else:
                                    # Convert NaN to None for numeric arrays
                                    trace_dict[field] = [v if not (isinstance(v, float) and np.isnan(v)) 
                                                        else None for v in value]
                        else:
                            trace_dict[field] = list(value)
                    else:
                        trace_dict[field] = list(value)
                else:
                    trace_dict[field] = value
        
        # 3. Create appropriate trace object based on trace type
        trace_type = trace_dict.pop('type', 'scatter')
        
        if trace_type == 'heatmap':
            # Regenerate Heatmap (text overwrite logic removed)
            heatmap_kwargs = {k: v for k, v in trace_dict.items() if k != 'type'}
            
            # ⚠️ Important: Do not regenerate text from z (preserve already formatted string array)
            # texttemplate safety mechanism
            if 'text' in heatmap_kwargs and heatmap_kwargs['text'] is not None:
                if 'texttemplate' not in heatmap_kwargs:
                    heatmap_kwargs['texttemplate'] = '%{text}'
                if 'textfont' not in heatmap_kwargs:
                    heatmap_kwargs['textfont'] = {'size': 9}
            
            # Add to subplot position if subplot position information exists
            if row is not None and col is not None:
                fig_clean.add_trace(go.Heatmap(**heatmap_kwargs), row=row, col=col)
            else:
                fig_clean.add_trace(go.Heatmap(**heatmap_kwargs))
        else:
            # Handle other trace types
            trace_cls_map = {
                'scatter': go.Scatter,
                'bar': go.Bar,
                'histogram': go.Histogram,
                'box': go.Box,
            }
            trace_cls = trace_cls_map.get(trace_type, go.Scatter)
            clean_kwargs = {k: v for k, v in trace_dict.items() if k != 'type'}
            
            if row is not None and col is not None:
                fig_clean.add_trace(trace_cls(**clean_kwargs), row=row, col=col)
            else:
                fig_clean.add_trace(trace_cls(**clean_kwargs))
    
    # 4. Copy layout and adjust size
    layout_dict = fig.layout.to_plotly_json()
    
    # CSS container size limit: #left is 65% of 1800px = 1170px, excluding 40px padding ≈ 1130px
    MAX_CONTAINER_WIDTH = 1130
    
    # Dynamically calculate size based on data
    calculated_width = None
    calculated_height = None
    
    # Calculate based on data size if Heatmap
    has_heatmap = any(hasattr(trace, 'type') and trace.type == 'heatmap' for trace in fig.data)
    
    if has_heatmap:
        # Check x, y data size from Heatmap trace
        max_x_len = 0
        max_y_len = 0
        for trace in fig.data:
            if hasattr(trace, 'type') and trace.type == 'heatmap':
                x_data = getattr(trace, 'x', None)
                y_data = getattr(trace, 'y', None)
                if x_data is not None:
                    if isinstance(x_data, (list, np.ndarray)):
                        max_x_len = max(max_x_len, len(x_data))
                    elif hasattr(x_data, '__len__'):
                        max_x_len = max(max_x_len, len(x_data))
                if y_data is not None:
                    if isinstance(y_data, (list, np.ndarray)):
                        max_y_len = max(max_y_len, len(y_data))
                    elif hasattr(y_data, '__len__'):
                        max_y_len = max(max_y_len, len(y_data))
        
        # Calculate Heatmap size: minimum 35px per cell, including margins
        if max_x_len > 0 and max_y_len > 0:
            if has_subplots and n_cols > 1:
                # Minimum width needed per subplot: 35px per month + 120px margin
                subplot_width = max(35 * max_x_len + 120, 350)  # Minimum 350px
                calculated_width = subplot_width * n_cols  # Sufficient size without limit
                calculated_height = max(35 * max_y_len + 150, 350) * n_rows
            else:
                calculated_width = max(35 * max_x_len + 120, 350)
                calculated_height = max(35 * max_y_len + 150, 350)
    
    # Set height
    if calculated_height is not None:
        layout_dict['height'] = calculated_height
    elif 'height' not in layout_dict or layout_dict['height'] is None:
        if has_subplots:
            layout_dict['height'] = 400 * n_rows
        else:
            layout_dict['height'] = 600  # Default height
    
    # Set width (ensure sufficient size for Heatmap, scrollable)
    if calculated_width is not None:
        # For Heatmap, ignore container limit and set sufficient size (scrollable)
        if has_heatmap:
            layout_dict['width'] = calculated_width  # Sufficient size without limit
        else:
            layout_dict['width'] = min(calculated_width, MAX_CONTAINER_WIDTH)
    elif 'width' not in layout_dict or layout_dict['width'] is None:
        if has_subplots:
            # If subplot exists, proportional to number of columns but with container size limit
            subplot_width = 500  # Default 500px per subplot
            layout_dict['width'] = min(subplot_width * n_cols, MAX_CONTAINER_WIDTH)
        else:
            layout_dict['width'] = min(1000, MAX_CONTAINER_WIDTH)  # Default width, apply limit
    
    # Adjust margin to prevent chart clipping
    if 'margin' not in layout_dict or layout_dict['margin'] is None:
        layout_dict['margin'] = dict(l=50, r=50, t=50, b=50)
    else:
        # If margin exists, ensure right and bottom margins
        if isinstance(layout_dict['margin'], dict):
            layout_dict['margin']['r'] = max(layout_dict['margin'].get('r', 50), 50)
            layout_dict['margin']['b'] = max(layout_dict['margin'].get('b', 50), 50)
    
    fig_clean.update_layout(**layout_dict)
    
    return fig_clean


def _show_tear_sheet_results(result_dict: Optional[Dict]) -> None:
    """
    Display all tables and figures from a tear sheet result dictionary.
    
    This helper function simplifies the repetitive pattern of checking
    for tables and figures in result dictionaries and displaying them.
    
    Parameters
    ----------
    result_dict : dict, optional
        Result dictionary containing 'table' and/or 'figures' keys
    """
    if not result_dict:
        return
    
    # Display tables
    if 'table' in result_dict and result_dict['table'] is not None:
        utils.print_table(result_dict['table'])
    
    if 'worst_periods' in result_dict and result_dict['worst_periods'] is not None:
        utils.print_table(result_dict['worst_periods'])
    
    # Display figures
    if 'figures' in result_dict:
        for fig_name, fig in result_dict['figures'].items():
            if fig is not None:
                fig.show()


def _display_results(
    quantile_stats: Optional[pd.DataFrame] = None,
    returns_result: Optional[Dict] = None,
    information_result: Optional[Dict] = None,
    turnover_result: Optional[Dict] = None,
    factor_data: Optional[pd.DataFrame] = None
) -> None:
    """
    Helper function to display computed results on screen.
    
    Function to separate computation (Compute) and presentation (View).
    
    Parameters
    ----------
    quantile_stats : pd.DataFrame, optional
        Quantile statistics table
    returns_result : dict, optional
        Returns tear sheet result
    information_result : dict, optional
        Information tear sheet result
    turnover_result : dict, optional
        Turnover tear sheet result
    factor_data : pd.DataFrame, optional
        Factor data (for quantile stats output)
    """
    # Output Quantile Statistics
    if quantile_stats is not None and not quantile_stats.empty and factor_data is not None:
        plotting.plot_quantile_statistics_table(factor_data)
    
    # Output Returns
    _show_tear_sheet_results(returns_result)
    
    # Output Information
    _show_tear_sheet_results(information_result)
    
    # Output Turnover
    _show_tear_sheet_results(turnover_result)


def _render_html(
    quantile_stats=None,
    returns_result=None,
    information_result=None,
    turnover_result=None,
    factor_name="Factor Analysis"
):
    """
    Renders a complete HTML file containing all tear sheet plots and tables.
    
    Implementation based on VertexLens style:
    - Use Jinja2 template (base_template.html)
    - Include Plotly.js from CDN only once
    - Convert each figure using pio.to_html(full_html=False, include_plotlyjs=False)
    - Organize tables and charts by section
    
    Parameters
    ----------
    quantile_stats : pd.DataFrame, optional
        Quantile statistics table
    returns_result : dict, optional
        Result from create_returns_tear_sheet
    information_result : dict, optional
        Result from create_information_tear_sheet
    turnover_result : dict, optional
        Result from create_turnover_tear_sheet
    factor_name : str
        Name of the factor for the title
    
    Returns
    -------
    str
        Complete HTML content as string
    """
    # Collect all charts and metrics
    charts_html = []
    metrics_html = []
    
    # Quantile Statistics Table
    if quantile_stats is not None and not quantile_stats.empty:
        # Round to 3 decimal places and format numeric columns
        quantile_stats_rounded = quantile_stats.round(3)
        # Format only numeric columns to 3 decimal places (preserve integers)
        numeric_cols = quantile_stats_rounded.select_dtypes(include=[np.number]).columns
        styled_html = quantile_stats_rounded.style.format(
            {col: "{:.3f}" for col in numeric_cols}
        ).set_caption("<h3>Quantile Statistics</h3>").to_html()
        metrics_html.append(styled_html)
    
    # Returns Tear Sheet
    if returns_result is not None:
        if 'table' in returns_result and returns_result['table'] is not None:
            # Round to 3 decimal places and format numeric columns
            table_rounded = returns_result['table'].round(3)
            numeric_cols = table_rounded.select_dtypes(include=[np.number]).columns
            styled_html = table_rounded.style.format(
                {col: "{:.3f}" for col in numeric_cols}
            ).set_caption("<h3>Returns Table</h3>").to_html()
            metrics_html.append(styled_html)
        
        if 'worst_periods' in returns_result and returns_result['worst_periods'] is not None:
            # Round to 3 decimal places and format numeric columns
            worst_rounded = returns_result['worst_periods'].round(3)
            numeric_cols = worst_rounded.select_dtypes(include=[np.number]).columns
            styled_html = worst_rounded.style.format(
                {col: "{:.3f}" for col in numeric_cols}
            ).set_caption("<h3>Worst 5 Periods</h3>").to_html()
            metrics_html.append(styled_html)
        
        # ⭐ VertexLens approach: convert figure objects to HTML (after figure is fully initialized)
        # ⚠️ Fix bdata compression issue: regenerate figure to convert to HTML without compression
        if 'figures' in returns_result:
            for fig_name, fig in returns_result['figures'].items():
                if fig is not None:
                    # Regenerate figure to prevent bdata compression
                    fig_clean = _rebuild_figure_without_compression(fig)
                    chart_html = pio.to_html(
                        fig_clean, 
                        full_html=False, 
                        include_plotlyjs=False
                    )
                    charts_html.append(chart_html)
    
    # Information Tear Sheet
    if information_result is not None:
        if 'table' in information_result and information_result['table'] is not None:
            # Round to 3 decimal places and format numeric columns
            table_rounded = information_result['table'].round(3)
            numeric_cols = table_rounded.select_dtypes(include=[np.number]).columns
            styled_html = table_rounded.style.format(
                {col: "{:.3f}" for col in numeric_cols}
            ).set_caption("<h3>Information Table</h3>").to_html()
            metrics_html.append(styled_html)
        
        # ⭐ VertexLens approach: convert figure objects to HTML (after figure is fully initialized)
        # ⚠️ Fix bdata compression issue: regenerate figure to convert to HTML without compression
        if 'figures' in information_result:
            for fig_name, fig in information_result['figures'].items():
                if fig is not None:
                    # Regenerate figure to prevent bdata compression
                    fig_clean = _rebuild_figure_without_compression(fig)
                    chart_html = pio.to_html(
                        fig_clean, 
                        full_html=False, 
                        include_plotlyjs=False
                    )
                    charts_html.append(chart_html)
    
    # Turnover Tear Sheet
    if turnover_result is not None:
        if 'table' in turnover_result and turnover_result['table'] is not None:
            # Round to 3 decimal places and format numeric columns
            table_rounded = turnover_result['table'].round(3)
            numeric_cols = table_rounded.select_dtypes(include=[np.number]).columns
            styled_html = table_rounded.style.format(
                {col: "{:.3f}" for col in numeric_cols}
            ).set_caption("<h3>Turnover Table</h3>").to_html()
            metrics_html.append(styled_html)
        
        # ⭐ VertexLens approach: convert figure objects to HTML (after figure is fully initialized)
        # ⚠️ Fix bdata compression issue: regenerate figure to convert to HTML without compression
        if 'figures' in turnover_result:
            for fig_name, fig in turnover_result['figures'].items():
                if fig is not None:
                    # Regenerate figure to prevent bdata compression
                    fig_clean = _rebuild_figure_without_compression(fig)
                    chart_html = pio.to_html(
                        fig_clean, 
                        full_html=False, 
                        include_plotlyjs=False
                    )
                    charts_html.append(chart_html)
    
    # Load Jinja2 template (using cached environment)
    env = _get_jinja_env()
    template = env.get_template('base_template.html')
    
    # Render template (VertexLens style)
    generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rendered_html = template.render(
        factor_name=factor_name,
        generation_date=generation_date,
        charts=charts_html,
        metrics=metrics_html
    )
    
    return rendered_html


