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

import pandas as pd
import numpy as np
import warnings
from typing import Optional, Union, List

import empyrical as ep
from pandas.tseries.offsets import BDay
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from . import utils

# Convenience variable for MultiIndex slicing
idx = pd.IndexSlice


def factor_information_coefficient(factor_data, group_adjust=False, by_group=False):
    """
    Computes the Spearman Rank Correlation based Information Coefficient (IC)
    between factor values and N period forward returns for each period in
    the factor index.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    group_adjust : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, compute period wise IC separately for each group.

    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between factor and
        provided forward returns.
    """

    date_idx = factor_data.index.names.index("date")
    # .levels still works but needed for freq attribute access
    freq = factor_data.index.levels[date_idx].freq

    factor_data = factor_data.copy()

    # Performance optimization: reuse get_level_values result
    date_level = factor_data.index.get_level_values("date")
    grouper = [date_level]

    if group_adjust:
        factor_data = utils.demean_forward_returns(factor_data, grouper + ["group"])
    if by_group:
        grouper.append("group")

    # [Optimization] Rank IC calculation: pre-compute ranks then use Pearson Correlation
    # Spearman Correlation = Pearson Correlation after ranking
    # This is much faster than calling spearmanr for each group
    forward_returns_cols = utils.get_forward_returns_columns(factor_data.columns)
    
    # 1. Convert Factor to Rank (vectorized)
    grouper_obj = factor_data.groupby(grouper, observed=True, sort=False)
    ranked_factor = grouper_obj["factor"].rank()
    
    # 2. Calculate Rank for each Forward Return column and compute Pearson Correlation
    ic_dict = {}
    for col in forward_returns_cols:
        # Convert Return to Rank
        ranked_returns = grouper_obj[col].rank()
        
        # Calculate Pearson Correlation for each group (equivalent to Spearman for ranked data)
        def calc_corr(group):
            f_rank = group["factor_rank"].values
            r_rank = group["return_rank"].values
            # Remove NaN
            mask = ~(np.isnan(f_rank) | np.isnan(r_rank))
            if mask.sum() < 2:
                return np.nan
            # Pearson correlation (equivalent to Spearman for ranked data)
            f_clean = f_rank[mask]
            r_clean = r_rank[mask]
            return np.corrcoef(f_clean, r_clean)[0, 1]
        
        # Construct temporary DataFrame with rank data (preserve original index)
        temp_df = pd.DataFrame({
            "factor_rank": ranked_factor,
            "return_rank": ranked_returns
        }, index=factor_data.index)
        
        # Calculate correlation by group (convert grouper to index level names)
        if by_group:
            groupby_keys = ["date", "group"]
        else:
            groupby_keys = ["date"]
        
        ic_series = temp_df.groupby(groupby_keys, observed=True, sort=False).apply(calc_corr)
        ic_dict[col] = ic_series
    
    ic = pd.DataFrame(ic_dict)
    
    if by_group:
        return ic
    else:
        return ic.asfreq(freq)


def mean_information_coefficient(
    factor_data, group_adjust=False, by_group=False, by_time=None
):
    """
    Get the mean information coefficient of specified groups.
    Answers questions like:
    What is the mean IC for each month?
    What is the mean IC for each group for our whole timerange?
    What is the mean IC for for each group, each week?

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    group_adjust : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, take the mean IC for each group.
    by_time : str (pd time_rule), optional
        Time window to use when taking mean IC.
        See http://pandas.pydata.org/pandas-docs/stable/timeseries.html
        for available options.

    Returns
    -------
    ic : pd.DataFrame
        Mean Spearman Rank correlation between factor and provided
        forward price movement windows.
    """

    ic = factor_information_coefficient(factor_data, group_adjust, by_group)

    grouper = []
    if by_time is not None:
        grouper.append(pd.Grouper(freq=by_time))
    if by_group:
        grouper.append("group")

    if len(grouper) == 0:
        ic = ic.mean()

    else:
        # Performance optimization: disable sorting with sort=False
        ic = ic.reset_index().set_index("date").groupby(grouper, sort=False).mean()

    return ic


def factor_weights(factor_data, demeaned=True, group_adjust=False, equal_weight=False):
    """
    Computes asset weights by factor values and dividing by the sum of their
    absolute value (achieving gross leverage of 1). Positive factor values will
    results in positive weights and negative values in negative weights.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    demeaned : bool
        Should this computation happen on a long short portfolio? if True,
        weights are computed by demeaning factor values and dividing by the sum
        of their absolute value (achieving gross leverage of 1). The sum of
        positive weights will be the same as the negative weights (absolute
        value), suitable for a dollar neutral long-short portfolio
    group_adjust : bool
        Should this computation happen on a group neutral portfolio? If True,
        compute group neutral weights: each group will weight the same and
        if 'demeaned' is enabled the factor values demeaning will occur on the
        group level.
    equal_weight : bool, optional
        if True the assets will be equal-weighted instead of factor-weighted
        If demeaned is True then the factor universe will be split in two
        equal sized groups, top assets with positive weights and bottom assets
        with negative weights

    Returns
    -------
    returns : pd.Series
        Assets weighted by factor value.
    """

    def to_weights_equal_weight(group, _demeaned):
        """Equal weight logic - still uses apply due to complex conditional logic"""
        group = group.copy()

        if _demeaned:
            # top assets positive weights, bottom ones negative
            group = group - group.median()

        negative_mask = group < 0
        group[negative_mask] = -1.0
        positive_mask = group > 0
        group[positive_mask] = 1.0

        if _demeaned:
            # positive weights must equal negative weights
            if negative_mask.any():
                group[negative_mask] /= negative_mask.sum()
            if positive_mask.any():
                group[positive_mask] /= positive_mask.sum()

        return group

    # Performance optimization: reuse get_level_values result
    date_level = factor_data.index.get_level_values("date")
    grouper = [date_level]
    if group_adjust:
        grouper.append("group")

    factor_series = factor_data["factor"].copy()

    # [Optimization 1] Demean operation: use transform (Cython optimized)
    if demeaned and not equal_weight:
        grouper_obj = factor_data.groupby(grouper, observed=True, sort=False)["factor"]
        factor_series = factor_series - grouper_obj.transform("mean")

    # [Optimization 2] When equal_weight is False: vectorized weight calculation
    if not equal_weight:
        # Calculate absolute value sum (Vectorized)
        # Grouping criteria depends on group_adjust flag
        if group_adjust:
            # Group by date + group
            abs_sum = factor_series.abs().groupby(grouper, observed=True, sort=False).transform("sum")
        else:
            # Group by date only
            abs_sum = factor_series.abs().groupby([date_level], observed=True, sort=False).transform("sum")
        
        # Division (Vectorized)
        weights = factor_series / abs_sum
    else:
        # equal_weight case: use apply due to complex conditional logic
        # Performance optimization: disable sorting with sort=False
        weights = factor_data.groupby(grouper, group_keys=False, observed=True, sort=False)[
            "factor"
        ].apply(to_weights_equal_weight, demeaned)

    if group_adjust:
        # [Optimization 3] group_adjust post-processing: use transform
        # Group neutralization by date (make sum of weights for each group equal to 0)
        date_grouper = weights.groupby(level="date", observed=True, sort=False)
        # Demean by date (neutralize group weights by averaging them per date)
        weights = weights - date_grouper.transform("mean")
        # Normalize by absolute sum
        abs_sum = weights.abs().groupby(level="date", observed=True, sort=False).transform("sum")
        weights = weights / abs_sum

    return weights


def factor_returns(
    factor_data,
    demeaned=True,
    group_adjust=False,
    equal_weight=False,
    by_asset=False,
):
    """
    Computes period wise returns for portfolio weighted by factor
    values.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    demeaned : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    group_adjust : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    equal_weight : bool, optional
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    by_asset: bool, optional
        If True, returns are reported separately for each asset.

    Returns
    -------
    returns : pd.DataFrame
        Period wise factor returns
    """

    date_idx = factor_data.index.names.index("date")
    # .levels still works but needed for freq attribute access
    freq = factor_data.index.levels[date_idx].freq

    weights = factor_weights(factor_data, demeaned, group_adjust, equal_weight)

    s = utils.get_forward_returns_columns(factor_data.columns)
    weighted_returns = factor_data[s].multiply(weights, axis=0)

    if by_asset:
        returns = weighted_returns
    else:
        # Requires at least one weighted return
        # Otherwise returns np.nan
        # Performance optimization: disable sorting with sort=False
        returns = weighted_returns.groupby(level="date", sort=False).sum(min_count=1).asfreq(freq)

    return returns


def factor_alpha_beta(
    factor_data,
    returns=None,
    demeaned=True,
    group_adjust=False,
    equal_weight=False,
):
    """
    Compute the alpha (excess returns), alpha t-stat (alpha significance),
    and beta (market exposure) of a factor. A regression is run with
    the period wise factor universe mean return as the independent variable
    and mean period wise return from a portfolio weighted by factor values
    as the dependent variable.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    returns : pd.DataFrame, optional
        Period wise factor returns. If this is None then it will be computed
        with 'factor_returns' function and the passed flags: 'demeaned',
        'group_adjust', 'equal_weight'
    demeaned : bool
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation
    group_adjust : bool
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation
    equal_weight : bool, optional
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation

    Returns
    -------
    alpha_beta : pd.Series
        A list containing the alpha, beta, a t-stat(alpha)
        for the given factor and forward returns.
    """

    if returns is None:
        returns = factor_returns(factor_data, demeaned, group_adjust, equal_weight)

    universe_ret = (
        factor_data.groupby(level="date")[
            utils.get_forward_returns_columns(factor_data.columns)
        ]
        .mean()
        .reindex(returns.index, axis=0)
    )

    if isinstance(returns, pd.Series):
        returns.name = universe_ret.columns.values[0]
        returns = pd.DataFrame(returns)

    alpha_beta = pd.DataFrame()
    for period in returns.columns.values:
        x = universe_ret[period].values
        y = returns[period].values
        x = add_constant(x)

        reg_fit = OLS(y, x, missing="drop").fit()
        try:
            alpha, beta = reg_fit.params
        except ValueError:
            alpha_beta.loc["Ann. alpha", period] = np.nan
            alpha_beta.loc["beta", period] = np.nan
        else:
            freq_adjust = pd.Timedelta("252Days") / pd.Timedelta(period)

            alpha_beta.loc["Ann. alpha", period] = (1 + alpha) ** freq_adjust - 1
            alpha_beta.loc["beta", period] = beta

    return alpha_beta


def cumulative_returns(returns):
    """
    Computes cumulative returns from simple daily returns.

    Parameters
    ----------
    returns: pd.Series
        pd.Series containing daily factor returns (i.e. '1D' returns).

    Returns
    -------
    Cumulative returns series : pd.Series
        Example:
            2015-01-05   1.001310
            2015-01-06   1.000805
            2015-01-07   1.001092
            2015-01-08   0.999200
    """

    return ep.cum_returns(returns, starting_value=1)


def positions(weights, period, freq=None):
    """
    Builds net position values time series, the portfolio percentage invested
    in each position.

    Parameters
    ----------
    weights: pd.Series
        pd.Series containing factor weights, the index contains timestamps at
        which the trades are computed and the values correspond to assets
        weights
        - see factor_weights for more details
    period: pandas.Timedelta or string
        Assets holding period (1 day, 2 mins, 3 hours etc). It can be a
        Timedelta or a string in the format accepted by Timedelta constructor
        ('1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset, optional
        Used to specify a particular trading calendar. If not present
        weights.index.freq will be used

    Returns
    -------
    pd.DataFrame
        Assets positions series, datetime on index, assets on columns.
        Example:
            index                 'AAPL'         'MSFT'          cash
            2004-01-09 10:30:00   13939.3800     -14012.9930     711.5585
            2004-01-09 15:30:00       0.00       -16012.9930     411.5585
            2004-01-12 10:30:00   14492.6300     -14624.8700       0.0
            2004-01-12 15:30:00   14874.5400     -15841.2500       0.0
            2004-01-13 10:30:00   -13853.2800    13653.6400      -43.6375
    """

    weights = weights.unstack()

    if not isinstance(period, pd.Timedelta):
        period = pd.Timedelta(period)

    if freq is None:
        freq = weights.index.freq

    if freq is None:
        freq = BDay()
        warnings.warn("'freq' not set, using business day calendar", UserWarning)

    #
    # weights index contains factor computation timestamps, then add returns
    # timestamps too (factor timestamps + period) and save them to 'full_idx'
    # 'full_idx' index will contain an entry for each point in time the weights
    # change and hence they have to be re-computed
    #
    trades_idx = weights.index.copy()
    returns_idx = utils.add_custom_calendar_timedelta(trades_idx, period, freq)
    weights_idx = trades_idx.union(returns_idx)

    #
    # Compute portfolio weights for each point in time contained in the index
    #
    portfolio_weights = pd.DataFrame(index=weights_idx, columns=weights.columns)
    active_weights = []

    for curr_time in weights_idx:

        #
        # fetch new weights that become available at curr_time and store them
        # in active weights
        #
        if curr_time in weights.index:
            assets_weights = weights.loc[curr_time]
            expire_ts = utils.add_custom_calendar_timedelta(curr_time, period, freq)
            active_weights.append((expire_ts, assets_weights))

        #
        # remove expired entry in active_weights (older than 'period')
        #
        if active_weights:
            expire_ts, assets_weights = active_weights[0]
            if expire_ts <= curr_time:
                active_weights.pop(0)

        if not active_weights:
            continue
        #
        # Compute total weights for curr_time and store them
        #
        tot_weights = [w for (ts, w) in active_weights]
        tot_weights = pd.concat(tot_weights, axis=1)
        tot_weights = tot_weights.sum(axis=1)
        tot_weights /= tot_weights.abs().sum()

        portfolio_weights.loc[curr_time] = tot_weights

    return portfolio_weights.fillna(0)


def mean_return_by_quantile(
    factor_data,
    by_date=False,
    by_group=False,
    demeaned=True,
    group_adjust=False,
):
    """
    Computes mean returns for factor quantiles across
    provided forward returns columns.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    by_date : bool
        If True, compute quantile bucket returns separately for each date.
    by_group : bool
        If True, compute quantile bucket returns separately for each group.
    demeaned : bool
        Compute demeaned mean returns (long short portfolio)
    group_adjust : bool
        Returns demeaning will occur on the group level.

    Returns
    -------
    mean_ret : pd.DataFrame
        Mean period wise returns by specified factor quantile.
    std_error_ret : pd.DataFrame
        Standard error of returns by specified quantile.
    """

    # Performance optimization: reuse get_level_values result
    date_level = factor_data.index.get_level_values("date")
    
    if group_adjust:
        grouper = [date_level] + ["group"]
        factor_data = utils.demean_forward_returns(factor_data, grouper)
    elif demeaned:
        factor_data = utils.demean_forward_returns(factor_data)
    # No copy needed in else block since factor_data is not modified

    grouper = ["factor_quantile", date_level]

    if by_group:
        grouper.append("group")

    # Performance optimization: disable sorting with sort=False, observed=True to process only explicit levels
    # "count" is required for std_error_ret calculation, cannot be omitted
    group_stats = factor_data.groupby(grouper, observed=True, sort=False)[
        utils.get_forward_returns_columns(factor_data.columns)
    ].agg(["mean", "std", "count"])

    mean_ret = group_stats.T.xs("mean", level=1).T

    if not by_date:
        # Performance optimization: reuse get_level_values result
        quantile_level = mean_ret.index.get_level_values("factor_quantile")
        grouper = [quantile_level]
        if by_group:
            grouper.append(mean_ret.index.get_level_values("group"))
        # Performance optimization: disable sorting with sort=False
        group_stats = mean_ret.groupby(grouper, sort=False).agg(["mean", "std", "count"])
        mean_ret = group_stats.T.xs("mean", level=1).T

    std_error_ret = group_stats.T.xs("std", level=1).T / np.sqrt(
        group_stats.T.xs("count", level=1).T
    )

    return mean_ret, std_error_ret


def compute_mean_returns_spread(mean_returns, upper_quant, lower_quant, std_err=None):
    """
    Computes the difference between the mean returns of
    two quantiles. Optionally, computes the standard error
    of this difference.

    Parameters
    ----------
    mean_returns : pd.DataFrame
        DataFrame of mean period wise returns by quantile.
        MultiIndex containing date and quantile.
        See mean_return_by_quantile.
    upper_quant : int
        Quantile of mean return from which we
        wish to subtract lower quantile mean return.
    lower_quant : int
        Quantile of mean return we wish to subtract
        from upper quantile mean return.
    std_err : pd.DataFrame, optional
        Period wise standard error in mean return by quantile.
        Takes the same form as mean_returns.

    Returns
    -------
    mean_return_difference : pd.Series
        Period wise difference in quantile returns.
    joint_std_err : pd.Series
        Period wise standard error of the difference in quantile returns.
        if std_err is None, this will be None
    """

    mean_return_difference = mean_returns.xs(
        upper_quant, level="factor_quantile"
    ) - mean_returns.xs(lower_quant, level="factor_quantile")

    if std_err is None:
        joint_std_err = None
    else:
        std1 = std_err.xs(upper_quant, level="factor_quantile")
        std2 = std_err.xs(lower_quant, level="factor_quantile")
        joint_std_err = np.sqrt(std1**2 + std2**2)

    return mean_return_difference, joint_std_err


def quantile_turnover(quantile_factor, quantile, period=1):
    """
    Computes the proportion of names in a factor quantile that were
    not in that quantile in the previous period.

    Parameters
    ----------
    quantile_factor : pd.Series
        DataFrame with date, asset and factor quantile.
    quantile : int
        Quantile on which to perform turnover analysis.
    period: int, optional
        Number of days over which to calculate the turnover.

    Returns
    -------
    quant_turnover : pd.Series
        Period by period turnover for that quantile.
    """

    # Keep DateTimeIndex frequency information
    date_idx = quantile_factor.index.names.index("date")
    # .levels still works but needed for freq attribute access
    freq = quantile_factor.index.levels[date_idx].freq

    quant_names = quantile_factor[quantile_factor == quantile]
    # Performance optimization: disable sorting with sort=False
    quant_name_sets = (
        quant_names.groupby(level=["date"], sort=False)
        .apply(lambda x: set(x.index.get_level_values("asset")))
        .asfreq(freq)
    )

    name_shifted = quant_name_sets.shift(periods=period)

    new_names = (quant_name_sets - name_shifted).dropna()

    def f(xs):
        return 0 if pd.isna(xs) else len(xs)

    def g(xs):
        return 1 if pd.isna(xs) else len(xs)

    quant_turnover = (new_names.apply(f) / quant_name_sets.apply(g)).rename(quantile)
    return quant_turnover


def factor_rank_autocorrelation(factor_data, period=1):
    """
    Computes autocorrelation of mean factor ranks in specified time spans.
    We must compare period to period factor ranks rather than factor values
    to account for systematic shifts in the factor values of all names or names
    within a group. This metric is useful for measuring the turnover of a
    factor. If the value of a factor for each name changes randomly from period
    to period, we'd expect an autocorrelation of 0.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    period: int, optional
        Number of days over which to calculate the turnover.

    Returns
    -------
    autocorr : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation of
        factor values.
    """
    date_idx = factor_data.index.names.index("date")
    # .levels still works but needed for freq attribute access
    freq = factor_data.index.levels[date_idx].freq

    # Performance optimization: disable sorting with sort=False
    asset_ranks_by_day = (
        factor_data.groupby(level="date", sort=False)["factor"]
        .rank()
        .reset_index()
        .pivot(index="date", columns="asset", values="factor")
        .asfreq(freq)
    )

    asset_shifted = asset_ranks_by_day.shift(period)

    return (
        asset_ranks_by_day.corrwith(asset_shifted, axis=1).rename(period).asfreq(freq)
    )


def common_start_returns(
    factor,
    returns,
    before,
    after,
    cumulative=False,
    mean_by_date=False,
    demean_by=None,
):
    """
    A date and equity pair is extracted from each index row in the factor
    dataframe and for each of these pairs a return series is built starting
    from 'before' the date and ending 'after' the date specified in the pair.
    All those returns series are then aligned to a common index (-before to
    after) and returned as a single DataFrame

    Parameters
    ----------
    factor : pd.DataFrame
        DataFrame with at least date and equity as index, the columns are
        irrelevant
    returns : pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets in the
        columns. Returns data should span the factor analysis time period
        plus/minus an additional buffer window corresponding to after/before
        period parameters.
    before:
        How many returns to load before factor date
    after:
        How many returns to load after factor date
    cumulative: bool, optional
        Whether the given returns are cumulative. If False the given
        returns are assumed to be daily.
    mean_by_date: bool, optional
        If True, compute mean returns for each date and return that
        instead of a return series for each asset
    demean_by: pd.DataFrame, optional
        DataFrame with at least date and equity as index, the columns are
        irrelevant. For each date a list of equities is extracted from
        'demean_by' index and used as universe to compute demeaned mean
        returns (long short portfolio)

    Returns
    -------
    aligned_returns : pd.DataFrame
        Dataframe containing returns series for each factor aligned to the same
        index: -before to after
    """
    if not cumulative:
        returns = returns.apply(cumulative_returns, axis=0)

    all_returns = []

    for timestamp, df in factor.groupby(level="date"):

        equities = df.index.get_level_values("asset")

        try:
            day_zero_index = returns.index.get_loc(timestamp)
        except KeyError:
            continue

        starting_index = max(day_zero_index - before, 0)
        ending_index = min(day_zero_index + after + 1, len(returns.index))

        equities_slice = set(equities)
        if demean_by is not None:
            demean_equities = demean_by.loc[timestamp].index.get_level_values("asset")
            equities_slice |= set(demean_equities)

        series = returns.loc[
            returns.index[starting_index:ending_index], list(equities_slice)
        ]
        series.index = range(
            starting_index - day_zero_index, ending_index - day_zero_index
        )

        if demean_by is not None:
            mean = series.loc[:, demean_equities].mean(axis=1)
            series = series.loc[:, equities]
            series = series.sub(mean, axis=0)

        if mean_by_date:
            series = series.mean(axis=1)

        all_returns.append(series)

    return pd.concat(all_returns, axis=1)


def average_cumulative_return_by_quantile(
    factor_data,
    returns,
    periods_before=10,
    periods_after=15,
    demeaned=True,
    group_adjust=False,
    by_group=False,
):
    """
    Plots average cumulative returns by factor quantiles in the period range
    defined by -periods_before to periods_after

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    returns : pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets in the
        columns. Returns data should span the factor analysis time period
        plus/minus an additional buffer window corresponding to periods_after/
        periods_before parameters.
    periods_before : int, optional
        How many periods before factor to plot
    periods_after  : int, optional
        How many periods after factor to plot
    demeaned : bool, optional
        Compute demeaned mean returns (long short portfolio)
    group_adjust : bool
        Returns demeaning will occur on the group level (group
        neutral portfolio)
    by_group : bool
        If True, compute cumulative returns separately for each group

    Returns
    -------
    cumulative returns and std deviation : pd.DataFrame
        A MultiIndex DataFrame indexed by quantile (level 0) and mean/std
        (level 1) and the values on the columns in range from
        -periods_before to periods_after
        If by_group=True the index will have an additional 'group' level
        ::
            ---------------------------------------------------
                        |       | -2  | -1  |  0  |  1  | ...
            ---------------------------------------------------
              quantile  |       |     |     |     |     |
            ---------------------------------------------------
                        | mean  |  x  |  x  |  x  |  x  |
                 1      ---------------------------------------
                        | std   |  x  |  x  |  x  |  x  |
            ---------------------------------------------------
                        | mean  |  x  |  x  |  x  |  x  |
                 2      ---------------------------------------
                        | std   |  x  |  x  |  x  |  x  |
            ---------------------------------------------------
                ...     |                 ...
            ---------------------------------------------------
    """

    def cumulative_return_around_event(q_fact, demean_by):
        return common_start_returns(
            q_fact,
            returns,
            periods_before,
            periods_after,
            cumulative=True,
            mean_by_date=True,
            demean_by=demean_by,
        )

    def average_cumulative_return(q_fact, demean_by):
        q_returns = cumulative_return_around_event(q_fact, demean_by)
        q_returns.replace([np.inf, -np.inf], np.nan, inplace=True)

        return pd.DataFrame(
            {
                "mean": q_returns.mean(skipna=True, axis=1),
                "std": q_returns.std(skipna=True, axis=1),
            }
        ).T

    if by_group:
        #
        # Compute quantile cumulative returns separately for each group
        # Deman those returns accordingly to 'group_adjust' and 'demeaned'
        #
        returns_bygroup = []

        # Performance optimization: disable sorting with sort=False
        for group, g_data in factor_data.groupby("group", observed=True, sort=False):
            g_fq = g_data["factor_quantile"]
            if group_adjust:
                demean_by = g_fq  # Demeans at group level
            elif demeaned:
                demean_by = factor_data["factor_quantile"]  # Demean by all
            else:
                demean_by = None
            #
            # Align cumulative return from different dates to the same index
            # then compute mean and std
            #
            # Performance optimization: disable sorting with sort=False
            avgcumret = g_fq.groupby(g_fq, sort=False).apply(average_cumulative_return, demean_by)
            if len(avgcumret) == 0:
                continue

            avgcumret["group"] = group
            avgcumret.set_index("group", append=True, inplace=True)
            returns_bygroup.append(avgcumret)

        return pd.concat(returns_bygroup, axis=0)

    else:
        #
        # Compute quantile cumulative returns for the full factor_data
        # Align cumulative return from different dates to the same index
        # then compute mean and std
        # Deman those returns accordingly to 'group_adjust' and 'demeaned'
        #
        if group_adjust:
            all_returns = []
            # Performance optimization: disable sorting with sort=False
            for group, g_data in factor_data.groupby("group", observed=True, sort=False):
                g_fq = g_data["factor_quantile"]
                # Performance optimization: disable sorting with sort=False
                avgcumret = g_fq.groupby(g_fq, sort=False).apply(
                    cumulative_return_around_event, g_fq
                )
                all_returns.append(avgcumret)
            q_returns = pd.concat(all_returns, axis=1)
            q_returns = pd.DataFrame(
                {"mean": q_returns.mean(axis=1), "std": q_returns.std(axis=1)}
            )
            return q_returns.unstack(level=1).stack(level=0)
        elif demeaned:
            fq = factor_data["factor_quantile"]
            # Performance optimization: disable sorting with sort=False
            return fq.groupby(fq, sort=False).apply(average_cumulative_return, fq)
        else:
            fq = factor_data["factor_quantile"]
            # Performance optimization: disable sorting with sort=False
            return fq.groupby(fq, sort=False).apply(average_cumulative_return, None)


def factor_cumulative_returns(
    factor_data,
    period,
    long_short=True,
    group_neutral=False,
    equal_weight=False,
    quantiles=None,
    groups=None,
):
    """
    Simulate a portfolio using the factor in input and returns the cumulative
    returns of the simulated portfolio

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    period : string
        'factor_data' column name corresponding to the 'period' returns to be
        used in the computation of porfolio returns
    long_short : bool, optional
        if True then simulates a dollar neutral long-short portfolio
        - see performance.create_pyfolio_input for more details
    group_neutral : bool, optional
        If True then simulates a group neutral portfolio
        - see performance.create_pyfolio_input for more details
    equal_weight : bool, optional
        Control the assets weights:
        - see performance.create_pyfolio_input for more details
    quantiles: sequence[int], optional
        Use only specific quantiles in the computation. By default all
        quantiles are used
    groups: sequence[string], optional
        Use only specific groups in the computation. By default all groups
        are used

    Returns
    -------
    Cumulative returns series : pd.Series
        Example:
            2015-07-16 09:30:00  -0.012143
            2015-07-16 12:30:00   0.012546
            2015-07-17 09:30:00   0.045350
            2015-07-17 12:30:00   0.065897
            2015-07-20 09:30:00   0.030957
    """
    fwd_ret_cols = utils.get_forward_returns_columns(factor_data.columns)

    if period not in fwd_ret_cols:
        raise ValueError("Period '%s' not found" % period)

    todrop = list(fwd_ret_cols)
    todrop.remove(period)
    portfolio_data = factor_data.drop(todrop, axis=1)

    if quantiles is not None:
        portfolio_data = portfolio_data[
            portfolio_data["factor_quantile"].isin(quantiles)
        ]

    if groups is not None:
        portfolio_data = portfolio_data[portfolio_data["group"].isin(groups)]

    returns = factor_returns(portfolio_data, long_short, group_neutral, equal_weight)

    return cumulative_returns(returns[period])


def factor_positions(
    factor_data,
    period,
    long_short=True,
    group_neutral=False,
    equal_weight=False,
    quantiles=None,
    groups=None,
):
    """
    Simulate a portfolio using the factor in input and returns the assets
    positions as percentage of the total portfolio.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    period : string
        'factor_data' column name corresponding to the 'period' returns to be
        used in the computation of porfolio returns
    long_short : bool, optional
        if True then simulates a dollar neutral long-short portfolio
        - see performance.create_pyfolio_input for more details
    group_neutral : bool, optional
        If True then simulates a group neutral portfolio
        - see performance.create_pyfolio_input for more details
    equal_weight : bool, optional
        Control the assets weights:
        - see performance.create_pyfolio_input for more details.
    quantiles: sequence[int], optional
        Use only specific quantiles in the computation. By default all
        quantiles are used
    groups: sequence[string], optional
        Use only specific groups in the computation. By default all groups
        are used

    Returns
    -------
    assets positions : pd.DataFrame
        Assets positions series, datetime on index, assets on columns.
        Example:
            index                 'AAPL'         'MSFT'          cash
            2004-01-09 10:30:00   13939.3800     -14012.9930     711.5585
            2004-01-09 15:30:00       0.00       -16012.9930     411.5585
            2004-01-12 10:30:00   14492.6300     -14624.8700       0.0
            2004-01-12 15:30:00   14874.5400     -15841.2500       0.0
            2004-01-13 10:30:00   -13853.2800    13653.6400      -43.6375
    """
    fwd_ret_cols = utils.get_forward_returns_columns(factor_data.columns)

    if period not in fwd_ret_cols:
        raise ValueError("Period '%s' not found" % period)

    todrop = list(fwd_ret_cols)
    todrop.remove(period)
    portfolio_data = factor_data.drop(todrop, axis=1)

    if quantiles is not None:
        portfolio_data = portfolio_data[
            portfolio_data["factor_quantile"].isin(quantiles)
        ]

    if groups is not None:
        portfolio_data = portfolio_data[portfolio_data["group"].isin(groups)]

    weights = factor_weights(portfolio_data, long_short, group_neutral, equal_weight)

    return positions(weights, period)


def create_pyfolio_input(
    factor_data,
    period,
    capital=None,
    long_short=True,
    group_neutral=False,
    equal_weight=False,
    quantiles=None,
    groups=None,
    benchmark_period="1D",
):
    """
    Simulate a portfolio using the input factor and returns the portfolio
    performance data properly formatted for Pyfolio analysis.

    For more details on how this portfolio is built see:
    - performance.cumulative_returns (how the portfolio returns are computed)
    - performance.factor_weights (how assets weights are computed)

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    period : string
        'factor_data' column name corresponding to the 'period' returns to be
        used in the computation of porfolio returns
    capital : float, optional
        If set, then compute 'positions' in dollar amount instead of percentage
    long_short : bool, optional
        if True enforce a dollar neutral long-short portfolio: asset weights
        will be computed by demeaning factor values and dividing by the sum of
        their absolute value (achieving gross leverage of 1) which will cause
        the portfolio to hold both long and short positions and the total
        weights of both long and short positions will be equal.
        If False the portfolio weights will be computed dividing the factor
        values and  by the sum of their absolute value (achieving gross
        leverage of 1). Positive factor values will generate long positions and
        negative factor values will produce short positions so that a factor
        with only posive values will result in a long only portfolio.
    group_neutral : bool, optional
        If True simulates a group neutral portfolio: the portfolio weights
        will be computed so that each group will weigh the same.
        if 'long_short' is enabled the factor values demeaning will occur on
        the group level resulting in a dollar neutral, group neutral,
        long-short portfolio.
        If False group information will not be used in weights computation.
    equal_weight : bool, optional
        if True the assets will be equal-weighted. If long_short is True then
        the factor universe will be split in two equal sized groups with the
        top assets in long positions and bottom assets in short positions.
        if False the assets will be factor-weighed, see 'long_short' argument
    quantiles: sequence[int], optional
        Use only specific quantiles in the computation. By default all
        quantiles are used
    groups: sequence[string], optional
        Use only specific groups in the computation. By default all groups
        are used
    benchmark_period : string, optional
        By default benchmark returns are computed as the factor universe mean
        daily returns but 'benchmark_period' allows to choose a 'factor_data'
        column corresponding to the returns to be used in the computation of
        benchmark returns. More generally benchmark returns are computed as the
        factor universe returns traded at 'benchmark_period' frequency, equal
        weighting and long only


    Returns
    -------
     returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902

     positions : pd.DataFrame
        Time series of dollar amount (or percentage when 'capital' is not
        provided) invested in each position and cash.
         - Days where stocks are not held can be represented by 0.
         - Non-working capital is labelled 'cash'
         - Example:
            index         'AAPL'         'MSFT'          cash
            2004-01-09    13939.3800     -14012.9930     711.5585
            2004-01-12    14492.6300     -14624.8700     27.1821
            2004-01-13    -13853.2800    13653.6400      -43.6375


     benchmark : pd.Series
        Benchmark returns computed as the factor universe mean daily returns.

    """

    #
    # Build returns:
    # We don't know the frequency at which the factor returns are computed but
    # pyfolio wants daily returns. So we compute the cumulative returns of the
    # factor, then resample it at 1 day frequency and finally compute daily
    # returns
    #
    cumrets = factor_cumulative_returns(
        factor_data,
        period,
        long_short,
        group_neutral,
        equal_weight,
        quantiles,
        groups,
    )
    cumrets = cumrets.resample("1D").last().fillna(method="ffill")
    returns = cumrets.pct_change().fillna(0)

    #
    # Build positions. As pyfolio asks for daily position we have to resample
    # the positions returned by 'factor_positions' at 1 day frequency and
    # recompute the weights so that the sum of daily weights is 1.0
    #
    positions = factor_positions(
        factor_data,
        period,
        long_short,
        group_neutral,
        equal_weight,
        quantiles,
        groups,
    )
    positions = positions.resample("1D").sum().fillna(method="ffill")
    positions = positions.div(positions.abs().sum(axis=1), axis=0).fillna(0)
    positions["cash"] = 1.0 - positions.sum(axis=1)

    # Transform percentage positions to dollar positions
    if capital is not None:
        positions = positions.mul(cumrets.reindex(positions.index) * capital, axis=0)

    #
    # Build benchmark returns as the factor universe mean returns traded at
    # 'benchmark_period' frequency
    #
    fwd_ret_cols = utils.get_forward_returns_columns(factor_data.columns)
    if benchmark_period in fwd_ret_cols:
        benchmark_data = factor_data.copy()
        # Make sure no negative positions
        benchmark_data["factor"] = benchmark_data["factor"].abs()
        benchmark_rets = factor_cumulative_returns(
            benchmark_data,
            benchmark_period,
            long_short=False,
            group_neutral=False,
            equal_weight=True,
        )
        benchmark_rets = benchmark_rets.resample("1D").last().fillna(method="ffill")
        benchmark_rets = benchmark_rets.pct_change().fillna(0)
        benchmark_rets.name = "benchmark"
    else:
        benchmark_rets = None

    return returns, positions, benchmark_rets


# ----------------------------------
# Additional Custom Functions (Extended beyond original alphalens)
# ----------------------------------
def yearly_win_rate(factor_data, group_adjust=False):
    """
    Calculates the proportion of years where IC was positive.
    
    Track A validation criterion: Pass if 70% or higher
    
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        factor_data from get_clean_factor_and_forward_returns
    group_adjust : bool
        Whether to calculate group-neutral IC
        
    Returns
    -------
    win_rate : pd.Series
        Yearly win rate for each forward period (0~1)
    yearly_ic : pd.DataFrame
        Yearly IC values (year × forward periods)
    """
    # Calculate average IC by year
    yearly_ic = mean_information_coefficient(
        factor_data,
        group_adjust=group_adjust,
        by_group=False,
        by_time="YE"  # Yearly (year-end basis, 'Y' is deprecated)
    )
    
    # Calculate proportion of years where IC > 0
    win_rate = (yearly_ic > 0).mean()
    
    return win_rate, yearly_ic


def cumulative_spread_drawdown(factor_data, period, long_short=True, group_neutral=False):
    """
    Calculates Maximum Drawdown (MDD) for Long-Short portfolio (Spread).
    
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        factor_data from get_clean_factor_and_forward_returns
    period : str
        Forward return period (e.g., '20D')
    long_short : bool
        Whether to use Long-Short portfolio
    group_neutral : bool
        Whether to use group-neutral portfolio
        
    Returns
    -------
    mdd : float
        Maximum drawdown (0~1, e.g., 0.15 = 15%)
    mdd_date : pd.Timestamp
        Date when MDD occurred
    drawdown_series : pd.Series
        Cumulative drawdown time series
    """
    # Calculate Spread portfolio returns (Q1 - Q5)
    mean_ret, _ = mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral
    )
    
    if period not in mean_ret.columns:
        raise ValueError(f"Period '{period}' not found in factor_data")
    
    # Difference between top/bottom quantile returns (Spread)
    top_quantile = factor_data['factor_quantile'].max()
    bottom_quantile = factor_data['factor_quantile'].min()
    
    spread_returns = mean_ret.loc[top_quantile, period] - mean_ret.loc[bottom_quantile, period]
    
    # Calculate cumulative returns
    cum_returns = cumulative_returns(spread_returns)
    
    # Calculate cumulative maximum
    cummax = cum_returns.expanding().max()
    
    # Drawdown calculation: (current - peak) / peak
    drawdown = (cum_returns - cummax) / cummax
    
    # MDD: Maximum drawdown (absolute value)
    mdd = abs(drawdown.min())
    mdd_date = drawdown.idxmin()
    
    return mdd, mdd_date, drawdown


def quantile_monotonicity_score(mean_quant_ret):
    """
    Quantifies Monotonicity of quantile returns.
    
    Measures monotonic relationship between quantile numbers and returns
    using Spearman Rank Correlation.
    
    Parameters
    ----------
    mean_quant_ret : pd.DataFrame
        Average returns by quantile (quantile × period)
        
    Returns
    -------
    monotonicity_scores : pd.Series
        Monotonicity score for each forward period (0~1, closer to 1 means perfect monotonic relationship)
    """
    scores = {}
    quantile_numbers = mean_quant_ret.index.values
    
    for period in mean_quant_ret.columns:
        returns = mean_quant_ret[period].values
        # Calculate Spearman Rank Correlation
        corr, _ = stats.spearmanr(quantile_numbers, returns)
        # Use absolute value (both monotonic increase/decrease become positive)
        scores[period] = abs(corr) if not np.isnan(corr) else 0.0
    
    return pd.Series(scores)


def spread_calmar_ratio(factor_data, period, long_short=True, group_neutral=False):
    """
    Calculates Calmar Ratio for Spread portfolio.
    
    Calmar Ratio = CAGR / MDD
    Higher is better (generally 1.0 or higher is good)
    
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        factor_data from get_clean_factor_and_forward_returns
    period : str
        Forward return period (e.g., '20D')
    long_short : bool
        Whether to use Long-Short portfolio
    group_neutral : bool
        Whether to use group-neutral portfolio
        
    Returns
    -------
    calmar_ratio : float
        Calmar Ratio (CAGR / MDD)
    cagr : float
        Annualized return (CAGR)
    mdd : float
        Maximum drawdown (MDD)
    """
    # Calculate Spread returns
    mean_ret, _ = mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral
    )
    
    if period not in mean_ret.columns:
        raise ValueError(f"Period '{period}' not found in factor_data")
    
    top_quantile = factor_data['factor_quantile'].max()
    bottom_quantile = factor_data['factor_quantile'].min()
    
    spread_returns = mean_ret.loc[top_quantile, period] - mean_ret.loc[bottom_quantile, period]
    
    # Calculate cumulative returns
    cum_returns = cumulative_returns(spread_returns)
    
    # Calculate CAGR (annualized)
    if len(cum_returns) > 0 and cum_returns.iloc[-1] > 0:
        # Convert period to years
        days_per_year = 252  # Trading days basis
        try:
            period_days = int(pd.Timedelta(period).days)
        except:
            period_days = int(period.replace('D', '').replace('d', '')) if period.replace('D', '').replace('d', '').isdigit() else 1
        
        total_periods = len(cum_returns)
        years = (total_periods * period_days) / days_per_year
        
        if years > 0:
            total_return = cum_returns.iloc[-1]
            cagr = (total_return ** (1 / years)) - 1 if total_return > 0 else 0.0
        else:
            cagr = 0.0
    else:
        cagr = 0.0
    
    # Calculate MDD
    mdd, _, _ = cumulative_spread_drawdown(factor_data, period, long_short, group_neutral)
    
    # Calculate Calmar Ratio
    if mdd > 0:
        calmar_ratio = abs(cagr) / mdd
    else:
        calmar_ratio = np.inf if cagr > 0 else 0.0
    
    return calmar_ratio, cagr, mdd


def ic_decay_ratio(ic):
    """
    Calculates the decay ratio of IC over time.
    
    IC Decay = IC(20D) / IC(1D) or IC(20D) / IC(5D)
    Higher is better (closer to 1.0 means predictive power maintained over time)
    
    Parameters
    ----------
    ic : pd.DataFrame
        IC time series data (date × forward_periods)
        
    Returns
    -------
    decay_ratios : pd.Series
        IC Decay ratio for each period combination
    """
    periods = ic.columns.tolist()
    decay_ratios = {}
    
    # Compare other periods with 1D as baseline
    if '1D' in periods:
        ic_1d = ic['1D'].mean()
        for period in periods:
            if period != '1D' and ic_1d != 0:
                ic_period = ic[period].mean()
                decay_ratios[f'{period}/1D'] = abs(ic_period / ic_1d) if not np.isnan(ic_period) else 0.0
    
    # Compare other periods with 5D as baseline
    if '5D' in periods:
        ic_5d = ic['5D'].mean()
        for period in periods:
            if period not in ['1D', '5D'] and ic_5d != 0:
                ic_period = ic[period].mean()
                decay_ratios[f'{period}/5D'] = abs(ic_period / ic_5d) if not np.isnan(ic_period) else 0.0
    
    return pd.Series(decay_ratios)


def breakeven_transaction_cost(factor_data, period, long_short=True, group_neutral=False):
    """
    Calculates Breakeven Transaction Cost.
    
    Formula: Spread (bps) / (Turnover * 2)
    This value should be at least 10~15bps or higher to cover slippage in practice.
    
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        factor_data from get_clean_factor_and_forward_returns
    period : str
        Forward return period (e.g., '20D')
    long_short : bool
        Whether to use Long-Short portfolio
    group_neutral : bool
        Whether to use group-neutral portfolio
        
    Returns
    -------
    breakeven_cost : float
        Breakeven Transaction Cost (bps)
    spread_bps : float
        Spread returns (bps)
    avg_turnover : float
        Average turnover rate
    """
    from . import plotting
    
    # Calculate Spread
    mean_ret, _ = mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral
    )
    
    if period not in mean_ret.columns:
        raise ValueError(f"Period '{period}' not found in factor_data")
    
    top_quantile = factor_data['factor_quantile'].max()
    bottom_quantile = factor_data['factor_quantile'].min()
    
    spread_returns = mean_ret.loc[top_quantile, period] - mean_ret.loc[bottom_quantile, period]
    spread_bps = spread_returns.mean() * plotting.DECIMAL_TO_BPS
    
    # Calculate Turnover (average of Top + Bottom Quantile)
    try:
        period_int = int(pd.Timedelta(period).days)
    except:
        period_int = int(period.replace('D', '').replace('d', '')) if period.replace('D', '').replace('d', '').isdigit() else 1
    
    quantile_factor = factor_data["factor_quantile"]
    top_turnover = quantile_turnover(quantile_factor, top_quantile, period_int).mean()
    bottom_turnover = quantile_turnover(quantile_factor, bottom_quantile, period_int).mean()
    avg_turnover = (top_turnover + bottom_turnover) / 2
    
    # Calculate Breakeven Transaction Cost
    # Long-Short so both buy/sell occur, hence * 2
    if avg_turnover > 0:
        breakeven_cost = spread_bps / (avg_turnover * 2)
    else:
        breakeven_cost = np.inf
    
    return breakeven_cost, spread_bps, avg_turnover
