#!/usr/bin/env python
# coding: utf-8

# Code Author: Neo Weihong
# Last Modified: 15 April 2022

# This portfolio builder provides the function build_portfolio, that builds a
# long-short portfolio for carry trade returns, given data to be used as signals.

# Imports the necessary packages.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def get_returns_from_dfs(dfs):
    """
    Retrieves data for carry trade returns from a map of currencies to predictor DataFrames. 

    :param dfs: a dictionary of separate [Xs | Y] DataFrames for each currency, where Xs
                represent predictors and Y represents carry trade returns.
    :return: a single DataFrame containing carry trade returns for each currency.
    """
    returns = pd.DataFrame()
    for k, v in dfs.items():
        returns_col = v.columns[-1]
        returns = returns.join(v[returns_col].to_frame(name=k), how='outer')
    return returns

def skew_series(x):
    """
    Applies the skewness function to values in a Series or DataFrame.    
    
    :param x: a Series containing numerical values.
    :return: a Series on which the skewness function has been applied.
    """    
    x = x.dropna()
    T = len(x)
    if T <= 2:
        return np.nan
    return (T/((T-1)*(T-2)))*np.sum(((x - np.mean(x, axis=0))/np.std(x, ddof=1, axis=0))**3, axis=0)

def kurt_series(x):
    """
    Applies the kurtosis function to values in a Series or DataFrame.    
    
    :param x: a Series containing numerical values.
    :return: a Series on which the kurtosis function has been applied.
    """   
    x = x.dropna()
    T = len(x)
    if T <= 3:
        return np.nan
    term1 = ((T*(T+1))/((T-1)*(T-2)*(T-3)))*np.sum(((x - np.mean(x, axis=0))/np.std(x, ddof=1, axis=0))**4, axis=0)
    term2 = (3*(T-1)**2)/((T-2)*(T-3))
    return term1 - term2

def long_short(signals, k=3): 
    """
    Generates portfolio weights for long and short positions with the k top and k bottom signals.
    
    :param signals: a DataFrame containing the values at time t-1 to be used to decide
                    whether long or short positions are to be made at time t-1 for returns at time t.
    :param k: an integer indicating the top and bottom number of currencies to 
              long and short (default set to 3).
    :return: a DataFrame containing equal-weighted portfolio weights for the 
             top 3 and bottom 3 currencies as per the given signals.
    """   
    # Ranks the assets.
    sig_ranks = signals.rank(axis = 1, method='first')
    temp = sig_ranks.copy()
    # Retrieves the cutoff for the top k performers across currencies at each time t.
    top_index = sig_ranks.max(axis = 1) - k
    top_index = pd.concat([top_index] * signals.shape[1], axis = 1)
    top_index.columns = sig_ranks.columns
    # Retrieves the cutoff for the bottom k performers across currencies at each time t.
    bot_index = sig_ranks.min(axis = 1) + k
    bot_index =  pd.concat([bot_index] * signals.shape[1], axis = 1)
    bot_index.columns = sig_ranks.columns
    weights = sig_ranks * 0
    weights[sig_ranks > top_index] = 1/k
    weights[sig_ranks < bot_index] = -1/k
    weights = weights.fillna(0)
    # Ensures that k cannot be greater than half the number of assets in the model at any time t.
    weights = weights.where(temp.max(axis=1) >= 2*k, np.nan)
    assert len(weights[weights.sum(axis=1).dropna().round(5) != 0]) == 0
    return weights

def backtest(backtest_rtn, weights):
    """
    Generates portfolio returns and statistics given all curencies' returns and 
    investment weights for each currency.

    :param backtest_rtn: a DataFrame containing the returns for each currencies 
                         for each interval during the sampling period. 
    :param weights: A DataFrame containing the weights to be invested in each currency 
                    at every time period t.
    :return: a tuple with a DataFrame containing portfolio returns based on the 
             given weights, a Series containing the contribution of each currency's returns to the portfolio, a Series containing the proportion of short positions for each currency and a Series containing the proportion of long positions for each currency 
    """   
    temp = backtest_rtn.copy()
    portfolio_returns = backtest_rtn * weights 
    
    # Computes and stores each currency's contributions to returns earned.
    contr = portfolio_returns.sum(axis=0)
    contr = contr/contr.sum()
    
    # Computes and stores each currency's proportion of long positions.
    prop_short = (weights > 0).astype(int).sum(axis=0)
    prop_short = prop_short/prop_short.sum()

    # Computes and stores each currency's proportion of short positions.
    prop_long = (weights < 0).astype(int).sum(axis=0)
    prop_long = prop_long/prop_long.sum()
    
    strategy_name = backtest_rtn.index[0].strftime('%Y-%m-%d')
    portfolio_returns = pd.DataFrame(portfolio_returns.dropna(how='all').sum(axis = 1))
    portfolio_returns.columns = [strategy_name]
    portfolio_returns = portfolio_returns.where(temp.mean(axis=1).notnull(), np.nan)
    portfolio_returns = portfolio_returns.iloc[:, 0].dropna()
    
    return portfolio_returns, contr, prop_short, prop_long

def mpl_rtn(data_ts, xlab = "Date", ylab = "(%)", title = "Portfolio Returns", 
             figsize = (15, 6), xlab_fontsize = 12, ylab_fontsize = 12, title_fontsize = 15, plot_name=None):
    """
    Plots the percentage returns of a given portfolio.

    :param data_ts: a Series containing portfolio returns for the sampling period.
    :param xlab: the horizontal title for the graph (default set to 'Date').
    :param ylab: the verticle title for the graph (default set to '(%)').
    :param title: the title for the graph (default set to 'Portfolio Returns').
    :param figsize: the size of the graph given by a tuple of its length and breadth 
                    (default set to (15, 6)).
    :param xlab_fontsize: the font size of the graph's horizontal title 
                          (default set to 12).
    :param ylab_fontsize: the font size of the graph's verticle title (default set to 12).
    :param title_fontsize: the font size of the graph's title (default set to 15).
    :param plot_name: the name of the graph's file path, where the graph will be 
                      saved under if its value is not None (default set to None).
    :return: None.
    """   
    data_ts = data_ts * 100
    plt.plot(data_ts.index, [0 for i in range(len(data_ts))], c='orange')
    data_ts.plot(figsize = figsize, c='b')
    plt.title(title, fontsize = title_fontsize)
    plt.xlabel(xlab, fontsize = xlab_fontsize)
    plt.ylabel(ylab, fontsize = ylab_fontsize)
    plt.legend()
    if plot_name is not None:
        plt.savefig(f'{plot_name}.png')
    plt.show()

def get_stats(portfolio_returns, n):

    """
    Calculates and returns statistics of portfolio returns.

    :param portfolio_returns: a Series containing portfolio returns for each 
                              interval during the sampling period.
    :param n: a character representing the interval for which calculations is to
              to be made. This character is defined to be 'D', 'W', 'M', 'Q' and 'Y' for the day, week, month, quarter and year, respectively.
    :return: a tuple containing the portfolio's sample mean, 
             sample standard deviation, Sharpe ratio, skewness and kurtosis.
    """   
    if portfolio_returns.isnull().values.all(axis=0): return
    if type(portfolio_returns) == pd.core.frame.DataFrame and len(portfolio_returns.columns) == 1:
        portfolio_returns = portfolio_returns.iloc[:, 0]
    
    periods = {'D': 252, 'W': 52, 'M': 12, 'Q': 4, 'Y': 1}
    
    mean = portfolio_returns.mean() * periods[n]
    std = portfolio_returns.std() * np.sqrt(periods[n])
    sr = mean/std
    skew = skew_series(portfolio_returns)
    kurt = kurt_series(portfolio_returns)
    
    return mean, std, sr, skew, kurt

def build_portfolio(returns, signals, n='Q', k=3, 
                    cols=None, start=None, end=None, 
                    plot=False, file_name=None, save_returns=False, weight_spec=None):
    """
    Builds the portfolio and returns statistics of the portfolio.

    :param returns: a DataFrame containing carry trade returns for each 
                    individual currency with date indices.
    :param signals: a DataFrame containing numerical values at time t-1 to 
                    be used as signals with date indices.
    :param n: a character representing the interval between carry trade returns. 
              This character is defined to be 'D', 'W', 'M', 'Q' and 'Y' for the day, week, month, quarter and year, respectively (default set to Q).
    :param k: an integer indicating the top and bottom number of currencies to 
              long and short (default set to 3).
    :param cols: a list containing three-letter codes of currencies to be used to 
                 filter currencies of returns and signals. If this is set to None, all columns would be used (default set to None).
    :param start: a string in the format "%Y-%m-%d" or a pandas DateTime object denoting 
                  the start (inclusive) of the sampling period. If this is set to None, the entire dataset would be used (default set to None).
    :param end: a string in the format "%Y-%m-%d" or a pandas DateTime object denoting 
                the end (inclusive) of the sampling period. If this is set to None, the entire dataset would be used (default set to None).
    :param plot: a boolean indicating whether a graph is to be plotted. This is also
                 required for the plot to be saved (default set to False).
    :param file_name: the filename to be used for saving portfolio returns and plot 
                      figure. If this is set to None, the returns and plot figure will not be saved (default set to None).
    :param save_returns: a boolean indicating whether portfolio returns is to be saved 
                         (default set to False).
    :param weight_spec: a string to indicate predefined weights to be set. Valid strings
                        currently include 'full_long' and 'full_short' for portfolios with equally-weighted long and short positions across all currencies respectively (default set to None).
    :return: a tuple containing the generated portfolio's sample mean, 
             sample standard deviation, Sharpe ratio, skewness, kurtosis, a Series containing the contribution of each currency's returns to the portfolio, a Series containing the proportion of short positions for each currency and a Series containing the proportion of long positions for each currency 
    :throw: an Exception if an invalid character is given for the interval or the 
            post-filtered columns of returns and signals do not match. 
    """   
    # Validates the given interval character.
    n = n.upper()
    if n not in ['D', 'M', 'W', 'Q', 'Y']:        
        raise Exception('n must be either D, M, W, Q or Y, for day, month, week, quarter or year respectively.')

    # Filters the given DataFrame to contain the given currencies (if any) or 
    # otherwise preserves the original columns.
    returns = returns[sorted(cols)] if cols is not None else returns[returns.columns.sort_values()]
    signals = signals[sorted(cols)] if cols is not None else signals[signals.columns.sort_values()]
    if not returns.columns.equals(signals.columns):
        raise Exception('Columns of returns must be the same as columns of signals.')
    
    # Calculates the start and end of the sampling period.
    if start is None and end is None:
        start, end = returns.index[0], returns.index[-1]
    elif start is None:
        start = returns.index[0]
        end = pd.to_datetime(end, format="%Y-%m-%d") if type(end) == str else end
    elif end is None:
        start = pd.to_datetime(end, format="%Y-%m-%d") if type(start) == str else start
        end = returns.index[-1]
    
    # Filters the given DataFrame to contain only data from the samping period (if any)
    # or otherwise preserves the original date indices.    
    returns = returns[(returns.index >= start) & (returns.index <= end)]
    signals = signals[(signals.index >= start) & (signals.index <= end)]

    # Assigns weights for each currency at every time t.
    weights = long_short(signals, k)
    if weight_spec == 'full_long':
        weights.iloc[:, :] = -1/len(weights.columns)
    elif weight_spec == 'full_short':
        weights.iloc[:, :] = 1/len(weights.columns) 
    
    # Retrives portfolio returns and statistics.
    portfolio_returns, contr, prop_short, prop_long = backtest(returns, weights)
    
    # Saves portfolio returns.
    if save_returns and file_name is not None:
        portfolio_returns.to_csv(f'{file_name}.csv')

    # Plots portfolio returns.
    if plot: mpl_rtn(portfolio_returns, plot_name=file_name)
    
    # Retrieves and returns portfolio statistics.
    mean, std, sr, skew, kurt = get_stats(portfolio_returns, n)
    return mean, std, sr, skew, kurt, contr, prop_short, prop_long
