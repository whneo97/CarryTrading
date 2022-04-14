#!/usr/bin/env python
# coding: utf-8

# Code Author: Neo Weihong
# Last Modified: 15 April 2022

# This Python file contains code that achieves the following purposes:

# 1. Generate and save carry trade summary statistics given Excel files containing Interest Rates and Exchange Rates.

# 2. Generate and save the following regressions for carry trade variables:
# 2.1. Exchange rate changes against interest rate differentials (to test for UIP).
# 2.2. Forward premium against interest rate differentials (to test for CIP).
# 2.3. Carry trade returns derived using exchange rate changes against interest rate changes. 
# (to determine if carry trade returns derived using exchange rate changes would be profitable)
# 2.4. Carry trade returns derived using forward premiums against interest rate changes. 
# (to determine if carry trade returns derived using forward premiums would be profitable)
# 2.5. Carry trade returns derived using exchange rate changes against variables of uncertainty.
# (to investigate effects of variables of uncertainty on carry trade returns derived using exchange rate changes)
# 2.5. Carry trade returns derived using forward premiums changes against variables of uncertainty.
# (to investigate effects of variables of uncertainty on carry trade returns derived using forward premiums)

# For each of the following regressions, the presence of interest rates was also included for:
# CHF, EUR, JPY and SEK (which experienced negative interest rates during various time periods from 2011 onwards)
# (to determine if negative interest rates have an effect on carry trade returns)

# Variables of uncertainty include Cboe VIX, TED Spread and Chicago Fed NCFI.

# Imports the necessary packages.
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
import statsmodels.formula.api as smf
from scipy.stats import skew
from dateutil.relativedelta import relativedelta
from num2words import num2words

# Imports exchange rates Excel file.

# This file contain exchange rate data from 1 January 1960 to 14 March 2022 pulled from Datastream, 
# although most data are from 1997 onwards.
# The Excel files contains tabs labelled Daily, Weekly, Monthly, Quarterly, Yearly.
# These tabs are for exchange rate daily collected with the respective intervals between each data point.
# Each of these tabs contain columns <curr>_S, <curr>_1W, <curr>_1M, <curr>_3M,
# where <curr> represents each G10 currency's 3-letter code (excluding the USD).
# S, 1W, 1M and 3M refer to spot exchange rates and 1-week, 1-month and 3-month forward rates respectively.
print('Reading exchange rate Excel file...')
er = pd.ExcelFile('exchange_rates.xlsx', engine='openpyxl')
er_dfs = {tab: er.parse(tab) for tab in ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']}

# Imports interest rates Excel file.

# This file contain interest rate data from 1 January 1960 to 14 March 2022 pulled from Datastream, 
# although most data are from 1997 onwards.
# The Excel files contains tabs labelled Daily, Weekly, Monthly, Quarterly, Yearly.
# These tabs are for exchange rate daily collected with the respective intervals between each data point.
# Each of these tabs contain columns <curr>_1W, <curr>_1M, <curr>_3M,
# where <curr> represents each G10 currency's 3-letter code (including the USD).
# 1W, 1M and 3M refer to 1-week, 1-month and 3-month interbank rates respectively.
# This was with exception to Sweden and Norway, eurocurrency deposit rates were used.
print('Reading interest rate Excel file...')
ir = pd.ExcelFile('interest_rates.xlsx', engine='openpyxl')
ir_dfs = {tab: ir.parse(tab) for tab in ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']}    

# Initialises the list of G10 currency three-letter codes.
print('Initialising currencies...')
currencies = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK', 'USD']

# Imports Excel file containing variables of uncertainty.
# File contains data from 1990, 2006 and 1986 onwards for VIX, TED Spread and NCFI respectively.
# The 'factors' tab in this file has daily data for variables of uncertainty with VIX, TED Spread and NCFI as columns.
print('Reading variables of uncertainty Excel file...')
uncertainties_excel = pd.ExcelFile('factors_of_uncertainty.xlsx', engine='openpyxl')
uncertainties_df = uncertainties_excel.parse('factors', header=1)
uncertainties_df['Code'] = pd.to_datetime(uncertainties_df['Code']).dt.date
uncertainties_df = uncertainties_df.set_index('Code')
uncertainties_df.columns = ['ted', 'vix', 'ncfi']
uncertainties_df = uncertainties_df[['vix', 'ted', 'ncfi']].astype('float64')

# Stores DataFrames for exchange and interest rates in dictionary for ease of reference.
# Converts datatypes to float and divide percentage interest rates by 100 to obtain decimal interest rates.
# Dictionary contains three keys: 7, 30 and 90.
# The keys correspond to data with weekly, monthly and quarterly intervals respectively.
# "full" in the name of the key for exchange rates refers to the DataFrame having all spot, 1-week, 1-month and 3-month data.
# "full" in the name of the key for interest rates refers to the DataFrame having all 1-week, 1-month and 3-month data.
print('Initialising data maps for interest and exchange rates...')
data_maps = {7: {}, 30: {}, 90: {}}
for n, period in zip([7, 30, 90], ['Weekly', 'Monthly', 'Quarterly']):
    ir_period = ir_dfs[period]
    ir_period['date'] = pd.to_datetime(ir_period['date'])
    ir_period = ir_period.set_index('date').astype('float64')/100
    er_period = er_dfs[period]
    er_period['date'] = pd.to_datetime(er_period['date'])
    er_period = er_period.set_index('date').astype('float64')
    
    ir_period = ir_period[ir_period.columns.sort_values()]
    er_period = er_period[er_period.columns.sort_values()]
    data_maps[n]['ir_full'] = ir_period
    data_maps[n]['er_full'] = er_period

def log(x):
    """
    Applies the ln function to values in a given Series or DataFrame.
    Converts any infinity values due to the division of zero to nan.
    
    :param x: a Series or DataFrame containing numerical values.
    :return: a Series or DataFrame on which the ln function has been applied.
    """
    try:
        df = np.log(x)
        df[df==-np.inf] = np.nan
        return df
    except Exception as e:
        raise (e, x)

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

# Creates a directory to store summary statistics results
# if it does not already exist.
csv_file_dir = 'summary_stats'
if not os.path.exists(csv_file_dir):
    os.makedirs(csv_file_dir)

def to_spelling(text):
    """
    Converts and returns a string where all numerals in the string are represented by its spelling.
    :param text: a string of text.
    :return: a string where all numerals in text has been converted to its spelling.
    """
    text = str(text)
    return '_'.join([num2words(text).replace('-', '_') if text.isnumeric() else text for text in text.split('_')])

# Initialises a dictionary of dictionary to store various processed data.
# These data includes data computed using interest and exchange rates for carry trade returns.
# Dictionary contains three keys: 7, 30 and 90.
# The keys correspond to data with weekly, monthly and quarterly intervals respectively.
df_map = {7: {}, 30: {}, 90: {}}

# Stores data for values of uncertainty shifted one period back.
# The periods 7, 30 and 90 correspond to data with weekly, monthly and quarterly intervals respectively.
print('Generating shifted data for variables of uncertainty...')
for n, n_text in zip([7, 30, 90], ['1W', '1M', '3M']):
    
    # Initialises time mappings for period n.
    # This mapping is required since 
    # it's only possible to initiate a trade on a trading day.
    # but the maturity for a trading day could end up on a non-trading day.
    t0_t1_map = {} # Maps the a given date at time t to the date at t+1
    t1_t0_map = {} # Maps the a given date at time t to the date at t-1.

    # Fills in the time maps using date indices of variables of ucnertainty 
    df = uncertainties_df
    for index in range(len(df.index)):
        date = df.index[index]
        if index != len(df.index) - 1:
            date_plus_one = df.index[index + 1]
        next_dates = df.index[df.index >= (date + pd.to_timedelta(n, unit='d'))]
        if next_dates.empty: continue

        if date not in t0_t1_map.keys():
            t0_t1_map[date] = [next_dates[0]]
        else:
            t0_t1_map[date].append(next_dates[0])

        if next_dates[0] not in t1_t0_map.keys():
            t1_t0_map[next_dates[0]] = [date]
        else:
            t1_t0_map[next_dates[0]].append(date)
    
    def shift_df(df, periods):
        """
        Shifts the data in a DataFrame by the number of given periods.
        
        :param df: a DataFrame consisting variables of uncertainty.
        :param periods: the number of periods for the data to be shifted by.
        :return: a DataFrame containing shifted data that replaces data at time t by that at t+periods.        
        """
        if periods == 0:
            return df.astype('float64')
        temp = pd.DataFrame(columns = df.columns, index = df.index)
        shift_map = t0_t1_map if periods > 0 else t1_t0_map
        for date in df.index:
            if date not in shift_map.keys(): continue
            if any([val not in df.index for val in shift_map[date]]): continue
            temp.loc[date, :] = np.mean([df.loc[val, :] for val in shift_map[date]], axis=0)
        return shift_df(temp, periods + (-1 if periods > 0 else 1))
    
    # Converts and stores data for variables of uncertainty at time t-1.
    uncertainties_minus_one = shift_df(uncertainties_df, -1)
    df_map[n]['uncertainties_minus_one'] = uncertainties_minus_one

# Stores data for carry trade variables and displays summary statistics for each time period.
# The periods 7, 30 and 90 correspond to data with weekly, monthly and quarterly intervals respectively.

# All shifting periods are with respect to the time period of carry trade returns
# Hence, for a shifting period of t+1,
# We have:
# - carry trade returns at time t+1,
# - logarithmic exchange rate changes at time t+1 
# (i.e. difference between exchange rates at time t+1 and time t),
# - logarithmic forward premiums at time t+1 
# (i.e. difference between forward rate and spot rate at time t+1 and time t),
# - interest rate differentials at time t,
# - interest rates at time t.
print('Generating and storing summary statistics for carry trade variables...')
for n, n_text in zip([7, 30, 90], ['1W', '1M', '3M']):
    print(f'{n} days: Creating table to store summary statistics...')
    
    # Initialises DataFrame to store summary statistics for the period.
    table = pd.DataFrame()
    
    def insert_mean_series(series, index):
        """
        Inserts the values of the given Series (as means) to the summary statistics DataFrame.
        
        :param series: a Series containing sample means of a carry trade variable 
                       across currencies with three-letter code of G10 currencies as indices.
        :param index: the index of summary statistics DataFrame for the means to be added to.
        :return: None
        """
        
        for col in series.index:
            c = col.split('_')[0]
            if c not in currencies: continue
            table.loc[index, f'{c}_mn'] = series[col]

    def insert_sd_series(series, index):
        """
        Inserts the values of the given Series (as standard deviation) to the summary statistics DataFrame.
        
        :param series: a Series containing sample standard deviations of a carry trade variable 
                       across currencies with three-letter code of G10 currencies as indices.
        :param index: the index of summary statistics DataFrame for the standard deviations to be added to.
        :return: None
        """
        for col in series.index:
            c = col.split('_')[0]
            if c not in currencies: continue
            table.loc[index, f'{c}_sd'] = series[col]
     
    def insert_mean_sd(df, index):
        """
        Computes and inserts the sample mean and standard deviation across all currencies 
        to the summary statistics DataFrame.
        
        :param df: a DataFrame containing values of a carry trade variable for which
                   sample means and standard deviations is to be computed across currencies 
                   with three-letter code of G10 currencies as columns.
        :param index: the index of summary statistics DataFrame for the computed statistics to be added to.
        :return: None
        """
        mean_df = df.mean(axis=0)
        std_df = df.std(axis=0)
        for col in df.columns:
            c = col.split('_')[0]
            if c not in currencies: continue
            table.loc[index, f'{c}_mn'] = mean_df[col]
            table.loc[index, f'{c}_sd'] = std_df[col]
    
    def shift_df(df, periods):
        """
        Shifts given interest or exchange rate data by the number of a given number of periods.
        
        :param df: a DataFrame consisting interest rate or exchange rate data.
        :param periods: an integer number of periods for the data to be shifted by.
        :return: a DataFrame containing shifted data that replaces data at time t by that at t+periods.        
        """
        return df.shift(-periods)

    def get_name(shift_val):
        """
        Generates a text-formatted representation of an integer number.
        
        :param shift_val: an integer for the number of periods data is to be shifted by.
        :return: a string that is the text-formatted representation of the give number,
                 where numerals are converted to their spelling.        
        """
        if shift_val == 0: return to_spelling(shift_val)
        elif shift_val > 0: return f'plus_{to_spelling(shift_val)}'
        else: return f'minus_{to_spelling(abs(shift_val))}'
        
    def get_er_full(shift_val):
        """
        Retrieves an exchange rate DataFrame that is shifted by the given number of periods and
        stores the computed data in the global data map.
        
        "full" in the name of the key for exchange rates refers to the DataFrame 
        having all spot, 1-week, 1-month and 3-month data.
        
        :param shift_val: an integer number of periods the originally imported exchange rate data is to be shifted by.
        :return: a DataFrame containing stored exchange rate data shifted by the specified number of periods.
        """
        global df_map
        try: return df_map[n][f'er_full_{get_name(shift_val)}']
        except KeyError:
            df_map[n][f'er_full_{get_name(shift_val)}'] = shift_df(data_maps[n]['er_full'], shift_val)
            return df_map[n][f'er_full_{get_name(shift_val)}']
    
    def get_ir_full(shift_val):
        """
        Retrieves an interest rate DataFrame that is shifted by the given number of periods and
        stores the computed data in the global data map.
        
        "full" in the name of the key for interest rates refers to the DataFrame 
        having all spot, 1-week, 1-month and 3-month data.
        
        :param shift_val: an integer number of periods the originally imported interest rate data is to be shifted by.
        :return: a DataFrame containing stored interest rate data shifted by the specified number of periods.
        """
        global df_map
        try: return df_map[n][f'ir_full_{get_name(shift_val)}']
        except KeyError:
            df_map[n][f'ir_full_{get_name(shift_val)}'] = shift_df(data_maps[n]['ir_full'], shift_val)
            return df_map[n][f'ir_full_{get_name(shift_val)}']

    def get_lnS1_minus_lnS0(shift_val):
        """
        Retrieves data on logarithmic exchange rate changes that is shifted by the given number of periods and
        stores the computed data in the global data map and as a CSV in created local directory.
        
        :param shift_val: an integer number of periods for the data to be shifted by, 
                    relative to data for logarithmic exchange rate changes at time t.
        :return: Returns a DataFrame containing logarithmic exchange rate changes across currencies
                 shifted by the specified number of periods as argument.
        """
        global df_map
        try: return df_map[n][f'lnS1_minus_lnS0_{get_name(shift_val)}']
        except KeyError:
            er_plus_one = get_er_full(shift_val)
            er_data = get_er_full(shift_val-1)
            logged_S1 = log(er_plus_one[er_plus_one.columns[er_plus_one.columns.str.contains('_S')]])
            logged_S0 = log(er_data[er_data.columns[er_data.columns.str.contains('_S')]])
            lnS1_minus_lnS0_df = logged_S1 - logged_S0
            lnS1_minus_lnS0_df.to_csv(f'{csv_file_dir}/lnS1_minus_lnS0_{n}days.csv')
            df_map[n][f'lnS1_minus_lnS0_{get_name(shift_val)}'] = lnS1_minus_lnS0_df
            return lnS1_minus_lnS0_df
        
    def get_lnf0_minus_lnS0(shift_val):
        """
        Retrieves data on logarithmic forward premiums that is shifted by the given number of periods and
        stores the computed data in the global data map and as a CSV in created local directory.
        
        :param shift_val: an integer number of periods for the data to be shifted by, 
                    relative to data for logarithmic forward premiums at time t.
        :return: Returns a DataFrame containing logarithmic forward premiums across currencies
                 shifted by the specified number of periods as argument.
        """
        global df_map
        try: return df_map[n][f'lnf0_minus_lnS0_{get_name(shift_val)}']
        except KeyError:
            er_data = get_er_full(shift_val-1)
            spots = er_data[er_data.columns[er_data.columns.str.contains('_S')]]
            spots = spots[spots.columns.sort_values()]
            spots.columns = [i for i in currencies if 'USD' not in i]
            forwards = er_data[er_data.columns[er_data.columns.str.contains(f'_F_{n_text}')]]
            forwards = forwards[forwards.columns.sort_values()]
            forwards.columns = [i for i in currencies if 'USD' not in i]
            df = log(forwards) - log(spots)
            lnf0_minus_lnS0_df = df
            lnf0_minus_lnS0_df.to_csv(f'{csv_file_dir}/lnf0_minus_lnS0_{n}days.csv')
            df_map[n][f'lnf0_minus_lnS0_{get_name(shift_val)}'] = lnf0_minus_lnS0_df  
            return lnf0_minus_lnS0_df
        
    def get_ir(shift_val):
        """
        Retrieves data on interest rates that is shifted by the given number of periods and
        stores the computed data in the global data map and as a CSV in created local directory.
        
        :param shift_val: an integer number of periods for the data to be shifted by, 
                    relative to data for interest rates at time t-1.
        :return: Returns a DataFrame containing interest rates across currencies
                 shifted by the specified number of periods as argument.
        """
        global df_map
        try: return df_map[n][f'ir_{get_name(shift_val)}']
        except KeyError:
            ir_data = get_ir_full(shift_val-1)
            ir = ir_data[ir_data.columns[ir_data.columns.str.contains(f'_{n_text}')]]
            df_map[n][f'ir_{get_name(shift_val)}'] = ir
            ir.to_csv(f'{csv_file_dir}/ir_{n}days.csv')
            return ir

    def get_i_star_minus_i(shift_val):
        """
        Retrieves data on interest rate differentials that is shifted by the given number of periods and
        stores the computed data in the global data map and as a CSV in created local directory.
        
        :param shift_val: an integer number of periods for the data to be shifted by, 
                    relative to data for interest rate differentials at time t-1.
        :return: Returns a DataFrame containing interest rate differentials across currencies
                 shifted by the specified number of periods as argument.
        """        
        global df_map
        try: return df_map[n][f'i_star_minus_i_{get_name(shift_val)}']
        except KeyError:
            ir_data = get_ir_full(shift_val-1)
            diff = (ir_data[ir_data.columns[ir_data.columns.str.contains(f'_{n_text}')]])
            df = diff.copy()
            for col in diff.columns:
                df[col] = diff[col] - diff[f'USD_I_{n_text}']
            i_star_minus_i_df = df
            i_star_minus_i_df.to_csv(f'{csv_file_dir}/i_star_minus_i_df_{n}days.csv')
            df_map[n][f'i_star_minus_i_{get_name(shift_val)}'] = i_star_minus_i_df   
            return i_star_minus_i_df
    
    def get_z(shift_val, fwd=False):
        """
        Retrieves data on carry trade returns that is shifted by the given number of periods and
        stores the computed data in the global data map and as a CSV in created local directory.
        
        :param shift_val: an integer number of periods for the data to be shifted by, 
                    relative to data for carry trade returns at time t.
        :param fwd: a boolean that indicates carry trade returns are to be computed using forward premiums
              if set to True and False if returns are to be computed using exchange rate changes
              (default set to False).
        :return: Returns a DataFrame containing carry trade returns across currencies
                 shifted by the specified number of periods as argument.
        """        
        
        global df_map
        name = 'zspot' if not fwd else 'zfwd'
        try: return df_map[n][f'{name}_{get_name(shift_val)}']
        except KeyError:
            i_star_minus_i_df = get_i_star_minus_i(shift_val)
            lnS1_minus_lnS0_df = get_lnS1_minus_lnS0(shift_val) if not fwd else get_lnf0_minus_lnS0(shift_val)
            first_term = i_star_minus_i_df[[i for i in i_star_minus_i_df.columns.sort_values() if 'USD' not in i]]
            second_term = lnS1_minus_lnS0_df[lnS1_minus_lnS0_df.columns.sort_values()]
            first_term.columns = [i for i in currencies if 'USD' not in i]
            second_term.columns = first_term.columns
            z_df = first_term - second_term
            z_df.to_csv(f'{csv_file_dir}/{name}_df_{n}days.csv')
            df_map[n][f'{name}_{get_name(shift_val)}'] = z_df
            return z_df
    
    # Stores the computed data for exchange rate changes, forward premiums,
    # interest rates, interest rate differentials, 
    # returns derived using exchange rate changes
    # and returns using forward premiums in the global data map.
    print(f'{n} days: Initialising computation of summary statistics for carry trade variables...')
    for i in [-1, 0, 1]:
        get_lnS1_minus_lnS0(i)
        get_lnf0_minus_lnS0(i)
        get_ir(i)
        get_i_star_minus_i(i)
        get_z(i)
        get_z(i, fwd=True)
    
    # Insert the means and standard deviations, across currencies,
    # for exchange rate changes, forward premiums,
    # interest rate differentials, 
    # returns derived using exchange rate changes
    # and returns using forward premiums.
    insert_mean_sd(df_map[n]['lnS1_minus_lnS0_plus_one'], 'lnS1_minus_lnS0')
    insert_mean_sd(df_map[n]['lnf0_minus_lnS0_plus_one'], 'lnf0_minus_lnS0')
    insert_mean_sd(df_map[n]['i_star_minus_i_plus_one'], 'i_star_minus_i')
    insert_mean_sd(df_map[n]['zspot_plus_one'], 'zspot')
    insert_mean_sd(df_map[n]['zfwd_plus_one'], 'zfwd')
        
    # Computes the skewness and kurtosis of 
    # carry trade returns derived using exchange returns.
    # Stores the computed data in the global data map and as a CSV in created local directory.
    skew_zspot = df_map[n]['zspot_plus_one'].apply(skew_series)
    df_map[n]['skew_zspot'] = skew_zspot
    skew_zspot.to_csv(f'{csv_file_dir}/skew_zspot_{n}days.csv')
    insert_mean_series(skew_zspot, 'skew_zspot')
    kurt_zspot = df_map[n]['zspot_plus_one'].copy().apply(kurt_series)
    df_map[n]['kurt_zspot'] = kurt_zspot
    kurt_zspot.to_csv(f'{csv_file_dir}/kurt_zspot_{n}days.csv')
    insert_mean_series(kurt_zspot, 'kurt_zspot')
    
    # Computes the skewness and kurtosis of 
    # carry trade returns derived using forward premiums.
    # Stores the computed data in the global data map and as a CSV in created local directory.
    skew_zfwd = df_map[n]['zfwd_plus_one'].copy().apply(skew_series)
    df_map[n]['skew_zfwd'] = skew_zfwd
    skew_zfwd.to_csv(f'{csv_file_dir}/skew_zfwd_{n}days.csv')
    insert_mean_series(skew_zfwd, 'skew_zfwd')    
    kurt_zfwd = df_map[n]['zfwd_plus_one'].copy().apply(kurt_series)
    df_map[n]['kurt_zfwd'] = kurt_zfwd
    kurt_zfwd.to_csv(f'{csv_file_dir}/kurt_zfwd_{n}days.csv')
    insert_mean_series(kurt_zfwd, 'kurt_zfwd')    
    
    # Displays and saves the summary statistics table for carry trade variables.
    # Computes and saves the table of correlation amongst carry trade variables.
    print(f'{n} days: Saving table...')
    table = table.astype('float64')
    df_map[n]['table'] = table    
    table.to_csv(f'{csv_file_dir}/summary_table_{n}days.csv')
    corr = table.transpose().astype('float64').corr()
    df_map[n]['corr'] = corr
    corr.to_csv(f'{csv_file_dir}/corr_{n}days.csv')

print('Generation and saving of summary statistics tables complete.\n')
    
# Joins the a given list of DataFrames
# by the union of their indices
# and returns the joined DataFrame.
def join_dfs(dfs):
    temp = pd.DataFrame()
    for df in dfs:
        temp = temp.join(df, how='outer')
    return temp

# Creates a directory to store regression results
# if it does not already exist.
results_file_dir = 'reg_results'
if not os.path.exists(results_file_dir):
    os.makedirs(results_file_dir)

# Initialises names of keys in the global map 'df_map' 
# containing DataFrames of carry trade variables.

# All shifting periods are with respect to the time period of carry trade returns
# Hence, for a shifting period of t+1,
# We have:
# - carry trade returns at time t+1,
# - logarithmic exchange rate changes at time t+1 
# (i.e. difference between exchange rates at time t+1 and time t),
# - logarithmic forward premiums at time t+1 
# (i.e. difference between forward rate and spot rate at time t+1 and time t),
# - interest rate differentials at time t,
# - interest rates at time t.

# The keys include:
# - lnS1_minus_lnS0_plus_one: Logarithmic exchange rate changes at time t+1 
# - lnf0_minus_lnS0_plus_one: Logarithmic forward premium at time t+1
# - ir_plus_one: Interest rate at time t 
# - i_star_minus_i_plus_one: Interest rate at time t 
# - zspot_plus_one: Carry trade returns derived using exchange rate changes at time t+1
# - zfwd_plus_one: Carry trade returns derived using forward premiums at time t+1
print('Running regressions for carry trade variables...')
frame_names = ['lnS1_minus_lnS0_plus_one',
               'lnf0_minus_lnS0_plus_one',
               'ir_plus_one',
               'i_star_minus_i_plus_one',
               'zspot_plus_one',
               'zfwd_plus_one']

# Where I and II are introduced, they refer to the following
# I: without negative dummy 
# II: with negative dummy
# where negative dummy is a variable that indicates the presence of negative interest rates.
# A value of 1 indicates negative interest rates at time t and 0 otherwise.
# This is only applicable to CHF, EUR, JPY and SEK, since other countrie did not experience
# negative interest rates during the sampling period.

# Initialises and stores regression equations for regression.
print('Initialising regression equations...')
eqn1a = 'lnS1_minus_lnS0_plus_one ~ i_star_minus_i_plus_one'
eqn1b = 'lnS1_minus_lnS0_plus_one ~ i_star_minus_i_plus_one + neg_dummy'
eqn2a = 'lnf0_minus_lnS0_plus_one ~ i_star_minus_i_plus_one'
eqn2b = 'lnf0_minus_lnS0_plus_one ~ i_star_minus_i_plus_one + neg_dummy'
eqn3a = 'zspot_plus_one ~ i_star_minus_i_plus_one'
eqn3b = 'zspot_plus_one ~ i_star_minus_i_plus_one + neg_dummy'
eqn4a = 'zfwd_plus_one ~ i_star_minus_i_plus_one'
eqn4b = 'zfwd_plus_one ~ i_star_minus_i_plus_one + neg_dummy'
eqn5a = 'zspot_plus_one ~ i_star_minus_i_plus_one + vix + ted + ncfi'
eqn5b = 'zspot_plus_one ~ i_star_minus_i_plus_one + vix + ted + ncfi + neg_dummy'
eqn6a = 'zfwd_plus_one ~ i_star_minus_i_plus_one + vix + ted + ncfi'
eqn6b = 'zfwd_plus_one ~ i_star_minus_i_plus_one + vix + ted + ncfi + neg_dummy'

eqns = [eqn1a, eqn1b, eqn2a, eqn2b, eqn3a, eqn3b, eqn4a, eqn4b, eqn5a, eqn5b, eqn6a, eqn6b]


# Runs regressions, displays and stores regression results for each time period.
# The periods 7, 30 and 90 correspond to data with weekly, monthly and quarterly intervals respectively.
for n, n_text in zip([7, 30, 90], ['1W', '1M', '3M']):
    # Initialises a mapping of equations to tables the results are to be stored in.
    print(f'{n} days: Creating table to store regression results...')
    reg_tables_map = {}

    # Initialises tables to store results in.
    # Table 1: Exchange rate changes against interest rate differentials (to test for UIP).
    # Table 2: Forward premium against interest rate differentials (to test for CIP).
    # Table 3: Carry trade returns derived using exchange rate changes against interest rate changes. 
    # (to determine if carry trade returns derived using exchange rate changes would be profitable)
    # Table 4: Carry trade returns derived using forward premiums against interest rate changes. 
    # (to determine if carry trade returns derived using forward premiums would be profitable)
    # Table 5: Carry trade returns derived using exchange rate changes against variables of uncertainty.
    # (to investigate effects of variables of uncertainty on carry trade returns derived using exchange rate changes)
    # Table 6: Carry trade returns derived using forward premiums changes against variables of uncertainty.
    # (to investigate effects of variables of uncertainty on carry trade returns derived using forward premiums)
    
    table1 = pd.DataFrame(columns=[j for i in [(f'{c}_I', f'{c}_II') 
                                               for c in currencies] 
                                   for j in i 
                                   if 'USD' not in currencies])
    table2 = table1.copy()
    table3 = table1.copy()
    table4 = table1.copy()
    table5 = table1.copy()
    table6 = table1.copy()
    
    # Stores mapping of equations to corresponding tables.
    reg_tables_map[eqn1a] = table1
    reg_tables_map[eqn1b] = table1
    reg_tables_map[eqn2a] = table2
    reg_tables_map[eqn2b] = table2
    reg_tables_map[eqn3a] = table3
    reg_tables_map[eqn3b] = table3
    reg_tables_map[eqn4a] = table4
    reg_tables_map[eqn4b] = table4 
    reg_tables_map[eqn5a] = table5
    reg_tables_map[eqn5b] = table5
    reg_tables_map[eqn6a] = table6
    reg_tables_map[eqn6b] = table6
    
    # Runs regressions for each G10 currency independently other than USD.
    # USD is omitted since returns are made in USD.
    print(f'{n} days: Initialising regression sequence...')
    for curr in currencies:
        if curr == 'USD': continue
        frames = []
        # Creates a DataFrame containing all the necessary regression variables,
        # including negative dummy variable indicating presence of negative interest rates.
        for name in frame_names:
            df = df_map[n][name]
            df = df[df.columns.sort_values()]
            df.columns = [i.split('_')[0] for i in df.columns]
            frames.append(df[curr].to_frame(name=name))
        reg_df = join_dfs(frames + [uncertainties_df])
        reg_df['neg_dummy'] = (reg_df['ir_plus_one'] < 0).astype(int)

        # Stores regression table for each G10 currency (other than USD)
        # into the global map of carry trade DataFrames.
        try:
            df_map[n]['reg_df'][curr] = reg_df
        except KeyError:
            df_map[n]['reg_df'] = {curr: reg_df}
        
        # Runs regressions for each equation.
        # Stores the results for each equation in the corresponding tables.
        # Regression results include coefficients and standard errors.
        for eqn in eqns:
            eqn_table = reg_tables_map[eqn]
            reg_df = df_map[n]['reg_df'][curr]
            eqn = to_spelling(eqn)
            reg_df.columns = [to_spelling(i) for i in reg_df.columns]
            y = eqn.split('~')[0].strip()
            x = [k.strip() for k in eqn.split('~')[1].split('+')]
            with_var = 'II' if 'neg_dummy' in x else 'I'
            if 'neg_dummy' in x and len(reg_df[reg_df['neg_dummy'] == 1].dropna()) == 0: continue 
            try: 
                reg = smf.ols(eqn, data=reg_df).fit()
                for beta_index in range(len(x)):
                    var_name = x[beta_index]
                    beta = reg.params[var_name]
                    b_se = reg.bse[var_name]
                    beta_val = f'{beta:.3f}\n({b_se:.3f})'
                    eqn_table.loc[f'B{beta_index+1}', f'{curr}_{with_var}'] = beta_val
            except Exception as e:
                continue

    # Displays tables containing regression results
    # and store them as CSVs in the specified local directory.
    table1.to_csv(f'{results_file_dir}/table1_{n_text}.csv')
    table2.to_csv(f'{results_file_dir}/table2_{n_text}.csv')
    table3.to_csv(f'{results_file_dir}/table3_{n_text}.csv')
    table4.to_csv(f'{results_file_dir}/table4_{n_text}.csv')
    table5.to_csv(f'{results_file_dir}/table5_{n_text}.csv')
    table6.to_csv(f'{results_file_dir}/table6_{n_text}.csv')

print('Running of regressions and saving of result tables complete.\n')

# Prepares and stores data on carry trade returns 
# and interest rates at time t for machine learning.
print('Preparing data for portfolio building and machine learning...')

# Creates a dictionary of dictionaries to store DataFrames 
# containing interest rate differentials, 
# variables of uncertainty (i.e., VIX, TED Spread and NCFI) 
# and carry trade returns derived using spot rates for each period.
# Each currency has its own separate DataFrame.
dfs = {7: {}, 30: {}, 90: {}}
for n in [7, 30, 90]:
    zspot_before = df_map[n]['zspot_zero']
    i_star_minus_i_before = df_map[n]['i_star_minus_i_plus_one'].copy()
    i_star_minus_i_before.columns = [i.split('_')[0] for i in i_star_minus_i_before.columns]
    for col in zspot_before:
        df = pd.DataFrame()
        df2 = df.copy()
        df['zspot_zero'] = zspot_before[col].dropna()
        df2['i_diff'] = i_star_minus_i_before[col].dropna()
        df = df2.join(df, how='outer')
        df = uncertainties_df.join(df, how='right')
        dfs[n][col] = df

# Stores quarterly data for 
# 1. Carry trade predictors (interest rate differentials and variables of uncertainty)
# and carry trade returns.
# 2. Interest rates at time t-1 to be used as signals for the traditional portfolio.
with open(f'i_signal.pkl', 'wb') as f:
    pickle.dump(df_map[90]['ir_zero'], f)
with open('dfs.pkl', 'wb') as f:
    pickle.dump(dfs[90], f)

print('Data preparation and storage complete.')
