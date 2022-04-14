#!/usr/bin/env python
# coding: utf-8

# Code Author: Neo Weihong
# Last Modified: 15 April 2022

# This model executor serves to run machine learning models by 
# training them based on the aggregated performance of carry trade portfolios.

# Imports the necessary packages.
import itertools
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from portfolio_builder import get_returns_from_dfs, build_portfolio
from time import time
from matplotlib import pyplot as plt

def plot_returns(y, y_hat):
    """
    Plots a currency's actual carry trade returns against predicted carry trade returns.    
    
    :param y: a Series containing a currency's actual carry trade returns.
    :param y_hat: a Series containing a currency's predicted carry trade returns.
    :return: None.
    """    
    plt.plot(y_hat.index, y_hat, label='predicted')
    plt.plot(y.index, y, label='actual')
    plt.legend()
    plt.show()
    
def get_interpolated_dfs_with_filled_na(dfs):
    """
    Interpolates missing values in currencies' predictor DataFrames and fills missing values with the mean.
    
    :param dfs: a dictionary of DataFrames for each currency, each in the form [Xs | Y],
                where Xs are predictors and Y is response variable containing carry trade returns.
    :return: a dictionary containing the same currencies as keys and DataFrames with 
             interpolated and imputed missing values.
    """
    for k, v in dfs.items():
        dfs[k] = v.interpolate()
        dfs[k] = dfs[k].fillna(dfs[k].mean())
    return dfs

def get_test_start_date(dfs):
    """
    Retrieves the latest test start date for a given dictionary of DataFrames, 
    such that it is the latest 80% cut-off date for the data of any given curency.

    This ensures that each currency has at least 80% of data to be used for training, 
    and that the training and test cut-offs for all currencies are the same.
    
    :param dfs: a dictionary of DataFrames for each currency, each in the form [Xs | Y],
                where Xs are predictors and Y is response variable containing carry trade returns.
    :return: a pandas DateTime that marks the latest start date for the testing phase 
             of machine learning.
    """
    test_start_date = None
    dfs = get_interpolated_dfs_with_filled_na(dfs)
    for k, v in dfs.items():
        if test_start_date is None or test_start_date > v.index[int(0.8*len(v))]:
            test_start_date = v.index[int(0.8*len(v))]
    assert test_start_date is not None
    return test_start_date

def get_train_test(dfs):
    """
    Retrieves the training and testing dataset of all currencies.

    This function may also be used for retrieving training and validation data.
    
    :param dfs: a dictionary of DataFrames for each currency, each in the form [Xs | Y],
                where Xs are predictors and Y is response variable containing carry trade returns.
    :return: two dictionary of DataFrames with currencies as keys mapped to DataFrames 
             for training and testing, respectively.
    """
    if type(dfs) == pd.core.frame.DataFrame:
        n_train = int(0.8*len(dfs))
        df_train = dfs.iloc[:n_train, :]
        df_test = dfs.iloc[n_train:, :]
        return df_train, df_test
    
    test_start_date = get_test_start_date(dfs)
    df_train = {}
    for k, v in dfs.items():
        df_train[k] = v[v.index < test_start_date]
    
    df_test = {}
    for k, v in dfs.items():
        df_test[k] = v[v.index >= test_start_date]
        
    return df_train, df_test

def get_hyperparam_combis(hyperparams):
    """
    Retrieves combinations of hyperparameters given possible values for each parameter attribute.

    :param hyperparams: a dictionary containing hyperparameter attributes as keys and 
                        list of values for the corresponding parameter to be tested as values.
    :return: a list of dictionaries, each containing keys of hyperparameter attributes 
             matched to specific values.
    """
    hyperparam_names, hyperparam_values = [], []
    for k, v in hyperparams.items():
        hyperparam_names.append(k)
        hyperparam_values.append(v)
    hyperparam_combis = list(itertools.product(*hyperparam_values))
    hyperparam_combis = [{name: v for v, name in zip(combi, hyperparam_names)} 
                         for combi in hyperparam_combis]
    return hyperparam_combis

def hms(seconds):
    """
    Generates a formatted time string in hms format given seconds.
    
    :param seconds: an integer number of seconds.
    :return: a string containing the duration in the format <hr>h<mn>m<s>s, 
             where hr, mn and s represent the number of hours, minutes and seconds in the given duration, respectivley, each only if non-zero.
    """
    hr = int(seconds // 3600)
    mn = int((seconds - hr * 3600) // 60)
    sc = seconds - hr * 3600 - mn * 60
    sc = round(sc, 2) if hr == 0 and mn == 0 and sc < 1 else int(sc)

    return ((f'{hr}h ' if hr != 0 else '')
            + (f'{mn}m ' if mn != 0 else '') 
            + f'{sc}s')

def log(statement):
    """
    Prints the given statement with the current timestamp.
    
    :param statement: a string statement to be printed with the timestamp.
    :return: None.
    """
    dt_string = datetime.now().strftime("%d/%m/%y %H:%M:%S")
    print(dt_string, '|', statement)
    
class Timer:
    """
    Serves as a utility to time and estimate the duration of a process with multiple iterations.    
    """
    def __init__(self, total_iterations, start=True):
        """
        Constructs a new Timer instance.

        :param total_iterations: an integer number of total iterations in the process 
                                 to be timed.
        :param start: a boolean indicating whether the timer is to be started immediately 
                      after the Timer is created (default set to True).
        :return: None.
        """
        self.started = False
        if start: self.start()
        self.total_elapsed = 0
        self.average_dur = None
        self.remaining_dur = None
        self.total_iterations = total_iterations
        self.curr_iteration = 0

    def start(self):
        """
        Starts the time of the current Timer or restart it with an unchanged 
        number of input iterations if it has already started.

        :return: None.
        """
        if self.started:
            self.__init__(self.total_iterations)
        self.start_time = time()
        self.prev_time = self.start_time
        self.started = True
        
    def stop(self):
        """
        Stops the time of the current Timer by reinstantiating the timer with
        an unchanged number of input iterations.

        :return: None.
        """
        self.__init__(self.total_iterations, start=False)
        
    def timestamp(self):
        """
        Laps the time for an iteration.

        Outputs the time taken for the iteration, total time elapsed, average time taken
        and estimated time remaining to complete all iterations, in this order.

        :return: None.
        """
        if not self.started:
            raise Exception('Timer not started.')
        self.curr_iteration += 1
        lap_time = time()
        iteration_time = lap_time - self.prev_time
        self.total_elapsed = lap_time - self.start_time
        self.average_dur = self.total_elapsed / self.curr_iteration
        self.remaining_dur = self.average_dur * (self.total_iterations - self.curr_iteration)
        log(f'Iter {self.curr_iteration}/{self.total_iterations} | ' 
            + f'Iter time: {hms(iteration_time)} | Elapsed: {hms(self.total_elapsed)} | ' 
            + f'Average: {hms(self.average_dur)} | Remaining: {hms(self.remaining_dur)}')
        self.prev_time = lap_time
        if self.curr_iteration == self.total_iterations:
            self.stop()
            
def execute_training_phase(model, hyperparam_combis, train_and_validate_dfs, returns, timer):
    """
    Conducts training of the given model using the input data and combinations of hyperparameters.

    :param model: an ML_Model instance for which the training phase is to be conducted.
    :param hyperparam_combis: a list of dictionaries, each containing keys of 
                              hyperparameter attributes matched to specific values.
    :param train_and_validate_dfs: a single dictionary of DataFrames with currencies as 
                                   keys mapped to DataFrames for training and validation.
    :param returns: a DataFrame containing carry trade returns for each 
                    individual currency with date indices.
    :param timer: a Timer instance for timing and estimating the durations remaining 
                  for the training.
    :return: a dictionary containing keys of hyperparameter attributes matched 
             to specific values representing the best combination of hyperparameters.
    """
    log('Executing training phase...')
    perfomance_table = pd.DataFrame(columns=['mean', 'sr', 'skew'])
    for combi_index, combi in enumerate(hyperparam_combis):
        train_dfs, validate_dfs = get_train_test(train_and_validate_dfs)
        signals = pd.DataFrame()
        error = False
        for currency, train_df in train_dfs.items():
            log(f'Training with hyperparameters combi {combi_index+1}...')
            try:
                predicted_returns = model.train(train_df, **combi).predict(train_and_validate_dfs[currency])
            except Exception as e:
                print(e)
                error = True
                break
            predicted_returns.columns = [currency]
            signals = signals.join(predicted_returns, how='outer')
            timer.timestamp()
        if error: continue
        mean, std, sr, skew, kurt, contr, prop_short, prop_long = build_portfolio(returns, signals, n='Q', k=3)
        perfomance_table.loc[combi_index] = mean, sr, skew
    
    # Retrieves model hyperparameters with the best aggregate performance.
    normalised = pd.DataFrame(MinMaxScaler().fit(perfomance_table).transform(perfomance_table))
    indicators = ((((normalised - 1)**2).sum(axis=1))**0.5)
    if len(perfomance_table['mean'] >= 0) != 0:
        indicators.where(perfomance_table['mean'] >= 0, 1)
    best_combi = hyperparam_combis[indicators.argmin()]
    log(f'Training phase complete with best combi {best_combi}.')
    
    return best_combi

def execute_testing_phase(model, combi, dfs, returns, timer, plot=False):
    """
    Conducts training of the given model using the input data and combinations of hyperparameters.

    :param model: an ML_Model instance for which the testing phase is to be conducted.
    :param combi: a dictionary containing keys of 
                  hyperparameter attributes matched to specific values.
    :param dfs: a dictionary of separate [Xs | Y] DataFrames for each currency, where Xs
                represent predictors and Y represents carry trade returns.
    :param returns: a DataFrame containing carry trade returns for each 
                    individual currency with date indices.
    :param timer: a Timer instance for timing and estimating the durations remaining 
                  for the training.
    :param plot: a boolean indicating whether a graph is to be plotted and saved 
                 (default set to False).
    :return: a tuple containing the model results in the form 
             <trained_model, combi, stats, mean_rmse>, storing a trained ML model instance, a dictionary containing a combination of hyperparameters, 
             statistics and the mean RMSE, respectively.
    """
    log('Executing testing phase...')
    train_and_validate_dfs, final_test_dfs = get_train_test(dfs)
    signals = pd.DataFrame()
    out_of_sample_rmses = []
    trained_models = {}
    for currency, train_and_validate_df in train_and_validate_dfs.items():
        log(f'Testing with currency {currency}...')
        trained_model = model.train(train_and_validate_dfs[currency], **combi)
        trained_models[currency] = trained_model
        y_test = trained_model.predict(final_test_dfs[currency])
        y_hat_test = final_test_dfs[currency].iloc[:, -1]

        y_test_df = pd.DataFrame(y_test)
        y_test_df.columns = ['y']
        y_hat_test_df = pd.DataFrame(y_hat_test)
        y_hat_test_df.columns = ['y_hat']
        y_df = y_test_df.join(y_hat_test_df)
        out_of_sample_rmses.append((mean_squared_error(y_df['y'], y_df['y_hat']))**0.5)
        predicted_returns = trained_model.predict(dfs[currency])
        predicted_returns.columns = [currency]
        signals = signals.join(predicted_returns, how='outer')
        timer.timestamp()
    stats = build_portfolio(returns, signals, n='Q', k=3, plot=plot,
                            file_name=trained_models[currency].name, save_returns=True)
    mean_rmse = np.array(out_of_sample_rmses).mean()
    log(f'Testing phase complete.')
    return trained_model, combi, stats, mean_rmse

def execute_carry_trade_returns_prediction_model(model, dfs, hyperparams={}):
    """
    Trains and narrows down on the best model for a given ML_Model instance to predict carry trade returns using the given data and combinations of hyperameters.

    :param model: an ML_Model instance using which carry trade returns are to 
                  be predicted.
    :param dfs: a dictionary of separate [Xs | Y] DataFrames for each currency, where Xs
                represent predictors and Y represents carry trade returns.
    :param hyperparams: a dictionary containing hyperparameter attributes as keys 
                        and lists of values for the corresponding parameter to be tested as dictionary values (default set to empty dictionary).
    :return: a tuple containing the model results in the form 
             <trained_model, combi, stats, mean_rmse>, storing a trained ML model instance, a dictionary containing a combination of hyperparameters, 
             statistics and the mean RMSE, respectively.
    """
    hyperparams = {k: ([v] if not hasattr(v, '__iter__') else v) for k, v in hyperparams.items()}
    train_and_validate_dfs, final_test_dfs = get_train_test(dfs)
    hyperparam_combis = get_hyperparam_combis(hyperparams)
    returns = get_returns_from_dfs(dfs)

    num_iterations = len(dfs)
    if len(hyperparam_combis) > 1:
        num_iterations = (len(hyperparam_combis) + 1) * len(dfs) 
        best_combi = execute_training_phase(model, hyperparam_combis, train_and_validate_dfs, returns, timer)
    elif len(hyperparam_combis) == 1:
        best_combi = hyperparam_combis[0]
    else:
        best_combi == {}

    timer = Timer(num_iterations)

    trained_model, combi, stats, mean_rmse = execute_testing_phase(model, best_combi, dfs, returns, timer)
    return trained_model, combi, stats, mean_rmse
