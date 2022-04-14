#!/usr/bin/env python
# coding: utf-8

# Code Author: Neo Weihong
# Last Modified: 15 April 2022

# This Python file contains code that trains data and build optimal portfolios using the following models
# 1. Traditional Model: Every quarter during the sampling period, take long positions for currencies with 
# on top 3 interest rates and short currences with the bottom 3 interest rates.
# 2. Full Long: Every quarter during the sampling period, take long positions for all currencies.
# 3. Full Short: Every quarter during the sampling period, take short positions for all currencies.

# For ML models, every quarter during the sampling period, take long positions for currencies with 
# on top 3 carry trade returns predicted using <ML_Model> and short currences with the bottom 3 carry trade 
# returns predicted using <ML_Model>, where <ML_Model> refers to the following models.
# 4. ARIMA=
# 5. CNN
# 6. LSTM

# Sampling period: 1997-2022
# Interval: Quarter
# Number of Long Positions taken: 3 (except for Full Long and Full Short portfolio)
# Number of Short Positions taken: 3 (except for Full Long and Full Short portfolio)
# Currencies used: CHF, EUR, GBP, JPY, NOK, SEK

# Imports the necessary packages.
import keras
import os
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime
from portfolio_builder import get_returns_from_dfs, build_portfolio
from lstm_model import LSTM_Model
from cnn_model import CNN_Model
from arima_model import ARIMA_Model
from model_executor import execute_carry_trade_returns_prediction_model

# Loads quarterly data for:
# 1. Carry trade predictors (interest rate differentials and variables of uncertainty)
# and carry trade returns.
# 2. Interest rates at time t-1 to be used as signals for the traditional portfolio.
print('Reading model inputs...')
with open('dfs.pkl', 'rb') as f:
    dfs = pickle.load(f)
    dfs = {k: v for k, v in dfs.items() if k not in ['AUD', 'CAD', 'NZD']}
with open('i_signal.pkl', 'rb') as f:
    i_signal = pickle.load(f)
    i_signal.columns = [i.split('_')[0] for i in i_signal.columns]
    i_signal = i_signal[[i for i in i_signal.columns if i not in ['AUD', 'CAD', 'NZD', 'USD']]]

# Initialises the list of G10 currency three-letter codes.
print('Initialising currencies...')
currencies = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK', 'USD']

# Retrieves a single DataFrame contaning carry trade returns from dictionary of 
# separate [Xs | Y] DataFrames for each currency, where Xs represent predictors
# and Y represents carry trade returns.
returns = get_returns_from_dfs(dfs)

# Creates a directory to store machine learning results
# if it does not already exist.
results_file_dir = 'ml_results'
if not os.path.exists(results_file_dir):
    os.makedirs(results_file_dir)
    
# Initiliases filenames of model and results pickle file to be saved.
model_filename = 'model'
results_filename = 'results.pkl'

# Defines portfolio constant names.
print('Defining portfolio names...')
TRADITIONAL = 'traditional'
FULL_LONG = 'full_long'
FULL_SHORT = 'full_short'
ARIMA = 'ARIMA'
CNN = 'CNN'
LSTM = 'LSTM'

# Stores names of existing portfolios.
portfolios = [TRADITIONAL, FULL_LONG, FULL_SHORT, ARIMA, CNN, LSTM]

# Stores names of machine learning portfolios.
ml_portfolio = [ARIMA, CNN, LSTM]

# Maps each machine learnign portfolio name to the corresponding ML Model.
ml_model_map = {ARIMA: ARIMA_Model, 
                CNN: CNN_Model, 
                LSTM: LSTM_Model}

# Initialises a map of hypermarameter combinations for each ML model to be used for testing.
# Smaller ranges of values may be used to speed up testing.
print('Mapping portfolio hyperparameters...')
hyperparams_map = {ARIMA: {'p': [i for i in range(1, 11)], 
                           'd': [i for i in range(1, 6)], 
                           'q': [i for i in range(1, 6)]}, 
                   CNN: {'filters': [2**i for i in range(5, 10)], 
                         'kernel_size': [2, 3, 5], 
                         'lookback': [3, 4, 5]}, 
                   LSTM: {'epochs': [i*100 for i in range(1, 6)], 
                          'batch_size': [2**i for i in range(5, 8)], 
                          'lookback': [i for i in range(1, 5)]}}

def save_model_results(model_name, results):
    """
    Saves the given model and results under a directory named by the model and timestamp.
    
    :param model_name: a string indiating the name of the model to be saved.
    :param results: a tuple containing the model results in the form <trained_model, combi, stats, mean_rmse>, 
                    storing an ML model object, dictionary containing a combination of hyperparameters, 
                    statistics and the mean RMSE, respectively.
    :return: the name of the directory that the model and results are saved in.
    """
    curr_time_stamp = datetime.now().strftime("%d_%m_%y_%H%M%S")
    model_dir = f'{results_file_dir}/{model_name}_{curr_time_stamp}'
    trained_model, combi, stats, mean_rmse = results
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if isinstance(trained_model.model, keras.engine.training.Model):
        tf.keras.models.save_model(trained_model.model, f'{model_dir}/{model_filename}')
        trained_model.model = None
    with open(f'{model_dir}/{results_filename}', 'wb') as f:
        pickle.dump(results, f)
    return model_dir

def get_model_results(model_dir):
    """
    Retrieves the saved model and results from the given model directory.

    :param model_dir: a string storing the file path to the directory containing the model and results.
    :return: a model result tuple in the form <trained_model, combi, stats, mean_rmse>, containing an 
             ML model object, dictionary containing a combination of hyperparameters, statistics and 
             the mean RMSE, respectively.
    """
    with open(f'{model_dir}/{results_filename}', 'rb') as f:
        results = pickle.load(f)
    trained_model, combi, stats, mean_rmse = results
    if trained_model.model is None:
        trained_model.model = tf.keras.models.load_model(f'{model_dir}/{model_filename}')
    return trained_model, combi, stats, mean_rmse

def load_and_run_model(model_dir, dfs, save=False):
    """
    Loads and runs a pre-trained and saved model from the given model directory on the given DataFrames.

    :param model_dir: a string storing the file path to the directory containing the model and results.
    :param dfs: a dictionary of DataFrames for each currency, each in the form [Xs | Y], where Xs are 
                predictors and Y is response variable containing carry trade returns.
    :param save: a boolean that is True if the new model results are to be saved and False otherwise.
                 (default set to False)
    :return: results of the newly-trained model, in the form <trained_model, combi, stats, mean_rmse>, 
             containing an ML model object, dictionary containing a combination of hyperparameters, 
             statistics and the mean RMSE, respectively.
    """
    trained_model, combi, stats, mean_rmse = get_model_results(model_dir)
    model_name = trained_model.name if trained_model.name is not None else 'model'
    new_results = execute_carry_trade_returns_prediction_model(trained_model, dfs, hyperparams=combi)
    if save: save_model_results(model_name, results)
    return new_results

def build_model_and_save_results(name, dfs, hyperparams=None):
    """
    Builds a portfolio and saves the results of the portfolio in the local directory.
    
    If the portfolio is to be trained using ML model, also save the model along with the results,
    in the form of the tuple <trained_model, combi, stats, mean_rmse>, containing an ML model object, 
    dictionary containing a combination of hyperparameters, statistics and the mean RMSE, respectively
    
    :param name: a string storing a name of the portfolio from the global portfolio.
    :param dfs: a dictionary of DataFrames for each currency, each in the form [Xs | Y], where Xs are 
                predictors and Y is response variable containing carry trade returns.
    :param hyperparams: a dictionary containing hyperparameter attributes as keys and list of values for 
                        the corresponding parameter to be tested as values (default set to None).
    :return: None.
    :throw: an Exception if the given name of the portfolio is not found in the global portfolio.
    """
    if name not in portfolios:
        raise Exception(f'Portfolio {name} not defined.')
    weight_spec = name if name in [FULL_LONG, FULL_SHORT] else None
    print(f'Building {name} portfolio...')

    # Retrieves model results.
    if name in ml_portfolio:
        model = ml_model_map[name]
        results = execute_carry_trade_returns_prediction_model(model(), dfs, hyperparams=hyperparams)
        trained_model, combi, stats, mean_rmse = results        
        save_model_results(name, results)
    else:
        stats = build_portfolio(returns, i_signal, n='Q', k=3, start=None, end=None, plot=False, 
                                file_name=f'{results_file_dir}/{name}', save_returns=True, weight_spec=weight_spec)
    
    # Stores model statistics.
    mean, std, sr, skew, kurt, contr, prop_short, prop_long = stats
    statistics = pd.DataFrame({f'{name}': [mean, std, sr, skew, kurt]})
    contr.to_csv(f'{results_file_dir}/{name}_contr.csv')
    prop_short.to_csv(f'{results_file_dir}/{name}_prop_short.csv')
    prop_long.to_csv(f'{results_file_dir}/{name}_prop_long.csv')
    statistics.to_csv(f'{results_file_dir}/{name}_stats.csv')

# Builds portfolios for the following and save results and/or models:
# 1. Traditional
# 2. Full Long
# 3. Full Short
# 4. ARIMA
# 5. CNN
# 6. LSTM
for portfolio_name in portfolios:
    print(f'Building portfolio and storing results for {portfolio_name}...')
    build_model_and_save_results(portfolio_name, dfs, hyperparams_map.get(portfolio_name))
    print()
print(f'Building of portfolio and saving of results complete.')    
