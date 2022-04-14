#!/usr/bin/env python
# coding: utf-8

# Code Author: Neo Weihong
# Last Modified: 15 April 2022

# Imports the necessary packages.
import numpy as np
import pandas as pd
from abstract_ml_model import ML_Model
from statsmodels.tsa.arima.model import ARIMA

class ARIMA_Model(ML_Model):
    """
    Implements ML_Model using ARIMA.
    """
    def __init__(self):
        """
        Constructs a new ARIMA_Model instance.

        :return: None.
        """
        self.name = 'ARIMA'
        self.train_df = None
        self.model = None
        
    def train(self, df, **kwargs):
        """
        Trains the ARIMA_Model using the given data and hyperparameters.

        :param df: a DataFrame having columns [Xs | Y], where Xs are 
                   predictors and Y is response variable containing carry trade returns.
        :param kwargs: variable keyword arguments to be passed in as parameters or 
                       hyperparameters of the model.
        :return: a trained ARIMA_Model instance using the given map of currencies 
                 to dataframes.
        """
        order = [1, 1, 1]
        for i, order_var in enumerate(['p', 'd', 'q']):
            if order_var in kwargs.keys():
                order[i] = kwargs[order_var]
        order = tuple(order)

        train_data = df.iloc[:, -1]
        self.train_df = train_data
        model = ARIMA(train_data.values, order=order)
        model_fit = model.fit()
        self.model = model_fit
        return self
            
    def predict(self, test_X):
        """
        Predicts carry trade returns for given data using a trained ARIMA_Model instance.

        :param df: a DataFrame having columns [Xs] for features.
        :return: a DataFrame with a single column containing data for the predicted response carry trade returns.
        """
        forecast = self.model.predict(start=len(self.train_df.values), 
                                     end=len(self.train_df.values)+len(test_X)-1)
        forecast = forecast.copy()
        return pd.DataFrame(forecast, columns=['y_hat'], index=test_X.index)
