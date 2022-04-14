#!/usr/bin/env python
# coding: utf-8

# Code Author: Neo Weihong
# Last Modified: 15 April 2022

# Imports the necessary packages.
from abc import ABC, abstractmethod
 
class ML_Model(ABC):
    """
    Serves as a template for other models to be built and trained
    for the prediction of carry trade returns.
    """
    @abstractmethod
    def train(self, df, **kwargs):
        """
        Trains the ML_Model using the given data and hyperparameters.

        :param df: a DataFrame having columns [Xs | Y], where Xs are 
                   predictors and Y is response variable containing carry trade returns.
        :param kwargs: variable keyword arguments to be passed in as parameters or 
                       hyperparameters of the model.
        :return: a trained ML_Model instance using the given map of currencies 
                 to dataframes.
        """
        pass
    
    @abstractmethod    
    def predict(self, df):
        """
        Predicts carry trade returns for given data using a trained ML_Model instance.

        :param df: a DataFrame having columns [Xs] for features.
        :return: a DataFrame with a single column containing data for the predicted response carry trade returns.
        """
        pass
