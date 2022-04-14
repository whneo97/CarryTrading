#!/usr/bin/env python
# coding: utf-8

# Code Author: Neo Weihong
# Last Modified: 15 April 2022

# Imports the necessary packages.
import numpy as np
import pandas as pd
from abstract_ml_model import ML_Model
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

class CNN_Model(ML_Model):
    """
    Implements ML_Model using CNN.

    With reference to: https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting
    """ 
    def __init__(self):
        """
        Constructs a new CNN_Model instance.

        :return: None.
        """
        self.name = 'CNN'
        self.model = None
        self.n_steps = None
        self.nfeatures = None
    
    def __split_sequences(self, sequences, n_steps):
        """
        Splits the data into groups of 3D predictor and 1D response variables for CNN.

        Each observation, instead of covering only data for a single date, now covers data for n_steps number of dates, mapped to corresponding carry trade returns.

        :param sequences: a n x m DataFrame having columns [Xs | Y], where Xs are 
                          predictors at time t-n_steps and Y is response variable at time t containing carry trade returns.
        :param n_steps: the number of lookback periods for CNN to utilise for training.
        :return: a pair containing arrays for (n-n_step+1) x (n_step) x (m-1) predictor 
                 and (n-n_step+1) response variables, respectively.
        """
        X, y = list(), list()
        for i in range(len(sequences)):
            end_ix = i + n_steps
            if end_ix > len(sequences): break
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)
    
    def train(self, df, **kwargs):
        """
        Trains the CNN_Model using the given data and hyperparameters.

        :param df: a DataFrame having columns [Xs | Y], where Xs are 
                   predictors and Y is response variable containing carry trade returns.
        :param kwargs: variable keyword arguments to be passed in as parameters or 
                       hyperparameters of the model.
        :return: a trained CNN_Model instance using the given map of currencies 
                 to dataframes.
        """
        self.model = 1

#       # Selects a number of time steps to lookback for the model.
        if 'lookback' in kwargs.keys():
            n_steps = kwargs['lookback']
            self.n_steps = n_steps
            del kwargs['lookback']      
        else:
            n_steps = 1
        train = df
        dataset = train.iloc[:, :-1].shift(-n_steps).join(train.iloc[:, -1]).dropna().values

        # Prepares the data into training by CNN.
        X, y = self.__split_sequences(dataset, n_steps)
        self.nfeatures = X.shape[2]

        # Defines the CNN model.
        model = Sequential()
        model.add(Conv1D(activation='relu', input_shape=(n_steps, self.nfeatures), **kwargs))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        # Fits the model using the given prepared training data.
        model.fit(X, y, epochs=1000, verbose=0)
        self.model = model
        return self
    
    def predict(self, test_X):
        """
        Predicts carry trade returns for given data using a trained CNN_Model instance.

        :param df: a DataFrame having columns [Xs] for features.
        :return: a DataFrame with a single column containing data for the predicted response carry trade returns.
        """
        if self.model is None:
            raise Exception('Model has not been trained!')
            
        if len(test_X.columns) == self.nfeatures - 1:
            test_X['zspot'] = np.nan # dummy column as the last column
            
        test_x = test_X.shift(-self.n_steps).dropna()
        x_input = self.__split_sequences(test_x.values, self.n_steps)[0]
        yhat = pd.DataFrame(self.model.predict(x_input, verbose=0), 
                            index=test_x.index[:len(x_input)], columns=['y_hat'])
        return yhat
