#!/usr/bin/env python
# coding: utf-8

# Code Author: Neo Weihong
# Last Modified: 15 April 2022

# Imports the necessary packages.
import numpy as np
import pandas as pd
from abstract_ml_model import ML_Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

class LSTM_Model(ML_Model):
    """
    Implements ML_Model using LSTM.

    With reference to: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras
    """ 
    def __init__(self):
        """
        Constructs a new LSTM_Model instance.

        :return: None.
        """
        self.name = 'LSTM'
        self.model = None
        self.scaler = None
        self.lookback = None
        self.nfeatures = None
        self.training_df = None
    
    def __series_to_supervised(self, df, n_in=1, n_out=1, dropnan=True, index=None):
        """
        Shifts the data for predictors to obtain lookback (and possibly look forward) values for predictors.

        :param df: a n x m DataFrame having columns [Xs | Y], where Xs are 
                   predictors at time t-n_steps and Y is response variable at time t containing carry trade returns.
        :param n_in: an integer number of lookback periods for LSTM to 
                     utilise for training (default set to 1).
        :param n_out: an integer number of look forward periods for generalisability 
                      (default set to 1).
        :dropnan: a boolean indicating whether to drop missing values in the 
                  output DataFrame (default set to True).
        :index: a list containing the indices to be used to set the n indices for the 
                output DataFrame. If set to None, default integer indices will be used (default set to None).
        :return: a n x (m x (n_in + n_out)) DataFrame containing shifted data.
        """        
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [f'{j}_minus_{i}' for j in df.columns]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [j for j in df.columns]
            else:
                names += [f'{j}_plus_{i}' for j in df.columns]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        agg.index = df.index if index is None else index
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
    def __run_ML(self, **kwargs):
        """
        Builds the machine learning model with the given hyperparameters.

        :param kwargs: variable keyword arguments to be passed in as parameters or 
                       hyperparameters of the model.
        :return: the trained LSTM model.
        """
        # Defines the LSTM model.
        train_X, train_y = self.train_X, self.train_y
        model = Sequential()
        model.add(LSTM(50, dropout=0.1, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        
        # Fits the model using the stored prepared training data.
        history = model.fit(train_X, train_y, 
                            verbose=0, shuffle=False, **kwargs)
        return model
    
    def __get_reframed_df(self, df, lookback=None):
        """
        Scales and reshapes and prepares the data for training by LSTM.

        :param df: a n x m DataFrame having columns [Xs | Y], where Xs are 
                   predictors and Y is response variable containing carry trade returns.
        :param lookback: the number of lookback periods for LSTM to utilise for training.
        :return: a tuple containning a reshaped n x lookback x (m-1) array for 
                 prediction, an array of length n for response variables, an integer containing m- and the reshaped DataFrame.
        """
        df = df.astype('float32')
        nfeatures = len(df.columns) if self.nfeatures is None else self.nfeatures 
        if self.scaler is None:
            self.scaler = MinMaxScaler()
        if self.lookback is None:
            assert lookback is not None
            self.lookback = lookback
        
        if self.training_df is not None:
            combined = pd.concat([self.training_df, df])
        else:
            combined = df
        scaled = pd.DataFrame(self.scaler.fit_transform(combined), columns = df.columns)
        if self.training_df is not None:
            scaled = scaled.iloc[len(self.training_df):]
        reframed = self.__series_to_supervised(scaled, self.lookback, 1, index=df.index)
        reframed = reframed[[i for i in reframed.columns if 'minus' in i or 'zero' in i]]
        assert len(reframed.columns) == nfeatures * self.lookback + 1

        train_X, train_y = reframed.values[:, :-1], reframed.values[:, -1]
        train_X = train_X.reshape((train_X.shape[0], self.lookback, nfeatures))
        return train_X, train_y, nfeatures, reframed
    
    def train(self, df, **kwargs):
        """
        Trains the LSTM_Model using the given data and hyperparameters.

        :param df: a DataFrame having columns [Xs | Y], where Xs are 
                   predictors and Y is response variable containing carry trade returns.
        :param kwargs: variable keyword arguments to be passed in as parameters or 
                       hyperparameters of the model.
        :return: a trained LSTM_Model instance using the given map of currencies 
                 to dataframes.
        """
        self.training_df = df
        if 'lookback' in kwargs.keys():
            lookback = kwargs['lookback']
            self.lookback = lookback
            del kwargs['lookback']      
        else:
            lookback = 1
        self.train_X, self.train_y, self.nfeatures, reframed = self.__get_reframed_df(df, lookback)
        self.model = self.__run_ML(**kwargs)
        return self
            
    def predict(self, test_X):
        """
        Predicts carry trade returns for given data using a trained LSTM_Model instance.

        :param df: a DataFrame having columns [Xs] for features.
        :return: a DataFrame with a single column containing data for the predicted response carry trade returns.
        """
        if self.model is None:
            raise Exception('Model has not been trained!')
            
        if len(test_X.columns) == self.nfeatures - 1:
            test_X['zspot'] = np.nan # dummy column as the last column
        
        test_X_input, dummy_y, nfeatures, reframed = self.__get_reframed_df(test_X, self.lookback)
        scaler = self.scaler

        # make a prediction
        yhat = self.model.predict(test_X_input)
        inv_yhat = np.broadcast_to(yhat, (yhat.shape[0], scaler.n_features_in_))
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,-1]
        inv_yhat = pd.DataFrame(inv_yhat, index=reframed.index, columns = ['y_hat'])

        return inv_yhat
