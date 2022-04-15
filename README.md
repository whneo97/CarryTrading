# Project Premise

Carry trades have traditionally been determined by interest rate differentials and exchange rates. However, long-standing difficulties in assessing the behaviour of the forward premium and exchange rate changes have rendered carry trade returns challenging to predict and subject to crash risk. This project first confirmed that carry trades remained profitable for G10 currencies using recent data and investigated the effects of negative interest rates on carry trade return using simple linear regressions. Thereafter, the performances of machine learning models in constructing equal-weighted portfolios were compared against the traditional portfolio for quarterly data of the CHF, EUR, JPY, GBP, NOK and SEK from 1997 to 2022. The traditional portfolio took long positions for the top 3 currencies (with the top 3 interest rates) and the bottom 3 currencies (with the bottom 3 interest rates) every quarter from 1997 to 2022.

# Implementation

ARIMA, CNN and LSTM were used to construct equal-weighted portfolios for comparison with the traditional portfolio. In addition to these models, a framework was also provided to build other machine learning models for the purpose of predicting carry trade returns. Unlike machine learning models that use the RMSE to tweak the models for optimised performance, the models in this setup were tweaked using measures of performance for the portfolio themselves, including mean returns, Sharpe ratio and negative skewness that were normalised between 0 to 1, with the Euclidean distance from (1, 1, 1) used to balance these performance measures. [abstract_ml_model.py](abstract_ml_model.py) contains the functions train and predict that can be used to build `ML_Model` classes. Instances of these models were passed into [model_engine.py](model_engine.py) to run the various models, including the building of the traditional portfolio. [model_executor.py](model_executor.py) was then used to run the training and execution sequence, which compared portfolio results using [portfolio_builder.py](portfolio_builder.py). The framework was created to be compartmentalised and easily extendible, such that should more machine learning models be required, additional `ML_Model` classes from [abstract_ml_model.py](abstract_ml_model.py) can be implemented.

# Key Finding

Carry trade returns were still found to be positive, with trends UIP still continuing to be violated for recent data. The traditional portfolio was found to exhibit negative skewness of returns, as past studies have shown. Machine learning methods, particularly LSTM, were shown to have the potential to construct a portfolio with positive returns and mitigate such negative skewness of returns. The methodology that utilised portfolio-based performance measures as metrics for models can be further explored for future research and may also be applicable to other financial instruments.
