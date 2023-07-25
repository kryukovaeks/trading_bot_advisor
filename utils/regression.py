#
# Python Module with Class
# for Vectorized Backtesting
# of Machine Learning-based Strategies
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#

import yfinance as yf

import numpy as np
import pandas as pd
from sklearn import linear_model

import random
from sklearn.preprocessing import StandardScaler, Normalizer,MinMaxScaler

class ScikitVectorBacktester(object):
    ''' Class for the vectorized backtesting of
    Machine Learning-based trading strategies.

    Attributes
    ==========
    symbol: str
        TR RIC (financial instrument) to work with
    start: str
        start date for data selection
    end: str
        end date for data selection
    amount: int, float
        amount to be invested at the beginning
    tc: float
        proportional transaction costs (e.g. 0.5% = 0.005) per trade
    model: str
        either 'regression' or 'logistic'

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    select_data:
        selects a sub-set of the data
    prepare_features:
        prepares the features data for the model fitting
    fit_model:
        implements the fitting step
    run_strategy:
        runs the backtest for the regression-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    '''

    def __init__(self, symbol, amount, tc, model,period=1, no_exp=True, std=True, lags=3,  feat=['lags','serial_mean_return_lags', 'ewa']):
        self.symbol = symbol
        #self.start = start
        #self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        self.lags = lags
        self.feat = feat
        self.std = std
        self.no_exp = no_exp
        self.period = period
        
    
        if model == 'linear regression':
            self.model = linear_model.LinearRegression()
        elif model == 'logistic regression':
            self.model = linear_model.LogisticRegression()
                #C=1e6,solver='lbfgs', multi_class='ovr', max_iter=1000)
        elif model == 'DecisionTreeClassifier':
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier()
        elif model == 'simple dnn':
            import tensorflow as tf
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.optimizers import adam_v2

            optimizer = adam_v2.Adam(learning_rate=0.0001)
            def set_seeds(seed=100):
                random.seed(seed)
                np.random.seed(seed)
                tf.random.set_seed(100)
            set_seeds()
            model = Sequential()
            model.add(Dense(64, activation='relu',
                    input_shape=(self.lags*len(self.feat),)))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(1, activation='sigmoid')) # <5>
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            self.model = model
            
        elif model == 'improved dnn':
            from sklearn.preprocessing import StandardScaler
            from keras.models import Sequential
            from keras.layers import Dense, Dropout, BatchNormalization, Activation
            from tensorflow.keras.optimizers import Adam

            from keras.callbacks import EarlyStopping
            model = Sequential()
            model.add(Dense(128, activation='relu', input_shape=(self.lags*len(self.feat),)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))

            # Compile the model with binary crossentropy loss and Adam optimizer
            optimizer = Adam(lr=0.001)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # Use early stopping to prevent overfitting
            early_stop = EarlyStopping(monitor='val_loss', patience=10)
            self.model = model

        elif model == 'lstm':
            from tensorflow.keras.layers import LSTM
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam


            optimizer = adam_v2.Adam(learning_rate=0.0001)
            def set_seeds(seed=100):
                random.seed(seed)
                np.random.seed(seed)
                tf.random.set_seed(100)
            set_seeds()
            model = Sequential()
            model.add(LSTM(units=64, activation='relu',
                    input_shape=(self.lags*len(self.feat), 1)))
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            self.model = model

        elif model == 'lstm_2':

            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout


            optimizer = adam_v2.Adam(learning_rate=0.0001)
            def set_seeds(seed=100):
                random.seed(seed)
                np.random.seed(seed)
                tf.random.set_seed(100)
            set_seeds()

            model = Sequential()
            model.add(LSTM(units=128, input_shape=(self.lags*len(self.feat), 1), return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=64, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=32))
            model.add(Dropout(0.2))
            model.add(Dense(units=1, activation='sigmoid'))

            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            self.model = model



        else:
            raise ValueError('Model not known or not yet implemented.')
        self.get_data()

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        
        stock = yf.Ticker(self.symbol)
        # Use the Ticker object to fetch historical data for the stock
        historical_data = stock.history(period='max') #start = self.start, end = self.end)
        
        raw = historical_data['Close'].reset_index().rename(columns={'Close': 'price'})
        raw['returns'] = np.log(raw['price'] / raw['price'].shift(self.period))
        self.data = raw.dropna().set_index('Date')

    def select_data(self, start=None, end=None):
        ''' Selects sub-sets of the financial data.
        '''
        if start == None:
            start = self.data.index.min()
        if end == None:
            end = self.data.index.max()
        data = self.data[(self.data.index >= start) &
                         (self.data.index <= end)].copy()
        return data

    def prepare_features(self,start, end):
        ''' Prepares the feature columns for the regression and prediction steps.
        '''

        self.data_subset = self.select_data(start, end)
        self.feature_columns = []
        for lag in range(1, self.lags + 1):
            if 'lags' in self.feat:
                col = 'lag_{}'.format(lag)
                
                self.data_subset[col] = self.data_subset['returns'].shift(lag)
                self.feature_columns.append(col)
            
                
            if 'serial_mean_return_lags' in self.feat:
                col = 'lag_{}'.format(lag)
                col_2 = 'serial_mean_return_lag_{}'.format(lag)
                if self.no_exp is True:
                    self.data_subset[col] = self.data_subset['returns'].apply(np.exp).shift(lag)
                self.data_subset[col_2] = self.data_subset[col] - self.data_subset[col].rolling(self.lags).mean()
                self.data_subset[col_2] = self.data_subset[col_2]/self.data_subset[col_2].rolling(window=self.lags).std()
                self.feature_columns.append(col_2)
            if ('serial_mean_return_lags' in self.feat) & ('ewa' in self.feat):
                col_3 = 'ewa_lag_{}'.format(lag)
                self.data_subset[col_3] = self.data_subset[col_2].ewm(span=self.lags).mean()
                self.feature_columns.append(col_3)
            if ('serial_mean_return_lags' in self.feat) & ('ewa' in self.feat) & ('ewd' in self.feat):
                col_4 = 'ewd_lag_{}'.format(lag)
                self.data_subset[col_4] = self.data_subset[col_2].ewm(span=self.lags).std()
                self.feature_columns.append(col_4)
            if ('serial_mean_return_lags' in self.feat) & ('rsi' in self.feat):
                from ta.momentum import RSIIndicator
                col_5 = 'ewd_lag_{}'.format(lag)
                rsi_indicator = RSIIndicator(self.data_subset[col_2], window=self.lags)
                self.data_subset[col_5] = rsi_indicator.rsi()
                self.feature_columns.append(col_5)

            

        self.data_subset.dropna(inplace=True)
        self.y = np.sign(self.data_subset['returns'])
        
    def fit_model(self, start, end):
        ''' Implements the fitting step.
        '''
        
        #scaler
        if self.std is True:
            self.mu, self.std = self.data_subset.mean(), self.data_subset.std()
            self.data_subset = (self.data_subset - self.mu) / self.std
        st.write(self.feature_columns)
        self.model.fit(self.data_subset[self.feature_columns],
                       self.y)

    def run_strategy(self, start_in, end_in, start_out, end_out):
        ''' Backtests the trading strategy.
        '''
        self.prepare_features(start_in, end_in)
        self.fit_model(start_in, end_in)
        # data = self.select_data(start_out, end_out)
        
        
        self.prepare_features(start_out, end_out) #test data
        if self.std is True:
            self.data_subset = (self.data_subset - self.mu) / self.std
        
        prediction = self.model.predict(self.data_subset[self.feature_columns])
        from sklearn.metrics import accuracy_score
        if self.std is True:
            self.data_subset = self.data_subset * self.std + self.mu
        self.data_subset['prediction'] = np.where(prediction>0,1,-1)
        st.write('mean_of_going up: ',self.data_subset[np.sign(self.data_subset['returns'])==1].shape[0]/self.data_subset.shape[0])
        st.write(self.data_subset[['prediction','returns']])
        st.write(accuracy_score(self.data_subset['prediction'], self.y))
        self.data_subset['strategy'] = (self.data_subset['prediction'] *
                                        self.data_subset['returns'])
        # determine when a trade takes place
        trades = self.data_subset['prediction'].diff().fillna(0) != 0
        
        # subtract transaction costs from return when trade takes place
        self.data_subset['strategy'][trades] -= self.tc
        self.data_subset['creturns'] = (self.amount *
                        self.data_subset['returns'].cumsum().apply(np.exp))
        self.data_subset['cstrategy'] = (self.amount *
                        self.data_subset['strategy'].cumsum().apply(np.exp))
        self.results = self.data_subset
        
        # absolute performance of the strategy
        aperf = self.results['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - self.results['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results is None:
            st.write('No results to plot yet. Run a strategy.')
        title = '%s | TC = %.4f' % (self.symbol, self.tc)
        st.line_chart(self.results[['creturns', 'cstrategy']], title=title)  




