#
# Python Script with Long Only Class
# for Event-based Backtesting
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
from utils.BacktestBase import *
import streamlit as st
import plotly.graph_objects as go
class BacktestLongOnly(BacktestBase):
        
    def run_sma_strategy(self, SMA1, SMA2):
        ''' Backtesting a SMA-based strategy.

        Parameters
        ==========
        SMA1, SMA2: int
            shorter and longer term simple moving average (in days)
        '''
        msg = f'\n\nRunning SMA strategy | SMA1={SMA1} & SMA2={SMA2}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        st.markdown(msg)
        st.markdown('=' * 55)
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        self.data['SMA1'] = self.data['price'].rolling(SMA1).mean()
        self.data['SMA2'] = self.data['price'].rolling(SMA2).mean()

        for bar in range(SMA2, len(self.data)):
            if self.position == 0:
                if self.data['SMA1'].iloc[bar] > self.data['SMA2'].iloc[bar]:
                    self.place_buy_order(bar, amount=self.amount)

                    self.position = 1  # long position
                    price_entry = self.data['price'].iloc[bar]
                    
            elif self.position == 1:
                if (self.data['SMA1'].iloc[bar] < self.data['SMA2'].iloc[bar]) & (price_entry<self.data['price'].iloc[bar]):
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
        self.close_out(bar)
    def run_sma_improved_strategy(self, SMA1, SMA2):
        ''' Backtesting a SMA-based strategy.

        Parameters
        ==========
        SMA1, SMA2: int
            shorter and longer term simple moving average (in days)
        '''
        msg = f'\n\nRunning SMA strategy | SMA1={SMA1} & SMA2={SMA2}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        st.markdown(msg)
        st.markdown('=' * 55)
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        self.data['SMA1'] = self.data['price'].rolling(SMA1).mean()
        self.data['SMA2'] = self.data['price'].rolling(SMA2).mean()

        for bar in range(SMA2, len(self.data)):
            if self.position == 0:
                if (self.data['SMA1'].iloc[bar] < self.data['SMA2'].iloc[bar]) & \
                (self.data['SMA1'].iloc[bar-1]<self.data['SMA1'].iloc[bar]) &\
                 (self.data['SMA1'].iloc[bar-2]<self.data['SMA1'].iloc[bar-1]):
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1  # long position
                    price_entry = self.data['price'].iloc[bar]
                    
            elif self.position == 1:
                if (self.data['SMA1'].iloc[bar] > self.data['SMA2'].iloc[bar]) & \
                (self.data['SMA1'].iloc[bar]<self.data['SMA1'].iloc[bar-1] ) \
                & (self.data['SMA1'].iloc[bar-1]<self.data['SMA1'].iloc[bar-2]):
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
        self.close_out(bar)        
        
    def run_momentum_strategy(self, momentum, hold):
        ''' Backtesting a momentum-based strategy.

        Parameters
        ==========
        momentum: int
            number of days for mean return calculation
        '''
        self.buy_dates = []
        self.sell_dates = []
        self.buy_prices = []
        self.sell_prices = []

        msg = f'\n\nRunning momentum strategy | {momentum} days'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        st.markdown(msg)
        st.markdown('=' * 55)
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        self.data['momentum'] = self.data['return'].rolling(momentum).mean()
        for bar in range(momentum, len(self.data)):
            if self.position == 0:
                if self.data['momentum'].iloc[bar] > 0:
                    self.place_buy_order(bar, amount=self.amount)
                    self.buy_dates.append(self.data.index[bar])  
                    self.buy_prices.append(self.data['price'].iloc[bar]) 
                    self.position = 1  # long position
                    price_entry = self.data['price'].iloc[bar]
                    bar_entry = bar
            elif self.position == 1:
                if (self.data['momentum'].iloc[bar]) < 0 & (price_entry<self.data['price'].iloc[bar]) & (bar>bar_entry+hold):
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
                    self.sell_dates.append(self.data.index[bar])  
                    self.sell_prices.append(self.data['price'].iloc[bar]) 
        self.close_out(bar)

        fig = go.Figure()

        # Plotting the Price data
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['price'],
                                mode='lines',
                                name='Price', 
                                line=dict(color='blue', width=2, dash='solid'),
                                opacity=0.6))

        # Plotting the buy points
        fig.add_trace(go.Scatter(x=self.buy_dates, y=self.buy_prices,
                                mode='markers',
                                marker=dict(symbol='triangle-up', size=10, color='green'),
                                name='Buy Signal'))

        # Plotting the sell points
        fig.add_trace(go.Scatter(x=self.sell_dates, y=self.sell_prices,
                                mode='markers',
                                marker=dict(symbol='triangle-down', size=10, color='red'),
                                name='Sell Signal'))

        # Update layout to match the given matplotlib's appearance
        fig.update_layout(title='Price and Buy/Sell Signals with Momentum Strategy',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        template="plotly_white",
                        showlegend=True)

        # Show the figure in Streamlit
        st.plotly_chart(fig)



    def run_mean_reversion_strategy_Richmond(self,SMA=200):
        #https://www.richmondquant.com/news/2018/11/30/using-mean-reversion-techniques-to-profit-in-volatile-markets
        '''Backtesting a run_mean_reversion_strategy_Richmond'''
        st.markdown("\nRunning run_mean_reversion_strategy_Richmond")
        st.markdown("=" * 55)
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        self.data['SMA'] = self.data['price'].rolling(window=SMA).mean()
        #self.data['4-Day Decline'] = self.data['price'].rolling(window=4).apply(lambda x: x[0] > x[-1]).fillna(0)
        for bar in range(SMA, len(self.data)):
            if self.position == 0:
                if self.data['price'].iloc[bar] > self.data['SMA'].iloc[bar] and \
                   self.data['price'].iloc[bar]<self.data['price'].iloc[bar-1] and\
                self.data['price'].iloc[bar-1]<self.data['price'].iloc[bar-2] and\
                self.data['price'].iloc[bar-2]<self.data['price'].iloc[bar-3]:
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1  # long position
                    price_entry = self.data['price'].iloc[bar]
            elif self.position == 1:
                if self.data['price'].iloc[bar]/price_entry-1>0.03:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
        self.close_out(bar)

    def run_mean_reversion_strategy(self, SMA, threshold):
        ''' Backtesting a mean reversion-based strategy.

        Parameters
        ==========
        SMA: int
            simple moving average in days
        threshold: float
            absolute value for deviation-based signal relative to SMA
        '''
        msg = f'\n\nRunning mean reversion strategy | '
        msg += f'SMA={SMA} & thr={threshold}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        st.markdown(msg)
        st.markdown('=' * 55)
        self.position = 0
        self.trades = 0
        self.amount = self.initial_amount

        self.data['SMA'] = self.data['price'].rolling(SMA).mean()

        for bar in range(SMA, len(self.data)):
            if self.position == 0:
                if (self.data['price'].iloc[bar] <
                        self.data['SMA'].iloc[bar] - threshold):
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1
                    price_entry = self.data['price'].iloc[bar]


            elif self.position == 1:
                if (self.data['price'].iloc[bar] >= self.data['SMA'].iloc[bar])& (price_entry<self.data['price'].iloc[bar]):
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0
        self.close_out(bar)


    def run_enhanced_momentum_strategy(self, momentum, hold, ma_period=200, threshold=0.01):
        """ Backtesting an enhanced momentum-based strategy.

        Parameters
        ==========
        momentum: int
            number of days for mean return calculation
        hold: int
            minimum number of days to hold after buying
        ma_period: int
            number of days for moving average trend confirmation
        threshold: float
            minimum momentum threshold to initiate a trade
        """
        self.buy_dates = []
        self.sell_dates = []
        self.buy_prices = []
        self.sell_prices = []

        msg = f'\n\nRunning enhanced momentum strategy | {momentum} days'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        st.markdown(msg)
        st.markdown('=' * 55)

        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        self.data['momentum'] = self.data['return'].rolling(momentum).mean()
        self.data['volatility'] = self.data['return'].rolling(momentum).std()
        self.data['sharpe'] = self.data['momentum'] / self.data['volatility']
        self.data['long_ma'] = self.data['price'].rolling(ma_period).mean()

        for bar in range(max(ma_period, momentum), len(self.data)):
            if self.position == 0:
                if (self.data['momentum'].iloc[bar] > threshold) and (self.data['sharpe'].iloc[bar] > 0) and (self.data['price'].iloc[bar] > self.data['long_ma'].iloc[bar]):
                    self.place_buy_order(bar, amount=self.amount)
                    self.buy_dates.append(self.data.index[bar])
                    self.buy_prices.append(self.data['price'].iloc[bar])
                    self.position = 1  # long position
                    price_entry = self.data['price'].iloc[bar]
                    bar_entry = bar

            elif self.position == 1:
                if ((self.data['momentum'].iloc[bar] < 0) and (price_entry < self.data['price'].iloc[bar]) and (bar > bar_entry + hold)):
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
                    self.sell_dates.append(self.data.index[bar])
                    self.sell_prices.append(self.data['price'].iloc[bar])

        self.close_out(bar)

        

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['price'], mode='lines', name='Price', line=dict(color='blue', width=2, dash='solid'), opacity=0.6))
        fig.add_trace(go.Scatter(x=self.buy_dates, y=self.buy_prices, mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy Signal'))
        fig.add_trace(go.Scatter(x=self.sell_dates, y=self.sell_prices, mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Signal'))
        fig.update_layout(title='Price and Buy/Sell Signals with Enhanced Momentum Strategy', xaxis_title='Date', yaxis_title='Price', template="plotly_white", showlegend=True)
        st.plotly_chart(fig)


    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler


    def run_regression_strategy(self, window=50, reg_type='linear'):
        ''' Backtesting a regression-based strategy.

        Parameters
        ==========
        window: int
            number of days to use for regression fitting
        reg_type: str
            type of regression ('linear', 'logistic', 'random_forest')
        '''
        msg = f'\n\nRunning {reg_type} regression strategy | Window={window}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        st.markdown(msg)
        st.markdown('=' * 55)
        
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        
        for bar in range(window, len(self.data)):
            train_data = self.data['price'].iloc[bar-window:bar]
            scaler = StandardScaler()
            X = scaler.fit_transform(np.arange(window).reshape(-1, 1))
            y = None
            
            if reg_type == 'linear':
                y = train_data.values
                model = LinearRegression()
                model.fit(X, y)
                slope = model.coef_[0]
                if self.position == 0 and slope > 0:  # Buy signal
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1
                elif self.position == 1 and slope <= 0:  # Sell signal
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0
            elif reg_type == 'logistic':
                 y = (train_data.pct_change().dropna() > 0).astype(int).values
                model = LogisticRegression()
                model.fit(X[:-1], y)  # Exclude the last observation
                proba = model.predict_proba(X[-1].reshape(1, -1))[0,1]
                if self.position == 0 and proba > 0.5:  # Buy signal
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1
                elif self.position == 1 and proba <= 0.5:  # Sell signal
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0
            elif reg_type == 'random_forest':
                y = (train_data.pct_change().dropna() > 0).astype(int).values
                model = RandomForestClassifier()
                model.fit(X[:-1], y)  # Exclude the last observation
                prediction = model.predict(X[-1].reshape(1, -1))
                if self.position == 0 and prediction == 1:  # Buy signal
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1
                elif self.position == 1 and prediction == 0:  # Sell signal
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0
                    
            self.close_out(bar)
