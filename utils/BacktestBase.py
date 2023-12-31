#
# Python Script with Base Class
# for Event-based Backtesting
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import numpy as np
import pandas as pd
from pylab import mpl, plt
import yfinance as yf
import streamlit as st

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
from ta.trend import MACD, AroonIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import StochasticOscillator


class BacktestBase(object):
    ''' Base class for event-based backtesting of trading strategies.

    Attributes
    ==========
    symbol: str
        TR RIC (financial instrument) to be used
    start: str
        start date for data selection
    end: str
        end date for data selection
    amount: float
        amount to be invested either once or per trade
    ftc: float
        fixed transaction costs per trade (buy or sell)
    ptc: float
        proportional transaction costs per trade (buy or sell)

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    plot_data:
        plots the closing price for the symbol
    get_date_price:
        returns the date and price for the given bar
    print_balance:
        prints out the current (cash) balance
    print_net_wealth:
        prints auf the current net wealth
    place_buy_order:
        places a buy order
    place_sell_order:
        places a sell order
    close_out:
        closes out a long or short position
    '''

    def __init__(self, symbol, 
                 start, end, 
                 amount,
                 ftc=0.0, ptc=0.0, verbose=False):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_amount = amount
        self.amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.units = 0
        self.position = 0
        self.trades = 0
        self.verbose = verbose
        self.get_data()
        self.get_data_full()

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        
        stock = yf.Ticker(self.symbol)
        # Use the Ticker object to fetch historical data for the stock
         #period='max'
        if self.start==False and self.end==False:
            historical_data= stock.history(period='max')
        else:
            historical_data = stock.history(start = self.start, end = self.end)
        raw = historical_data['Close'].reset_index().rename(columns={'Close': 'price'})
        raw['return'] = np.log(raw['price'] / raw['price'].shift(1))
        self.data = raw.dropna().set_index('Date')
    # FEATURE ENGINEERING START
    def moving_average(self, df, window):
        return df['Close'].rolling(window=window).mean()

    def RSI(self, df, window):
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def volatility(self, df, window):
        return df['Close'].rolling(window=window).std()

    def volume_roc(self, df):
        return df['Volume'].pct_change()

    def get_data_full(self):
        ''' Retrieves and prepares the data.
        '''
        stock = yf.Ticker(self.symbol)
        
        # Fetch data
        if self.start==False and self.end==False:
            historical_data = stock.history(period='max')
        else:
            historical_data = stock.history(start=self.start, end=self.end)
        
        raw = historical_data[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()

        # Feature Engineering
        raw['7_day_MA'] = self.moving_average(raw, 7)
        raw['15_day_MA'] = self.moving_average(raw, 15)
        raw['30_day_MA'] = self.moving_average(raw, 30)
        raw['RSI_14'] = self.RSI(raw, 14)
        raw['Volatility_7'] = self.volatility(raw, 7)
        raw['Volume_ROC'] = self.volume_roc(raw)
        raw['return'] = np.log(raw['Close'] / raw['Close'].shift(1))
        
        # MACD
        macd = MACD(raw['Close'])
        raw['macd_diff'] = macd.macd_diff()

        # Aroon
        aroon = AroonIndicator(raw['Close'])
        raw['aroon_up'] = aroon.aroon_up()
        raw['aroon_down'] = aroon.aroon_down()

        # OBV
        obv = OnBalanceVolumeIndicator(raw['Close'], raw['Volume'])
        raw['obv'] = obv.on_balance_volume()

        # Ichimoku Cloud
        raw['ichi_a'] = (raw['High'].rolling(window=9).max() + raw['Low'].rolling(window=9).min()) / 2  # Tenkan-sen
        raw['ichi_b'] = (raw['High'].rolling(window=26).max() + raw['Low'].rolling(window=26).min()) / 2  # Kijun-sen

        # Stochastic Oscillator
        stoch = StochasticOscillator(raw['High'], raw['Low'], raw['Close'])
        raw['stoch'] = stoch.stoch()

        # For Fibonacci Retracement, we need only levels. Actual retracement lines will be drawn based on significant highs/lows
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        raw['high_prev'] = raw['High'].shift(1)
        raw['low_prev'] = raw['Low'].shift(1)
        for level in fib_levels:
            raw[f'fib_{level*100}%'] = raw['low_prev'] + (raw['high_prev'] - raw['low_prev']) * level

        self.full_data = raw.dropna().set_index('Date').replace([np.inf, -np.inf], 0)
    # FEATURE ENGINEERING END       
    def plot_data(self, cols=None):
        ''' Plots the closing prices for symbol.
        '''
        if cols is None:
            cols = ['price']
        #self.data['price'].plot(figsize=(10, 6), title=self.symbol)
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Scatter(x=self.data.index, y=self.data['price'])])
        fig.update_layout(title=self.symbol, xaxis_title='Date', yaxis_title='Price')
        return fig

    def plot_sma(self, cols=None, sma1=20, sma2=50):
        ''' Plots the closing prices, sma1, and sma2 for symbol.
        '''
        if cols is None:
            cols = ['price']

        # Calculate the sma1 and sma2 values
        self.data['sma1'] = self.data['price'].rolling(window=sma1).mean()
        self.data['sma2'] = self.data['price'].rolling(window=sma2).mean()

        # Plot the price data and sma1 and sma2 on the same plot
        ax = self.data[cols].plot(figsize=(10, 6), title=self.symbol)
        ax.plot(self.data['sma1'], label=f"SMA{str(sma1)}")
        ax.plot(self.data['sma2'], label=f"SMA{str(sma2)}")
        ax.legend()


    def get_date_price(self, bar):
        ''' Return date and price for bar.
        '''
        date = str(self.data.index[bar])[:10]
        price = self.data.price.iloc[bar]
        return date, price

    def print_balance(self, bar):
        ''' Print out current cash balance info.
        '''
        date, price = self.get_date_price(bar)
        st.text(f'{date} | current balance {self.amount:.2f}')

    def print_net_wealth(self, bar):
        ''' Print out current cash balance info.
        '''
        date, price = self.get_date_price(bar)
        net_wealth = self.units * price + self.amount
        st.text(f'{date} | current net wealth {net_wealth:.2f}')

    def place_buy_order(self, bar, units=None, amount=None):
        ''' Place a buy order.
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            units = amount / price #int
        self.amount -= (units * price) * (1 + self.ptc) + self.ftc
        self.units += units
        self.trades += 1
        if self.verbose:
            st.text(f'{date} | buying {units} units at {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def place_sell_order(self, bar, units=None, amount=None):
        ''' Place a sell order.
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
        self.amount += (units * price) * (1 - self.ptc) - self.ftc
        self.units -= units
        self.trades += 1
        if self.verbose:
            st.text(f'{date} | selling {units} units at {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def close_out(self, bar):
        ''' Closing out a long or short position.
        '''
        date, price = self.get_date_price(bar)
        self.amount += self.units * price
        self.units = 0
        self.trades += 1
        if self.verbose:
            st.text(f'{date} | inventory {self.units} units at {price:.2f}')
            st.text('=' * 55)
        st.text('Final balance   [$] {:.2f}'.format(self.amount))
        perf = ((self.amount - self.initial_amount) /
                self.initial_amount * 100)
        st.text('Net Performance [%] {:.2f}'.format(perf))
        st.text('Trades Executed [#] {:.2f}'.format(self.trades-1))
        st.text('=' * 55)
    def print_hold(self):
        return ((self.data['price'][-1]/self.data['price'][0]-1)*100,'%')

