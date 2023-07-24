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


plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'


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
                 ftc=0.0, ptc=0.0, verbose=True):
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

    def plot_data(self, cols=None):
        ''' Plots the closing prices for symbol.
        '''
        if cols is None:
            cols = ['price']
        #self.data['price'].plot(figsize=(10, 6), title=self.symbol)
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Scatter(x=self.data.index, y=self.data['price'])])
        fig.update_layout(title=self.symbol, xaxis_title='Date', yaxis_title='Price')
        fig.show()

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
        print(f'{date} | current balance {self.amount:.2f}')

    def print_net_wealth(self, bar):
        ''' Print out current cash balance info.
        '''
        date, price = self.get_date_price(bar)
        net_wealth = self.units * price + self.amount
        print(f'{date} | current net wealth {net_wealth:.2f}')

    def place_buy_order(self, bar, units=None, amount=None):
        ''' Place a buy order.
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
        self.amount -= (units * price) * (1 + self.ptc) + self.ftc
        self.units += units
        self.trades += 1
        if self.verbose:
            print(f'{date} | buying {units} units at {price:.2f}')
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
            print(f'{date} | selling {units} units at {price:.2f}')
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
            print(f'{date} | inventory {self.units} units at {price:.2f}')
            print('=' * 55)
        print('Final balance   [$] {:.2f}'.format(self.amount))
        perf = ((self.amount - self.initial_amount) /
                self.initial_amount * 100)
        print('Net Performance [%] {:.2f}'.format(perf))
        print('Trades Executed [#] {:.2f}'.format(self.trades))
        print('=' * 55)
    def print_hold(self):
        return ((self.data['price'][-1]/self.data['price'][0]-1)*100,'%')

