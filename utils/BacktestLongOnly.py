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
                    self.buy_dates.append(self.data.index.values[bar])  
                    self.buy_prices.append(self.data['price'].iloc[bar]) 
                    self.position = 1  # long position
                    price_entry = self.data['price'].iloc[bar]
                    ber_enrty = bar
            elif self.position == 1:
                if (self.data['momentum'].iloc[bar]) < 0 & (price_entry<self.data['price'].iloc[bar]) & (bar>ber_enrty+hold):
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
                    self.sell_dates.append(self.data.index.values[bar])  
                    self.sell_prices.append(self.data['price'].iloc[bar]) 
        self.close_out(bar)
        import plotly.graph_objects as go

        fig = go.Figure()



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
        fig.update_layout(title='Price and Buy/Sell Signals with Plotly',
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


"""if __name__ == '__main__':
    def run_strategies():
        lobt.run_sma_strategy(42, 252)
        lobt.run_momentum_strategy(60)
        lobt.run_mean_reversion_strategy(50, 5)
    lobt = BacktestLongOnly('AAPL.O', '2010-1-1', '2019-12-31', 10000,
                            verbose=False)
    run_strategies()
    # transaction costs: 10 USD fix, 1% variable
    lobt = BacktestLongOnly('AAPL.O', '2010-1-1', '2019-12-31',
                            10000, 10.0, 0.01, False)
    run_strategies()"""
