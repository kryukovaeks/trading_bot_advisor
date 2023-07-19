import streamlit as st
import matplotlib.pyplot as plt
import textwrap
import openai
import numpy as np
import datetime as dt
from pycoingecko import CoinGeckoAPI
from GoogleNews import GoogleNews
import os
import plotly.graph_objects as go
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import requests
import time

st.set_page_config(layout='wide') 


#openai.api_key = os.getenv('OPENAI_API_KEY')
#openai.api_key = st.secrets["OPENAI_API_KEY"]
#st.write("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])

# Initialize CoinGecko 
cg = CoinGeckoAPI()

# Define streamlit elements
st.title('Crypto Trading Bot Advisor')
st.write('Enter your parameters below:')
# Ask user for OpenAI API Key
openai_key_input = st.text_input("Enter your OpenAI API Key:", type="password")  # Using type="password" hides the entered characters
if openai_key_input:
    openai.api_key = openai_key_input

max_budget = st.number_input('Maximum budget ($):', min_value=10.0, max_value=10000.0, value=100.0)
cryptos_input = st.text_input('Enter cryptos (comma separated):')
days_input = st.slider('Number of days for price analysis:', min_value=1, max_value=365, value=30)

if cryptos_input:
    cryptos = [crypto.strip() for crypto in cryptos_input.split(',')]
    crypto_data=''
    from plotly.subplots import make_subplots

    # Create subplots: each row represents a different crypto
    fig = make_subplots(rows=len(cryptos), cols=1)

    # Add traces (one trace per crypto)
    for i, crypto in enumerate(cryptos, start=1):
        try:
            data = cg.get_coin_market_chart_by_id(crypto, vs_currency='usd', days=days_input)
            prices = data['prices']
            prices_only = [price[1] for price in prices]
            time_stamps = [price[0] for price in prices]
            dates = [dt.datetime.fromtimestamp(time_stamp/1000) for time_stamp in time_stamps]
            high = max(prices_only)
            low = min(prices_only)
            avg = np.mean(prices_only)
            crypto_data += f" {crypto} prices for the past {days_input} days: High={high}, Low={low}, Average={avg}\n"
            st.write(f"{crypto} prices for the past {days_input} days: High={high}, Low={low}, Average={avg}")

            # Add a trace for this cryptocurrency to the i-th subplot
            fig.add_trace(go.Scatter(x=dates, y=prices_only, mode='lines', name=crypto), row=i, col=1)

        except Exception as e:
            st.error(f"An error occurred when fetching data for {crypto}: {str(e)}")

    # Show the figure with the graphs
    st.plotly_chart(fig)






    news_dict = {}
    try:
        googlenews = GoogleNews(period='7d')
        googlenews.enableException(True)

        for term in cryptos:
            
            googlenews.search(term.capitalize())
            googlenews.get_page(1)
            news_dict[term] = googlenews.results()
            googlenews.clear()

            #st.write(term)
            #st.write(news_dict[term])
            time.sleep(5)
        # Create a list to hold the news data
        news_data = pd.DataFrame()

        # Append the news data to the list
        for term, news_list in news_dict.items():
            for news in news_list:
                news_data_i = pd.DataFrame([news])
                news_data_i['crypto']=term
                news_data = pd.concat([news_data, news_data_i], axis=0, ignore_index=True)
            

        # Get the date input from the user
        #date_input = pd.to_datetime(date_input)  # Replace with the actual date input

        # Sort the dataframe by the 'datetime' column
        #st.write(news_data)

        df = news_data.sort_values(by =['crypto','datetime'], ascending=False).groupby(['title']).head(1).drop_duplicates()



        # Expand the maximum width of each cell to display more content
        pd.set_option('display.max_colwidth', None)

    
        # Check if the selected columns are in the session state. If not, initialize it.
        if 'selected_columns' not in st.session_state:
            st.session_state.selected_columns = ['title', 'date', 'crypto']

        # Update the multiselect widget to use session state.
        selected_columns = st.multiselect("Select columns", df.columns, default=st.session_state.selected_columns)
        st.session_state.selected_columns = selected_columns

        if selected_columns:
            df_selected = df[selected_columns]
            html_table = df_selected.to_html(escape=False, index=False)

            # Wrap the HTML table in a div with fixed height and overflow
            html_table_with_scroll = f"""
            <div style="height:300px;overflow:auto;">
                {html_table}
            </div>
            """

            # Use Streamlit's markdown renderer to display the wrapped table
            st.markdown(html_table_with_scroll, unsafe_allow_html=True)
    except Exception as e:
        st.error(e)








    # AI prompt
    base_prompt = f"""
    You are in control of my crypto trading profile. You should take into consideration the factors you have to determine the best trade. Here is the info:

    You can execute these commands:

    1. buy_crypto_price(symbol, amount)
    2. buy_crypto_limit(symbol, amount, limit)
    3. sell_crypto_price(symbol, amount)
    4. sell_crypto_limit(symbol, amount, limit)
    5. do_nothing()
    You have to provide amount.
    Use this when you don't see any necessary changes.

    You also have access to this data:

    1. Historical data
    2. News Headlines

    The current date and time is {dt.datetime.today()}

    You are called once every 30 minutes, keep this in mind.

    The only cryptos you can trade are {', '.join(cryptos)}.

    Here are the data sources:

    """

    info_str = f"Historical statistics for {days_input} days: {crypto_data}\n News: {news_data[['title','crypto']].drop_duplicates()}"
    prompt = base_prompt + "\n\n" + info_str
    user_prompt = """
    What should we do to make the most amount of profit based on the info? Here are your options for a response.

    1. buy_crypto_price(symbol, amount) This will buy the specified amount of the specified cryptocurrency.
    2. buy_crypto_limit(symbol, amount, limit) This will set a limit order to buy the specified amount of the specified cryptocurrency if it reaches the specified limit.
    3. sell_crypto_price(symbol, amount) This will sell the specified amount of the specified cryptocurrency.
    4. sell_crypto_limit(symbol, amount, limit) This will set a limit order to sell the specified amount of the specified cryptocurrency if it reaches the specified limit.
    5. do_nothing() Use this when you don't see any necessary changes.
    
    Choose one (firstly write the execution command) and explain
    CRITICAL: RESPOND IN ONLY THE ABOVE FORMAT. EXAMPLE: buy_crypto_price("ETH", 100). 
    ALSO IN THE AMOUNT FIELD, USE THE UNIT SYSTEM OF DOLLARS. ASSUME WE HAVE A BUDGET of UP TO ${max_budget} WORTH OF dollar PER TRADE for 24 hours.

    !give execution for every crypto at the beginning
    !do not forget to explain
        """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature = 0.2,
    )

    res = response.choices[0].message["content"]
    res = res.replace("\\", "")
    st.write("ChatGPT advise:")
    st.write(textwrap.fill(str(res), width=50))

