import streamlit as st
import matplotlib.pyplot as plt
import textwrap
import openai
import numpy as np
import datetime as dt
from pycoingecko import CoinGeckoAPI
from GoogleNews import GoogleNews

# Initialize CoinGecko and GoogleNews
cg = CoinGeckoAPI()
googlenews = GoogleNews()

# Define streamlit elements
st.title('Crypto Analyzer')
st.write('Enter your parameters below:')

cryptos_input = st.text_input('Enter cryptos (comma separated):')
days_input = st.slider('Number of days for price analysis:', min_value=1, max_value=365, value=30)
date_input = st.date_input('News date:')
if cryptos_input:
    cryptos = [crypto.strip() for crypto in cryptos_input.split(',')]


    # Get historical data and plot it
    crypto_data = ''
    for crypto in cryptos:
        data = cg.get_coin_market_chart_by_id(crypto, vs_currency='usd', days=days_input)
        prices = data['prices']
        dates = [dt.datetime.utcfromtimestamp(price[0]/1000).date() for price in prices]
        prices_only = [price[1] for price in prices]
        high = max(prices_only)
        low = min(prices_only)
        avg = np.mean(prices_only)
        crypto_data += f" {crypto} data for the past {days_input} days: High={high}, Low={low}, Average={avg}\n"
        st.line_chart(data={"price": prices_only}, width=0, height=0, use_container_width=True)

    # Get news
    news_dict = {}
    for term in cryptos:
        googlenews.search(term)
        googlenews.get_page(date_input.strftime('%m/%d/%Y'))
        news_dict[term] = googlenews.results()
        googlenews.clear()

    # Print news
    news_output = ''
    for term, news_list in news_dict.items():
        news_output += f"News for {term}:\n"
        for news in news_list:
            news_output += f"{news['title']} - {news['media']}\n"

    # AI prompt
    openai.api_key = 'YOUR_OPENAI_API_KEY'

    base_prompt = f"""
    You are in control of my crypto trading profile. You should take into consideration the factors you have to determine the best trade. Here is the info:

    You can execute these commands:

    1. buy_crypto_price(symbol, amount)
    2. buy_crypto_limit(symbol, amount, limit)
    3. sell_crypto_price(symbol, amount)
    4. sell_crypto_limit(symbol, amount, limit)
    5. do_nothing()

    Use this when you don't see any necessary changes.

    You also have access to this data:

    1. Historical data
    2. News Headlines

    The current date and time is {dt.datetime.today()}

    You are called once every 30 minutes, keep this in mind.

    The only cryptos you can trade are {', '.join(cryptos)}.

    Here are the data sources:

    """

    info_str = f"Historical data: {crypto_data}\n News: {news_output}"
    prompt = base_prompt + "\n\n" + info_str
    user_prompt = """
    What should we do to make the most amount of profit based on the info? Here are your options for a response.

    1. buy_crypto_price(symbol, amount) This will buy the specified amount of the specified cryptocurrency.
    2. buy_crypto_limit(symbol, amount, limit) This will set a limit order to buy the specified amount of the specified cryptocurrency if it reaches the specified limit.
    3. sell_crypto_price(symbol, amount) This will sell the specified amount of the specified cryptocurrency.
    4. sell_crypto_limit(symbol, amount, limit) This will set a limit order to sell the specified amount of the specified cryptocurrency if it reaches the specified limit.
    5. do_nothing() Use this when you don't see any necessary changes.

    Choose one and explain
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
    st.write(textwrap.fill(str(res), width=50))

