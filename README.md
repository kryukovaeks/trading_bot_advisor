
# Crypto Trading Bot Advisor

## Description
The Crypto Trading Bot Advisor is a Streamlit application that fetches real-time cryptocurrency market data and news articles. It leverages the CoinGecko API for the market data, Google News for recent news articles, and the OpenAI's GPT-3 model to provide insights based on the current market situation.

## Prerequisites
- Python 3.7 or later
- Streamlit
- OpenAI Python client
- PyCoinGecko
- GoogleNews
- Plotly

Install all prerequisites with pip:
```
pip install streamlit openai pycoingecko GoogleNews plotly pandas
```

## Setup

Clone the repository:
```
git clone http://github.com/yourusername/cryptotradingbot
```

Navigate to the project directory:
```
cd cryptotradingbot
```

Create a `secrets.toml` file in your project's root directory and insert your OpenAI API key as shown below:
```
[default]
OPENAI_API_KEY = "your_api_key_here"
```
**Important**: Remember to add `secrets.toml` to your `.gitignore` file to prevent exposing your secret keys.

## Usage

To start the Streamlit application, run the following command:
```
streamlit run app.py
```

## Demo

Check the live application at: [Streamlit App](https://tradingbotadvisor.streamlit.app)

