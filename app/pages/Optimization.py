import json

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

import sys
import os
import io

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model import MarkowitzMeanVarOptimization
from utils import *
from report_analysis import extract_text_with_pdfplumber, ask_gpt, ask_gpt_gamma


st.title('Portfolio Optimization')

with st.expander("Stock Picking"):

    options = st.multiselect(
        "Choose from dow jones stocks:",
        ["INTC", "AAPL", "MSFT", "AMZN", "WMT", "JPM", "V", "UNH", "HD", "PG", "JNJ", "CRM", "CVX", "KO", "MRK", "CSCO",
         "MCD", "AXP", "IBM", "GS", "CAT", "DIS", "VZ", "AMGN", "HON", "NKE", "BA", "SHW", "MMM", "TRV",],
        ["INTC", "AAPL", "MSFT"],
    )

    options_crypto = st.multiselect(
        "Choose from crypto currencies:",
        ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD", "XRP-USD"],
        ["BTC-USD"]
    )

    options_bonds = st.multiselect(
        "Choose from bonds:",
        ["^TNX", "^FVX", "^IRX"],
        ["^TNX"]
    )


    ticker = options + options_crypto + options_bonds
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data['Close']

    # Transform the data to a daily return and Crypto has no closing day so we need to drop rows with NaN
    data = data.pct_change().dropna().iloc[1:, :]

    # Or if 0 instead of nans
    data = data[(data != 0).all(axis=1)]

    st.dataframe(data.head())

    constraints = []

    st.subheader("Constraints")
    FI = st.checkbox("Fully invested")
    if FI:
        constraints.append("Fully Invested")

    NS = st.checkbox("No short selling")
    if NS:
        constraints.append("No Short Selling")

    MV = MarkowitzMeanVarOptimization(data, constraints)
    data = MV.get_data()
    print(data)
    mu = MV.get_mu()
    print(mu)
    vol = MV.get_vol()
    print(vol)
    cov = MV.get_cov_matrix()
    print(cov)

    frontier = MV.efficient_frontier_points()
    print(frontier[0])
    print(frontier[1])

    # Make a dataframe with the frontier
    frontier_df = pd.DataFrame(frontier.T, columns=["Return", "Volatility"])

    # Merge the individual stock data with the frontier, to plot them together, add a column to frontier df to indicate
    # that these are the efficient frontier points or the individual stocks
    frontier_df["Type"] = "Efficient Frontier"
    stocks_df = pd.DataFrame({"Return": mu, "Volatility": vol, "Type": "Individual Stocks"})
    combined_df = pd.concat([frontier_df, stocks_df])

    st.scatter_chart(x="Volatility", y="Return", data=combined_df, color="Type")

list_of_gtp_responses = []
with st.expander("Views computations"):
    uploaded_files = st.file_uploader("Choose a pdf file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = io.BytesIO(uploaded_file.read())
        pdf_tokenized = extract_text_with_pdfplumber(bytes_data)
        response = ask_gpt(pdf_tokenized)
        print(response)
        print(response.content)
        response = json.loads(response.content)
        print(response)
        print(type(response))
        list_of_gtp_responses.append(response)

    # Make a dataframe with the responses
    responses_df = pd.DataFrame(list_of_gtp_responses,
                                columns=["Stock_analysed", "Ticker", "Report_date",
                                         "Company_writing_report", "Actual_price",
                                         "Expected_price", "Date_expected_price", "Currency"])

    print(responses_df.head())
    st.dataframe(responses_df.head(), height=300)


with st.expander("Computation of the Risk Aversion"):

    proportion_risky_asset = st.select_slider(
        "What maximal percentage of your portfolio should be invested in risky assets?",
        options=[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5
        ],
    )

    opti_weights = MV.efficient_frontier(proportion_risky_asset)[1]
    tickers = MV.get_tickers()

    # Make a dataframe with the optimal weights
    weights_df = pd.DataFrame(opti_weights, index=tickers, columns=["Weights"])

    st.bar_chart(weights_df)
