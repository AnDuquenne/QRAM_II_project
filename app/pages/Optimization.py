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
from stockflow_prediction import predict_stockflow

from sklearn.preprocessing import MinMaxScaler


st.title('Portfolio Optimization')

# ------------------------------------------------ Stock picking ------------------------------------------------ #

with st.expander("Stock Picking"):

    options = st.multiselect(
        "Choose from dow jones stocks:",
        ["INTC", "AAPL", "MSFT", "AMZN", "WMT", "JPM", "V", "UNH", "HD", "PG", "JNJ", "CRM", "CVX", "KO", "MRK", "CSCO",
         "MCD", "AXP", "IBM", "GS", "CAT", "DIS", "VZ", "AMGN", "HON", "NKE", "BA", "SHW", "MMM", "TRV", "NVDA"],
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
    end_date = "2023-10-31"
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

# ------------------------------------------------ Views computations ------------------------------------------------ #

list_of_gtp_responses = []
list_of_views_from_gpt = []
with st.expander("Views computations"):
    uploaded_files = st.file_uploader("Choose a pdf file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = io.BytesIO(uploaded_file.read())
        pdf_tokenized = extract_text_with_pdfplumber(bytes_data)
        response = ask_gpt(pdf_tokenized)
        response = json.loads(response.content)
        list_of_gtp_responses.append(response)

    # Make a dataframe with the responses
    responses_df = pd.DataFrame(list_of_gtp_responses,
                                columns=["Stock_analysed", "Ticker", "Report_date",
                                         "Company_writing_report", "Actual_price",
                                         "Expected_price", "Forecasting_horizon", "Currency"])

    # Compute the return as pct change
    responses_df["total_return"] = (responses_df["Expected_price"] - responses_df["Actual_price"]) / responses_df[
        "Actual_price"]

    # Compute the annualized return
    responses_df["annualized_return"] = compute_return_365(responses_df["total_return"],
                                                           responses_df["Forecasting_horizon"])

    # Daily return
    responses_df["daily_return"] = compute_return_daily(responses_df["total_return"])

    st.dataframe(responses_df.head(), height=300)

    tickers = data.columns

    for i in range(responses_df['Ticker'].shape[0]):
        tick_ = responses_df['Ticker'][i]
        if tick_ in tickers:
            # Create a view for the stock, the view is a vector of the expected return for that stock
            view = np.zeros(len(tickers))
            # Find the index of the stock in the tickers pandas columns list
            idx = np.where(tickers == tick_)[0][0]
            # Set the expected return for that stock
            view[idx] = responses_df['daily_return'][i]

        print(view)

        list_of_views_from_gpt.append(view)

    # Compute views using stockflow
    date_ = end_date
    # Recover columns that are included in the "option" list
    stock_ = data[options].iloc[-8:, :]
    stock_ = stock_.dropna().iloc[1:, :].values

    results = []
    for i in range(stock_.shape[1]):
        stockflow_input = stock_[:, i]
        # Min max scaling without using MinMaxScaler
        min_ = stockflow_input.min()
        max_ = stockflow_input.max()
        stockflow_input = (stockflow_input - min_) / (max_ - min_)

        stockflow_result = predict_stockflow(date_, stockflow_input)
        # Reverse the min max scaling
        stockflow_result = stockflow_result * (max_ - min_) + min_
        results.append(stockflow_result)

    # Make a view from the stockflow prediction
    view_stockflow = np.zeros(len(tickers))
    for i in range(len(options)):
        idx = np.where(tickers == options[i])[0][0]
        view_stockflow[idx] = results[i]

    # Transform the views into separate vectors
    # [x, 0, 0, y, 0, z] -> [x, 0, 0, 0, 0, 0], [0, 0, 0, y, 0, 0], [0, 0, 0, 0, 0, z]
    list_of_views_from_gpt = split_vector((np.array(list_of_views_from_gpt, dtype=np.float64).squeeze()))
    list_of_views_from_stockflow = split_vector(view_stockflow.astype(np.float64))

    print(f"List of views from gpt: {list_of_views_from_gpt}")
    print(f"List of views from stockflow: {list_of_views_from_stockflow}")

    # Make a dataframe with the views
    views_df_gpt = pd.DataFrame(list_of_views_from_gpt, columns=tickers)
    views_df_stockflow = pd.DataFrame(list_of_views_from_stockflow, columns=tickers)

    # concatenate the two dataframes on the rows
    views_df = pd.concat([views_df_gpt, views_df_stockflow])

    st.write("View from the rep analysis")
    st.dataframe(list_of_views_from_gpt)
    st.write("View from the stockflow prediction")
    st.dataframe(view_stockflow)

    st.dataframe(views_df.head(10))
