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

from model import MarkowitzMeanVarOptimization, BlackLittermanOptimization
from utils import *
from report_analysis import extract_text_with_pdfplumber, ask_gpt
from stockflow_prediction import predict_stockflow

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


from sklearn.preprocessing import MinMaxScaler

# Load csv credit rating data
ratings = pd.read_csv("app/io/Dow_Jones_Credit_Ratings_Oct_2023.csv")

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
    start_date = "2019-01-01"
    end_date = "2023-10-31"
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data['Close']

    # Transform the data to a daily return and Crypto has no closing day so we need to drop rows with NaN
    data = data.pct_change().dropna().iloc[1:, :]

    # Or if 0 instead of nans
    data = data[(data != 0).all(axis=1)]

    # Ratings
    ratings = ratings[ratings['Ticker'].isin(data.columns)][['Ticker', 'S&P Rating', 'S&P Rating Number']]

    st.dataframe(data.head())

    # Risk-free rate
    rf_rate = st.number_input("Risk free rate")

# ------------------------------------------------ Constraints ------------------------------------------------ #

with st.expander("Constraints"):
    constraints = []

    st.subheader("Constraints")
    FI = st.checkbox("Fully invested")
    if FI:
        constraints.append("Fully Invested")

    NS = st.checkbox("No short selling")
    if NS:
        constraints.append("No Short Selling")

    checkbox = st.checkbox("Credit rating constraints")
    percentage_rating = st.number_input(
        "Minimum percentage of the portfolio held in the range",
        value=0.8, placeholder="Percentage of the portfolio", min_value=0.0, max_value=1.0
    )
    start_rating, end_rating = st.select_slider(
        "Select the range of credit rating",
        options=[
            "AAA",
            "AA+",
            "AA",
            "AA-",
            "A+",
            "A",
            "A-",
            "BBB+",
            "BBB",
            "BBB-",
            "BB+",
            "BB",
            "BB-",
            "B+",
            "B",
            "B-",
            "CCC+",
            "CCC",
            "CCC-",
            "CC",
            "C",
            "D"
        ],
        value=("AAA", "BBB"),
    )
    # st.write(f"Start rating: {start_rating}")
    # st.write(f"End rating: {end_rating}")
    # st.write(f"Number: {percentage_rating}")

# ---------------------------------------------- Efficient Frontier --------------------------------------------- #

MV = MarkowitzMeanVarOptimization(data, constraints, ratings, rf_rate=rf_rate)
if checkbox:
    MV.set_special_constraints(sp_rating_to_number[start_rating], sp_rating_to_number[end_rating], percentage_rating)
data = MV.get_data()
mean = MV.get_mu()
vol = MV.get_vol()
cov = MV.get_cov_matrix()

frontier = MV.efficient_frontier_points()

# Make a dataframe with the frontier
frontier_df = pd.DataFrame(frontier.T, columns=["Return", "Volatility"])

# Merge the individual stock data with the frontier, to plot them together, add a column to frontier df to indicate
# that these are the efficient frontier points or the individual stocks
frontier_df["Type"] = "Efficient Frontier"
stocks_df = pd.DataFrame({"Return": mean, "Volatility": vol, "Type": "Individual Stocks"})
for i in range(stocks_df.shape[0] - 1):
    stocks_df["Type"].iloc[i] = MV.get_tickers()[i]
stocks_df["Type"].iloc[-1] = "Risk Free Asset"

combined_df = pd.concat([frontier_df, stocks_df])

with st.expander("Mean Variance Optimization"):
    proportion_risky_asset = st.select_slider(
        "Risk Aversion:",
        options=([i] for i in range(0, 1000, 10))
    )

    col_efficient_frontier, col_individual_stocks = st.columns(2)
    with col_efficient_frontier:
        st.scatter_chart(x="Volatility", y="Return", data=combined_df, color="Type")

    with col_individual_stocks:

        opti_weights = MV.efficient_frontier(proportion_risky_asset)[1]
        tickers = np.append(MV.get_tickers(), "Risk Free Asset")

        # Make a dataframe with the optimal weights
        weights_df = pd.DataFrame(opti_weights, index=tickers, columns=["Weights"])

        st.bar_chart(weights_df)

# ------------------------------------------------ Views computations ------------------------------------------------ #

list_of_gtp_responses = []
list_of_views_from_gpt = []
list_std_stockflow = []
with (st.expander("Views computations")):
    ak_ = st.text_input("Your key")
    # if ak_ is none use secret
    if len(ak_) <= 5 or ak_ is None:
        ak_ = st.secrets["OAK"]

    uploaded_files = st.file_uploader("Choose a pdf file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = io.BytesIO(uploaded_file.read())
        pdf_tokenized = extract_text_with_pdfplumber(bytes_data)
        response = ask_gpt(pdf_tokenized, ak_)
        response = json.loads(response.content)
        list_of_gtp_responses.append(response)

    # Make a dataframe with the responses
    responses_df = pd.DataFrame(list_of_gtp_responses,
                                columns=["Stock_analysed", "Ticker", "Report_date",
                                         "Company_writing_report", "Actual_price",
                                         "Expected_price", "Min_expected_price", "Max_expected_price",
                                         "Forecasting_horizon", "Currency"])

    # Compute the return as pct change
    responses_df["total_return"] = (responses_df["Expected_price"] - responses_df["Actual_price"]) / responses_df[
        "Actual_price"]

    # Compute the annualized return
    responses_df["annualized_return"] = compute_return_365(responses_df["total_return"],
                                                           responses_df["Forecasting_horizon"])

    # Daily return
    responses_df["daily_return"] = compute_return_daily(responses_df["total_return"])

    # Omega scaling
    responses_df['omega'] \
        = (responses_df['Max_expected_price'] - responses_df['Min_expected_price']) / responses_df['Expected_price']

    st.dataframe(responses_df.head(), height=150)

    tickers = data.columns

    for i in range(responses_df['Ticker'].shape[0]):
        tick_ = responses_df['Ticker'][i]
        view = None
        if tick_ in tickers:
            # Create a view for the stock, the view is a vector of the expected return for that stock
            view = np.zeros(len(tickers))
            # Find the index of the stock in the tickers pandas columns list
            idx = np.where(tickers == tick_)[0][0]
            # Set the expected return for that stock
            view[idx] = responses_df['daily_return'][i]

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

        stockflow_result, stockflow_result_std = predict_stockflow(date_, stockflow_input)
        # Reverse the min max scaling
        stockflow_result = stockflow_result * (max_ - min_) + min_
        results.append(stockflow_result)
        list_std_stockflow.append(stockflow_result_std)

    # Make a view from the stockflow prediction
    view_stockflow = np.zeros(len(tickers))
    for i in range(len(options)):
        idx = np.where(tickers == options[i])[0][0]
        view_stockflow[idx] = results[i]

    # User input views
    list_user_input_views = []
    list_user_omega = []

    for i in range(len(tickers)):
        with st.container():
            st.write(f"{tickers[i]}")
            col1, col2 = st.columns(2)
            with col1:
                user_input = st.number_input(f"Expected daily return", value=0.0, key=f"edr_{i}")
            with col2:
                user_input_vol = st.number_input(f"Omega", value=0.0, key=f"omega_{i}")
            # user_input = compute_return_daily(user_input)
            list_user_input_views.append(user_input)
            list_user_omega.append(user_input_vol)

    # Transform the views into separate vectors
    # [x, 0, 0, y, 0, z] -> [x, 0, 0, 0, 0, 0], [0, 0, 0, y, 0, 0], [0, 0, 0, 0, 0, z]
    list_of_views_from_gpt = list_of_views_from_gpt
    list_of_views_from_stockflow = split_vector(view_stockflow.astype(np.float64))
    list_of_user_input_views = split_vector(np.array(list_user_input_views).astype(np.float64))

    # Make a dataframe with the views
    views_df_gpt = pd.DataFrame(list_of_views_from_gpt, columns=tickers)
    views_df_stockflow = pd.DataFrame(list_of_views_from_stockflow, columns=tickers)
    views_df_user = pd.DataFrame(list_of_user_input_views, columns=tickers)

    # concatenate the two dataframes on the rows
    views_df = pd.concat([views_df_stockflow, views_df_gpt, views_df_user])

    # st.write("View from the rep analysis")
    # st.dataframe(list_of_views_from_gpt)
    # st.write("View from the stockflow prediction")
    # st.dataframe(view_stockflow)
    # st.write("User input views")
    # st.dataframe(list_user_input_views)

    coldata, colupdated = st.columns(2)
    with coldata:
        st.write("Views dataframe")
        st.dataframe(views_df.head(10))
    with colupdated:
        st.write("Updated dataframe")
        views_df, removed_rows = remove_redundent_rows(views_df)
        st.dataframe(views_df.head(10))

    P = binary_transform(views_df.values)

    Q = combine_vectors(views_df.values.T)

    # ------------------------------------------------ Omega ------------------------------------------------ #
    # Omega from gpt
    omega_gpt = responses_df['omega'].values * 0.1

    # Omega from stockflow
    omega_stockflow = np.array(list_std_stockflow)

    # Omega from user input
    list_user_omega = np.array(list_user_omega).astype(np.float64)
    omega_user = list_user_omega[list_user_omega != 0]

    # Combine the two omegas
    omega_sacling = np.concatenate((omega_stockflow, omega_gpt, omega_user), axis=0)

    # Removes the rows that are redundant
    omega_sacling = np.delete(omega_sacling, removed_rows, axis=0)

    # st.dataframe(omega_sacling)
    omega = np.diag(omega_sacling) * (0.1 ** 1)


    colP, colQ, colOmega = st.columns(3)
    with colP:
        st.write("P")
        st.dataframe(P)
    with colQ:
        st.write("Q")
        st.dataframe(Q)
    with colOmega:
        st.write("Omega")
        st.dataframe(omega)

    MV_black_litterman = BlackLittermanOptimization(data, constraints, rf_rate=rf_rate, ratings=ratings,
                                                    P=P, Q=Q, omega=omega, tau=0.05)
    if checkbox:
        MV_black_litterman.set_special_constraints(sp_rating_to_number[start_rating], sp_rating_to_number[end_rating],
                                   percentage_rating)
    _, opti_weights = MV_black_litterman.optimize_black_litterman()

    # Make a dataframe with the optimal weights
    opti_weights = pd.DataFrame(opti_weights, index=tickers, columns=["Weights"])

with st.expander("Black-Litterman Optimization"):
    st.write("Optimal weights")
    st.bar_chart(opti_weights)
