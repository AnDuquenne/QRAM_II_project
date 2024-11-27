import pandas as pd
import sys
import os
import io

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Creating the table as a DataFrame
data = {
    "Company Name": [
        "3M Company", "American Express", "Amgen", "Apple", "Boeing", "Caterpillar",
        "Chevron", "Cisco Systems", "Coca-Cola", "Dow Inc.", "Goldman Sachs",
        "Home Depot", "Honeywell International", "Intel", "International Business Machines (IBM)",
        "Johnson & Johnson", "JPMorgan Chase", "McDonald's", "Merck & Co.",
        "Microsoft", "Nike", "Procter & Gamble", "Salesforce", "Travelers Companies",
        "UnitedHealth Group", "Verizon Communications", "Visa", "Walgreens Boots Alliance",
        "Walmart", "Walt Disney"
    ],
    "S&P Rating": [
        "AA-", "BBB+", "A-", "AA+", "BBB-", "A", "AA", "AA-", "A+", "BBB+",
        "BBB+", "A", "A", "A+", "A-", "AAA", "A-", "BBB+", "A+", "AAA",
        "AA-", "AA-", "A", "AA", "A+", "BBB+", "AA-", "BBB", "AA", "A-"
    ],
    "Moody's Rating": [
        "A1", "A3", "Baa1", "Aaa", "Baa2", "A2", "Aa2", "A1", "Aa3", "Baa2",
        "A2", "A2", "A2", "A1", "A3", "Aaa", "Aa2", "Baa1", "A1", "Aaa",
        "A1", "Aa3", "A2", "Aa2", "A3", "Baa1", "Aa3", "Baa2", "Aa2", "A2"
    ],
    "Fitch Rating": [
        "AA-", "A-", "A-", "NR", "BBB", "A", "AA", "A+", "A", "BBB+",
        "A", "A", "A", "A+", "A-", "AAA", "AA-", "BBB+", "A+", "AAA",
        "NR", "NR", "NR", "AA", "A+", "BBB+", "NR", "BBB", "AA", "A-"
    ]
}

df = pd.DataFrame(data)

tickers = [
    "MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DOW", "GS",
    "HD", "HON", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "PG",
    "CRM", "TRV", "UNH", "VZ", "V", "WBA", "WMT", "DIS"
]

df['Ticker'] = tickers

sp_rating_to_number = {
    "AAA": 1,
    "AA+": 2,
    "AA": 3,
    "AA-": 4,
    "A+": 5,
    "A": 6,
    "A-": 7,
    "BBB+": 8,
    "BBB": 9,
    "BBB-": 10,
    "BB+": 11,
    "BB": 12,
    "BB-": 13,
    "B+": 14,
    "B": 15,
    "B-": 16,
    "CCC+": 17,
    "CCC": 18,
    "CCC-": 19,
    "CC": 20,
    "C": 21,
    "D": 22
}

df['S&P Rating Number'] = df['S&P Rating'].map(sp_rating_to_number)

# Saving the DataFrame as a CSV file
file_path = "app/io/Dow_Jones_Credit_Ratings_Oct_2023.csv"
df.to_csv(file_path, index=False)

