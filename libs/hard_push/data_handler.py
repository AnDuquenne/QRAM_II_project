import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------- XLSX to CSV --------------------------------------------------- #
#
# # import the xlsx files from data folder
# df = pd.read_excel('Data/Baselines.xlsx', sheet_name='Baselines')
#
# # rename the first column to 'Date'
# df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
#
# # Change the date column to datetime
# df['Date'] = pd.to_datetime(df['Date'])
#
# # Save the data to a new csv file
# df.to_csv('Data/Baselines.csv', index=False)
#
# # Import the second xlsx file
# df = pd.read_excel('Data/Baselines.xlsx', sheet_name='SP500')
#
# # rename the first column to 'Date'
# df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
#
# # Change the date column to datetime
# df['Date'] = pd.to_datetime(df['Date'])
#
# # Save the data to a new csv file
# df.to_csv('Data/SP500.csv', index=False)
#
# # Import the third xlsx file
# df = pd.read_excel('Data/Baselines.xlsx', sheet_name='NASDAQ100')
#
# # rename the first column to 'Date'
# df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
#
# # Change the date column to datetime
# df['Date'] = pd.to_datetime(df['Date'])
#
# # Save the data to a new csv file
# df.to_csv('Data/NASDAQ100.csv', index=False)

# --------------------------------------------------- Check data --------------------------------------------------- #

df_baselines = pd.read_csv('Data/Baselines.csv')
df_sp500 = pd.read_csv('Data/SP500.csv')
df_nasdaq100 = pd.read_csv('Data/NASDAQ100.csv')

# # print the first 5 rows of the data
# print(df_baselines.head())
# print(df_sp500.head())
# print(df_nasdaq100.head())
#
# # print summary of the data
# print(df_baselines.info())
# print(df_sp500.info())
# print(df_nasdaq100.info())

# --------------------------------------------------- Clean data --------------------------------------------------- #
#
# 1. Baselines
# Check for missing values
print(df_baselines.isnull().sum())

# histogram of the data
df_baselines.hist()
plt.show()

# Handle 0 values by replacing them with the previous value
df_baselines.replace(0, np.nan, inplace=True)
df_baselines.ffill(inplace=True)

# 2. SP500
# Check for missing values
print(df_sp500.isnull().sum())

# Remove all columns with missing values
df_sp500.dropna(inplace=True, axis=1)
print(df_sp500.isnull().sum())
print(df_sp500.info())

# ----------------------------------------------- Prepocess the data ----------------------------------------------- #

# 1. Baselines
# Set the date as the index
df_baselines.set_index('Date', inplace=True)

# Compute the returns
df_baselines = df_baselines.pct_change().iloc[1:, :]

# Normalize the data to take values between 0 and 1 using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_baselines.iloc[:, :] = pd.DataFrame(scaler.fit_transform(df_baselines))

# 2. SP500
# Set the date as the index
df_sp500.set_index('Date', inplace=True)

# Compute the returns
df_sp500 = df_sp500.pct_change().iloc[1:, :]

# Normalize the data to take values between 0 and 1 using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_sp500.iloc[:, :] = pd.DataFrame(scaler.fit_transform(df_sp500))

print(df_baselines.head())
print(df_sp500.head())

# Save the data
df_baselines.to_csv('Data/Baselines_cleaned.csv')
df_sp500.to_csv('Data/SP500_cleaned.csv')
