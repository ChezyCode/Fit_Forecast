import pandas as pd
import yfinance as yf



# Getting data from Yahoo finance
def get_data(stock_name, start_date, end_date):
    data: pd.DataFrame = yf.download(stock_name, start=start_date, end=end_date)
    closing_prices = data['Close']
    s_obs = closing_prices.to_numpy()
    return s_obs

# Getting data from Yahoo finance
# def get_data(stock_name, start_date, end_date):
#     data: pd.DataFrame = yf.download(stock_name, start=start_date, end=end_date, auto_adjust=False)
#     closing_prices = data['Close']
#     s_act = closing_prices.to_numpy()

#     filtered_prices = [s_act[0]]
#     for i in range(1, len (s_act)):
#         if s_act[i] != s_act[i-1]:
#             filtered_prices.append(s_act[i])
#         return filtered_prices
#     return filtered_prices



#  closing_prices = gd.get_data(stock_name, start_date, end_date)
#     closing_prices = [price.item() for price in closing_prices]
#     closing_prices = filter_prices_duplicates(closing_prices)
#     forecast_days = len(closing_prices) - len(S_n_list)


# def filter_prices_duplicates(closing_prices):
#     filtered_prices = [closing_prices[0]]

#     for i in range(1, len(closing_prices)):
#         if closing_prices[i] != closing_prices[i-1]:
#             filtered_prices.append(closing_prices[i])
#     return filtered_prices