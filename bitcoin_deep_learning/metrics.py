from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_array
from numpy import mean, abs
import numpy as np
import pandas as pd

def mean_absolute_scaled_error(y_true, y_pred, y_baseline):
    e_t = y_true - y_pred
    scale = mean_absolute_error(y_baseline[1:], y_baseline[:-1])
    return mean(abs(e_t / scale))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = check_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#### Compute ROI, Returns, Sharp Ratio, based on the selected play_strategy ####

def compute_roi(play_strategy):
    return (list(play_strategy)[-1] / list(play_strategy)[0]) - 1

def compute_returns(play_strategy):
    return play_strategy.pct_change()[1:]

def compute_sharp_ratio(play_strategy):
    return compute_roi(play_strategy) / compute_returns(play_strategy).std()

#### Define the play_strategies (hold and trader) ####

def play_hodl_strategy(y_true,
                       y_pred,
                       total_investment = 400,
                       investment_frequency = 7,
                       exchange_fee = 0.005):
    """
    Takes y_true and y_pred pd.Series, total_investment, investment_frequency (in days), exchange_fee.
    Returns a pd.Series of the daily portfolio positions).
    """

    # Enter total USD amount to be invested over the period (i.e., budget)
    # total_investment = 400

    # Enter the investment horizon in days (e.g., 7 for a weekly investement strategy)
    # investment_frequency = 7

    # Enter the exchange fee applicable to each trade (e.g., 0.5% on Coinbase Pro)
    # exchange_fee = 0.005

    # Periodic investment amount (e.g., total investment / number of weeks)
    investment_amount = total_investment / (len(y_true) / investment_frequency)

    # List of daily percent change of y_true
    y_true_daily_pct_change = pd.Series(y_true).pct_change()

    # List of daily_portfolio position (usd + btc_usd)
    daily_portfolio_position = []

    # List of the price percent change over the investment horizon (e.g., D7 price / D0 price)
    y_true_percent_change = [(value / list(y_true)[index - investment_frequency])-1 for index, value in enumerate(list(y_true)) if index % investment_frequency == 0][1:]

    # List of returns over the investment horizon (e.g., D7 price / D0 price)
    investment_frequency_returns = []

    # USD balance
    usd_balance = total_investment

    # Bitcoin balance in USD
    btc_usd_balance = 0

    # Counters for indexes during for-loop
    counter = 0
    counter_percent_change = 0

    # Invest at each period of the investment horizon (e.g., weekly)
    for value in y_true[::investment_frequency]:

        usd_balance -= investment_amount

        if btc_usd_balance > 0:
            investment_frequency_returns.append(btc_usd_balance * y_true_percent_change[counter_percent_change])
            btc_usd_balance += investment_amount + investment_frequency_returns[counter_percent_change] - (investment_amount * exchange_fee)
            counter_percent_change += 1

        else:
            btc_usd_balance += investment_amount - (investment_amount * exchange_fee)

        for days in range(investment_frequency):
            if days == 0:
                daily_portfolio_position.append(usd_balance + btc_usd_balance)
            else:
                daily_portfolio_position.append(usd_balance + (btc_usd_balance + (btc_usd_balance * list(y_true_daily_pct_change)[days])))


        counter += 1

        # print(btc_usd_balance)
        #print(investment_frequency_returns)

    # Return On Investment = btc_usd_balance / total invested
    # roi = (btc_usd_balance / total_investment) -1

    assert btc_usd_balance == daily_portfolio_position[- investment_frequency]

    return pd.Series(daily_portfolio_position)


def play_trader_strategy(y_true,
                         y_pred,
                         total_investment = 400,
                         investment_frequency = 7,
                         buy_threshold = 0.05,
                         sell_threshold = -0.10,
                         buy_multiplicator = 2,
                         sell_multiplicator = 2,
                         exchange_fee = 0.005):
    """
    Takes y_true and y_pred pd.Series, total_investment, investment_frequency (in days), investment_threshold, exchange_fee.
    Returns a pd.Series of the daily portfolio positions).
    """

    # Enter total USD amount to be invested over the period (i.e., budget)
    # total_investment = 400

    # Enter the investment horizon in days (e.g., 7 for a weekly investement strategy)
    # investment_frequency = 7

    # Enter a buy threshold (i.e., buy if prediction is above/below a certain percentage)
    # buy_threshold = 0.05

    # Enter a sell threshold (i.e., sell if prediction is above/below a certain percentage)
    # sell_threshold = -0.10

    # Enter a buyer multiplicator (e.g., 2 times the price increase rate predicted: 2 * 15%)
    # buy_multiplicator = 2

    # Enter a seller multiplicator (e.g., 2 times the price decrease rate predicted: 2 * 15%)
    # sell_multiplicator = 2

    # Enter the exchange fee applicable to each trade (e.g., 0.5% on Coinbase Pro)
    # exchange_fee = 0.005

    # Periodic investment amount (e.g., total investment / number of weeks)
    # investment_amount = total_investment / (len(y_true) / investment_frequency)

    # List of daily percent change of y_true
    y_true_daily_pct_change = pd.Series(y_true).pct_change()

    # List of daily_portfolio position (usd + btc_usd)
    daily_portfolio_position = []

    # List of the price percent change over the investment horizon (e.g., D7 price / D0 price)
    # y_true_percent_change = [(value / list(y_true)[index - investment_frequency])-1 for index, value in enumerate(list(y_true)) if index % investment_frequency == 0][1:]

    # List of returns over the investment horizon (e.g., D7 price / D0 price)
    # investment_frequency_returns = []

    # USD balance
    usd_balance = total_investment

    # Bitcoin balance in USD
    btc_usd_balance = 0

    # Counters for indexes during for-loop
    counter = 0
    # counter_percent_change = 0

    # Invest at each period of the investment horizon (e.g., weekly)
    for value in y_pred[::investment_frequency]:

        # Buy/sell only based on a prediction
        if len(y_pred[::investment_frequency]) > counter + 1:
            #print((list(y_pred)[::investment_frequency][counter + 1] / list(y_true)[::investment_frequency][counter]) - 1 < -investment_threshold)

            predicted_move_ratio = (list(y_pred)[::investment_frequency][counter + 1] / list(y_true)[::investment_frequency][counter]) - 1

            print(predicted_move_ratio)

            if predicted_move_ratio > buy_threshold:
                if usd_balance <= 0:
                    pass
                else:
                    btc_usd_balance += usd_balance * (predicted_move_ratio * buy_multiplicator) - ((usd_balance * (predicted_move_ratio * buy_multiplicator)) * exchange_fee)
                    usd_balance -= usd_balance * (predicted_move_ratio * buy_multiplicator)
                    print(btc_usd_balance)

            if predicted_move_ratio < sell_threshold:
                if btc_usd_balance == 0:
                    pass
                else:
                    usd_balance += btc_usd_balance * (- predicted_move_ratio * sell_multiplicator) - ((btc_usd_balance * (- predicted_move_ratio * sell_multiplicator)) * exchange_fee)
                    btc_usd_balance -= btc_usd_balance * (- predicted_move_ratio * sell_multiplicator)
                    print(btc_usd_balance)

        for days in range(investment_frequency):
            if days == 0:
                daily_portfolio_position.append(usd_balance + btc_usd_balance)
            else:
                daily_portfolio_position.append(usd_balance + (btc_usd_balance + (btc_usd_balance * list(y_true_daily_pct_change)[days])))

        counter += 1

        # print(btc_usd_balance)
        #print(investment_frequency_returns)

    # Return On Investment = btc_usd_balance / total invested
    # roi = (btc_usd_balance / total_investment) -1


    #assert btc_usd_balance == daily_portfolio_position[- investment_frequency]

    return pd.Series(daily_portfolio_position)


#### NOTES ####

# y_true = (90, 1)
# y_pred = (90, 1) # dollar value
# .predict here
