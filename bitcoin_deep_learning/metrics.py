from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_array
from numpy import mean, abs
import numpy as np
import pandas as pd

from bitcoin_deep_learning.call_api import ApiCall
from bitcoin_deep_learning.model import LinearRegressionBaselineModel
from bitcoin_deep_learning.cross_val import cross_val
from bitcoin_deep_learning.cross_val import cross_val_metrics

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

def compute_sharpe_ratio(play_strategy):
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

    # assert btc_usd_balance == daily_portfolio_position[- investment_frequency]

    return pd.Series(daily_portfolio_position)

def play_trader_strategy_2(y_true,
                           y_pred,
                           total_investment = 400,
                           investment_frequency = 7,
                           buy_threshold = 0.05,
                           sell_threshold = -0.10,
                           buy_multiplicator = 3,
                           sell_multiplicator = 3,
                           exchange_fee = 0.005,
                           tax_rate = 0.30):
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
    # sell_multiplicator = 3

    # Enter the exchange fee applicable to each trade (e.g., 0.5% on Coinbase Pro)
    # exchange_fee = 0.005

    # Enter the applicable tax rate (e.g., 30% for financial gains in France)
    # tax_rate = 0.30

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

    # Cost_basis list for tax calculation purposes
    cost_basis = []

    # Amount bought for cost_basis weighting
    amount_bought = []

    # Taxable basis (i.e., profits taken)
    taxable_basis = []

    taxes = 0

    # Counters for indexes during for-loop
    counter = 0
    counter_percent_change = 0

    # Invest at each period of the investment horizon (e.g., weekly)
    for value in y_pred[::investment_frequency]:

        # Buy/sell only based on a prediction
        if len(y_pred[::investment_frequency]) > counter + 1:
            #print((list(y_pred)[::investment_frequency][counter + 1] / list(y_true)[::investment_frequency][counter]) - 1 < -investment_threshold)

            predicted_move_ratio = (list(y_pred)[::investment_frequency][counter + 1] / list(y_true)[::investment_frequency][counter]) - 1

            #print(predicted_move_ratio)

            if predicted_move_ratio > buy_threshold:
                if usd_balance <= 0:
                    pass
                else:

                    ### Tax ###
                    cost_basis.append(list(y_true)[::investment_frequency][counter])
                    amount_bought.append(usd_balance * (predicted_move_ratio * buy_multiplicator))
                    ### Tax ###

                    btc_usd_balance += usd_balance * (predicted_move_ratio * buy_multiplicator) - ((usd_balance * (predicted_move_ratio * buy_multiplicator)) * exchange_fee)
                    usd_balance -= usd_balance * (predicted_move_ratio * buy_multiplicator)
                    # print(btc_usd_balance)


                # TODO (buy because price is predicted to go up)

            if predicted_move_ratio < sell_threshold:
                if btc_usd_balance == 0:
                    pass

                else:

                    ### Tax ###
                    amount_sold = btc_usd_balance * (- predicted_move_ratio * sell_multiplicator) - ((btc_usd_balance * (- predicted_move_ratio * sell_multiplicator)) * exchange_fee)
                    wa_cost_basis = (np.array(amount_bought) * np.array(cost_basis)).sum() / np.array(amount_bought).sum()
                    taxable_basis.append(amount_sold * ((list(y_true)[::investment_frequency][counter] / wa_cost_basis) - 1))

                    for index in range(len(amount_bought)):
                        amount_bought[index] -= amount_sold / len(amount_bought)

                    ### Tax ###
                    # print(amount_bought)

                    usd_balance += btc_usd_balance * (- predicted_move_ratio * sell_multiplicator) - ((btc_usd_balance * (- predicted_move_ratio * sell_multiplicator)) * exchange_fee)
                    btc_usd_balance -= btc_usd_balance * (- predicted_move_ratio * sell_multiplicator)
                    #print(btc_usd_balance)
                # TODO (buy because price is predicted to go down)


        for days in range(investment_frequency):
            if days == 0:
                daily_portfolio_position.append(usd_balance + btc_usd_balance)
            else:
                daily_portfolio_position.append(usd_balance + (btc_usd_balance + (btc_usd_balance * list(y_true_daily_pct_change)[days])))
            #print(btc_usd_balance)
        counter += 1

        # print(btc_usd_balance)
        # print(investment_frequency_returns)

    # Return On Investment = btc_usd_balance / total invested
    # roi = (btc_usd_balance / total_investment) -1


    # Return On Investment after taxes on trades

    if np.array(taxable_basis).sum() > 0:
        taxes += np.array(taxable_basis).sum() * tax_rate
        daily_portfolio_position[-1] -= taxes

    # roi_after_taxes_on_trades = ((btc_usd_balance + usd_balance - taxes) / total_investment) - 1

    #assert btc_usd_balance == daily_portfolio_position[- investment_frequency]

    #return pd.Series(daily_portfolio_position)
    return pd.Series(daily_portfolio_position)

def iterate_cross_val_results(model = LinearRegressionBaselineModel(),
                              df = ApiCall().read_local()):

    roi_hodl = []
    roi_trader = []
    sharpe_hodl = []
    sharpe_trader = []

    for reality, prediction, score in cross_val_metrics(model, df):
        y_true, y_pred, score = reality, prediction, score

        roi_hodl.append(compute_roi(play_hodl_strategy(y_true, y_pred)))
        roi_trader.append(compute_roi(play_trader_strategy_2(y_true, y_pred)))
        sharpe_hodl.append(compute_sharpe_ratio(play_hodl_strategy(y_true, y_pred)))
        sharpe_trader.append(compute_sharpe_ratio(play_trader_strategy_2(y_true, y_pred)))

    return np.array(roi_hodl).mean(), np.array(roi_trader).mean(), np.array(sharpe_hodl).mean(), np.array(sharpe_trader)

if __name__ == '__main__':
    # df = ApiCall().read_local()
    # y_true = df["[+]_[T]_Bitcoin_Price"].copy().tail(450)
    # y_pred = y_true
    # model = LinearRegressionBaselineModel()
    # df = ApiCall().read_local()
    # print(cross_val(model,df=df))
    # print("Hodl strategy (1) ROI and (2) Sharpe Ratio:")
    # print(compute_roi(play_hodl_strategy(y_true, y_pred)))
    # print(compute_sharpe_ratio(play_hodl_strategy(y_true, y_pred)))
    # print("Trader strategy (1) ROI and (2) Sharpe Ratio:")
    # print(compute_roi(play_trader_strategy_2(y_true, y_pred)))
    # print(compute_sharpe_ratio(play_trader_strategy_2(y_true, y_pred)))
    model = LinearRegressionBaselineModel()
    df = ApiCall().read_local()
    roi_hold, roi_trader, sharpe_hodl, sharpe_trader = iterate_cross_val_results(
        model = model,
        df = df)
    print("Hodler roi is: ", roi_hold)
    print("Hodler sharpe ratio is: ", sharpe_hodl)
    print("Trader roi is:  ", roi_trader)
    print("Trader sharpe ratio is: ", sharpe_trader)

#### NOTES ####

# y_true = (90, 1)
# y_pred = (90, 1) # dollar value
# .predict here
