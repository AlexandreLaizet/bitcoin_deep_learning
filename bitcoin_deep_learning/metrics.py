from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_array
from numpy import mean, abs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from bitcoin_deep_learning.call_api import ApiCall
from bitcoin_deep_learning.model import LinearRegressionBaselineModel
from bitcoin_deep_learning.model import DummyModel
from bitcoin_deep_learning.model import RnnDlModel

from bitcoin_deep_learning.cross_val import cross_val_trade

class Mean_absolute_percentage_error():
    def __init__(self):
        self.name = "MAPE"

    def compute(self,y_true,y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_absolute_scaled_error(y_true, y_pred, y_baseline):
    #
    e_t = y_true - y_pred
    scale = mean_absolute_error(y_baseline[1:], y_baseline[:-1])
    return mean(abs(e_t / scale))

def mean_absolute_percentage_error(y_true, y_pred):
    # Alex, WHAT DOES CHECK ARRAY DO ? IS IT NECESSARY ?
    #y_true, y_pred = check_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#### Compute ROI, Returns, Sharp Ratio, based on the selected play_strategy ####

def compute_roi(select_strategy):
    return (list(select_strategy)[-1] / list(select_strategy)[0]) - 1

def compute_returns(select_strategy):
    return select_strategy.pct_change()[1:]

def compute_sharpe_ratio(select_strategy):
    if compute_returns(select_strategy).sum() == 0:
        return 0
    else:
        return compute_roi(select_strategy) / compute_returns(select_strategy).std()

#### Define the play_strategies ####

def play_hodler_strategy(y_true,
                        y_pred,
                        total_investment = 100,
                        investment_horizon = 7,
                        exchange_fee = 0.005,
                        tax_rate = 0.30):
    """
    Hodler strategy:
    (1) Buys the same amount following a regular investment horizon without consideration of price prediction.
    (2) Invests equal portions of total_investment at each investment horizon step (i.e., Dollar Cost Averaging - "DCA").
    (3) Never sells ("diamond hands"). However, in order to calculate potential returns, a tax rate (in case of profit taking) will be applied on the last day of the hodler's portfolio.
    """

    # Lists
    daily_usd_position = []
    daily_btc_position = []
    daily_btc_usd_position = []
    daily_portfolio_position = []
    investments = []
    cost_basis = []
    taxable_basis = []

    # Trackers
    investment = total_investment / (len(list(y_true)) / investment_horizon)
    usd_balance = total_investment
    btc_balance = 0
    btc_usd_balance = 0
    taxes = 0
    counter = 0

    # Loop
    for value in list(y_true)[::investment_horizon]:

        if counter == 0:
            usd_balance -= investment
            btc_usd_balance += investment - (investment * exchange_fee)
            btc_balance = btc_usd_balance / value

            # Tax-prep-start
            investments.append(investment)
            cost_basis.append(value)
            # Tax-prep-end

        if counter > 0 and counter < len(list(y_true)[::investment_horizon]) - 1 and usd_balance > investment:
            btc_usd_balance = btc_balance * value
            usd_balance -= investment
            btc_usd_balance += investment - (investment * exchange_fee)
            btc_balance = btc_usd_balance / value

            # Tax-prep-start
            investments.append(investment)
            cost_basis.append(value)
            # Tax-prep-end

        if counter > 0 and counter == len(list(y_true)[::investment_horizon]) - 1:
            btc_usd_balance = btc_balance * value
            usd_balance += btc_usd_balance - (btc_usd_balance * exchange_fee)
            btc_usd_balance = 0
            btc_balance = 0

            # Tax-calc-start
            wa_cost_basis = (np.array(investments) * np.array(cost_basis)).sum() / np.array(investments).sum()
            taxable_basis.append(usd_balance * ((value / wa_cost_basis) - 1))
            investments.clear()
            cost_basis.clear()
            # Tax-calc-end

        if counter < int(len(list(y_true)) / investment_horizon):
            for day in range(investment_horizon):
                daily_usd_position.append(usd_balance)
                daily_btc_position.append(btc_balance)
                btc_usd_balance = btc_balance * list(y_true)[counter * investment_horizon + day]
                daily_btc_usd_position.append(btc_usd_balance)
                daily_portfolio_position.append(usd_balance + btc_usd_balance)

        if counter == int(len(list(y_true)) / investment_horizon):
            for day in range((counter) - (len(list(y_true)) - (counter * investment_horizon))):
                daily_usd_position.append(usd_balance)
                daily_btc_position.append(btc_balance)
                btc_usd_balance = btc_balance * list(y_true)[counter * investment_horizon + day]
                daily_btc_usd_position.append(btc_usd_balance)
                daily_portfolio_position.append(usd_balance + btc_usd_balance)

        ### print(counter)

        counter += 1

    # Tax-pay-start
    if np.array(taxable_basis).sum() > 0:
        taxes += np.array(taxable_basis).sum() * tax_rate
        daily_portfolio_position[-1] -= taxes
    # Tax-pay-end

    ### print(np.array(taxable_basis).sum() * tax_rate)

    return pd.Series(daily_portfolio_position)


def play_trader_strategy(y_true,
                         y_pred,
                         total_investment = 100,
                         investment_horizon = 7,
                         buy_threshold = 0.10,
                         sell_threshold = - 0.05,
                         exchange_fee = 0.005,
                         tax_rate = 0.30):
    """
    Trader strategy:
    (1) Assess daily if the predicted price will reach the buy_threshold or the sell_threshold over the investment_horizon.
    (2) Invest total_investment if price is predicted to increase by at least the buy_threshold over the investment_horizon.
    (3) After reaching the investment horizon, sells total_investment if price is predicted to decrease by at least the sell_threshold over the investment_horizon.
    (4) Repeat process from (1).
    """

    # Lists
    daily_usd_position = []
    daily_btc_position = []
    daily_btc_usd_position = []
    daily_portfolio_position = []
    investments = []
    cost_basis = []
    taxable_basis = []

    # Trackers
    usd_balance = total_investment
    btc_balance = 0
    btc_usd_balance = 0
    investment = 0
    reassessment_day = 0
    taxes = 0
    counter = 0

    # Loop
    y_true = list(y_true)
    y_pred = list(y_pred)
    for value in y_true:
        if len(list(y_pred)) > counter + investment_horizon:

            if ((list(y_pred)[counter + investment_horizon] / value) -1) > buy_threshold:
            #if list(y_pred)[counter + investment_horizon] > buy_threshold:

                if usd_balance > 0:

                    investment = usd_balance - (usd_balance * exchange_fee)

                    # Tax-prep-start
                    investments.append(investment)
                    cost_basis.append(value)
                    # Tax-prep-end

                    usd_balance = 0
                    btc_usd_balance += investment
                    btc_balance += btc_usd_balance / value
                    reassessment_day = counter + investment_horizon

                    ### print(f"price bought {value}")
                    ### print(f"price predicted {list(y_pred)[counter + investment_horizon]}")

                else:
                    btc_usd_balance = btc_balance * value

            if usd_balance == 0:

                if counter >= reassessment_day:

                    if ((list(y_pred)[counter + investment_horizon] / value) -1) < sell_threshold:
                    #if list(y_pred)[counter + investment_horizon] < sell_threshold:

                        btc_usd_balance = btc_balance * value
                        usd_balance += btc_usd_balance - (btc_usd_balance * exchange_fee)
                        btc_usd_balance = 0
                        btc_balance = 0

                        ### print(f"price sold {value}")

                        # Tax-calc-start
                        wa_cost_basis = (np.array(investments) * np.array(cost_basis)).sum() / np.array(investments).sum()
                        taxable_basis.append(usd_balance * ((value / wa_cost_basis) - 1))
                        investments.clear()
                        cost_basis.clear()
                        # Tax-calc-end

                    else:
                        btc_usd_balance = btc_balance * value

                else:
                    btc_usd_balance = btc_balance * value

        if len(list(y_pred)) <= counter + investment_horizon:

            if counter >= reassessment_day:

                if btc_usd_balance > 0:

                    btc_usd_balance = btc_balance * value
                    usd_balance += btc_usd_balance - (btc_usd_balance * exchange_fee)
                    btc_usd_balance = 0
                    btc_balance = 0

                    ### print(f"price sold {value}")

                    # Tax-calc-start
                    wa_cost_basis = (np.array(investments) * np.array(cost_basis)).sum() / np.array(investments).sum()
                    taxable_basis.append(usd_balance * ((value / wa_cost_basis) - 1))
                    investments.clear()
                    cost_basis.clear()
                    # Tax-calc-end

                else:
                    btc_usd_balance = btc_balance * value
            else:
                btc_usd_balance = btc_balance * value

        daily_usd_position.append(usd_balance)
        daily_btc_position.append(btc_balance)
        daily_btc_usd_position.append(btc_usd_balance)
        daily_portfolio_position.append(usd_balance + btc_usd_balance)

        counter += 1

        ### print(counter)

    # Tax-pay-start
    if np.array(taxable_basis).sum() > 0:
        taxes += np.array(taxable_basis).sum() * tax_rate
        daily_portfolio_position[-1] -= taxes
    # Tax-pay-end

    ### print(np.array(taxable_basis).sum() * tax_rate)

    return pd.Series(daily_portfolio_position)

def play_whale_strategy(y_true,
                        y_pred,
                        total_investment = 100,
                        investment_horizon = 7,
                        buy_threshold = 0.15,
                        sell_threshold = - 0.05,
                        exchange_fee = 0.005,
                        tax_rate = 0.30):
    """
    Whale strategy:
    (1) Assess daily if the predicted price will reach the buy_threshold or the sell_threshold over the investment_horizon.
    (2) Invest total_investment if price is predicted to increase by at least the buy_threshold over the investment_horizon.
    (3) After reaching the investment horizon, sells total_investment if price is predicted to decrease by at least the sell_threshold over the investment_horizon.
    (4) Repeat process from (1).
    """

    # Lists
    daily_usd_position = []
    daily_btc_position = []
    daily_btc_usd_position = []
    daily_portfolio_position = []
    investments = []
    cost_basis = []
    taxable_basis = []

    # Trackers
    usd_balance = total_investment
    btc_balance = 0
    btc_usd_balance = 0
    investment = 0
    reassessment_day = 0
    taxes = 0
    counter = 0

    # Loop
    for value in list(y_true):

        if len(list(y_pred)) > counter + investment_horizon:

            if ((list(y_pred)[counter + investment_horizon] / value) -1) > buy_threshold:
            #if list(y_pred)[counter + investment_horizon] > buy_threshold:

                if usd_balance > 0:

                    investment = usd_balance - (usd_balance * exchange_fee)

                    # Tax-prep-start
                    investments.append(investment)
                    cost_basis.append(value)
                    # Tax-prep-end

                    usd_balance = 0
                    btc_usd_balance += investment
                    btc_balance += btc_usd_balance / value
                    reassessment_day = counter + investment_horizon

                    ### print(f"price bought {value}")
                    ### print(f"price predicted {list(y_pred)[counter + investment_horizon]}")

                else:
                    btc_usd_balance = btc_balance * value

            if usd_balance == 0:

                if counter >= reassessment_day:

                    if ((list(y_pred)[counter + investment_horizon] / value) -1) < sell_threshold:
                    #if list(y_pred)[counter + investment_horizon] < sell_threshold:

                        btc_usd_balance = btc_balance * value
                        usd_balance += btc_usd_balance - (btc_usd_balance * exchange_fee)
                        btc_usd_balance = 0
                        btc_balance = 0

                        ### print(f"price sold {value}")

                        # Tax-calc-start
                        wa_cost_basis = (np.array(investments) * np.array(cost_basis)).sum() / np.array(investments).sum()
                        taxable_basis.append(usd_balance * ((value / wa_cost_basis) - 1))
                        investments.clear()
                        cost_basis.clear()
                        # Tax-calc-end

                    else:
                        btc_usd_balance = btc_balance * value

                else:
                    btc_usd_balance = btc_balance * value

        if len(list(y_pred)) <= counter + investment_horizon:

            if counter >= reassessment_day:

                if btc_usd_balance > 0:

                    btc_usd_balance = btc_balance * value
                    usd_balance += btc_usd_balance - (btc_usd_balance * exchange_fee)
                    btc_usd_balance = 0
                    btc_balance = 0

                    ### print(f"price sold {value}")

                    # Tax-calc-start
                    wa_cost_basis = (np.array(investments) * np.array(cost_basis)).sum() / np.array(investments).sum()
                    taxable_basis.append(usd_balance * ((value / wa_cost_basis) - 1))
                    investments.clear()
                    cost_basis.clear()
                    # Tax-calc-end

                else:
                    btc_usd_balance = btc_balance * value
            else:
                btc_usd_balance = btc_balance * value

        daily_usd_position.append(usd_balance)
        daily_btc_position.append(btc_balance)
        daily_btc_usd_position.append(btc_usd_balance)
        daily_portfolio_position.append(usd_balance + btc_usd_balance)

        counter += 1

        ### print(counter)

    # Tax-pay-start
    if np.array(taxable_basis).sum() > 0:
        taxes += np.array(taxable_basis).sum() * tax_rate
        daily_portfolio_position[-1] -= taxes
    # Tax-pay-end

    ### print(np.array(taxable_basis).sum() * tax_rate)

    return pd.Series(daily_portfolio_position)

def play_hodler_whale_strategy(y_true,
                               y_pred,
                               total_investment = 100,
                               investment_horizon = 7,
                               buy_threshold = 0.15,
                               sell_threshold = - 1,
                               exchange_fee = 0.005,
                               tax_rate = 0.30):
    """
    Hodler whale strategy:
    (1) Assess daily if the predicted price will reach the buy_threshold or the sell_threshold over the investment_horizon.
    (2) Invest total_investment if price is predicted to increase by at least the buy_threshold over the investment_horizon.
    (3) After reaching the investment horizon, sells total_investment if price is predicted to decrease by at least the sell_threshold over the investment_horizon.
    (4) Repeat process from (1).
    """

    # Lists
    daily_usd_position = []
    daily_btc_position = []
    daily_btc_usd_position = []
    daily_portfolio_position = []
    investments = []
    cost_basis = []
    taxable_basis = []

    # Trackers
    usd_balance = total_investment
    btc_balance = 0
    btc_usd_balance = 0
    investment = 0
    reassessment_day = 0
    taxes = 0
    counter = 0

    # Loop
    for value in list(y_true):

        if len(list(y_pred)) > counter + investment_horizon:

            if ((list(y_pred)[counter + investment_horizon] / value) -1) > buy_threshold:
            #if list(y_pred)[counter + investment_horizon] > buy_threshold:

                if usd_balance > 0:

                    investment = usd_balance - (usd_balance * exchange_fee)

                    # Tax-prep-start
                    investments.append(investment)
                    cost_basis.append(value)
                    # Tax-prep-end

                    usd_balance = 0
                    btc_usd_balance += investment
                    btc_balance += btc_usd_balance / value
                    reassessment_day = counter + investment_horizon

                    ### print(f"price bought {value}")
                    ### print(f"price predicted {list(y_pred)[counter + investment_horizon]}")

                else:
                    btc_usd_balance = btc_balance * value

            if usd_balance == 0:

                if counter >= reassessment_day:

                    if ((list(y_pred)[counter + investment_horizon] / value) -1) < sell_threshold:
                    #if list(y_pred)[counter + investment_horizon] < sell_threshold:

                        btc_usd_balance = btc_balance * value
                        usd_balance += btc_usd_balance - (btc_usd_balance * exchange_fee)
                        btc_usd_balance = 0
                        btc_balance = 0

                        ### print(f"price sold {value}")

                        # Tax-calc-start
                        wa_cost_basis = (np.array(investments) * np.array(cost_basis)).sum() / np.array(investments).sum()
                        taxable_basis.append(usd_balance * ((value / wa_cost_basis) - 1))
                        investments.clear()
                        cost_basis.clear()
                        # Tax-calc-end

                    else:
                        btc_usd_balance = btc_balance * value

                else:
                    btc_usd_balance = btc_balance * value

        if len(list(y_pred)) <= counter + investment_horizon:

            if counter >= reassessment_day:

                if btc_usd_balance > 0:

                    btc_usd_balance = btc_balance * value
                    usd_balance += btc_usd_balance - (btc_usd_balance * exchange_fee)
                    btc_usd_balance = 0
                    btc_balance = 0

                    ### print(f"price sold {value}")

                    # Tax-calc-start
                    wa_cost_basis = (np.array(investments) * np.array(cost_basis)).sum() / np.array(investments).sum()
                    taxable_basis.append(usd_balance * ((value / wa_cost_basis) - 1))
                    investments.clear()
                    cost_basis.clear()
                    # Tax-calc-end

                else:
                    btc_usd_balance = btc_balance * value
            else:
                btc_usd_balance = btc_balance * value

        daily_usd_position.append(usd_balance)
        daily_btc_position.append(btc_balance)
        daily_btc_usd_position.append(btc_usd_balance)
        daily_portfolio_position.append(usd_balance + btc_usd_balance)

        counter += 1

        ### print(counter)

    # Tax-pay-start
    if np.array(taxable_basis).sum() > 0:
        taxes += np.array(taxable_basis).sum() * tax_rate
        daily_portfolio_position[-1] -= taxes
    # Tax-pay-end

    ### print(np.array(taxable_basis).sum() * tax_rate)

    return pd.Series(daily_portfolio_position)

def play_charles_strategy(y_true,
                          y_pred,
                          total_investment = 100,
                          investment_horizon = 7,
                          buy_threshold = 0.05,
                          sell_threshold = - 0.30,
                          exchange_fee = 0.005,
                          tax_rate = 0.30):
    """
    Charles strategy:
    (1) Assess daily if the predicted price will reach the buy_threshold or the sell_threshold over the investment_horizon.
    (2) Invest total_investment if price is predicted to increase by at least the buy_threshold over the investment_horizon.
    (3) After reaching the investment horizon, sells total_investment if price is predicted to decrease by at least the sell_threshold over the investment_horizon.
    (4) Repeat process from (1).
    """

    # Lists
    daily_usd_position = []
    daily_btc_position = []
    daily_btc_usd_position = []
    daily_portfolio_position = []
    investments = []
    cost_basis = []
    taxable_basis = []

    # Trackers
    usd_balance = total_investment
    btc_balance = 0
    btc_usd_balance = 0
    investment = 0
    reassessment_day = 0
    taxes = 0
    counter = 0

    # Loop
    for value in list(y_true):

        if len(list(y_pred)) > counter + investment_horizon:

            if ((list(y_pred)[counter + investment_horizon] / value) -1) > buy_threshold:
            #if list(y_pred)[counter + investment_horizon] > buy_threshold:

                if usd_balance > 0:

                    investment = usd_balance - (usd_balance * exchange_fee)

                    # Tax-prep-start
                    investments.append(investment)
                    cost_basis.append(value)
                    # Tax-prep-end

                    usd_balance = 0
                    btc_usd_balance += investment
                    btc_balance += btc_usd_balance / value
                    reassessment_day = counter + investment_horizon

                    ###print(f"price bought {value}")
                    ###print(f"price predicted {list(y_pred)[counter + investment_horizon]}")

                else:
                    btc_usd_balance = btc_balance * value

            if usd_balance == 0:

                if counter >= reassessment_day:

                    if ((list(y_pred)[counter + investment_horizon] / value) -1) < sell_threshold:
                    #if list(y_pred)[counter + investment_horizon] < sell_threshold:

                        btc_usd_balance = btc_balance * value
                        usd_balance += btc_usd_balance - (btc_usd_balance * exchange_fee)
                        btc_usd_balance = 0
                        btc_balance = 0

                        ###print(f"price sold {value}")

                        # Tax-calc-start
                        wa_cost_basis = (np.array(investments) * np.array(cost_basis)).sum() / np.array(investments).sum()
                        taxable_basis.append(usd_balance * ((value / wa_cost_basis) - 1))
                        investments.clear()
                        cost_basis.clear()
                        # Tax-calc-end

                    else:
                        btc_usd_balance = btc_balance * value

                else:
                    btc_usd_balance = btc_balance * value

        if len(list(y_pred)) <= counter + investment_horizon:

            if counter >= reassessment_day:

                if btc_usd_balance > 0:

                    btc_usd_balance = btc_balance * value
                    usd_balance += btc_usd_balance - (btc_usd_balance * exchange_fee)
                    btc_usd_balance = 0
                    btc_balance = 0

                    ###print(f"price sold {value}")

                    # Tax-calc-start
                    wa_cost_basis = (np.array(investments) * np.array(cost_basis)).sum() / np.array(investments).sum()
                    taxable_basis.append(usd_balance * ((value / wa_cost_basis) - 1))
                    investments.clear()
                    cost_basis.clear()
                    # Tax-calc-end

                else:
                    btc_usd_balance = btc_balance * value
            else:
                btc_usd_balance = btc_balance * value

        daily_usd_position.append(usd_balance)
        daily_btc_position.append(btc_balance)
        daily_btc_usd_position.append(btc_usd_balance)
        daily_portfolio_position.append(usd_balance + btc_usd_balance)

        counter += 1

        ###print(counter)

    # Tax-pay-start
    if np.array(taxable_basis).sum() > 0:
        taxes += np.array(taxable_basis).sum() * tax_rate
        daily_portfolio_position[-1] -= taxes
    # Tax-pay-end

    ### print(np.array(taxable_basis).sum() * tax_rate)

    ###print(daily_portfolio_position[-1])

    return pd.Series(daily_portfolio_position)

def iterate_cross_val_results(model = LinearRegressionBaselineModel(),
                              df = ApiCall().read_local(data = 'train')):
    roi_hodler = []
    roi_trader = []
    roi_whale = []
    roi_hodler_whale = []
    roi_charles = []
    sharpe_hodler = []
    sharpe_trader = []
    sharpe_whale = []
    sharpe_hodler_whale = []
    sharpe_charles = []
    score_list = []

    portfolio_positions = []

    #realities, predictions, = cross_val(model, df)
    #past_realities, realities, realities_diff, prediction_diff = cross_val_trade(model, df)
    past_realities, realities, realities_diff, prediction_diff = cross_val_trade(model, df)
    preds_arr = []
    for past_prices, diffs in zip(past_realities,prediction_diff):
        preds_arr.append(past_prices * diffs.reshape(1,-1)+ past_prices )

    #for reality, prediction in zip(realities,prediction_diff):
    for reality, prediction in zip(realities, preds_arr):
        y_true, y_pred = reality, prediction

        roi_hodler.append(compute_roi(play_hodler_strategy(y_true, y_pred)))
        roi_trader.append(compute_roi(play_trader_strategy(y_true, y_pred)))
        roi_whale.append(compute_roi(play_whale_strategy(y_true, y_pred)))
        roi_hodler_whale.append(compute_roi(play_hodler_whale_strategy(y_true, y_pred)))
        roi_charles.append(compute_roi(play_charles_strategy(y_true, y_pred)))
        sharpe_hodler.append(compute_sharpe_ratio(play_hodler_strategy(y_true, y_pred)))
        sharpe_trader.append(compute_sharpe_ratio(play_trader_strategy(y_true, y_pred)))
        sharpe_whale.append(compute_sharpe_ratio(play_whale_strategy(y_true, y_pred)))
        sharpe_hodler_whale.append(compute_sharpe_ratio(play_hodler_whale_strategy(y_true, y_pred)))
        sharpe_charles.append(compute_sharpe_ratio(play_charles_strategy(y_true, y_pred)))
        #score_list.append(np.array(score).mean())

        #portfolio_positions.append()
    print(roi_charles)

    return np.array(roi_hodler).mean(), np.array(roi_trader).mean(), np.array(roi_whale).mean(), np.array(roi_hodler_whale).mean(), np.array(roi_charles).mean(), np.array(sharpe_hodler).mean(), np.array(sharpe_trader).mean(), np.array(sharpe_whale).mean(), np.array(sharpe_hodler_whale).mean(), np.array(sharpe_charles).mean()

def iterate_portfolio_positions(model = LinearRegressionBaselineModel(alpha = 0.05 , l1_ratio = 0.0001),
                                df = ApiCall().read_local(data = 'all'),
                                cv=False, verbose=True):

    portfolio_positions_hodler = []
    portfolio_positions_trader = []
    portfolio_positions_whale = []
    portfolio_positions_hodler_whale = []
    portfolio_positions_charles = []

    past_realities, realities, realities_diff, prediction_diff = cross_val_trade(model, df, cv=cv, verbose=verbose)
    preds_arr = []
    for past_prices, diffs in zip(past_realities,prediction_diff):
        preds_arr.append(past_prices * diffs.reshape(1,-1) + past_prices )


    for reality, prediction in zip(realities, preds_arr):
        y_true, y_pred = reality, prediction
        #print(y_true)
    for y_true, y_pred in zip(realities, preds_arr):
        #y_true, y_pred = reality, prediction
        portfolio_positions_hodler.append(play_hodler_strategy(y_true, y_pred))
        portfolio_positions_trader.append(play_trader_strategy(y_true, y_pred))
        portfolio_positions_whale.append(play_whale_strategy(y_true, y_pred))
        portfolio_positions_hodler_whale.append(play_hodler_whale_strategy(y_true, y_pred))
        portfolio_positions_charles.append(play_charles_strategy(y_true, y_pred))

    return portfolio_positions_hodler, portfolio_positions_trader, portfolio_positions_whale, portfolio_positions_hodler_whale, portfolio_positions_charles

def plot_portolio_positions(fold = 0, model = LinearRegressionBaselineModel(alpha = 0.05 , l1_ratio = 0.0001),
                            df = ApiCall().read_local(data = 'all'),
                                                      cv = False,
                                                      verbose = 0):

    positions = iterate_portfolio_positions(model = model,
                                df = df,
                                cv=cv,
                                verbose=verbose)

    hodler, trader, whale, hodler_whale, charles = positions

    figure(figsize=(10, 6), dpi=100)

    plt.plot(hodler[fold], label="Hodler")
    plt.plot(trader[fold], label="Trader")
    plt.plot(whale[fold], label="Whale")
    plt.plot(hodler_whale[fold], label="Hodler whale")
    plt.plot(charles[fold], label="Charles")

    plt.title("Investment strategies' results by investment strategy")
    plt.xlabel("Days")
    plt.ylabel("USD portfolio value")
    plt.legend()

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

    #past_realities, realities, realities_diff, prediction_diff = cross_val_trade(LinearRegressionBaselineModel(alpha = 0.5 , l1_ratio = 0.01), ApiCall().read_local(data = 'train'))
    #ct = 0
    #preds = []
    #preds_arr = []
    #for past_prices, diffs in zip(past_realities,prediction_diff):
    #    for diff in diffs:
    #        preds.append(past_prices[ct] + past_prices[ct] * diff)
    #        ct += 1
    #    preds_arr.append(np.array(preds))
    #    preds.clear()
    #    ct = 0

    #print(len(preds_arr))

    print(iterate_portfolio_positions())



    #model = LinearRegressionBaselineModel(alpha = 0.05 , l1_ratio = 0.0001)
    #model = DummyModel()
    #model = RnnDlModel()

    #roi_hodler, roi_trader, roi_whale, roi_hodler_whale, roi_charles, sharpe_hodler, sharpe_trader, sharpe_whale, sharpe_hodler_whale, sharpe_charles = iterate_cross_val_results(model = model)
    #print("---")
    #print("Hodler roi: ", roi_hodler)
    #print("Hodler sharpe ratio: ", sharpe_hodler)
    #print("---")
    #print("Trader roi: ", roi_trader)
    #print("Trader sharpe ratio: ", sharpe_trader)
    #print("---")
    #print("Whale roi: ", roi_whale)
    #print("Whale sharpe ratio: ", sharpe_whale)
    #print("---")
    #print("Hodler whale roi: ", roi_hodler_whale)
    #print("Hodler whale sharpe ratio: ", sharpe_hodler_whale)
    #print("---")
    #print("Charles roi: ", roi_charles)
    #print("Charles sharpe ratio: ", sharpe_charles)

    # df = ApiCall().read_local()
    # model = LinearRegressionBaselineModel()
    # print(len(cross_val_metrics(model=model,df=df)[1]))
    plot_portolio_positions(fold = 0, model = RnnDlModel(epochs=2,patience=2),
                            df = ApiCall().read_local(data = 'all'),
                                                      cv = False,
                                                      verbose = 0)

#### NOTES ####

# y_true = (90, 1)
# y_pred = (90, 1) # dollar value
# .predict here
