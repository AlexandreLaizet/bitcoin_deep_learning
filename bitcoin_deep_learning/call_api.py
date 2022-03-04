import requests
import pandas as pd
import numpy as np
import time
import datetime
from datetime import date, timedelta
#from dotenv import load_dotenv, find_dotenv
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('API_KEY')

class ApiCall():
    def __init__(self,API_KEY=API_KEY):
        self.name_and_url={'[+]_[NH]_Circulating_Supply':'https://api.glassnode.com/v1/metrics/supply/current',
            '[+]_[NH]_Issuance':"https://api.glassnode.com/v1/metrics/supply/issued",
            '[+]_[NH]_Number_of_Active_Addresses':"https://api.glassnode.com/v1/metrics/addresses/active_count",
            # DONE :  find a way to process the Mean Hash Rate
            #'[AVG]_[NH]_Mean_Hash_Rate':"https://api.glassnode.com/v1/metrics/mining/hash_rate_mean",
            '[+]_[NH]_Mean_Block_Interval':"https://api.glassnode.com/v1/metrics/blockchain/block_interval_median",
            '[+]_[NH]_Number_of_Transactions':"https://api.glassnode.com/v1/metrics/transactions/count",
            '[+]_[NH]_Number_of_Addresses_with_a_Non-Zero_Balance':"https://api.glassnode.com/v1/metrics/addresses/non_zero_count",
            '[+]_[NH]_Number_of_Addresses_with_Balance_≥_0.01':"https://api.glassnode.com/v1/metrics/addresses/min_point_zero_1_count",
            '[+]_[NH]_Number_of_Addresses_with_Balance_≥_0.1':"https://api.glassnode.com/v1/metrics/addresses/min_point_1_count",
            '[+]_[NH]_Number_of_Addresses_with_Balance_≥_1':"https://api.glassnode.com/v1/metrics/addresses/min_1_count",
            '[+]_[NH]_Number_of_Addresses_with_Balance_≥_10':"https://api.glassnode.com/v1/metrics/addresses/min_10_count",
            '[+]_[NH]_Number_of_Addresses_with_Balance_≥_100':"https://api.glassnode.com/v1/metrics/addresses/min_100_count",
            '[+]_[NH]_Number_of_Addresses_with_Balance_≥_1k':"https://api.glassnode.com/v1/metrics/addresses/min_1k_count",
            '[+]_[NH]_Number_of_Addresses_with_Balance_≥_10k':"https://api.glassnode.com/v1/metrics/addresses/min_10k_count",
            '[%]_[BSB]_Percent_of_Supply_Last_Active_1+_Years_Ago':"https://api.glassnode.com/v1/metrics/supply/active_more_1y_percent",
            '[//]_[BSB]_Realized_HODL_Ratio':"https://api.glassnode.com/v1/metrics/indicators/rhodl_ratio",
            '[AVG]_[BSB]_Average_Spent_Output_Lifespan_(ASOL)':"https://api.glassnode.com/v1/metrics/indicators/asol",
            '[//]_[BSB]_Liveliness':"https://api.glassnode.com/v1/metrics/indicators/liveliness",
            '[%]_[BSB]_Percent_Balance_on_Exchanges_-_All_Exchanges':"https://api.glassnode.com/v1/metrics/distribution/balance_exchanges_relative",
            '[$]_[BSB]_Exchange_Net_Position_Change_-_All_Exchanges':"https://api.glassnode.com/v1/metrics/distribution/exchange_net_position_change",
            '[//]_[BSB]_Realized_Profit/Loss_Ratio':"https://api.glassnode.com/v1/metrics/indicators/realized_profit_loss_ratio",
            '[$]_[BSB]_Net_Unrealized_Profit/Loss_(NUPL)':"https://api.glassnode.com/v1/metrics/indicators/net_realized_profit_loss",
            '[$]_[BSB]_Realized_Price':"https://api.glassnode.com/v1/metrics/market/price_realized_usd",
            '[%]_[BSB]_Price_Drawdown_from_ATH':"https://api.glassnode.com/v1/metrics/market/price_drawdown_relative",
            # This api answer has a different shape than the others :
            #'[//]_[AV]_Stock-to-Flow_Ratio':"https://api.glassnode.com/v1/metrics/indicators/stock_to_flow_ratio",
            '[//]_[AV]_Market_Value_to_Realized_Value_Ratio_(MVRV)':"https://api.glassnode.com/v1/metrics/market/mvrv",
            '[//]_[AV]_Puell_Multiple':"https://api.glassnode.com/v1/metrics/indicators/puell_multiple",
            '[//]_[AV]_Realized_Profits-to-Value_(RPV)_Ratio':"https://api.glassnode.com/v1/metrics/indicators/realized_profits_to_value_ratio",
            '[+]_[T]_Bitcoin_Price':"https://api.glassnode.com/v1/metrics/market/price_usd_close"}

        self.params = {'a': 'BTC', 'api_key': API_KEY}

    def get_raw_glassnode_data(self,verbose=0):
        # We make a first api call for the stock to flow ratio, since it is
        # has a diffenrent response.
        res = requests.get("https://api.glassnode.com/v1/metrics/indicators/stock_to_flow_ratio",
            params=self.params)
        test_df = pd.read_json(res.text,convert_dates=["t"])
        date_serie = pd.DataFrame(test_df['o'].apply(lambda x: x['daysTillHalving'])).merge(pd.DataFrame(test_df['t']), how = 'inner', left_index=True, right_index=True)
        date_serie.rename(columns={"o": "[+]_[NH]_Days_Till_Halving"},
                          inplace=True)
        ratio_serie = pd.DataFrame(test_df['o'].apply(lambda x: x['ratio'])).merge(pd.DataFrame(test_df['t']), how = 'inner', left_index=True, right_index=True)
        #date_serie.rename(columns={"o":"date"},inplace=True)
        global_df = pd.merge(date_serie,ratio_serie,how="inner",on="t").rename(columns={"o":"[//]_[AV]_Stock-to-Flow_Ratio"})

        #Making API requests for most of the blockchains data
        for name in list(self.name_and_url.keys()):
            # make API request
            res = requests.get(self.name_and_url[name],
                params=self.params)
            if verbose:
                print("requesting ",name," ...")
            df = pd.read_json(res.text, convert_dates=['t'])
            df.rename(columns={"v":name},inplace=True)
            global_df = pd.merge(global_df,df,how="inner",on="t")
            time.sleep(1)


        #Making Api request for the hash rate, high number answer need special treatment
        hash_params = {'a': 'BTC', 'api_key': API_KEY,"f":"CSV","timestamp_format":"unix"}
        res = requests.get("https://api.glassnode.com/v1/metrics/mining/hash_rate_mean",params=hash_params)
        tmp = res.content.decode("utf-8").split("\n")
        ls = []
        date = []
        A = [x.split(",") for x in tmp]
        for x in A[1:-1]:
            ls.append(float(x[1]))
            date.append(x[0])
        data = pd.DataFrame(data={"t": date, "[AVG]_[NH]_mean_hash_rate": ls})
        data["t"] = pd.to_datetime(data["t"].apply(
            lambda x:datetime.datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d')))

        #Merging
        pd.merge(data,global_df,how='inner',on="t").head()
        return global_df.rename(columns={"t":"date"})

    def get_fear_and_greed(self):
        # make API request
        res = requests.get('https://api.alternative.me/fng/?limit=0&date_format=DD/MM/YYYY').json()

        # convert to pandas dataframe (note: starting at 1 because 0 is the
        # current day in process and does not have the same format)
        df = pd.DataFrame(
            res["data"][1:]).rename(columns={"timestamp":"date",
                                            "value":"fear_greed_value",
                                            "value_classification":"fear_greed_value_class"})

        #change the date to a datetime format
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_raw_data(self,short=True):
        '''if short = True return 2018-01-02 as first date'''
        #TODO FEAR AND GREED IS Y_D_M merge not working !!!!!
        df_api = self.get_raw_glassnode_data()
        df_fear_greed = self.get_fear_and_greed()
        if not short :
            return pd.merge(df_api,df_fear_greed,how="left",right_on="date",left_on="date")
        return pd.merge(df_api,df_fear_greed,how="inner",right_on="date",left_on="date")


    def get_clean_data(self, raw_data_start_date = "2018/01/26", columns_to_drop = ["fear_greed_value_class"]):

        df = self.get_raw_data()

        # Enter start dataframe (2018/01/26 is the start of the Lightning network feature)
        #raw_data_start_date = "2018/01/26"

        # Enter columns to drop from dataframe
        #columns_to_drop = ["fear_greed_value_class"]

        # Clean dataframe
        df_clean = df.drop(columns=columns_to_drop)
        df_clean["fear_greed_value"] = df_clean["fear_greed_value"].apply(np.int64)
        df_clean = df_clean[df["date"] > raw_data_start_date].copy()

        # Create supply buckets
        df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_0.01 - 0.1"] = df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_≥_0.01"] - df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_≥_0.1"]
        df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_0.1 - 1"] = df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_≥_0.1"] - df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_≥_1"]
        df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_1 - 10"] = df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_≥_1"] - df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_≥_10"]
        df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_10 - 100"] = df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_≥_10"] - df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_≥_100"]
        df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_100 - 1k"] = df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_≥_100"] - df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_≥_1k"]
        df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_1k - 10k"] = df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_≥_1k"] - df_clean["[+]_[NH]_Number_of_Addresses_with_Balance_≥_10k"]

        df_clean = df_clean.drop(columns=["[+]_[NH]_Number_of_Addresses_with_Balance_≥_0.01",
                                          "[+]_[NH]_Number_of_Addresses_with_Balance_≥_0.1",
                                          "[+]_[NH]_Number_of_Addresses_with_Balance_≥_1",
                                          "[+]_[NH]_Number_of_Addresses_with_Balance_≥_10",
                                          "[+]_[NH]_Number_of_Addresses_with_Balance_≥_100",
                                          "[+]_[NH]_Number_of_Addresses_with_Balance_≥_1k"])

        return df_clean.reindex(columns=(['date'] +
                                list(df_clean.copy().drop(columns = ['date','[+]_[T]_Bitcoin_Price']).columns)
                                + ["[+]_[T]_Bitcoin_Price"]))


    #TODO implement a way to stock a df_train, df_test, df_val
    def data_to_csv(self, df=False,name="BTC_df", short=True):
        if not type(df)== pd.core.frame.DataFrame :
            df = self.get_clean_data()
        #TODO ASK A TA IF THIS WILL BROKE AS SOON AS WE PUSH IT TO GIT
        # loca path/ absolute ?
        df.to_csv('../data_raw/'+name+'.csv',index=False)
        return df

    def read_local(self,name="BTC_df"):

        df = pd.read_csv('../data_raw/'+name+'.csv')
        # We verify if the local data is up to date
        if not  (pd.Timestamp(df["date"].iloc[-1]) ==
                            pd.Timestamp(date.today()- timedelta(days = 1))) :
            print("Downloading data..")
            df = self.data_to_csv(name)
        print("Data is up to date and has been loaded from local")
        return df


if __name__=="__main__":
    df = ApiCall().get_raw_data()
    print(len(df.columns))
    assert len(df.columns)==32, "df should have 32 columns"
