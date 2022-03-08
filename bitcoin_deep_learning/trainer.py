import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import csv
import os
################################################################################
#            LOCAL IMPORTS
################################################################################
from bitcoin_deep_learning.cross_val import cross_val
from bitcoin_deep_learning.model import LinearRegressionBaselineModel,DummyModel
#from bitcoin_deep_learning.call_api import ApiCall
from bitcoin_deep_learning.metrics import Mean_absolute_percentage_error
from bitcoin_deep_learning.params import (ROOT_DIR, FOLD_TEST_SIZE,
                                          FOLD_TRAIN_SIZE, HORIZON, API_KEY)


#model = LinearRegressionBaselineModel()


model = LinearRegressionBaselineModel({"alpha":1,"l1_ratio":0.5})
def train(model,
          df,
          metric=Mean_absolute_percentage_error(),
          save:bool=True,
          precision:int=5
          ):
    reality,prediction = cross_val(model,df)
    fold_score = [round(metric.compute(Y_true,Y_pred),precision)
                            for Y_true,Y_pred in zip(reality,prediction)]
    score =round(np.mean(np.array(fold_score)),precision)
    # Option to save results
    if save == True :
        file_path = os.path.join(ROOT_DIR,
                                        "cross_val_data",
                                        'test.csv')
        # Check if file is there and create it otherwise
        if not os.path.isfile(file_path):
            fieldnames = ["name",'fold_score',"mean_score","min_score","max_score",'hyperparams','date']
            pd.DataFrame(columns=fieldnames).to_csv(file_path,index=False)
        # Append a new line with current CV results
        with open(file_path , 'a', newline='') as csvfile:
            fieldnames = ["name",'fold_score',"mean_score","min_score","max_score",'hyperparams','date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({"name":model.name, "fold_score":fold_score,
                            "mean_score":score,"min_score":min(fold_score),
                            "max_score":max(fold_score),
                            "hyperparams":model.hyperparams,
                            'date':datetime.now().strftime("%d-%m %H:%M:%S")})
            print("Training done")
        return fold_score, score


def train_with_trader(model,
          df,
          metric=Mean_absolute_percentage_error(),
          save:bool=False,
          strategies:list=[],
          precision:int=5
          ):
    reality,prediction = cross_val(model,df)
    fold_score = [round(metric.compute(Y_true,Y_pred),precision)
                            for Y_true,Y_pred in zip(reality,prediction)]
    score =round(np.mean(np.array(fold_score)),precision)

    for strat in strategies :
        pass
    if save == True :
        with open(r'test.csv', 'a', newline='') as csvfile:
            fieldnames = ["name",'fold_score',"mean_score","min_score",
                          "max_score",'hyperparams','date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({"name":model.name, "fold_score":fold_score,
                            "mean_score":score,"min_score":min(fold_score),
                            "max_score":max(fold_score),
                            "hyperparams":model.hyperparams,
                            'date':datetime.datetime.now().strftime("%d-%m %H:%M:%S")})
            print("Training done")
        return fold_score, score


# Need a function to read the test.csv file
def read_result(file="test.csv"):
    df = pd.read_csv(os.path.join(ROOT_DIR,
                                  "cross_val_data",
                                  'test.csv'))
    return df
    # How to read the hyperparams column as a dict
    # import json
    #text = json.loads(a["hyperparams"][0].replace("\'", "\""))



if __name__ == "__main__":
    # # print("BITCOIN HUNT IS OPEN")
    # df = ApiCall().read_local(data="train")
    # model1 = LinearRegressionBaselineModel()
    # # # model2 = DummyModel()
    # # print(cross_val(model1,df,verbose=1))
    # score = [15.38210528111765, 25.720944945045964, 39.102210516931635, 40.37771438907446,
    # 33.601183893795216, 19.269462918652618, 12.503808074277556, 16.667250967243195,
    # 26.094812000754313, 19.00642774677545, 12.38672368897159, 15.18933141906669, 17.9440152105882,
    # 16.377828127702763, 10.57480668359942, 13.594896621814472, 15.96493955320262, 10.471524967662697,
    # 14.814035123980737, 14.576536875855092, 25.12837343681645, 41.69384220283879, 49.72615496353173,
    # 49.33069449116819, 41.184365503077, 27.61567803357174, 17.29660301838008, 26.821202649834873, 25.15024156775706]
    # # print("training 1 done")
    # # cross_val(model2,df,verbose=1)
    # metric = Mean_absolute_percentage_error()
    # result = {"model":model1.name,
    #           "score":{metric.name:score}}
    # a = pd.DataFrame.from_dict(result)
    # print(a.columns)
    # print(a.iloc[0,1])
    pass
