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
from bitcoin_deep_learning.model import LinearRegressionBaselineModel,DummyModel, RnnDlModel
#from bitcoin_deep_learning.call_api import ApiCall
from bitcoin_deep_learning.metrics import Mean_absolute_percentage_error
from bitcoin_deep_learning.params import (ROOT_DIR, FOLD_TEST_SIZE,
                                          FOLD_TRAIN_SIZE, HORIZON, API_KEY)
from sklearn.metrics import mean_absolute_error


def train(model,
          df,
          metric=Mean_absolute_percentage_error(),
          save:bool=True,
          precision:int=5
          ):
    reality,prediction = cross_val(model,df)
    fold_score = [round(mean_absolute_error(Y_true,Y_pred),precision)
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
    print("BITCOIN HUNT IS OPEN")
    from bitcoin_deep_learning.call_api import ApiCall
    df = ApiCall().read_local(data="train")
    model = RnnDlModel(epochs=2,patience=1)
    train(model,df)
    pass
