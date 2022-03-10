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
from bitcoin_deep_learning.metrics import Mean_absolute_percentage_error,iterate_cross_val_results
from bitcoin_deep_learning.params import (ROOT_DIR, FOLD_TEST_SIZE,
                                          FOLD_TRAIN_SIZE, HORIZON, API_KEY)
from sklearn.metrics import mean_absolute_error


def cv_train(model,
          df,
          metric=Mean_absolute_percentage_error(),
          save:bool=True,
          precision:int=5,
          with_trader:bool = True
          ):
    reality,prediction = cross_val(model,df)
    fold_score = [round(mean_absolute_error(Y_true,Y_pred),precision)
                            for Y_true,Y_pred in zip(reality,prediction)]
    score =round(np.mean(np.array(fold_score)),precision)
    # Option to save results
    if save == True :
        if not with_trader :
            fieldnames = ["name",'fold_score',"mean_score","min_score","max_score",'hyperparams','date']
            file_path = os.path.join(ROOT_DIR,
                                            "cross_val_data",
                                            'cv_train.csv')
            # Check if file is there and create it otherwise
            if not os.path.isfile(file_path):
                pd.DataFrame(columns=fieldnames).to_csv(file_path,index=False)
            # Append a new line with current CV results
            with open(file_path , 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writerow({"name":model.name, "fold_score":fold_score,
                                "mean_score":score,"min_score":min(fold_score),
                                "max_score":max(fold_score),
                                "hyperparams":model.hyperparams,
                                'date':datetime.now().strftime("%d-%m %H:%M:%S")})
                print("Training done")
            return fold_score, score
        # if with_trader is true we compute the ROI and sharpe metrics
        fieldnames = ["name",'fold_score',"mean_score","min_score","max_score",'hyperparams','date',
             "roi_hodler", "sharpe_hodler", "roi_trader", "sharpe_trader","roi_whale", "sharpe_whale",
              "roi_hodler_whale",  "sharpe_hodler_whale", "roi_charles",
                "sharpe_charles"]
        file_path = os.path.join(ROOT_DIR,
                                        "cross_val_data",
                                        'cv_train_with_trader.csv')
        # Check if file is there and create it otherwise
        if not os.path.isfile(file_path):
            pd.DataFrame(columns=fieldnames).to_csv(file_path,index=False)
        # Append a new line with current CV results
        with open(file_path , 'a', newline='') as csvfile:
            (roi_hodler, roi_trader, roi_whale, roi_hodler_whale, roi_charles,
             sharpe_hodler, sharpe_trader, sharpe_whale, sharpe_hodler_whale,
             sharpe_charles) = iterate_cross_val_results(model=model,df=df)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({"name":model.name, "fold_score":fold_score,
                            "mean_score":score,"min_score":min(fold_score),
                            "max_score":max(fold_score),
                            "hyperparams":model.hyperparams,
                            'date':datetime.now().strftime("%d-%m %H:%M:%S"),
                            "roi_hodler":roi_hodler,
                            "sharpe_hodler": sharpe_hodler,
                            "roi_trader": roi_trader,
                            "sharpe_trader": sharpe_trader,
                            "roi_whale": roi_whale,
                            "sharpe_whale": sharpe_whale,
                            "roi_hodler_whale": roi_hodler_whale,
                            "sharpe_hodler_whale": sharpe_hodler_whale,
                            "roi_charles": roi_charles,
                            "sharpe_charles":sharpe_charles})
            print("Training with trader done")


#train(model,df,)



# Need a function to read the test.csv file
def read_result(file="cv_train_with_trader.csv"):
    ''' How to read the hyperparams column as a dict
    import json
    text = json.loads(a["hyperparams"][{index}].replace("\'", "\""))'''
    df = pd.read_csv(os.path.join(ROOT_DIR,
                                  "cross_val_data",
                                  file))
    return df




if __name__ == "__main__":
    print("BITCOIN HUNT IS OPEN")
    from bitcoin_deep_learning.call_api import ApiCall
    df = ApiCall().read_local(data="train")
    model = RnnDlModel(epochs=2,patience=1)
    cv_train(model,df,with_trader=False)
    pass
