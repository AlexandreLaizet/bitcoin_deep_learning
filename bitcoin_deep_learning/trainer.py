#from msilib import sequence
from bitcoin_deep_learning.cross_val import sequence_indexes,fold_indexes
import pandas as pd
import numpy as np
from tqdm import tqdm


df = pd.DataFrame(np.random.rand(1275,30))


def train(hyperparams=None,
          df=df):
    start_fold_train,end_fold_train,start_fold_test,end_fold_test = fold_indexes()
    for i in tqdm(range(len(start_fold_train))) :
        train_fold_df = df.loc[start_fold_train[i]:end_fold_train[i]].copy().reset_index()

        sequence_start,sequence_stop,target_idx = sequence_indexes(shape=train_fold_df.shape)
        X_train = []
        Y_true = []
        #print(len(train_fold_df.shape))
        for j in range(len(sequence_start)):
            X_train_seq = np.array(train_fold_df.iloc[sequence_start[j]:sequence_stop[j]])
            y_true = train_fold_df.iloc[target_idx[j],-1]
            X_train.append(X_train_seq)
            Y_true.append(y_true)

        Y_true = np.array(y_true)
        X_train = np.array(X_train)
        print(X_train.shape)
        #model.fit(X_train,Y_pred)
        #print("the model is fitting",{i},"eme iteration")

        # test_fold_df = df.loc[start_fold_test[i]:end_fold_test[i]].copy().reset_index()

        # a,b,c=sequence_indexes(test_fold_df)

        # X_test=[]
        # Y_true=[]
        # for j in range(len(a)):
        #     x_test_seq = test_fold_df.iloc[a[j]:b[j]]
        #     y_pred = test_fold_df.iloc[c[j],-1]
        #     X_test.append(x_test_seq)
        #     Y_true.append(y_pred)

        # X_test=np.array(X_test)
        # Y_true=np.array(Y_true)
        # print(X_test.shape,Y_true.shape)



        #print("Validation in process")


if __name__ == "__main__":
    print("BITCOIN HUNT IS OPEN")
    train()
    print("Training is done")
