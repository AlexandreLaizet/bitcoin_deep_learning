from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
####################################################
#            LOCAL IMPORTS
####################################################

from bitcoin_deep_learning.model import LinearRegressionBaselineModel
from bitcoin_deep_learning.call_api import ApiCall

api = ApiCall()

#df =  api.read_local()
#print("dummy shape is ",df.shape)

fold_train_size=12*30
fold_test_size=3*30
horizon = 7
gap = horizon-1
sequence_lenght = 90
fold_step = 30

def mae(y_pred,y_true):
    return np.mean(np.absolute(y_pred-y_true))


def fold_indexes(df,
                 df_shape=(1275,30),
                 fold_train_size=fold_train_size,
                 fold_test_size=fold_test_size,
                 horizon=horizon,
                 sequence_lenght = 90,
                 fold_step=30
                 ):
    '''
    Entry df or df shape or both ?
    Return a tuple of 4 list of indices that will be used to
    divide the (train) df in folds'''

    total_fold_size = fold_train_size+(horizon-1)+fold_test_size
    max_fold_amount = (df.shape[0]-total_fold_size)//fold_step
    a,b =  zip(*[((n*fold_step,n*fold_step+fold_train_size-1),
            (n*fold_step+fold_train_size-sequence_lenght,n*fold_step+fold_test_size+fold_train_size+(horizon-1)))
            for n in range(max_fold_amount+1)])
    start_fold_train, end_fold_train = zip(*list(a))
    start_fold_test, end_fold_test = zip(*list(b))
    return list(start_fold_train),list(end_fold_train),list(start_fold_test),list(end_fold_test)


#TODO sequence_indexe should take a df or a df.shape
def sequence_indexes(df,
                    shape=None,

                     fold_train_size=fold_train_size,
                     fold_test_size = fold_test_size,
                     sequence_lenght=sequence_lenght,
                     horizon=horizon,
                     sample_step=1
                     ):
    '''Take a sub_df in entry and return a X_train indexes sequences and a y_true index '''
    if shape==None:
        shape = df.shape
    total_fold_size = fold_train_size+(horizon-1)+fold_test_size
    #print(shape)
    max_seq =((shape[0]-sequence_lenght-horizon)//sample_step)
    #assert fold_shape[0]>=total_fold_size,'''df is too smal compare to fold
    #dimensions '''
    seq_start_stop , index_pred =  zip(*[((n*sample_step,n*sample_step+sequence_lenght),n*sample_step+horizon+sequence_lenght)
                 for n in range(max_seq)])
    seq_start, seq_stop = zip(*list(seq_start_stop))
    return list(seq_start), list(seq_stop), list(index_pred)

#from charles import Model


def cross_val(model, df, hyperparams=None):
    df = df.drop(columns=["date"])
    prediction = []
    reality = []
    score = []
    start_fold_train, end_fold_train, start_fold_test, end_fold_test = fold_indexes(
        df=df)
    for i in range(len(start_fold_train)):
        #reinitialise the model between two fold
        model.set_model()
        train_fold_df = df.loc[start_fold_train[i]:end_fold_train[i]].copy(
        ).reset_index(drop=True)
        #print(train_fold_df)
        #print("train fold shape is ",train_fold_df.shape)
        #print(train_fold_df.columns)

        sequence_start, sequence_stop, target_idx = sequence_indexes(
            df=train_fold_df)
        X_train = []
        Y_train = []
        #print(len(train_fold_df.shape))
        for j in range(len(sequence_start)):
            X_train_seq = np.array(
                train_fold_df.iloc[sequence_start[j]:sequence_stop[j]])
            y_train = train_fold_df.iloc[target_idx[j], -1]
            X_train.append(np.array(X_train_seq))
            Y_train.append(np.array(y_train))
        Y_train = np.array(Y_train)

        X_train = np.array(X_train)
        #print("X_train shape is",X_train.shape)
        #print("Y_train shape is ",Y_train.shape)
        #print(Y_train)
        #model.fit(model.preproc(X_train),Y_train)
        #print("the model is fitting",{i},"eme iteration")

        #print("Validation in process")

        test_fold_df = df.loc[start_fold_test[i]:end_fold_test[i]].copy(
        ).reset_index(drop=True)
        #print("test fold shape is ",test_fold_df.shape)
        #print(test_fold_df)

        a, b, c = sequence_indexes(df=test_fold_df)
        Y_test = []
        X_test = []
        for j in range(len(a)):
            X_test_seq = test_fold_df.iloc[a[j]:b[j]]
            y_test = test_fold_df.iloc[c[j], -1]
            #print("y_test shape is",y_test.shape)
            X_test.append(np.array(X_test_seq))
            Y_test.append(np.array(y_test))
        Y_test = np.array(Y_test)
        #print("Y_test shape is ",Y_test.shape)
        X_test = np.array(X_test)
        #print("X_test",X_test.shape)

        #CREATE A BATCH ARRAY
        Y_pred = model.run(X_test, X_train, Y_train)
        #print(X_test)
        #print(Y_pred)

        #FROM METRICS.PY IMPORT MAE
        # metrics(Y_pred,Y_true)
        reality.append(Y_test)
        prediction.append(Y_pred)
        score.append(mae(Y_test, Y_pred))

    return score, np.mean(score)


if __name__ == "__main__":
    print("Start of test")
    model = LinearRegressionBaselineModel()
    df = ApiCall().read_local()
    print("INITIAL SHAPE IS ",df.shape)
    print(cross_val(model,df=df))







# def alex_cross_val(model,
#               hyperparams=None,
#               df=None):
#     df = df.drop(columns=["date"])
#     prediction = []
#     reality = []
#     start_fold_train,end_fold_train,start_fold_test,end_fold_test = fold_indexes(df=df)
#     for i in range(len(start_fold_train)) :
#         train_fold_df = df.loc[start_fold_train[i]:end_fold_train[i]].copy().reset_index(drop=True)

#         sequence_start,sequence_stop,target_idx = sequence_indexes(df = train_fold_df)
#         X_train = []
#         Y_true = []
#         for j in range(len(sequence_start)):
#             X_train_seq = np.array(train_fold_df.iloc[sequence_start[j]:sequence_stop[j]])
#             y_true = train_fold_df.iloc[target_idx[j],-1]
#             X_train.append(X_train_seq)
#             Y_true.append(y_true)

#         Y_true = np.array(Y_true)

#         X_train = np.array(X_train)

#         model.fit(model.preproc(X_train),Y_true)



#         test_fold_df = df.loc[start_fold_test[i]:end_fold_test[i]].copy().reset_index(drop=True)


#         a,b,c=sequence_indexes(df=test_fold_df)
#         Y_test = []
#         X_test = []
#         for j in range(len(a)):
#             X_test_seq = test_fold_df.iloc[a[j]:b[j]]
#             y_test = test_fold_df.iloc[c[j],-1]
#             X_test.append(X_test_seq)
#             Y_test.append(y_test)


#         Y_test = np.array(Y_true)
#         X_test = np.array(X_test)


#         Y_pred = model.predict(X_test)

#         reality.append(Y_test)
#         prediction.append(Y_pred)



#     return reality,prediction




def cross_val_charles(model, df, hyperparams=None) :
    df = df.drop(columns=["date"])
    prediction = []
    reality = []
    score = []
    start_fold_train, end_fold_train, start_fold_test, end_fold_test = fold_indexes(
        df=df)
    for i in range(len(start_fold_train)):
        #reinitialise the model between two fold
        model.set_model()
        train_fold_df = df.loc[start_fold_train[i]:end_fold_train[i]].copy(
        ).reset_index(drop=True)
        #print(train_fold_df)
        #print("train fold shape is ",train_fold_df.shape)
        #print(train_fold_df.columns)

        sequence_start, sequence_stop, target_idx = sequence_indexes(
            df=train_fold_df)
        X_train = []
        Y_train = []
        #print(len(train_fold_df.shape))
        for j in range(len(sequence_start)):
            X_train_seq = np.array(
                train_fold_df.iloc[sequence_start[j]:sequence_stop[j]])
            y_train = train_fold_df.iloc[target_idx[j], -1]
            X_train.append(X_train_seq)
            Y_train.append(y_train)
        Y_train = np.array(Y_train)

        X_train = np.array(X_train)
        #print("X_train shape is",X_train.shape)
        #print("Y_train shape is ",Y_train.shape)
        #print(Y_train)
        #model.fit(model.preproc(X_train),Y_train)
        #print("the model is fitting",{i},"eme iteration")

        #print("Validation in process")

        test_fold_df = df.loc[start_fold_test[i]:end_fold_test[i]].copy(
        ).reset_index(drop=True)
        #print("test fold shape is ",test_fold_df.shape)
        #print(test_fold_df)

        a, b, c = sequence_indexes(df=test_fold_df)
        Y_test = []
        X_test = []
        for j in range(len(a)):
            X_test_seq = test_fold_df.iloc[a[j]:b[j]]
            y_test = test_fold_df.iloc[c[j], -1]
            #print("y_test shape is",y_test.shape)
            X_test.append(np.array(X_test_seq))
            Y_test.append(np.array(y_test))
        Y_test = np.array(Y_test)
        #print("Y_test shape is ",Y_test.shape)
        X_test = np.array(X_test)
        #print("X_test",X_test.shape)

        #CREATE A BATCH ARRAY
        Y_pred = model.run(X_test, X_train, Y_train)
        #print(X_test)
        #print(Y_pred)

        #FROM METRICS.PY IMPORT MAE
        # metrics(Y_pred,Y_true)
        reality.append(Y_test)
        prediction.append(Y_pred)
        score.append(mae(Y_test, Y_pred))

    return score, np.mean(score)
