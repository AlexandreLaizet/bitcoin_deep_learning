from xmlrpc.client import boolean
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
################################################################################
#            LOCAL IMPORTS
################################################################################

from bitcoin_deep_learning.model import LinearRegressionBaselineModel, RnnDlModel
from bitcoin_deep_learning.call_api import ApiCall
from bitcoin_deep_learning.params import(FOLD_TRAIN_SIZE,FOLD_TEST_SIZE,
                                         HORIZON, SEQUENCE_LENGHT)
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


fold_train_size=FOLD_TRAIN_SIZE
fold_test_size=FOLD_TEST_SIZE
horizon = HORIZON
gap = horizon-1
sequence_lenght = 90
fold_step = 30
sample_step = 1

def mae(y_pred,y_true):
    return np.mean(np.absolute(y_pred-y_true))


def fold_indexes(df,
                 fold_train_size:int = fold_train_size,
                 fold_test_size:int = fold_test_size,
                 horizon:int = horizon,
                 sequence_lenght:int = sequence_lenght,
                 fold_step:int = fold_step,
                 verbose:int = 0
                 ):
    '''
    Return a tuple of 4 list of indexes that will be used to
    divide the (train) df in a train_fold and a test_fold'''

    # We compute the amount of fold we can do
    total_fold_size = fold_train_size+(horizon-1)+fold_test_size
    max_fold_amount = (df.shape[0]-total_fold_size)//fold_step
    # The verbose option allow the user to print the folds amount
    if verbose >=2 :
        print(f"The set is being split in {max_fold_amount} folds")
    # Creating a list of tuples with all the indexes (index of folds)
    # zipping in two intermediary lists
    a,b =  zip(*[((n*fold_step,
                   n*fold_step+fold_train_size-1),
            (n*fold_step+fold_train_size-sequence_lenght,
             n*fold_step+fold_test_size+fold_train_size+(horizon-1)))
            for n in range(max_fold_amount+1)])
    # zipping again to return a tupple of list
    start_fold_train, end_fold_train = zip(*list(a))
    start_fold_test, end_fold_test = zip(*list(b))
    return list(start_fold_train),list(end_fold_train),list(start_fold_test),list(end_fold_test)


def sequence_indexes(df:pd.DataFrame,
                     sequence_lenght:int=sequence_lenght,
                     horizon:int=horizon,
                     sample_step:int=sample_step,
                     verbose:int = 0,
                     ):
    '''Take a sub_df in entry and return a list of x_train_seq sequences and a
     list of y_true indexes '''
    shape = df.shape
    # We compute the amount of sequences we can create
    max_seq =((shape[0]-sequence_lenght-horizon)//sample_step)
    if verbose >= 10 :
        print(f"{max_seq} sequences have been created, it's something !")
    # Creating a list of tuples with all the indexes (index of folds)
    # with and intermediate list seq_start_stop to have the correct format
    seq_start_stop, index_pred = zip(*[((n * sample_step,
                                         n * sample_step + sequence_lenght),
                                        n * sample_step + horizon + sequence_lenght)
                                       for n in range(max_seq)])
    # zipping again to return a tupple of 3 lists
    seq_start, seq_stop = zip(*list(seq_start_stop))
    return list(seq_start), list(seq_stop), list(index_pred)

def cross_val(model, df,
              verbose:int=0,
              saving:bool=False,
              metrics=[],
              trader_metrics=[],
              hyperparams=None):
    '''Compute and process a complete cross validation of a given model,
    taking personalised metrics into account
    params :
    verbose range from 0 to 10 and allow to print a few informations during the
        process
    return reality, prediction '''
    df = df.drop(columns=["date"])
    # Initializing the variable to return
    prediction, reality, score = [], [], []
    # Setting the indexes to cut the df into folds
    start_fold_train, end_fold_train, start_fold_test, end_fold_test = fold_indexes(
        df=df,verbose=verbose)
    # Starting the iteration on folds
    for i in tqdm(range(len(start_fold_train))):
        # reinitialise the model between two folds to reset training
        model.set_model()
        # instantiating train fold
        if verbose >= 10 :
            print("Creating sequences in the train_fold")
        train_fold_df = df.loc[start_fold_train[i]:
                                 end_fold_train[i]].copy().reset_index(drop=True)
        # Setting the indexes to cut the train_fold in regular sequences and targets
        sequence_starts, sequence_stops, target_idx = sequence_indexes(
            df=train_fold_df,verbose=verbose)
        # Initializing the X_train, Y_train
        X_train, Y_train = [], []
        # Starting the iteration on the sequences to create X,Y_train
        for j in range(len(sequence_starts)):

            X_train_seq = np.array(
                train_fold_df.iloc[sequence_starts[j]:sequence_stops[j]])
            y_train = train_fold_df.iloc[target_idx[j], -1]
            #Converting the little df to np array
            X_train.append(np.array(X_train_seq))
            Y_train.append(np.array(y_train))
        # Converting the list of array to an array
        Y_train = np.array(Y_train)
        X_train = np.array(X_train)

        #Same process as ahead but on the test_fold
        test_fold_df = df.loc[start_fold_test[i]:end_fold_test[i]].copy(
        ).reset_index(drop=True)
        sequence_starts, sequence_stops, target_idx = sequence_indexes(df=test_fold_df,verbose=verbose)
        Y_test,X_test = [],[]
        for j in range(len(sequence_starts)):
            X_test_seq = test_fold_df.iloc[sequence_starts[j]:sequence_stops[j]]
            y_test = test_fold_df.iloc[target_idx[j], -1]
            X_test.append(np.array(X_test_seq))
            Y_test.append(np.array(y_test))
        Y_test = np.array(Y_test)
        X_test = np.array(X_test)
        # Now we have an X_test,Y_test , X_train,Y_train ready to be processed

        #TODO SHUFFLING THE X,y
        Y_pred = model.run(X_test,X_train, Y_train)

        # Keeping these lines in case we want to use Y_test, Y_pred in the futur
        reality.append(Y_test)
        prediction.append(Y_pred)

    if verbose :
        print(f"{model.name} has been cross-validated")
    return reality, prediction

def get_cross_XY(df=None, data="train", verbose = 0):
    '''    return (X_train_list, Y_train_list, X_test_list,Y_test_list)
    data=None
    '''
    if data == "train":
        df = ApiCall().read_local(data=data)
        df = df.drop(columns=["date"])
    # Initializing the variable to return
    X_train_list, Y_train_list, X_test_list,Y_test_list = [], [], [],[]
    # Setting the indexes to cut the df into folds
    start_fold_train, end_fold_train, start_fold_test, end_fold_test = fold_indexes(
        df=df,verbose=verbose)
    # Starting the iteration on folds
    for i in range(len(start_fold_train)):
        # reinitialise the model between two folds to reset training
        train_fold_df = df.loc[start_fold_train[i]:
                                 end_fold_train[i]].copy().reset_index(drop=True)
        # Setting the indexes to cut the train_fold in regular sequences and targets
        sequence_starts, sequence_stops, target_idx = sequence_indexes(
            df=train_fold_df,verbose=verbose)
        # Initializing the X_train, Y_train
        X_train, Y_train = [], []
        # Starting the iteration on the sequences to create X,Y_train
        for j in range(len(sequence_starts)):

            X_train_seq = np.array(
                train_fold_df.iloc[sequence_starts[j]:sequence_stops[j]])
            y_train = train_fold_df.iloc[target_idx[j], -1]
            #Converting the little df to np array
            X_train.append(np.array(X_train_seq))
            Y_train.append(np.array(y_train))
        # Converting the list of array to an array>
        Y_train = np.array(Y_train)
        X_train = np.array(X_train)

        X_train_list.append(X_train)
        Y_train_list.append(Y_train)

        #Same process as ahead but on the test_fold
        test_fold_df = df.loc[start_fold_test[i]:end_fold_test[i]].copy(
        ).reset_index(drop=True)
        sequence_starts, sequence_stops, target_idx = sequence_indexes(df=test_fold_df,verbose=verbose)
        Y_test,X_test = [],[]
        for j in range(len(sequence_starts)):
            X_test_seq = test_fold_df.iloc[sequence_starts[j]:sequence_stops[j]]
            y_test = test_fold_df.iloc[target_idx[j], -1]
            X_test.append(np.array(X_test_seq))
            Y_test.append(np.array(y_test))
        Y_test = np.array(Y_test)
        X_test = np.array(X_test)

        Y_test_list.append(Y_test)
        X_test_list.append(X_test)

    return (X_train_list, Y_train_list, X_test_list,Y_test_list)

def one_fold_cross_val(model, df,
              verbose:int=0,
              saving:bool=False,
              metrics=[],
              trader_metrics=[],
              hyperparams=None):
    '''Compute and process a complete cross validation of a given model,
    taking personalised metrics into account
    params :
    verbose range from 0 to 10 and allow to print a few informations during the
        process
    return reality, prediction '''
    df = df.drop(columns=["date"])
    # Initializing the variable to return
    prediction, reality, score = [], [], []
    # Setting the indexes to cut the df into folds
    start_fold_train, end_fold_train, start_fold_test, end_fold_test = fold_indexes(
        df=df,verbose=verbose)
    # Starting the iteration on folds
    for i in tqdm(range(len(start_fold_train))):
        # reinitialise the model between two folds to reset training
        model.set_model()
        # instantiating train fold
        if verbose >= 10 :
            print("Creating sequences in the train_fold")
        train_fold_df = df.loc[start_fold_train[i]:
                                 end_fold_train[i]].copy().reset_index(drop=True)
        # Setting the indexes to cut the train_fold in regular sequences and targets
        sequence_starts, sequence_stops, target_idx = sequence_indexes(
            df=train_fold_df,verbose=verbose)
        # Initializing the X_train, Y_train
        X_train, Y_train = [], []
        # Starting the iteration on the sequences to create X,Y_train
        for j in range(len(sequence_starts)):

            X_train_seq = np.array(
                train_fold_df.iloc[sequence_starts[j]:sequence_stops[j]])
            y_train = train_fold_df.iloc[target_idx[j], -1]
            #Converting the little df to np array
            X_train.append(np.array(X_train_seq))
            Y_train.append(np.array(y_train))
        # Converting the list of array to an array
        Y_train = np.array(Y_train)
        X_train = np.array(X_train)

        #Same process as ahead but on the test_fold
        test_fold_df = df.loc[start_fold_test[i]:end_fold_test[i]].copy(
        ).reset_index(drop=True)
        sequence_starts, sequence_stops, target_idx = sequence_indexes(df=test_fold_df,verbose=verbose)
        Y_test,X_test = [],[]
        for j in range(len(sequence_starts)):
            X_test_seq = test_fold_df.iloc[sequence_starts[j]:sequence_stops[j]]
            y_test = test_fold_df.iloc[target_idx[j], -1]
            X_test.append(np.array(X_test_seq))
            Y_test.append(np.array(y_test))
        Y_test = np.array(Y_test)
        X_test = np.array(X_test)
        # Now we have an X_test,Y_test , X_train,Y_train ready to be processed

        #TODO SHUFFLING THE X,y
        Y_pred = model.run(X_test,X_train, Y_train)

        # Keeping these lines in case we want to use Y_test, Y_pred in the futur
        reality.append(Y_test)
        prediction.append(Y_pred)

    if verbose :
        print(f"{model.name} has been cross-validated")
    return reality, prediction

def cross_val_trade(model, df,
                    cv=True,
              verbose:int=0,
              hyperparams=None):
    '''Compute and process a complete cross validation of a given model,
    taking personalised metrics into account
    params :
    verbose range from 0 to 10 and allow to print a few informations during the
        process
    return reality, prediction
    if cv = False this function will train on the whole dataframe without making fold and
    predict the last 90 days you can plot with :
    Y_true_past, Y_true, Y_diff_test, Y_diff_pred = cross_val_trade(model,df,cv=False,verbose=True)
    fig,axs = plt.subplots(2,1)
    axs[0].plot(Y_test)
    axs[0].plot(Y_pred, c= "r")
    plt.show
    axs[1].plot(Y_true)
    axs[1].plot(Y_true_past + Y_true_past* Y_pred, c= "r")
    '''
    df = df.drop(columns=["date"])
    # Initializing the variable to return
    prediction_diff, past_reality, reality, reality_diff = [], [], [], []
    # Setting the indexes to cut the df into folds
    if cv == True :
        start_fold_train, end_fold_train, start_fold_test, end_fold_test = fold_indexes(
            df=df,verbose=verbose)
        # Starting the iteration on folds
        for i in range(len(start_fold_train)):
            # reinitialise the model between two folds to reset training
            model.set_model()
            # instantiating train fold
            train_fold_df = df.loc[start_fold_train[i]:
                                    end_fold_train[i]].copy().reset_index(drop=True)
            # Setting the indexes to cut the train_fold in regular sequences and targets
            sequence_starts, sequence_stops, target_idx = sequence_indexes(
                df=train_fold_df,verbose=verbose)
            # Initializing the X_train, Y_train
            X_train, Y_train = [], []
            # Starting the iteration on the sequences to create X,Y_train
            for j in range(len(sequence_starts)):

                X_train_seq = np.array(
                    train_fold_df.iloc[sequence_starts[j]:sequence_stops[j]])
                y_train = train_fold_df.iloc[target_idx[j], -1]
                #Converting the little df to np array
                X_train.append(np.array(X_train_seq))
                Y_train.append(np.array(y_train))
            # Converting the list of array to an array
            Y_train = np.array(Y_train)
            X_train = np.array(X_train)

            #Same process as ahead but on the test_fold
            test_fold_df = df.loc[start_fold_test[i]:end_fold_test[i]].copy(
            ).reset_index(drop=True)
            sequence_starts, sequence_stops, target_idx = sequence_indexes(df=test_fold_df,verbose=verbose)
            Y_test,X_test,Y_true_price_past, Y_true_price = [],[],[], []

            for j in range(len(sequence_starts)):
                X_test_seq = test_fold_df.iloc[sequence_starts[j]:sequence_stops[j]]
                y_test = test_fold_df.iloc[target_idx[j], -1]
                y_true_price_past = test_fold_df.iloc[target_idx[j]-HORIZON, -2]
                y_true_price = test_fold_df.iloc[target_idx[j], -2]

                #y_true_price_minus_horizon = test_fold_df.iloc[target_idx[j], -2]
                X_test.append(np.array(X_test_seq))
                Y_true_price_past.append(y_true_price_past)
                Y_true_price.append(y_true_price)
                Y_test.append(np.array(y_test))

            Y_test = np.array(Y_test)
            Y_true_past = np.array(Y_true_price_past)
            Y_true = np.array(Y_true_price)
            X_test = np.array(X_test)
            # Now we have an X_test,Y_test , X_train,Y_train ready to be processed

            #TODO SHUFFLING THE X,y
            Y_pred = model.run(X_test,X_train, Y_train)

            # Keeping these lines in case we want to use Y_test, Y_pred in the futur
            prediction_diff.append(Y_pred)
            reality_diff.append(Y_test)
            past_reality.append(Y_true_past)
            reality.append(Y_true)

        # NOTA BENE PAST_REALITY HERE IS {HORIZON DAYS BEHIND REALITY}
        return past_reality, reality,reality_diff, prediction_diff
    else :
        train_fold_df = df.iloc[:-90]
        test_fold_df = df.iloc[-(90+SEQUENCE_LENGHT+HORIZON):]
    # Setting the indexes to cut the train_fold in regular sequences and targets
        sequence_starts, sequence_stops, target_idx = sequence_indexes(
            df=train_fold_df,verbose=verbose)
        # Initializing the X_train, Y_train
        X_train, Y_train = [], []
        # Starting the iteration on the sequences to create X,Y_train
        for j in range(len(sequence_starts)):

            X_train_seq = np.array(
                train_fold_df.iloc[sequence_starts[j]:sequence_stops[j]])
            y_train = train_fold_df.iloc[target_idx[j], -1]
            #Converting the little df to np array
            X_train.append(np.array(X_train_seq))
            Y_train.append(np.array(y_train))
        # Converting the list of array to an array
        Y_train = np.array(Y_train)
        X_train = np.array(X_train)

        Y_test,X_test,Y_true_price_past, Y_true_price = [],[],[], []

        # Setting the indexes to cut the test_fold in regular sequences and targets
        sequence_starts, sequence_stops, target_idx = sequence_indexes(
            df=test_fold_df,verbose=verbose)
        for j in range(len(sequence_starts)):
            X_test_seq = test_fold_df.iloc[sequence_starts[j]:sequence_stops[j]]
            y_test = test_fold_df.iloc[target_idx[j], -1]
            y_true_price_past = test_fold_df.iloc[target_idx[j]-HORIZON, -2]
            y_true_price = test_fold_df.iloc[target_idx[j], -2]

            #y_true_price_minus_horizon = test_fold_df.iloc[target_idx[j], -2]
            X_test.append(np.array(X_test_seq))
            Y_true_price_past.append(y_true_price_past)
            Y_true_price.append(y_true_price)
            Y_test.append(np.array(y_test))

        Y_diff_test = np.array(Y_test)
        Y_true_past = np.array(Y_true_price_past)
        Y_true = np.array(Y_true_price)
        X_test = np.array(X_test)
        # Now we have an X_test,Y_test , X_train,Y_train ready to be processed

        #TODO SHUFFLING THE X,y
        Y_diff_pred = model.run(X_test,X_train, Y_train)
        # returned in a list shape
        return [Y_true_past], [Y_true], [Y_diff_test], [Y_diff_pred]








if __name__ == "__main__":
    print("1")
    df = ApiCall().read_local()
    model = RnnDlModel(epochs=1, patience=1)
    cross_val(model,df)
    print("debug done")
