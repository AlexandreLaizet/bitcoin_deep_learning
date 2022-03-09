from bitcoin_deep_learning.call_api import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np

# Dummy model will be a model using the last price value the predict the next price value

class DummyModel():
    """
    Return the last value on the price column as ypred
    """

    def __init__(self):
        self.set_model()
        self.name = 'Dummy'
        self.hyperparams = None

    def preproc(self, X_test, X_train):
        return X_test, X_train

    def set_model(self):
        self.model = self
        return self

    def fit(self,X_train,y_train):
        return self

    def predict(self, X_test):
        y_pred = X_test[:, -7, -1]
        return y_pred

    def run(self, X_test, X_train, y_train):
        X_test, X_train = self.preproc(X_test, X_train)
        self.fit(X_train, y_train)
        return self.predict(X_test)

# Baseline model will use a simple linear regression on the last value of the sequence

class LinearRegressionBaselineModel():
    """
    Predict y_pred based on a linear regression
    """

    def __init__(self, alpha = 1, l1_ratio = 0.5):
        self.name = "LinearReg"
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = 10_000
        self.hyperparams = {"alpha":alpha,"l1_ratio":l1_ratio}
        self.set_model()

    def preproc(self, X_test, X_train):
        scaler = MinMaxScaler()
        X_train = X_train[:, -7, :]
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = X_test[:, -7, :]
        X_test = scaler.transform(X_test)
        #scaling y_train ?
        return X_test, X_train

    def set_model(self):
        self.model = ElasticNet(alpha =self.alpha,
                                l1_ratio = self.l1_ratio,
                                fit_intercept=True,
                                precompute=False,
                                max_iter=self.max_iter,
                                copy_X=True,
                                tol=0.0001,
                                warm_start=False,
                                positive=False,
                                random_state=None,
                                selection='cyclic')
        return self

    def fit(self, X_train, y_train = None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def run(self, X_test, X_train, y_train):
        X_test, X_train = self.preproc(X_test, X_train)
        self.fit(X_train, y_train)
        return self.predict(X_test)

# RNN model is our deep learning model that we will tune to beat our both dummy and baseline models

loss = 'mse'
optimizer = 'rmsprop'
#metrics = ['mae, mape']
metrics = 'mae'

class RnnDlModel():
    """
    Return the last value on the price column
    """

    def __init__(self, L1 = 0.01, L2 = 0.01,
                 loss = loss, optimizer = optimizer, metrics = metrics,
                 epochs=50, patience=10):
        self.name = "RNN"
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.L1 = L1
        self.L2 = L2
        self.epochs = epochs
        self.patience = patience
        self.history = None
        self.model = None
        self.hyperparams = {"L1":self.L1,"L2":self.L2,
                            "epochs":self.epochs,"patience":self.patience}

    def preproc(self, X_test, X_train, y_train = None):
        #Compute X_train.min
        #Compute X_train.max
        mins = np.expand_dims(np.min(X_train, axis = (0,1)),(0,1))
        maxes = np.expand_dims(np.max(X_train, axis= (0, 1)),(0,1))
        X_train_scaled = (X_train - mins) / (maxes - mins)
        #Scaler X_train
        #Scaler X_test
        X_test_scaled = (X_test - mins) / (maxes - mins)
        #print(np.max(X_train_scaled, axis = (0,1)))#
        return X_train_scaled, X_test_scaled

    def set_model(self):
        self.model = Sequential()

        reg_l1 = regularizers.L1(self.L1)
        reg_l2 = regularizers.L2(self.L2)
        reg_l1_l2 = regularizers.l1_l2(l1=0.005, l2=0.0005)

        self.model.add(GRU(units=128, return_sequences=True, activation='relu'))
        #self.model.add(layers.Dropout(rate=0.2))
        self.model.add(GRU(units=64, return_sequences=True, activation='relu'))
        #self.model.add(layers.Dropout(rate=0.2))
        self.model.add(GRU(units=32, activation='relu'))
        #self.model.add(layers.Dropout(rate=0.2))

        self.model.add(
            layers.Dense(32, activation="relu", kernel_regularizer=reg_l1))
        #self.model.add(layers.Dropout(rate=0.2))
        self.model.add(
            layers.Dense(16, activation="relu", bias_regularizer=reg_l2))
        #self.model.add(layers.Dropout(rate=0.2))
        self.model.add(
            layers.Dense(8, activation="relu",
                         activity_regularizer=reg_l1_l2))
        #self.model.add(layers.Dropout(rate=0.2))
        self.model.add(layers.Dense(1, activation="linear"))
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return self

    def fit(self, X_train, y_train, verbose = 1):
        #print(X_train.shape)
        es = EarlyStopping(patience=self.patience, restore_best_weights=True)
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size = 32,  # Too small --> no generalization. Too large --> compute slowly
            epochs=self.epochs,
            validation_split=0.2,
            #validation_data = (X_test,Y_test),
            callbacks=[es],
            workers=6,
            use_multiprocessing=True,
            verbose=verbose)
        return self

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def run(self, X_test, X_train, y_train):
        X_test,X_train = self.preproc(X_test, X_train)
        self.set_model()
        self.fit(X_train, y_train)
        return self.predict(X_test)


class RandomForestReg():
    """
    Predict y_pred based on a linear regression
    """
    def __init__(self,n_estimators=1000):
        self.name = "ClassicLinearReg"
        self.estimators = n_estimators
        self.hyperparams = {"n_estimator":self.estimators}
        self.set_model()

    def preproc(self, X_test, X_train):
        scaler = MinMaxScaler()
        X_train = X_train[:, -1, :]
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = X_test[:, -1, :]
        X_test = scaler.transform(X_test)
        #scaling y_train ?
        return X_test, X_train

    def set_model(self):
        self.model = RandomForestRegressor()
        return self

    def fit(self, X_train, y_train = None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def run(self, X_test, X_train, y_train):
        X_test, X_train = self.preproc(X_test, X_train)
        breakpoint()
        self.fit(X_train, y_train)
        return self.predict(X_test)




if __name__ == '__main__':
    # n_sequences = 263
    # n_sequences_t = 18
    # sequence_length = 90
    # n_features = 30
    # X_train = np.random.rand(n_sequences, sequence_length, n_features)
    # y_train = np.random.rand(n_sequences)
    # X_test = np.random.rand(n_sequences_t, sequence_length, n_features)
    # y_test = np.random.rand(n_sequences_t)

    #Call API
    from bitcoin_deep_learning.cross_val import cross_val
    from bitcoin_deep_learning.call_api import ApiCall
    from bitcoin_deep_learning.trainer import train
    # df = ApiCall().get_clean_data()
    # ApiCall().data_to_csv(df)
    # df = ApiCall().read_local()

    # Dummy model
    # dummy_model = DummyModel()
    # print(cross_val(dummy_model, df))

    # Regression model
    # reg_lin_model = LinearRegressionBaselineModel()
    # print(cross_val(reg_lin_model, df))

    #RNN model
    # rnn_model = RnnDlModel()
    # print(cross_val(rnn_model, df))

    df = ApiCall().read_local(data="train")
    model = LinearRegressionBaselineModel()

    train(model,df)
