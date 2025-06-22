import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report,roc_auc_score, make_scorer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.regularizers import l1, l2
from keras.optimizers import Adam

from sklearn import tree
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from hypopt import GridSearch
import keras_tuner as kt

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.losses import BinaryCrossentropy

from optuna import Trial, create_study, create_trial

def instantiate_random_trees(trial) -> RandomForestClassifier:
  params = {
    'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
    'max_depth': trial.suggest_int('max_depth', 10, 25),
    'min_samples_split': trial.suggest_float('min_samples_split', 0,1),
    'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0,1),
    'max_features': trial.suggest_float('max_features', 0,1),
    'n_jobs': -1,
    'random_state': 42
  }
  return RandomForestClassifier(**params)

def instantiate_model(trial : Trial) -> Pipeline:

  random_tree_model = instantiate_random_trees(trial)

  pipeline = Pipeline(
      [
          # ("scaler", scaler),
          ("classifier", random_tree_model),
      ]
  )

  return pipeline

def objective(trial : Trial, X : pd.DataFrame, y : np.ndarray | pd.Series, random_state : int=42) -> float:

  model = instantiate_model(trial)

  split = TimeSeriesSplit(n_splits= 5)
  roc_auc_scorer = make_scorer(roc_auc_score)
  scores = cross_val_score(model, X,y, scoring=roc_auc_scorer, cv= split)

  return np.min([np.mean(scores), np.median([scores])])


# def train_random_forest_optuna(trainX,trainY):

#   study = create_study(study_name='optimization', direction='maximize')

#   study.optimize(lambda trial: objective(trial, trainX, trainY), n_trials=100)

#   best_trial = study.best_trial

#   model = instantiate_model(best_trial)
#   model.fit(trainX, trainY)

#   return model

def train_random_forest(trainX,trainY):
  ''' Hàm train mô hình Random Forest
      INPUT: trainX,trainY(dataframe): các tập train/test đã được chia ra dùng để huấn luyện
      OUTPUT: pipeline(pipeline): mô hình đã được train
  '''

  # Tạo pipeline bao gồm 2 bước: scale các feature và Random Forest
  pipeline = Pipeline(
      [
          # ("scaler", TimeSeriesScalerDF('VN30')),
          # ("scaler", scaler),
          ("classifier", RandomForestClassifier()),
          # ("classifier", DecisionTreeClassifier()),

      ]
  )

  # Fit mô hình
  pipeline.fit(trainX,trainY)

  return pipeline



class Lasso_supervised(kt.HyperModel):
    def __init__(self, k, num_feature,binary):
        self.k = k
        self.binary = binary
        self.num_feature = num_feature

    def build(self,hp):
        model = Sequential([
            Dense(1, input_shape = (self.num_feature,),kernel_regularizer = l1(hp.Choice("l1_weight", [1e-4, 1e-3, 1e-2, 0.1,])),activation= 'softmax' if self.binary else None)
        ])
        
        if self.binary == True:
            loss = 'binary_crossentropy'
        else: loss = tf.keras.metrics.RootMeanSquaredError()

        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice("learning_rate", [1e-3, 1e-1, 1e-2, 0.1,1.0]),
                clipnorm = hp.Choice("max_grad_norm", [1e-2, 0.1, 1.0, 10.0])
            ),
            loss= loss,
        )
        return model
    def fit(self, hp, model, *args, **kwargs):
    
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [256,512,1024]),
            **kwargs, epochs = 100, verbose=1
        )

def train_Lasso_supervised(X_train,y_train,k,h,num_feature,binary = True):
    tuner = kt.GridSearch(
        Lasso_supervised(k = k,binary=binary,num_feature = num_feature),
        objective="loss",
        max_trials=100,
        overwrite=True,
        directory="tuning_dir",
        project_name= f"tune_Lasso_supervised_{'binary' if binary else 'reg'}",

    )

    es = EarlyStopping(monitor='loss', verbose=1, patience=25)

    checkpoint_filepath = (
        'Checkpoint/checkpoint_lasso_sup_binary.model.keras' if binary
        else 'Checkpoint/checkpoint_lasso_sup_reg.model.keras'
    )
    model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor = 'loss',
    save_best_only=True)    


    tuner.search(X_train, y_train, callbacks = [es])

    hypermodel = Lasso_supervised(k = k,binary=binary, num_feature = num_feature)
    best_hp = tuner.get_best_hyperparameters()[0]
    model = hypermodel.build(best_hp)

    history = hypermodel.fit(best_hp,model,X_train, y_train,callbacks = [model_checkpoint_callback])
    
    return model,history



class MLP_supervised(kt.HyperModel):
    def __init__(self, k,num_feature,binary):
        self.k = k
        self.binary = binary
        self.num_feature = num_feature

    def build(self,hp):
        model = Sequential([
            Dropout(0, input_shape=(self.num_feature,)),
            Dense(units=hp.Choice(f"units", [5, 20, 40]),activation = hp.Choice('activation', ['relu'])),
            Dropout(rate=hp.Choice("dropout", [0.1, 0.3, 0.5])),
            Dense(1,activation = 'softmax' if self.binary else None),
        ])

        if self.binary == True:
            loss =  'binary_crossentropy'
        else: loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice("learning_rate", [1e-3, 1e-1, 1.0]),
                clipnorm = hp.Choice("max_grad_norm", [1e-2, 0.1, 1.0, 10.0])
            ),
            loss= loss,
        )
        return model
    def fit(self, hp, model, *args, **kwargs):        

        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [256,512,1024]),
            **kwargs, epochs = 100, verbose=1
        )
    


def train_MLP_supervised(X_train,y_train,k,h,num_feature,binary = True):
    tuner = kt.GridSearch(
        MLP_supervised(k = k,binary = binary,num_feature = num_feature),
        objective="loss",
        max_trials=100,
        overwrite=True,
        directory="tuning_dir",
        project_name= f"tune_MLP_supervised_{'binary' if binary else 'reg'}",
    )

    es = EarlyStopping(monitor='loss', verbose=1, patience=25)

    checkpoint_filepath = (
        'Checkpoint/checkpoint_mlp_sup_binary.model.keras' if binary
        else 'Checkpoint/checkpoint_mlp_sup_reg.model.keras'
    )
    model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor = 'loss',
    save_best_only=True, verbose = 1)    

    tuner.search(X_train, y_train, callbacks = [es])

    hypermodel = MLP_supervised(k = k,binary = binary, num_feature= num_feature)
    best_hp = tuner.get_best_hyperparameters()[0]
    model = hypermodel.build(best_hp)

    history = hypermodel.fit(best_hp,model,X_train, y_train,callbacks = [model_checkpoint_callback])
    
    return model,history    


    
class LSTM_supervised(kt.HyperModel):
    def __init__(self, k, num_feature,binary):
        self.k = k  # number of timesteps
        self.binary = binary
        self.num_feature = num_feature

    def build(self, hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Choice("units", [16, 32, 64]),
            input_shape=(self.num_feature,1),
            return_sequences=False
        ))
        model.add(Dropout(rate=hp.Choice("dropout", [0.1, 0.3, 0.5])))

        if self.binary:
            model.add(Dense(1, activation="softmax"))
            loss = 'binary_crossentropy'
        else:
            model.add(Dense(1))
            loss = "mse"

        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice("learning_rate", [1e-4, 1e-3, 1e-2]),
                clipnorm=hp.Choice("max_grad_norm", [1e-2, 0.1, 1.0, 10.0])
            ),
            loss=loss
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [64, 128, 256]),
            epochs=100,
            verbose=1,
            **kwargs
        )



def train_LSTM_supervised(X_train, y_train, k ,num_feature, binary=True):
    X_train = X_train.to_numpy()
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    tuner = kt.GridSearch(
        LSTM_supervised(k=k, binary=binary, num_feature = num_feature),
        objective="loss",
        max_trials=100,
        overwrite=True,
        directory="tuning_dir",
        project_name=f"tune_LSTM_supervised_{'binary' if binary else 'reg'}",
    )

    es = EarlyStopping(monitor='loss', verbose=1, patience=25)

    checkpoint_filepath = (
        'Checkpoint/checkpoint_lstm_sup_binary.model.keras' if binary
        else 'Checkpoint/checkpoint_lstm_sup_reg.model.keras'
    )
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='loss',
        save_best_only=True
    )

    tuner.search(X_train, y_train, callbacks=[es])

    hypermodel = LSTM_supervised(k=k, binary=binary, num_feature = num_feature)
    best_hp = tuner.get_best_hyperparameters()[0]
    model = hypermodel.build(best_hp)

    history = hypermodel.fit(best_hp, model, X_train, y_train, callbacks=[model_checkpoint_callback])

    return model, history
