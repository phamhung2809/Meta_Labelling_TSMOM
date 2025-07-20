import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report,roc_auc_score, make_scorer, log_loss, accuracy_score
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
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.regularizers import l1, l2
from keras.optimizers import Adam

from sklearn import tree
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.losses import BinaryCrossentropy

from optuna import Trial, create_study, create_trial

def objective_rf(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
        'max_depth': trial.suggest_int('max_depth', 10, 25),
        'min_samples_split': trial.suggest_float('min_samples_split', 0,1),
        'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0,1),
        'max_features': trial.suggest_float('max_features', 0,1),
    }
    model = RandomForestClassifier(**params)

    model.fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)
    val_loss = log_loss(y_val, val_proba)

    return val_loss  # Optuna sẽ tối thiểu hoá log_loss

def train_rf_optuna(X_train, y_train, X_val, y_val, n_trials=100):
    # Tối ưu hoá hyperparameter
    study = create_study(direction='minimize')
    study.optimize(lambda trial: objective_rf(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)

    print("Best hyperparameters:", study.best_params)

    # Huấn luyện mô hình tốt nhất
    best_model = RandomForestClassifier(
        **study.best_params,
        random_state=42,
        n_jobs=-1
    )
    best_model.fit(X_train, y_train)

    return best_model, study


def build_lasso_model(trial, num_feature):
    model = Sequential([
        Dense(
            1,
            input_shape=(num_feature,),
            activation='sigmoid',
            kernel_regularizer=l1(trial.suggest_categorical("l1_weight", [1e-4, 1e-3, 1e-2, 0.1]))
        )
    ])

    optimizer = Adam(
        learning_rate=trial.suggest_categorical("learning_rate", [1e-3, 1e-2, 1e-1, 1.0]),
        clipnorm=trial.suggest_categorical("max_grad_norm", [1e-2, 0.1, 1.0, 10.0])
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy'
    )

    return model

def objective_lasso(trial, X_train, y_train, X_val, y_val, num_feature):
    model = build_lasso_model(trial, num_feature)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

    es = EarlyStopping(monitor='val_loss', patience=25, verbose=0, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    y_pred = model.predict(X_val)
    loss = log_loss(y_val, y_pred)

    return loss

def train_lasso_optuna(X_train, y_train, X_val, y_val, num_feature, n_trials=100):
    study = create_study(direction="minimize")

    # Tối ưu hoá
    study.optimize(
        lambda trial: objective_lasso(trial, X_train, y_train, X_val, y_val, num_feature),
        n_trials=n_trials
    )

    best_trial = study.best_trial
    print("Best hyperparameters:", best_trial.params)

    # Huấn luyện lại mô hình tốt nhất
    best_model = build_lasso_model(best_trial, num_feature)
    best_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=best_trial.params["batch_size"],
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)
        ],
        verbose=1
    )

    return best_model, study

def build_mlp_model(trial, num_feature):
    model = Sequential([
        Dropout(0, input_shape=(num_feature,)),
        Dense(
            units=trial.suggest_categorical("units", [5, 20, 40]),
            activation=trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
        ),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(
        learning_rate=trial.suggest_categorical("learning_rate", [1e-3, 1e-2, 1e-1]),
        clipnorm=trial.suggest_categorical("max_grad_norm", [1e-2, 0.1, 1.0, 10.0])
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
    )

    return model



def objective_mlp(trial, X_train, y_train, X_val, y_val, num_feature):
    model = build_mlp_model(trial, num_feature)

    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    es = EarlyStopping(monitor='val_loss', patience=25, verbose=0, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    y_pred = model.predict(X_val)
    return log_loss(y_val, y_pred)

def train_mlp_optuna(X_train, y_train, X_val, y_val, num_feature, n_trials=100):
    study = create_study(direction='minimize')
    study.optimize(
        lambda trial: objective_mlp(trial, X_train, y_train, X_val, y_val, num_feature),
        n_trials=n_trials
    )

    print("Best hyperparameters:", study.best_params)

    # Dùng hyperparameter tốt nhất để train lại mô hình
    best_trial = study.best_trial
    best_model = build_mlp_model(best_trial, num_feature)
    best_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=best_trial.params["batch_size"],
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)
        ],
        verbose=1
    )

    return best_model, study

def build_lstm_model(trial, input_shape):
    model = Sequential()
    model.add(LSTM(
        units=64,
        input_shape=input_shape,
        return_sequences=True
    ))
    model.add(LSTM(
        units=128,
        return_sequences=True
    ))
    model.add(LSTM(
        units=64,
        return_sequences=False
    ))
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(
        learning_rate=trial.suggest_categorical("learning_rate", [1e-3, 1e-2, 1e-1]),
        clipnorm=trial.suggest_categorical("max_grad_norm", [1e-2, 0.1, 1.0, 10.0])
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy"
    )

    return model

def objective_lstm(trial, X_train, y_train, X_val, y_val):
    input_shape = X_train.shape[1:]
    model = build_lstm_model(trial, input_shape)

    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    es = EarlyStopping(monitor="val_loss", patience=25, verbose=0, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    y_pred = model.predict(X_val)
    return log_loss(y_val, y_pred)


def train_lstm_optuna(X_train, y_train, X_val, y_val, n_trials=100):
    study = create_study(direction="minimize")
    study.optimize(
        lambda trial: objective_lstm(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials
    )

    best_trial = study.best_trial
    print("Best hyperparameters:", best_trial.params)

    input_shape = X_train.shape[1:]
    best_model = build_lstm_model(best_trial, input_shape)

    best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=best_trial.params["batch_size"],
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=25, verbose=1, restore_best_weights=True)
        ],
        verbose=1
    )

    return best_model, study


