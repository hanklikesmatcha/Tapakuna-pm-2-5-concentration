import numpy as np
import pandas as pd
import warnings
from requests import options

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from keras.layers import LSTM
import keras
import time

import pre_processing
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
from sklearn.model_selection import train_test_split

ROOT = "data"


def pearson_correlation():
    df = pd.read_csv(f"{ROOT}/cleaned_dataset.csv").dropna(axis=0)
    # Correlation Matrix
    corr = df.corr(method="pearson").round(2)
    corr.to_csv(f"{ROOT}/corr.csv")
    sns.heatmap(corr, annot=True)
    plt.subplots_adjust(wspace=0.7, bottom=0.40, left=0.3)
    plt.savefig(f"{ROOT}/Plotting_Correlation_HeatMap.png")
    plt.show()


def variance():
    df = pd.read_csv(f"{ROOT}/cleaned_dataset.csv").dropna(axis=0)
    df.plot.scatter(
        x="timestamp",
        y="pm_25 - Value (µg/m³)",
        c="air_temp - Value (°C)",
        colormap="viridis",
    )
    pd.to_datetime()
    variance = df[["timestamp", "pm_25 - Value (µg/m³)", "wind_speed - Value (m/s)"]]
    plt.plot()
    plt.show()


def summary():
    df = pd.read_csv(f"{ROOT}/cleaned_dataset.csv").dropna(axis=0)
    # Summary
    summary = df[
        [
            "pm_25 - Value (µg/m³)",
            "wind_speed - Value (m/s)",
            "wind_direction - Value (°)",
            "no2 - Value (µg/m³)",
            "no - Value (µg/m³)",
            "air_temp - Value (°C)",
        ]
    ]
    summary[["pm_25 - Value (µg/m³)"]].plot.kde()
    summary.sum().round(2).to_csv(f"{ROOT}/sum.csv")
    summary.mean().round(2).to_csv(f"{ROOT}/mean.csv")
    summary.median().round(2).to_csv(f"{ROOT}/median.csv")
    summary.describe().round(2).to_csv(f"{ROOT}/describe.csv")
    summary.groupby("pm_25 - Value (µg/m³)").describe().round(2).to_csv(
        f"{ROOT}/describe_group_by.csv"
    )
    plt.show()


def training_for_MLP():
    df = pd.read_csv(f"{ROOT}/cleaned_dataset.csv").dropna(axis=0)[
        [
            "pm_25 - Value (µg/m³)",
            "wind_speed - Value (m/s)",
            "wind_direction - Value (°)",
            "no2 - Value (µg/m³)",
            "no - Value (µg/m³)",
            "air_temp - Value (°C)",
        ]
    ]
    X = df[["pm_25 - Value (µg/m³)"]]
    y = df[
        [
            "wind_speed - Value (m/s)",
            "wind_direction - Value (°)",
            "no2 - Value (µg/m³)",
            "no2 - Value (µg/m³)",
            "air_temp - Value (°C)",
        ]
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=1, shuffle=True
    )
    regr = MLPRegressor(
        random_state=1,
        hidden_layer_sizes=(2, 1),
        activation="logistic",
        batch_size=5000,
    ).fit(X_train, y_train)
    predict = regr.predict(X_test)
    RMSE_1 = mean_squared_error(
        X_test["pm_25 - Value (µg/m³)"][:5], predict[0], squared=True
    )
    MAE_1 = mean_absolute_error(X_test["pm_25 - Value (µg/m³)"][:5], predict[0])
    print("1", RMSE_1.round(2))
    print("1", MAE_1.round(2))
    RMSE_2 = mean_squared_error(
        X_test["pm_25 - Value (µg/m³)"][:5], predict[1], squared=True
    )
    MAE_2 = mean_absolute_error(X_test["pm_25 - Value (µg/m³)"][:5], predict[1])
    print("2", RMSE_2.round(2))
    print("2", MAE_2.round(2))
    RMSE_3 = mean_squared_error(
        X_test["pm_25 - Value (µg/m³)"][:5], predict[2], squared=True
    )
    MAE_3 = mean_absolute_error(X_test["pm_25 - Value (µg/m³)"][:5], predict[2])
    print("3", RMSE_3.round(2))
    print("3", MAE_3.round(2))
    RMSE_4 = mean_squared_error(
        X_test["pm_25 - Value (µg/m³)"][:5], predict[3], squared=True
    )
    MAE_4 = mean_absolute_error(X_test["pm_25 - Value (µg/m³)"][:5], predict[3])
    print("4", RMSE_4.round(2))
    print("4", MAE_4.round(2))
    RMSE_5 = mean_squared_error(
        X_test["pm_25 - Value (µg/m³)"][:5], predict[4], squared=True
    )
    MAE_5 = mean_absolute_error(X_test["pm_25 - Value (µg/m³)"][:5], predict[4])
    print("5", RMSE_5.round(2))
    print("5", MAE_5.round(2))


def normalize(input: np.ndarray) -> np.ndarray:
    # normalize input data
    scaler = MinMaxScaler()
    x = scaler.fit_transform(input)
    return x


def max_regressor():
    """
    Code for MLP part 2)
    """
    df = pd.read_csv(f"{ROOT}/cleaned_dataset.csv").dropna(axis=0)

    # get input variables
    x = df[
        [
            "wind_speed - Value (m/s)",
            "wind_direction - Value (°)",
            "no2 - Value (µg/m³)",
            "no - Value (µg/m³)",
            "air_temp - Value (°C)",
        ]
    ].values

    # pm_25 - Value (µg/m³) is the target
    y = df["pm_25 - Value (µg/m³)"].values

    # split into train and test sets in time-series fashion 30% test set
    # get last 30% as test set
    len_dataset = len(x)
    x_train, x_test = x[: int(len_dataset * 0.7)], x[int(len_dataset * 0.7) :]
    y_train, y_test = y[: int(len_dataset * 0.7)], y[int(len_dataset * 0.7) :]

    # normalize input data
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # create and train the model with a single hidden layer of 25 neurons
    model = MLPRegressor(
        hidden_layer_sizes=(25),
        batch_size=16,
    ).fit(x_train, y_train)

    # make predictions
    y_pred = model.predict(x_test)

    # evaluate the model
    RMSE = mean_squared_error(y_test, y_pred, squared=True)
    MAE = mean_absolute_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)
    print("RMSE", RMSE.round(2))
    print("MAE", MAE.round(2))
    print("R2", R2.round(2))


def max_iterate_regressor():
    """
    Code for MLP part 3)
    """
    df = pd.read_csv(f"{ROOT}/cleaned_dataset.csv").dropna(axis=0)

    # get input variables
    x = df[
        [
            "wind_speed - Value (m/s)",
            "wind_direction - Value (°)",
            "no2 - Value (µg/m³)",
            "no - Value (µg/m³)",
            "air_temp - Value (°C)",
        ]
    ].values

    # pm_25 - Value (µg/m³) is the target
    y = df["pm_25 - Value (µg/m³)"].values

    # split into train and test sets in time-series fashion 30% test set
    # get last 30% as test set
    len_dataset = len(x)
    x_train, x_test = x[: int(len_dataset * 0.7)], x[int(len_dataset * 0.7) :]
    y_train, y_test = y[: int(len_dataset * 0.7)], y[int(len_dataset * 0.7) :]

    # normalize input data
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    rmse_record = []
    mae_record = []
    r2_record = []

    # iterate over the different combinations of layer sizes
    for i in range(1, 25):
        # create and train the model with a single hidden layer of 25 neurons
        model = MLPRegressor(
            hidden_layer_sizes=(25 - i, i),
            batch_size=16,
        ).fit(x_train, y_train)

        # make predictions
        y_pred = model.predict(x_test)

        # evaluate the model
        print(f"Metrics for Layer1: {25-i}, Layer2: {i}")
        RMSE = mean_squared_error(y_test, y_pred, squared=True)
        MAE = mean_absolute_error(y_test, y_pred)
        R2 = r2_score(y_test, y_pred)
        print("RMSE", RMSE.round(2))
        print("MAE", MAE.round(2))
        print("R2", R2.round(2))

        rmse_record.append(RMSE)
        mae_record.append(MAE)
        r2_record.append(R2)

    # plot the results
    plt.plot(range(1, 25), rmse_record, label="RMSE")
    plt.plot(range(1, 25), mae_record, label="MAE")
    plt.xlabel("Number of Hidden Layers in Second Layer")
    plt.ylabel("RMSE/MAE")
    plt.show()

    # pplot r2 record
    plt.plot(range(1, 25), r2_record, label="R2")
    plt.xlabel("Number of Hidden Layers in Second Layer")
    plt.ylabel("R2")
    plt.show()


def max_lstm_model():
    """
    Code for LSTM part 2)
    """
    import keras  # bad practice but needed to make it clear what I added lol

    df = pd.read_csv(f"{ROOT}/cleaned_dataset.csv").dropna(axis=0)

    # make 2 dimensional time series data inputs and targets from dataframe
    x = df[
        [
            "wind_speed - Value (m/s)",
            "wind_direction - Value (°)",
            "no2 - Value (µg/m³)",
            "no - Value (µg/m³)",
            "air_temp - Value (°C)",
        ]
    ].values
    y = df["pm_25 - Value (µg/m³)"].values

    # split into train and test sets in time-series fashion 30% test set
    # get last 30% as test set
    len_dataset = len(x)
    x_train, x_test = x[: int(len_dataset * 0.7)], x[int(len_dataset * 0.7) :]
    y_train, y_test = y[: int(len_dataset * 0.7)], y[int(len_dataset * 0.7) :]

    # reshape input data to be 3D [batch_size, timesteps, features]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    # fit the model, for each epoch record train and test loss
    # run the model 30 times, and record the loss at each epoch
    # plot the mean, standard deviation, minimum and maximum of the loss
    # at each epoch
    train_loss = []
    test_loss = []
    times = []

    for i in range(30):
        start = time.time()
        # create and train the model
        model = keras.Sequential()
        model.add(
            keras.layers.LSTM(
                units=50,
                return_sequences=True,
                input_shape=(x_train.shape[1], x_train.shape[2]),
            )
        )
        model.add(keras.layers.LSTM(units=25))
        model.add(keras.layers.Dense(1))
        opt = keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss="mean_squared_error")
        history = model.fit(
            x_train,
            y_train,
            batch_size=16,
            epochs=100,
            verbose=0,
            validation_data=(x_test, y_test),
        )
        # record the per epoch loss for all runs
        train_loss.append(history.history["loss"])
        test_loss.append(history.history["val_loss"])
        times.append(time.time() - start)

    # plot mean, standard deviation, minimum and maximum of the loss at each epoch
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    plt.plot(np.mean(train_loss, axis=0), label="train_loss", color="blue", linewidth=2)
    plt.plot(np.mean(test_loss, axis=0), label="test_loss", color="red", linewidth=2)
    plt.fill_between(
        range(len(train_loss[0])),
        np.mean(train_loss, axis=0) - np.std(train_loss, axis=0),
        np.mean(train_loss, axis=0) + np.std(train_loss, axis=0),
        alpha=0.2,
        color="blue",
    )
    plt.fill_between(
        range(len(test_loss[0])),
        np.mean(test_loss, axis=0) - np.std(test_loss, axis=0),
        np.mean(test_loss, axis=0) + np.std(test_loss, axis=0),
        alpha=0.2,
        color="red",
    )
    plt.plot(
        np.min(train_loss, axis=0), label="train_loss_min", color="blue", linewidth=1
    )
    plt.plot(np.min(test_loss, axis=0), label="test_loss_min", color="red", linewidth=1)
    plt.plot(
        np.max(train_loss, axis=0), label="train_loss_max", color="blue", linewidth=1
    )
    plt.plot(np.max(test_loss, axis=0), label="test_loss_max", color="red", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # plot time taken to train each run
    plt.plot(times)
    plt.xlabel("Run")
    plt.ylabel("Time (s)")
    plt.show()


def max_iterate_lstm():
    """
    code for LSTM part 3
    """
    best_epoch = 40  # this should be changed to whatever the results of part 2 LSTM is

    df = pd.read_csv(f"{ROOT}/cleaned_dataset.csv").dropna(axis=0)

    # make 2 dimensional time series data inputs and targets from dataframe
    x = df[
        [
            "wind_speed - Value (m/s)",
            "wind_direction - Value (°)",
            "no2 - Value (µg/m³)",
            "no - Value (µg/m³)",
            "air_temp - Value (°C)",
        ]
    ].values
    y = df["pm_25 - Value (µg/m³)"].values

    # split into train and test sets in time-series fashion 30% test set
    # get last 30% as test set
    len_dataset = len(x)
    x_train, x_test = x[: int(len_dataset * 0.7)], x[int(len_dataset * 0.7) :]
    y_train, y_test = y[: int(len_dataset * 0.7)], y[int(len_dataset * 0.7) :]

    # reshape input data to be 3D [batch_size, timesteps, features]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    # iterate batch size pseudo-logirithmically from 2 to 60
    # have a total of 30 different batch sizes
    batch_sizes = [i for i in range(1, 30)]
    batch_sizes = [i * 2 for i in batch_sizes]

    train_loss = []
    test_loss = []
    times = []
    for batch_size in batch_sizes:
        start = time.time()

        model = keras.Sequential()
        model.add(
            keras.layers.LSTM(
                units=25,
                return_sequences=True,
                input_shape=(x_train.shape[1], x_train.shape[2]),
            )
        )
        model.add(keras.layers.LSTM(units=25))
        model.add(keras.layers.Dense(1))
        opt = keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss="mean_squared_error")
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=best_epoch,
            verbose=0,
            validation_data=(x_test, y_test),
        )
        # record loss of final epoch
        train_loss.append(history.history["loss"][-1])
        test_loss.append(history.history["val_loss"][-1])

        end = time.time()
        times.append(end - start)

    # plot train and test loss at each batch size
    plt.plot(batch_sizes, train_loss, label="train_loss", color="blue", linewidth=2)
    plt.plot(batch_sizes, test_loss, label="test_loss", color="red", linewidth=2)
    plt.xlabel("Batch Size")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # plot time taken to train each batch size
    plt.plot(batch_sizes, times)
    plt.xlabel("Batch Size")
    plt.ylabel("Time (s)")
    plt.show()


def max_iterate_lstm2():
    """
    code for LSTM part 4
    """
    best_epoch = 40  # this should be changed to whatever the results of part 2 LSTM is
    best_batch_size = (
        16  # this should be changed to whatever the results of part 3 LSTM is
    )

    df = pd.read_csv(f"{ROOT}/cleaned_dataset.csv").dropna(axis=0)

    # make 2 dimensional time series data inputs and targets from dataframe
    x = df[
        [
            "wind_speed - Value (m/s)",
            "wind_direction - Value (°)",
            "no2 - Value (µg/m³)",
            "no - Value (µg/m³)",
            "air_temp - Value (°C)",
        ]
    ].values
    y = df["pm_25 - Value (µg/m³)"].values

    # split into train and test sets in time-series fashion 30% test set
    # get last 30% as test set
    len_dataset = len(x)
    x_train, x_test = x[: int(len_dataset * 0.7)], x[int(len_dataset * 0.7) :]
    y_train, y_test = y[: int(len_dataset * 0.7)], y[int(len_dataset * 0.7) :]

    # reshape input data to be 3D [batch_size, timesteps, features]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    # iterate neurons pseudo-logirithmically from 2 to 60
    # have a total of 30 different batch sizes
    neurons = [i for i in range(1, 30)]
    neurons = [i * 2 for i in neurons]

    train_loss = []
    test_loss = []
    times = []
    for num_neurons in neurons:
        start = time.time()
        model = keras.Sequential()
        model.add(
            keras.layers.LSTM(
                units=25,
                return_sequences=True,
                input_shape=(x_train.shape[1], x_train.shape[2]),
            )
        )
        model.add(keras.layers.LSTM(units=num_neurons))
        model.add(keras.layers.Dense(1))
        opt = keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss="mean_squared_error")
        history = model.fit(
            x_train,
            y_train,
            batch_size=best_batch_size,
            epochs=best_epoch,
            verbose=0,
            validation_data=(x_test, y_test),
        )
        # record loss of final epoch
        train_loss.append(history.history["loss"][-1])
        test_loss.append(history.history["val_loss"][-1])

        end = time.time()
        times.append(end - start)

    # plot train and test loss at each batch size
    plt.plot(num_neurons, train_loss, label="train_loss", color="blue", linewidth=2)
    plt.plot(num_neurons, test_loss, label="test_loss", color="red", linewidth=2)
    plt.xlabel("Batch Size")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # plot time taken to train each neuron
    plt.plot(num_neurons, times)
    plt.xlabel("Batch Size")
    plt.ylabel("Time (s)")
    plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # pre_processing.__init__(directory=ROOT)
    # pearson_correlation()
    # variance()
    # summary()
    # training_for_MLP()

    # max_regressor()
    # max_iterate_regressor()
    # max_lstm_model()
    # max_iterate_lstm()
    # max_iterate_lstm2()
