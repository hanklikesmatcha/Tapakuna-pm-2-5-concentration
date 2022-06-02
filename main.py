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

import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
from sklearn.model_selection import train_test_split

ROOT = "/Users/hank/Downloads"
ROOT = "data"

def pre_processing():
    pm_25_df = pd.read_csv(f'{ROOT}/PM2,5.Hourly Aggregate (ตg_mณ) - Unverified@23-20220528121258.csv')
    pm_25_df.columns = ['timestamp', "pm_25 - " + pm_25_df.iloc[0][1], None, None, None]
    pm_25_df['timestamp'] = pd.to_datetime(pm_25_df['timestamp'], errors='coerce', exact=True,
                                           format='%Y-%m-%d %H:%M:%S')
    pm_25_df = pm_25_df.iloc[1:, :-3].dropna(axis=0)
    for index, row in pm_25_df.iterrows():
        if float(row['pm_25 - Value (µg/m³)']) > 100 or float(row['pm_25 - Value (µg/m³)']) <= 0:
            pm_25_df.drop(index, inplace=True)
    pm_25_df.set_index('timestamp')
    wind_speed_df = pd.read_csv(f'{ROOT}/Wind Speed.Hourly Aggregate (m_s) - Unverified@23-20220528121743.csv')
    wind_speed_df.columns = ['timestamp', "wind_speed - " + wind_speed_df.iloc[0][1], None, None, None]
    wind_speed_df['timestamp'] = pd.to_datetime(wind_speed_df['timestamp'], errors='coerce', exact=True,
                                                format='%Y-%m-%d %H:%M:%S')
    wind_speed_df = wind_speed_df.iloc[1:, :-3].dropna(axis=0)
    wind_speed_df.set_index('timestamp')
    for index, row in wind_speed_df.iterrows():
        if float(row['wind_speed - Value (m/s)']) <= 0:
            wind_speed_df.drop(index, inplace=True)
    wind_speed_df.set_index('timestamp')
    merged_df = pd.merge_asof(pm_25_df, wind_speed_df, on='timestamp', tolerance=pd.Timedelta(nanoseconds=1),
                              direction='nearest')
    wind_direction_df = pd.read_csv(f'{ROOT}/Wind Dir.Hourly Aggregate (°) - Unverified@23-20220528121658.csv')
    wind_direction_df.columns = ['timestamp', "wind_direction - " + wind_direction_df.iloc[0][1], None, None, None]
    wind_direction_df['timestamp'] = pd.to_datetime(wind_direction_df['timestamp'], errors='coerce', exact=True,
                                                    format='%Y-%m-%d %H:%M:%S')
    wind_direction_df = wind_direction_df.iloc[1:, :-3]
    wind_direction_df.set_index('timestamp')
    merged_df = pd.merge_asof(merged_df, wind_direction_df, on='timestamp', tolerance=pd.Timedelta(nanoseconds=1),
                              direction='nearest')
    solar_rad_df = pd.read_csv(f'{ROOT}/Solar Rad.Hourly Aggregate (W_mē) - Unverified@23-20220528121404.csv')
    solar_rad_df.columns = ['timestamp', "solar_rad - " + solar_rad_df.iloc[0][1], None, None, None]
    solar_rad_df['timestamp'] = pd.to_datetime(solar_rad_df['timestamp'], errors='coerce', exact=True,
                                               format='%Y-%m-%d %H:%M:%S')
    solar_rad_df = solar_rad_df.iloc[1:, :-3].dropna(axis=0)
    solar_rad_df.set_index('timestamp')
    for index, row in solar_rad_df.iterrows():
        if float(row['solar_rad - Value (kW/m^2)']) > 1 or float(row['solar_rad - Value (kW/m^2)']) < 0:
            solar_rad_df.drop(index, inplace=True)
    merged_df = pd.merge_asof(merged_df, solar_rad_df, on='timestamp', tolerance=pd.Timedelta(nanoseconds=1),
                              direction='nearest')
    rel_humidity_df = pd.read_csv(f'{ROOT}/Rel Humidity.Hourly Aggregate (%) - Unverified@23-20220528121602.csv')
    rel_humidity_df.columns = ['timestamp', "rel_humidity - " + rel_humidity_df.iloc[0][1], None, None, None]
    rel_humidity_df['timestamp'] = pd.to_datetime(rel_humidity_df['timestamp'], errors='coerce', exact=True,
                                                  format='%Y-%m-%d %H:%M:%S')
    rel_humidity_df = rel_humidity_df.iloc[1:, :-3].dropna(axis=0)
    for index, row in rel_humidity_df.iterrows():
        if float(row['rel_humidity - Value (%)']) > 100 or float(row['rel_humidity - Value (%)']) < 0:
            rel_humidity_df.drop(index, inplace=True)
    rel_humidity_df.set_index('timestamp')
    merged_df = pd.merge_asof(merged_df, rel_humidity_df, on='timestamp', tolerance=pd.Timedelta(nanoseconds=1),
                              direction='nearest')
    no2_df = pd.read_csv(f'{ROOT}/NO2.Hourly Aggregate (ตg_mณ) - Unverified@23-20220528120844.csv')
    no2_df.columns = ['timestamp', "no2 - " + no2_df.iloc[0][1], None, None, None]
    no2_df['timestamp'] = pd.to_datetime(no2_df['timestamp'], errors='coerce', exact=True, format='%Y-%m-%d %H:%M:%S')
    no2_df = no2_df.iloc[1:, :-3].dropna(axis=0)
    for index, row in no2_df.iterrows():
        if float(row['no2 - Value (µg/m³)']) > 100 or float(row['no2 - Value (µg/m³)']) <= 0:
            no2_df.drop(index, inplace=True)
    no2_df.set_index('timestamp')
    merged_df = pd.merge_asof(merged_df, no2_df, on='timestamp', tolerance=pd.Timedelta(nanoseconds=1),
                              direction='nearest')
    no_df = pd.read_csv(f'{ROOT}/NO.Hourly Aggregate (ตg_mณ) - Unverified@23-20220528120642.csv')
    no_df.columns = ['timestamp', "no - " + no_df.iloc[0][1], None, None, None]
    no_df['timestamp'] = pd.to_datetime(no_df['timestamp'], errors='coerce', exact=True, format='%Y-%m-%d %H:%M:%S')
    no_df = no_df.iloc[1:, :-3].dropna(axis=0)
    for index, row in no_df.iterrows():
        if float(row['no - Value (µg/m³)']) > 100 or float(row['no - Value (µg/m³)']) <= 0:
            no_df.drop(index, inplace=True)
    no_df.set_index('timestamp')
    merged_df = pd.merge_asof(merged_df, no_df, on='timestamp', tolerance=pd.Timedelta(nanoseconds=1),
                              direction='nearest')
    air_temp_df = pd.read_csv(f'{ROOT}/Air Temp.Hourly Aggregate (°C) - Unverified@23-20220528121515.csv')
    air_temp_df.columns = ['timestamp', "air_temp - " + air_temp_df.iloc[0][1], None, None, None]
    air_temp_df['timestamp'] = pd.to_datetime(air_temp_df['timestamp'], errors='coerce', exact=True,
                                              format='%Y-%m-%d %H:%M:%S')
    air_temp_df = air_temp_df.iloc[1:, :-3].dropna(axis=0)
    air_temp_df.set_index('timestamp')
    for index, row in air_temp_df.iterrows():
        if float(row['air_temp - Value (°C)']) > 35 or float(row['air_temp - Value (°C)']) < 2:
            air_temp_df.drop(index, inplace=True)
    merged_df = pd.merge_asof(merged_df, air_temp_df, on='timestamp', tolerance=pd.Timedelta(nanoseconds=1),
                              direction='nearest')
    merged_df.insert(2, "lag1", " ")
    merged_df.insert(3, "lag2", " ")
    for index, row in merged_df.iterrows():
        if index > 0:
            merged_df.iloc[0:]['lag1'][index] = merged_df.iloc[0:]['pm_25 - Value (µg/m³)'][index - 1]
        if index > 1:
            merged_df.iloc[0:]['lag2'][index] = merged_df.iloc[0:]['lag1'][index - 1]
    for index, row in merged_df.iterrows():
        if str(row['timestamp'])[-6:] != ":00:00":
            merged_df.drop(index, inplace=True)
    # merged_df.set_index('timestamp')
    merged_df.to_csv(f'{ROOT}/cleaned_dataset.csv', index=False)


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
    variance_wind_speed = df[["timestamp", "pm_25 - Value (µg/m³)", "wind_speed - Value (m/s)"]]
    sns.relplot(x="timestamp", y="pm_25 - Value (µg/m³)", hue="wind_speed - Value (m/s)", data=variance_wind_speed)
    variance_wind_direction = df[["timestamp", "pm_25 - Value (µg/m³)", "wind_direction - Value (°)"]]
    sns.relplot(x="timestamp", y="pm_25 - Value (µg/m³)", hue="wind_direction - Value (°)", data=variance_wind_direction)
    variance_no2 = df[["timestamp", "pm_25 - Value (µg/m³)", "no2 - Value (µg/m³)"]]
    sns.relplot(x="timestamp", y="pm_25 - Value (µg/m³)", hue="no2 - Value (µg/m³)",
                data=variance_no2)
    variance_no = df[["timestamp", "pm_25 - Value (µg/m³)", "no - Value (µg/m³)"]]
    sns.relplot(x="timestamp", y="pm_25 - Value (µg/m³)", hue="no - Value (µg/m³)",
                data=variance_no)
    variance_air_temp = df[["timestamp", "pm_25 - Value (µg/m³)", "air_temp - Value (°C)"]]
    sns.relplot(x="timestamp", y="pm_25 - Value (µg/m³)", hue="air_temp - Value (°C)",
                data=variance_air_temp)
    plt.show()


def summary():
    df = pd.read_csv(f"{ROOT}/cleaned_dataset.csv").dropna(axis=0)
    statistics_summary = df[
        [
            "pm_25 - Value (µg/m³)",
            "wind_speed - Value (m/s)",
            "wind_direction - Value (°)",
            "no2 - Value (µg/m³)",
            "no - Value (µg/m³)",
            "air_temp - Value (°C)",
        ]
    ]
    statistics_summary[["pm_25 - Value (µg/m³)"]].plot.kde()
    statistics_summary.sum().round(2).to_csv(f"{ROOT}/sum.csv")
    statistics_summary.mean().round(2).to_csv(f"{ROOT}/mean.csv")
    statistics_summary.median().round(2).to_csv(f"{ROOT}/median.csv")
    statistics_summary.describe().round(2).to_csv(f"{ROOT}/describe.csv")
    statistics_summary.groupby("pm_25 - Value (µg/m³)").describe().round(2).to_csv(
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
    # pre_processing()
    # pearson_correlation()
    variance()
    # summary()
    # training_for_MLP()

    # max_regressor()
    # max_iterate_regressor()
    # max_lstm_model()
    # max_iterate_lstm()
    # max_iterate_lstm2()
