import pandas as pd


def __init__(directory: str):
    pm_25_df = pd.read_csv(f'{directory}/PM2,5.Hourly Aggregate (ตg_mณ) - Unverified@23-20220528121258.csv')
    pm_25_df.columns = ['timestamp', "pm_25 - " + pm_25_df.iloc[0][1], None, None, None]
    pm_25_df['timestamp'] = pd.to_datetime(pm_25_df['timestamp'], errors='coerce', exact=True,
                                           format='%Y-%m-%d %H:%M:%S')
    pm_25_df = pm_25_df.iloc[1:, :-3].dropna(axis=0)
    for index, row in pm_25_df.iterrows():
        if float(row['pm_25 - Value (µg/m³)']) > 100 or float(row['pm_25 - Value (µg/m³)']) <= 0:
            pm_25_df.drop(index, inplace=True)
    pm_25_df.set_index('timestamp')
    wind_speed_df = pd.read_csv(f'{directory}/Wind Speed.Hourly Aggregate (m_s) - Unverified@23-20220528121743.csv')
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
    wind_direction_df = pd.read_csv(f'{directory}/Wind Dir.Hourly Aggregate (°) - Unverified@23-20220528121658.csv')
    wind_direction_df.columns = ['timestamp', "wind_direction - " + wind_direction_df.iloc[0][1], None, None, None]
    wind_direction_df['timestamp'] = pd.to_datetime(wind_direction_df['timestamp'], errors='coerce', exact=True,
                                                    format='%Y-%m-%d %H:%M:%S')
    wind_direction_df = wind_direction_df.iloc[1:, :-3]
    wind_direction_df.set_index('timestamp')
    merged_df = pd.merge_asof(merged_df, wind_direction_df, on='timestamp', tolerance=pd.Timedelta(nanoseconds=1),
                              direction='nearest')
    solar_rad_df = pd.read_csv(f'{directory}/Solar Rad.Hourly Aggregate (W_mē) - Unverified@23-20220528121404.csv')
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
    rel_humidity_df = pd.read_csv(f'{directory}/Rel Humidity.Hourly Aggregate (%) - Unverified@23-20220528121602.csv')
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
    no2_df = pd.read_csv(f'{directory}/NO2.Hourly Aggregate (ตg_mณ) - Unverified@23-20220528120844.csv')
    no2_df.columns = ['timestamp', "no2 - " + no2_df.iloc[0][1], None, None, None]
    no2_df['timestamp'] = pd.to_datetime(no2_df['timestamp'], errors='coerce', exact=True, format='%Y-%m-%d %H:%M:%S')
    no2_df = no2_df.iloc[1:, :-3].dropna(axis=0)
    for index, row in no2_df.iterrows():
        if float(row['no2 - Value (µg/m³)']) > 100 or float(row['no2 - Value (µg/m³)']) <= 0:
            no2_df.drop(index, inplace=True)
    no2_df.set_index('timestamp')
    merged_df = pd.merge_asof(merged_df, no2_df, on='timestamp', tolerance=pd.Timedelta(nanoseconds=1),
                              direction='nearest')
    no_df = pd.read_csv(f'{directory}/NO.Hourly Aggregate (ตg_mณ) - Unverified@23-20220528120642.csv')
    no_df.columns = ['timestamp', "no - " + no_df.iloc[0][1], None, None, None]
    no_df['timestamp'] = pd.to_datetime(no_df['timestamp'], errors='coerce', exact=True, format='%Y-%m-%d %H:%M:%S')
    no_df = no_df.iloc[1:, :-3].dropna(axis=0)
    for index, row in no_df.iterrows():
        if float(row['no - Value (µg/m³)']) > 100 or float(row['no - Value (µg/m³)']) <= 0:
            no_df.drop(index, inplace=True)
    no_df.set_index('timestamp')
    merged_df = pd.merge_asof(merged_df, no_df, on='timestamp', tolerance=pd.Timedelta(nanoseconds=1),
                              direction='nearest')
    air_temp_df = pd.read_csv(f'{directory}/Air Temp.Hourly Aggregate (°C) - Unverified@23-20220528121515.csv')
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
    merged_df.to_csv(f'{directory}/cleaned_dataset.csv', index=False)
