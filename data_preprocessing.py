import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

df = pd.read_csv('data/airquality.csv', dtype={'station_id': str})
station_ids = df['station_id'].unique()

location_df = pd.read_csv('data/station.csv', dtype={'station_id': str})
location_df = location_df.drop(columns=['name_chinese', 'name_english', 'district_id'])
location_ids = location_df['station_id'].unique()

df = pd.merge(df, location_df, how='left', on='station_id')
concentration_cols = ['PM25_Concentration', 'PM10_Concentration', 'NO2_Concentration', 
                     'CO_Concentration', 'O3_Concentration', 'SO2_Concentration']

mean_vals = df[concentration_cols].mean()
std_vals = df[concentration_cols].std()

for col in concentration_cols:
    df[col] = (df[col] - mean_vals[col]) / std_vals[col]

print(df)

longest = 0
longest_subdf = None
for station_id, subdf in df.groupby('station_id'):
    subdf = subdf.drop(columns=['station_id', 'longitude', 'latitude'])
    subdf = subdf.set_index('time')
    subdf.sort_index(inplace=True)
    dates = subdf.index.str[:-9].unique()
    if len(dates) > longest:
        longest = len(dates)
        longest_station = station_id
        longest_subdf = subdf

dates = longest_subdf.index.str[:-9].unique()
date_to_int = {date: i for i, date in enumerate(dates)}
int_to_date = {i: date for date, i in date_to_int.items()}

hour_to_int = {f'{i:02d}:00:00': i for i in range(24)}
int_to_hour = {i: hour for hour, i in hour_to_int.items()}

station_id_to_int = {sid: i for i, sid in enumerate(station_ids)}
int_to_station_id = {i: sid for sid, i in station_id_to_int.items()}

data = np.full((len(station_ids), 365, 24, 6), -10, dtype=np.float32)
for station_id, subdf in df.groupby('station_id'):
    subdf = subdf.drop(columns=['station_id', 'longitude', 'latitude'])
    subdf = subdf.set_index('time')
    subdf.sort_index(inplace=True)
    for time in subdf.index:
        date = time[:-9]
        hour = time[-8:]
        if date in date_to_int and hour in hour_to_int:
            date_idx = date_to_int[date]
            hour_idx = hour_to_int[hour]
            station_idx = station_id_to_int[station_id]
            data[station_idx, date_idx, hour_idx, 0] = subdf.at[time, 'PM25_Concentration']
            data[station_idx, date_idx, hour_idx, 1] = subdf.at[time, 'PM10_Concentration']
            data[station_idx, date_idx, hour_idx, 2] = subdf.at[time, 'NO2_Concentration']
            data[station_idx, date_idx, hour_idx, 3] = subdf.at[time, 'CO_Concentration']
            data[station_idx, date_idx, hour_idx, 4] = subdf.at[time, 'O3_Concentration']
            data[station_idx, date_idx, hour_idx, 5] = subdf.at[time, 'SO2_Concentration']
data[np.isnan(data)] = -10

print(f"data.shape: {data.shape}")

data_flat = data.reshape(len(station_ids), 365*24, 6)
data_windowed = sliding_window_view(data_flat, window_shape=24, axis=1)
data_windowed = data_windowed.transpose(1, 0, 3, 2).copy()

invalid_samples = [6612, 6613, 6614, 6615]
data_windowed = np.delete(data_windowed, invalid_samples, axis=0)

original_data_windowed = data_windowed.copy()

print(f"data_windowed.shape: {data_windowed.shape}")

# Count and identify time series where all values are -10
null_time_series_count = np.sum(np.all(original_data_windowed == -10, axis=(2)))
print(f'Number of time series with all values -10: {null_time_series_count}')

# Get indices of null time series
null_time_series = np.array(np.where(np.all(original_data_windowed == -10, axis=(2)))).T
print(f'Shape of null_time_series: {null_time_series.shape}')

null_samples = set()
for pair in null_time_series:
    null_samples.add((pair[0], pair[1]))

null_samples = np.array(list(null_samples))
print(f"null_samples.shape: {null_samples.shape}")

original_data_windowed = original_data_windowed.transpose(0,1,3,2)
original_filtered = original_data_windowed[~np.all(original_data_windowed == -10, axis=(3)), :]
original_data_windowed = original_data_windowed.transpose(0,1,3,2)

print(f'Shape of filtered data: {original_filtered.shape}')
print(f'Number of -10 values in data_windowed: {np.sum(original_data_windowed == -10)}')
print(f'Number of -10 values in filtered data: {np.sum(original_filtered == -10)}')

for cycle_idx in range(data_windowed.shape[0]):
    for station_idx in range(data_windowed.shape[1]):
        sample = data_windowed[cycle_idx, station_idx, :, :]
        sample = sample.transpose(1, 0)
        for channel in range(sample.shape[0]):
            arr = sample[channel, :].copy()
            
            if np.all(arr != -10) or np.all(arr == -10):
                continue
            left = 0
            right = 0
            while right < len(arr):
                while left < len(arr) and arr[left] != -10:
                    left += 1
                right = left
                while right < len(arr) and arr[right] == -10:
                    right += 1


                if left == 0:
                    arr[left:right] = arr[right]
                elif right == len(arr):
                    arr[left:right] = arr[left - 1]
                else:
                    arr[left:right] = (arr[left - 1] + arr[right]) / 2

            data_windowed[cycle_idx, station_idx, :, channel] = arr

print(f"Number of -10 values in data_windowed after filling: {np.sum(data_windowed == -10)}")
data_windowed = data_windowed.transpose(0, 1, 3, 2)
filtered = data_windowed[~np.all(original_data_windowed == -10, axis=(2)), :]
data_windowed = data_windowed.transpose(0, 1, 3, 2)

print(f'Shape of filtered data: {filtered.shape}')
print(f'Number of -10 values in data_windowed: {np.sum(data_windowed == -10)}')
print(f'Number of -10 values in filtered data: {np.sum(filtered == -10)}')

np.save('data/data_windowed.npy', data_windowed)

np.save('data/null_stations.npy', null_samples)

static = np.zeros((len(station_ids), 2), dtype=np.float32)
for i in range(len(station_ids)):
    station_id = int_to_station_id[i]
    longitude = location_df.loc[location_df['station_id'] == station_id, 'longitude']
    latitude = location_df.loc[location_df['station_id'] == station_id, 'latitude']

    static[i, 0] = longitude.values[0]
    static[i, 1] = latitude.values[0]

print(f"static.shape: {static.shape}")

np.save('data/static.npy', static)