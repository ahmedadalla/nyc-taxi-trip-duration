import pandas as pd
import numpy as np
from geopy.distance import geodesic

def geopy_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    coord1=(pickup_latitude,pickup_longitude)
    coord2=(dropoff_latitude, dropoff_longitude)
    return geodesic(coord1, coord2).kilometers
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def get_direction(lat_1, long_1, lat_2, long_2):
    """
    Calculates the angle or direction of 2 points with consideration of the roundness of earth.
    """

    AVG_EARTH_RADIUS = 6371  # in km
    long_delta_rad = np.radians(long_2 - long_1)
    lat_1, long_1, lat_2, long_2 = map(np.radians, (lat_1, long_1, lat_2, long_2))
    y = np.sin(long_delta_rad) * np.cos(lat_2)
    x = np.cos(lat_1) * np.sin(lat_2) - np.sin(lat_1) * np.cos(lat_2) * np.cos(long_delta_rad)

    return np.degrees(np.arctan2(y, x))


def add_geographical_features(df, pickup_lat='pickup_latitude', pickup_lon='pickup_longitude',
                              dropoff_lat='dropoff_latitude', dropoff_lon='dropoff_longitude',
                              ):
    center_lat, center_lon = df[pickup_lat].median(), df[pickup_lon].median()
    df['pickup_center_dist'] = haversine(df[pickup_lat], df[pickup_lon],
                                         center_lat, center_lon)
    df['dropoff_center_dist'] = haversine(df[dropoff_lat], df[dropoff_lon],
                                          center_lat, center_lon)
    return df

def prepare(df):
    df=df.drop(columns=['id','store_and_fwd_flag'])
    df['log_trip_duration'] = np.log1p(df.trip_duration)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dayofweek'] = df.pickup_datetime.dt.dayofweek
    df['month'] = df.pickup_datetime.dt.month
    df['hour'] = df.pickup_datetime.dt.hour
    df['dayofyear'] = df.pickup_datetime.dt.dayofyear
    df["distance_km"] = df.apply(
        lambda row: geopy_distance(
            row["pickup_latitude"],
            row["pickup_longitude"],
            row["dropoff_latitude"],
            row["dropoff_longitude"]
        ),
        axis=1
    )
    df["direction"] = df.apply(
        lambda row: get_direction(
            row["pickup_latitude"],
            row["pickup_longitude"],
            row["dropoff_latitude"],
            row["dropoff_longitude"]
        ),
        axis=1
    )
    df['distance_km']=np.log1p(df['distance_km'])
    df['dayofyear_cos']=np.cos(df['dayofyear'])



    df=add_geographical_features(df)
    df['pickup_center_dist']=np.log1p(df['pickup_center_dist'])
    df['dropoff_center_dist']=np.log1p(df['dropoff_center_dist'])

    df['is_weekend'] = df['dayofweek'].isin([5, 6])
    df['work_day']=~df['is_weekend']
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 10)) | ((df['hour'] >= 16) & (df['hour'] <= 19))

    return df

def remove_outliers(df):
    Q1 = df['log_trip_duration'].quantile(0.25)
    Q3 = df['log_trip_duration'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bond = 8.470511843477752
    lower_bond = 4.49601610236845
    df=df[df['log_trip_duration']>lower_bond]
    df = df[df['log_trip_duration'] < upper_bond]
    return df

def remove_unreal(df):
    df = df[df['distance_km'] > 0]
    df = df[~((df['distance_km'] < 0.2) & (df['log_trip_duration'] > 5))]
    return df