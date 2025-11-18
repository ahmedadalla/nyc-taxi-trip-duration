import pandas as pd
import numpy as np
import os

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from geopy.distance import geodesic

def geopy_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    coord1=(pickup_latitude,pickup_longitude)
    coord2=(dropoff_latitude, dropoff_longitude)
    return geodesic(coord1, coord2).miles

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def add_geographical_features(df, pickup_lat='pickup_latitude', pickup_lon='pickup_longitude',
                              dropoff_lat='dropoff_latitude', dropoff_lon='dropoff_longitude',
                              ):
    center_lat, center_lon = df[pickup_lat].median(), df[pickup_lon].median()
    df['pickup_center_dist'] = haversine(df[pickup_lat], df[pickup_lon],
                                         center_lat, center_lon)
    df['dropoff_center_dist'] = haversine(df[dropoff_lat], df[dropoff_lon],
                                          center_lat, center_lon)


def predict_eval(model, train, train_features, name):
    y_train_pred = model.predict(train[train_features])
    rmse = mean_squared_error(train.log_trip_duration, y_train_pred, squared=False)
    r2 = r2_score(train.log_trip_duration, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")


def approach1(train, test): # direct
    numeric_features = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']
    categorical_features = ['dayofweek', 'month', 'hour', 'passenger_count','vendor_id']

    minmax=['distance_km','dayofyear','pickup_center_dist','dropoff_center_dist']
    train_features = categorical_features + numeric_features + minmax
    column_transformer = ColumnTransformer([
        ('scaling', StandardScaler(), numeric_features),
        ('minmax',MinMaxScaler(),minmax)
        ]
        , remainder = 'passthrough'
    )

    pipeline = Pipeline(steps=[
        ('data', column_transformer),
        ('regression', Ridge())
    ])

    model = pipeline.fit(train[train_features], train.log_trip_duration)
    predict_eval(model, train, train_features, "train")
    predict_eval(model, test, train_features, "test")


def prepare_data(train):
    train.drop(columns=['id'], inplace=True)

    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    train['dayofweek'] = train.pickup_datetime.dt.dayofweek
    train['month'] = train.pickup_datetime.dt.month
    train['hour'] = train.pickup_datetime.dt.hour
    train['dayofyear'] = train.pickup_datetime.dt.dayofyear
    train['log_trip_duration'] = np.log1p(train.trip_duration)
    train["distance_km"] = train.apply(
        lambda row: geopy_distance(
            row["pickup_latitude"],
            row["pickup_longitude"],
            row["dropoff_latitude"],
            row["dropoff_longitude"]
        ),
        axis=1
    )
    train=add_geographical_features(train)


if __name__ == '__main__':
    root_dir = r'C:\Users\ahmed\Downloads\1 project-nyc-taxi-trip-duration\1 project-nyc-taxi-trip-duration\Data'
    train = pd.read_csv(os.path.join(root_dir, 'split_sample/train.csv'))
    test = pd.read_csv(os.path.join(root_dir, 'split_sample/val.csv'))

    prepare_data(train)
    prepare_data(test)

    approach1(train, test)
