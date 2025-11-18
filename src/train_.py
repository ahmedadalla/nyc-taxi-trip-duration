import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import pickle

numeric_features = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','direction']
min_max=['distance_km','pickup_center_dist','dropoff_center_dist']

categorical_features = ['passenger_count','vendor_id','dayofweek', 'month', 'hour','is_weekend','is_rush_hour','dayofyear','work_day'] #

train_features = categorical_features + numeric_features + min_max
train_features.append('dayofyear_cos')


def train_model(train):
    cloumn_transformer=ColumnTransformer([
        ('scaling', StandardScaler(), numeric_features),
        ('poly',PolynomialFeatures(degree=2),min_max),
        ('minmax',MinMaxScaler(),min_max),
        ('encoding',OneHotEncoder(handle_unknown="ignore"),categorical_features)
        ],
        remainder = 'passthrough'
    )

    model=Pipeline(steps=[
        ('preparation',cloumn_transformer),
        ('model',Ridge(alpha=1))
    ])

    model.fit(train[train_features], train.log_trip_duration)

    root_dir = r'C:\Users\ahmed\Downloads\1 project-nyc-taxi-trip-duration\1 project-nyc-taxi-trip-duration'
    with open(os.path.join(root_dir, 'Models/model.pkl'), "wb") as file:
        pickle.dump(model, file)