from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os

root_dir = r'C:\Users\ahmed\Downloads\1 project-nyc-taxi-trip-duration\1 project-nyc-taxi-trip-duration'
with open(os.path.join(root_dir, 'Models/model.pkl'), "rb") as file:
    model=pickle.load(file)

numeric_features = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','direction']
min_max=['distance_km','pickup_center_dist','dropoff_center_dist']
categorical_features = ['passenger_count','vendor_id','dayofweek', 'month', 'hour','is_weekend','is_rush_hour','dayofyear','work_day']

train_features = categorical_features + numeric_features + min_max
train_features.append('dayofyear_cos')

def evaluate(df,message,model=model):
    y_p=model.predict(df[train_features])
    mse=mean_squared_error(df['log_trip_duration'],y_p,squared=False)
    r2=r2_score(df['log_trip_duration'],y_p)
    with open(os.path.join(root_dir,'results.txt'),'a+') as f:
        f.writelines(f'{message} " MSE :",{round(mse,2)} ,"R2 :",{round(r2,2)}\n')
    print(message+" MSE :",{mse} ,"R2 :",r2)


