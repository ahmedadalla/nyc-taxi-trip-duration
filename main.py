from src import data_preparation
from  src import train_
from  src import evaluate
import pandas as pd
import os

if __name__ == '__main__':
    root_dir = r'C:\Users\ahmed\Downloads\1 project-nyc-taxi-trip-duration\1 project-nyc-taxi-trip-duration\Data'
    train = pd.read_csv(os.path.join(root_dir, 'split/train.csv'))
    val = pd.read_csv(os.path.join(root_dir, 'split/val.csv'))

    #train = pd.concat([train, val], ignore_index=True)

    test = pd.read_csv(os.path.join(root_dir, 'split/test.csv'))
    #train=data_preparation.prepare(train)
    test=data_preparation.prepare(test)

    #train=data_preparation.remove_outliers(train)
    test=data_preparation.remove_outliers(test)

    #train=data_preparation.remove_unreal(train)
    test=data_preparation.remove_unreal(test)

    #train_.train_model(train)

    #evaluate.evaluate(train,"train")
    evaluate.evaluate(test,"test")




