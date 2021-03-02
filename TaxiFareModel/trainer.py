# imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
import pandas as pd
import numpy as np

from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import TimeFeaturesTransformer, DistanceTransformer
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())
        pipe_time = make_pipeline(TimeFeaturesTransformer(time_column='pickup_datetime'), StandardScaler())
        time_cols = ['pickup_datetime']
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        preproc = ColumnTransformer([('time', pipe_time, time_cols),
                                  ('distance', pipe_distance, dist_cols)]
                                  , remainder='drop' )
        self.pipeline = Pipeline(steps=[('preproc', preproc),
                            ('regressor', LinearRegression())])
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)
        return self.pipeline


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    # get data
    df = get_data(nrows=1000)
    # clean data
    df = clean_data(df, test=False)

    # set X and y
    X = df.drop(columns=['fare_amount','key'])
    y = df['fare_amount']
    # hold out
    X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    trainer = Trainer(X_train,y_train)
    # train
    trainer.set_pipeline()
    trainer.run()
    # evaluate

    print(trainer.evaluate(X_test, y_test))
