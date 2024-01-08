
import pandas as pd

# Benchmarking libs
import os 
import psutil
import timeit
import time
import resource


from memory_profiler import profile
from line_profiler import LineProfiler


import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.linear_model import LinearRegression


#@LineProfiler
@profile
def fit_function():
    model.fit(X_train, y_train)


df = pd.read_csv('Real estate.csv')


X = df.drop('Y house price of unit area', axis=1)
y = df['X4 number of convenience stores']


print("X=",X.shape, "\ny=",y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
model = LinearRegression()


# Time the training section
start_time = timeit.default_timer()
fit_function()
end_time = timeit.default_timer()


# Convert elapsed time from seconds to milliseconds
training_time_seconds = end_time - start_time
training_time_milliseconds = training_time_seconds * 1000




# Time the prediction section
y_pred = model.predict(X_test)

MAE= metrics.mean_absolute_error(y_test, y_pred)
MSE=metrics.mean_squared_error(y_test, y_pred)
RMSE= np.sqrt(MSE)

print(f"Training time: {training_time_milliseconds} milliseconds")
print("MAE: " ,MAE)
print("MSE: " ,MSE)
print("RMSE: " , RMSE)


