import pandas as pd


# Benchmarking libs
import os 
import psutil
import timeit
import time
import resource
from memory_profiler import profile
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("heart.csv")

X  = df.drop('target', axis=1)
y = df['target']

@profile
def fit_function():
    Classifier.fit(X_train, y_train)

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.30, random_state=1) 

Classifier = LogisticRegression(solver='liblinear')

start_time = timeit.default_timer()
fit_function()
end_time = timeit.default_timer()

elapsedTimeInMs = (end_time - start_time) * 1000

y_test_cat = Classifier.predict(X_test)

Results = pd.DataFrame({'actual': y_test, 'prediction': y_test_cat})

print(Results.head(3))

print(accuracy_score(y_test, y_test_cat))


print(f"Training Time: {elapsedTimeInMs} milliseconds")
