# Multiple-Linear-Regresstion--Start-Up-dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('50_Startups.csv')
dataset
dataset.isna().sum()
dataset.info()
### 0.4 Split into X & y
X = dataset.drop('Profit', axis=1)
X
y = dataset['Profit']
y
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_feature = ["State"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                  one_hot,
                                  categorical_feature)],
                                 remainder="passthrough")

transformed_X = transformer.fit_transform(X)
pd.DataFrame(transformed_X).head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size = 0.25, random_state = 2509)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_test,y_test)
y_pred = regressor.predict(X_test)
d = {'y_pred': y_pred, 'y_test': y_test}
d = {'y_pred': y_pred, 'y_test': y_test}
