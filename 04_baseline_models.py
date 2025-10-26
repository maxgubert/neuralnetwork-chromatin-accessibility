import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor


#Load data
def loadData(tsv_fn):
    data = pd.read_csv(tsv_fn, sep = "\t")
    seq = data.iloc[:,4:808].values
    values = data.iloc[:,808:809].values
    values_arcsinh = np.arcsinh(values) # Arcsinh transformation
    return seq, values_arcsinh


# Splits
X_train, y_train = loadData("~/nnPib2021/group_MM/Splits/C02M02/file_train_u.tsv")
X_test, y_test = loadData("~/nnPib2021/group_MM/Splits/C02M02/file_test_u.tsv")


# Linear Regression model with Mean
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print("Dummy Regression with Mean MSE:", mean_squared_error(y_test, dummy_pred))


# Linear Regression model with Median
dummy_md = DummyRegressor(strategy="median")
dummy_md.fit(X_train, y_train)
dummy_md_pred = dummy_md.predict(X_test)
print("Dummy Regression with Median MSE:", mean_squared_error(y_test, dummy_md_pred))


# Linear Regression
linreg = LinearRegression().fit(X_train, y_train)
linreg_pred = linreg.predict(X_test)
print('Linear Regression MSE:', mean_squared_error(y_test, linreg_pred))


# Lasso Regression (L1)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler_test = StandardScaler().fit(X_test)
X_test_scaled = scaler_test.transform(X_test)

lassocv = LassoCV(alphas=np.arange(0, 1, 0.05), cv=7).fit(X_train_scaled,y_train)
lasso_pred = lassocv.predict(X_test_scaled)
mse = mean_squared_error(y_test, lasso_pred)
print("Lasso Regression with Alpha: {0:.2f}, MSE: {1:.2f}".format(lassocv.alpha_, mse))


# Ridge Regression (L2)
ridgecv = RidgeCV(alphas=np.arange(0, 1, 0.05), cv=7).fit(X_train_scaled,y_train)
ridge_pred = ridgecv.predict(X_test_scaled)
mse = mean_squared_error(y_test, ridge_pred)
print("Ridge Regression with Alpha: {0:.2f}, MSE: {1:.2f}".format(ridgecv.alpha_, mse))


# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train_scaled,y_train)
rf_pred = rf.predict(X_test_scaled)
print("Random Forest MSE:", mean_squared_error(y_test, rf_pred))




