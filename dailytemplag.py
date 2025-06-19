import pandas as pd
import numpy as nm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

df=pd.read_csv("dailytemp.csv")

features=["Date", "temp"]
x=df[features]
y=df["temp"]

def create_lagged_features(series, n_lags):
    df_lagged=pd.DataFrame()
    for lag in range(1, n_lags+1):
        df_lagged[f'lag_{lag}'] = series.shift(lag)
    df_lagged["target"]=series.values
    df_lagged.dropna(inplace=True)
    return df_lagged

n_lags=3
df_lagged= create_lagged_features(df["temp"], n_lags)

train_size=int(len(df_lagged)*0.8)
train, test= df_lagged.iloc[:train_size], df_lagged.iloc[train_size:]


x_train, y_train= train.drop("target", axis=1), train["target"]
x_test, y_test= test.drop("target", axis=1), test["target"]

model=RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test.values, label='Actual temp', marker='o')
plt.plot(y_test.index, y_pred, label='Predicted temp', marker='x')
plt.title('temp Prediction using Random Forest Surrogate Model')
plt.xlabel('Time Step')
plt.ylabel('temp')
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R² Score (Accuracy): {r2:.2%}")