import pandas as pd     # Importing PANDAS

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv')

df

type(df)


df.info()

df.isnull().sum()    # Searching for Missing Values

df.rename(columns = {'Sales':'y','Month':'ds'},inplace = True)
# Renaming the Month and Sales columns as 'ds' and 'y'

df.info()

#  As the columns are not in the required datatype formats
#  We have to convert  'ds' to TIMESTAMP FORMAT or DATETIME LIBRARY
#  And convert 'y' to INTEGER TYPE (int or float)

df['ds'] = pd.to_datetime(df['ds'])
df['y'] = df['y'].astype(int)

df.info()

df

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.scatter(df['ds'],df['y'])

import numpy as np
#Creating An Outliner

df['y'] = np.where(df['y']>25000, np.nan, df['y'])

df['y'] = np.where(df['y']<5600, np.nan, df['y'])

df.isnull().sum()

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.scatter(df['ds'],df['y'])

# Implementing a forecasting model using "FBProphet" for the given dataset

import prophet
model = prophet.Prophet(changepoint_prior_scale=1)
model.fit(df)

# Predicting according to "Months" (Keeping freq = 'M')
# Forecasting for 500 Days

forecast = model.make_future_dataframe(periods = 17, freq='M',include_history=True)
forecast = model.predict(forecast)

model.plot(forecast,xlabel = 'Date',ylabel = 'y', figsize=(24,10),)
plt.show()

 **Saving the model using joblib**

import joblib

joblib.dump(model,'model_jlib')

joblib.load('model_jlib')
