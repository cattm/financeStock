# Description - python artificial recurrant neural network LSTM to predict stock price
# using Apple as our thing for now
# LSTM is Long Short Term Memory
# https://www.youtube.com/watch?v=QIUxPv5PJOY is my source

import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Import
import matplotlib.pyplot as plt
import seaborn
import yfinance as yf

from jproperties import Properties

configs = Properties()
with open('../inputs/app-config.properties', 'rb') as config_file:
    configs.load(config_file)
print(f'The Share to process is: {configs.get("SHARE").data}')
items_view = configs.items()
db_configs_dict = {}
for item in items_view:
    db_configs_dict[item[0]] = item[1].data
print(db_configs_dict)

data_df = yf.download(db_configs_dict.get("SHARE"), db_configs_dict.get("START"), db_configs_dict.get("END"))
# print(data_df)


# print(data_df['Close'])
data_df.to_csv(db_configs_dict.get("OUTP"))
print(data_df.shape)
plt.figure(figsize=(16, 8))
plt.title("Close Price History")
plt.plot(data_df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price USD', fontsize=18)
# plt.show()

# only want the closing price
# this is our new dataframe myData - will be used later......"data"
myData = data_df.filter(['Close'])
dataset = myData.values
training_data_len = math.ceil(len(dataset) * 0.8)
print("training data length is", training_data_len)

# scale the data - dont know why - but it will now be between 0 and 1 only
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

# create TRAINING data
train_data = scaled_data[0:training_data_len, :]
# split the data
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 60:
        print(x_train)
        print(y_train)
        print('**********')

# convert the trains to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# of course - Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# and wait a while

# create the TESTING data set - a new array
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60: i, 0])

# convert the data and reshape it
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# get predicted price value for test data set
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# get the RMS error
# rmse = np.sqrt(np.mean((predictions - y_test)**2 ))
# rmse = np.sqrt(np.mean(predictions - y_test) **2)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

print("RMSE", rmse)

# Plot the data
train = myData[:training_data_len]
valid = myData[training_data_len:]
valid['Predictions'] = predictions

# Visualise
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

print(valid)
valid.to_csv(db_configs_dict.get("PREDICTED"))

# get a new quote
app_quote = yf.Ticker(db_configs_dict.get("SHARE"))
print(app_quote.fast_info)
my_quote = yf.download(db_configs_dict.get("SHARE"), db_configs_dict.get("START"), db_configs_dict.get("END"))
new_df = my_quote.filter(['Close'])
# get last 60 days
last_60_days = new_df[-60:].values
# scale it
last_60_days_scaled = scaler.transform(last_60_days)

X_test = [last_60_days_scaled]
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print("Predicted Stock Price ", pred_price)

# check PREDICTION against Actual
my_quote2 = yf.download(db_configs_dict.get("SHARE"), db_configs_dict.get("PSTART"), db_configs_dict.get("PEND"))
print("Real Stock Price ", my_quote2['Close'])

# need to retrain
# split into different timeframes
# save the model data so I dont have to rebuild it every time as it takes a while!
# what are technical indicator values and how do I use them?
# check what I am scaling the data set with - should not use a test or validation set
# https://www.infinox.co.uk/fca/en/ix-intel/what-are-trading-indicators
# https://medium.com/geekculture/top-4-python-libraries-for-technical-analysis-db4f1ea87e09
# https://python.stockindicators.dev/indicators/#content
# https://www.youtube.com/watch?v=fJ3CfEwr39k
# https://www.youtube.com/watch?v=vT0-eLOw5Uk
# https://www.youtube.com/watch?v=eynxyoKgpng
# https://www.youtube.com/watch?v=zu2q28h9Uvc&list=PLNzr8RBV5nboxi7Hg3cv-CuTF2tO70r_P
# https://www.youtube.com/watch?v=I5unWZBldus
