#%%
import keras
import pandas as pd
import numpy as np


#%%
q = pd.read_csv('q_consistida.csv', parse_dates=['data']).set_index('data')


#%% Normaliza
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(q.to_numpy())


#%% Cria o dataset
look_back = 365
forecast_steps = 1
dataX, dataY = [], []
for i in range(len(dataset)-look_back):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
dataX, dataY = np.array(dataX), np.array(dataY)
# Input features are generally represented as 3D arrays
# [batch size, time steps, features]
dataX = dataX.reshape(dataX.shape[0], dataX.shape[1], 1)
dataY =  np.array(dataY).reshape(-1,1)


#%% Separacao em tres sets
X_train, y_train = dataX[:2499], dataY[:2499]
X_valid, y_valid = dataX[2499:3213], dataY[2499:3213]
X_test, y_test = dataX[3213:3570], dataY[3213:3570]


#%% Modelo 0 - Naive
y_pred0 = X_valid[:,-1]
mse0 = np.mean(keras.losses.mean_squared_error(y_valid, y_pred0))
q_pred0 = scaler.inverse_transform(y_pred0)


#%% Modelo 1 - ANN ridicula
model1 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[365,1]),
    keras.layers.Dense(1)
])
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.fit(X_train, y_train, epochs=50, batch_size=20, verbose=2)
y_pred1 = model1.predict(X_valid)
mse1 = np.mean(keras.losses.mean_squared_error(y_valid, y_pred1))
q_pred1 = scaler.inverse_transform(y_pred1)


#%% Modelo 2 - Simple RNN
model2 = keras.models.Sequential([
    keras.layers.SimpleRNN(1, return_sequences=True, input_shape=[None, 1])
])
model2.compile(loss='mean_squared_error', optimizer='adam')
model2.fit(x=X_train, y=y_train, batch_size=100, epochs=20, validation_data=(X_valid, y_valid), verbose=2)
y_pred2 = model2.predict(X_valid)[:,-1]
mse2 = np.mean(keras.losses.mean_squared_error(y_valid, y_pred2))
q_pred2 = scaler.inverse_transform(y_pred2)


#%% Modelo 3 - Deep RNN
model3 = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.Dense(1)
])
model3.compile(loss='mean_squared_error', optimizer='adam')
model3.fit(X_train, y_train, epochs=50, batch_size=20, verbose=2)
y_pred3 = model3.predict(X_valid)
mse3 = np.mean(keras.losses.mean_squared_error(y_valid, y_pred3))
q_pred3 = scaler.inverse_transform(y_pred3)


#%% Plota tudo
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt 
with PdfPages('figs.pdf') as pdf:
    for i, x_valid in enumerate(X_valid):
        print(i)
        plt.figure(figsize=(14,6))
        # Features
        plt.plot(range(1,366), scaler.inverse_transform(x_valid), color='black')
        # Target
        plt.scatter(366, scaler.inverse_transform(y_valid[[i]]), color='red', marker='.')
        # Predicoes
        # ?plt.scatter(366, q_pred0[i], color='red', marker='*')
        # plt.scatter(366, q_pred1[i], color='orange', marker='*')
        plt.scatter(366, q_pred2[i], color='blue', marker='*')
        pdf.savefig()
        plt.close()
# %%
