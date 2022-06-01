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
forecast_steps = 10
dataX, dataY = [], []
for i in range(len(dataset)-look_back-forecast_steps+1):
    dataX.append(dataset[i:(i+look_back),0])
    y = []
    for j in range(forecast_steps):
        y.append(dataset[i+j+1:i+j+1+look_back])
    dataY.append(y)
    # dataY.append(dataset[(i+look_back):(i+look_back+forecast_steps), 0])
dataX, dataY = np.array(dataX), np.array(dataY)
# # Input features are generally represented as 3D arrays
# # [batch size, time steps, features]
# dataX = dataX.reshape(dataX.shape[0], dataX.shape[1], 1)
# dataY = dataY.reshape(-1,10)


#%% Separacao em tres sets
X_train, Y_train = dataX[:2499], dataY[:2499]
X_valid, Y_valid = dataX[2499:3213], dataY[2499:3213]
X_test, Y_test = dataX[3213:3570], dataY[3213:3570]


#%% Modelo 3 - Deep RNN 10 steps forward
model3 = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10)
])
model3.compile(loss='mean_squared_error', optimizer='adam')
model3.fit(X_train, Y_train, epochs=20, batch_size=100, verbose=2)
Y_pred3 = model3.predict(X_valid)
mse3 = np.mean(keras.losses.mean_squared_error(Y_valid, Y_pred3))
# Q_pred3 = scaler.inverse_transform(Y_pred3)


#%% Modelo 3 - Deep RNN 10 steps progressiva
model3 = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10)
])
model3.compile(loss='mean_squared_error', optimizer='adam')
model3.fit(X_train, Y_train, epochs=20, batch_size=100, verbose=2)
Y_pred3 = model3.predict(X_valid)
mse3 = np.mean(keras.losses.mean_squared_error(Y_valid, Y_pred3))
# Q_pred3 = scaler.inverse_transform(Y_pred3)


#%% Modelo 4 - LSTM
lstm = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
lstm.compile(loss='mean_squared_error', optimizer='adam')
lstm.fit(X_train, Y_train, epochs=20, batch_size=100, verbose=2)
# Y_pred_lstm = lstm.predict(X_valid)
# mse4 = np.mean(keras.losses.mean_squared_error(Y_valid, Y_pred_lstm))
# Q_pred3 = scaler.inverse_transform(Y_pred3)



#%% Plota tudo
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt 
with PdfPages('previsao10.pdf') as pdf:
    for i, x_valid in enumerate(X_valid):
        print(i)
        plt.figure(figsize=(14,6))
        # Observado
        idx = range(1,376)
        passado = scaler.inverse_transform(x_valid).flatten()
        futuro = scaler.inverse_transform(Y_valid[[i]]).flatten()
        obs = np.concatenate((passado, futuro))
        plt.plot(idx, obs, color='black')
        
        # Predicoes
        # Modelo 3
        idx = range(365,376)
        ultimo_obs = np.array([obs[364]])
        pred = scaler.inverse_transform(Y_pred3[[i]]).flatten()
        pred = np.concatenate((ultimo_obs, pred))
        plt.plot(idx, pred, color='red', label='Modelo 3')
        pdf.savefig()
        plt.close()



# %%
n_steps = 50
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)
series = generate_time_series(10000, n_steps + 10)
# %%
Y = np.empty((10000, n_steps, 10))
for step_ahead in range(1, 10 + 1):
    Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]
# %%
