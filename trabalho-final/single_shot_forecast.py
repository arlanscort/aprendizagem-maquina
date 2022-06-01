''''
Single-shot models: the model makes the entire sequence prediction in a single step
'''

#%%
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#%% Leitura
q = pd.read_csv('q_consistida.csv', parse_dates=['data']).set_index('data')


#%% Normalizacao
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
q = scaler.fit_transform(q.to_numpy())


#%% Dataset
steps_back = 365
steps_forward = 10
dataset = []
for i in range(len(q)-steps_back-steps_forward+1):
    dataset.append(q[i:i+steps_back+steps_forward])
dataset = np.array(dataset)


#%% 1 - RNN Simples 1D
X_train, Y_train = dataset[:2490, :steps_back], dataset[:2490, -steps_forward:,0]
X_valid, Y_valid = dataset[2490:, :steps_back], dataset[2490:, -steps_forward:,0]
model1 = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10)
])
model1.compile(loss='mse', optimizer='adam')
model1.fit(X_train, Y_train, epochs=25, batch_size=100, verbose=2)
Y_pred1 = model1.predict(X_valid)


#%% 2 - RNN Simples 10D
X = dataset[:, :steps_back]
Y = np.empty((dataset.shape[0], steps_back, steps_forward))
for step_ahead in range(1, steps_forward+1):
    Y[:, :, step_ahead-1] = dataset[:, step_ahead:step_ahead + steps_back, 0]
X_train, Y_train = X[:2490], Y[:2490]
X_valid, Y_valid = X[2490:], Y[2490:]
def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:,-1], Y_pred[:,-1])
model2 = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
model2.compile(loss='mse', optimizer='adam', metrics=[last_time_step_mse])
model2.fit(X_train, Y_train, epochs=25, batch_size=50, verbose=2)
Y_pred2 = model2.predict(X_valid)


#%% 3 - LSTM 10D
model3 = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
model3.compile(loss='mse', optimizer='adam', metrics=[last_time_step_mse])
model3.fit(X_train, Y_train, epochs=50, batch_size=25, verbose=2)
Y_pred3 = model3.predict(X_valid)


#%% Plot
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt 
with PdfPages('single_shot_forecast.pdf') as pdf:
    for i in range(len(X_valid)):
        print(i)
        plt.figure(figsize=(14,6))
        # Obs
        idx = range(375)
        q = scaler.inverse_transform(dataset[i,:])
        plt.plot(idx[-45:], q[-45:], color='black')
        
        # Predicoes - Modelo 1
        q_pred = scaler.inverse_transform(Y_pred1[i].reshape(-1,1))
        q_pred = np.insert(q_pred, 0, q[364])
        plt.plot(idx[-11:], q_pred, color='red', label='RNN Simples 1D', marker='.')
        
        # Predicoes - Modelo 2
        q_pred = scaler.inverse_transform(Y_pred2[i,-1,:].reshape(-1,1))
        q_pred = np.insert(q_pred, 0, q[364])
        plt.plot(idx[-11:], q_pred, color='blue', label='RNN Simples 10D', marker='.')

        # Predicoes - Modelo 3
        q_pred = scaler.inverse_transform(Y_pred3[i,-1,:].reshape(-1,1))
        q_pred = np.insert(q_pred, 0, q[364])
        plt.plot(idx[-11:], q_pred, color='orange', label='LSTM', marker='.')

        pdf.savefig()
        plt.close()

# %%
