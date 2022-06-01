#%% Imports
import pandas as pd
import numpy as np
import plotly.graph_objects as go 
from tensorflow import keras
#%% Leitura
df = pd.read_csv('peq_jaguari_buenopolis.csv', parse_dates=['data']).set_index('data')
df['qjus'] = df.loc[:,'qjus'].interpolate(method='spline', order=1)
#%% Configuracoes
ds = df.copy()
n_passos = 365
vars_entrada = ['pme', 'etp', 'qmon']
var_saida = 'qjus'
treino_ini = '2012-01-01'
treino_fim = '2018-12-31'
validacao_ini = '2019-01-01'
validacao_fim = '2021-08-10'
# teste_ini = 
# teste_fim =
#%% Normalizacao
ds_treino = df.loc[treino_ini:treino_fim]
medias_norm = ds_treino.mean()
desvios_norm = ds_treino.std()
#%%ds_norm = (ds - medias_norm)/desvios_norm
X_treino = []
X_validacao = []
for var in vars_entrada:
    ds_temp = []
    for i in range(0, n_passos):
        ds_temp.append(ds_norm[var].shift(i).rename(f'{var}(t-{i})'))
    ds_temp = pd.concat(ds_temp, axis=1).iloc[:,::-1]
    x_treino = ds_temp.loc[treino_ini:treino_fim].dropna().values
    x_validacao = ds_temp.loc[validacao_ini:validacao_fim].dropna().values
    X_treino.append(x_treino)
    X_validacao.append(x_validacao)
X_treino = np.stack(X_treino, axis=2)
X_validacao = np.stack(X_validacao, axis=2)    
#     ds_var = pd.concat(ds_var, axis=1).dropna()
#     x_var = ds_var.values
#     x_var = x_var.reshape(x_var.shape[0], x_var.shape[1], 1)
#     x_var = np.flip(x_var, 1)
#     x.append(x_var)
# x = np.concatenate(x, axis=2)
y_treino = ds_norm.loc[treino_ini:treino_fim, 'qjus'].iloc[n_passos-1:].values.reshape(-1,1)
y_validacao = ds_norm.loc[validacao_ini:validacao_fim, 'qjus'].values.reshape(-1,1)
# #%% Modelo de base
# base = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[X_train.shape[1], X_train.shape[2]]),
#     keras.layers.Dense(1)
# ])
# base.compile(loss='mean_squared_error', optimizer='adam')
# historico_base = base.fit(X_train, y_train, epochs=50, batch_size=2000, verbose=2, validation_data=[X_test, y_test])
# y_pred_base = base.predict(X_test)
#%% LSTM
lstm = keras.models.Sequential([
    keras.layers.LSTM(20, input_shape=[None, X_treino.shape[2]], return_sequences=True, dropout=0.1),
    keras.layers.LSTM(20, dropout=0.1),
    keras.layers.Dense(1)
])
opt = keras.optimizers.Adam(learning_rate=0.001)
lstm.compile(loss='mean_squared_error', optimizer=opt)
historico_lstm = lstm.fit(X_treino, y_treino, epochs=50, batch_size=10, verbose=1, shuffle=True, validation_data=[X_validacao, y_validacao])
y_pred_lstm = lstm.predict(X_validacao)
#%%
import matplotlib.pyplot as plt 
plt.figure()
plt.plot(historico_lstm.history['loss'], label='loss')
plt.plot(historico_lstm.history['val_loss'], label='val_loss')
fig, ax = plt.subplots(2,1, figsize=(16,16), sharex=True)
ax[0].plot(X_validacao[:,-1,0], color='blue')
ax[1].plot(y_validacao, color='black')
ax[1].plot(y_pred_lstm, color='red')

# plt.legend() 
# fig = plt.figure(figsize=(16,8))
# plt.plot(y_test, color='black')
# # plt.plot(y_pred_base, color='green')
# plt.plot(y_pred_lstm, color='red')
# # %%




# %%

#%% Granja Garota - dados
df = pd.read_csv('64362000_qsim.csv', skiprows=4, parse_dates=['data']).set_index('data')
df = df.loc[:,['prcp(mm/dia)', 'etp(mm/dia)', 'qmon(m3/s)', 'q(m3/s)']]
df.columns = ['pme', 'etp', 'qmon', 'qjus']
df['qjus'] = df.loc[:,'qjus'].interpolate(method='spline', order=1)
n_passos = 365
X = []
# for var in ['pme', 'etp', 'qmon']:
for var in ['pme', 'etp']:
    df_var = []
    for i in range(n_passos, 0, - 1):
        if i != 0:
            df_var.append(df[var].shift(i).rename(f'{var}(t-{i})'))
        else: 
            df_var.append(df[var].rename(f'{var}(t)'))
    df_var = pd.concat(df_var, axis=1)
    X_var = df_var.dropna().values
    X_var = StandardScaler().fit_transform(X_var)
    n_amostra, _ = X_var.shape
    X_var = X_var.reshape(n_amostra, n_passos, -1)
    X.append(X_var)
X = np.concatenate(X, axis=2)
y = df['qjus'].shift(n_passos).dropna().values.reshape(-1,1)
scaler_y = StandardScaler().fit(y)
y = scaler_y.transform(y)
X_train, y_train = X[:12000], y[:12000]
X_test, y_test = X[12000:], y[12000:]
#%% Granja Garota - LSTM
lstm = keras.models.Sequential([
    keras.layers.LSTM(64, input_shape=[X_train.shape[1], X_train.shape[2]], return_sequences=True, dropout=0.1),
    keras.layers.LSTM(128, input_shape=[X_train.shape[1], X_train.shape[2]]),
    keras.layers.Dense(1)
])
opt = keras.optimizers.Adam(learning_rate=0.01)
lstm.compile(loss='mean_squared_error', optimizer=opt)
historico_lstm = lstm.fit(X_train, y_train, epochs=20, batch_size=72, verbose=1, shuffle=True, validation_data=[X_test, y_test])
y_pred_lstm = lstm.predict(X_test)
#%% Grana Garota - plot
fig = plt.figure(figsize=(16,8))
plt.plot(historico_lstm.history['loss'], label='loss')
plt.plot(historico_lstm.history['val_loss'], label='val_loss')
plt.legend() 
fig = plt.figure(figsize=(16,8))
plt.plot(y_test, color='black')
# plt.plot(y_pred_base, color='green')
plt.plot(y_pred_lstm, color='red')
# %%