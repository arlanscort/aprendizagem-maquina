#%% Imports
import pandas as pd
import numpy as np
import plotly.graph_objects as go 
from tensorflow import keras
import matplotlib.pyplot as plt 

#%% Dados
ds = pd.read_csv('dados.csv', parse_dates=['data']).set_index('data')

#%% Experimento SAC-SMA-LSTM
lista = []
for n_passos in [1, 100, 200, 365]:
    print(n_passos)
    # Configuracoes
    vars_entrada = ['pme', 'qin', 'tmed']
    var_saida = 'res'
    treino_ini = '2013-01-01'
    treino_fim = '2017-12-31'
    validacao_ini = '2019-01-01'
    validacao_fim = '2022-05-29'
    # Normalizacao
    ds_treino = ds.loc[treino_ini:treino_fim]
    medias_norm = ds_treino.mean()
    desvios_norm = ds_treino.std()
    ds_norm = (ds - medias_norm)/desvios_norm
    # Dataset de treino
    X_treino = []
    X_validacao = []
    for var in vars_entrada:
        ds_var = []
        for i in range(0, n_passos):
            ds_var.append(ds_norm[var].shift(i).rename(f'{var}(t-{i})'))
        ds_var = pd.concat(ds_var, axis=1).iloc[:,::-1]
        x_treino = ds_var.loc[treino_ini:treino_fim].values
        x_validacao = ds_var.loc[validacao_ini:validacao_fim].values
        X_treino.append(x_treino)
        X_validacao.append(x_validacao)
    X_treino = np.stack(X_treino, axis=2)
    X_validacao = np.stack(X_validacao, axis=2)
    y_treino = ds_norm.loc[treino_ini:treino_fim, var_saida].values.reshape(-1,1)
    y_validacao = ds_norm.loc[validacao_ini:validacao_fim, var_saida].values.reshape(-1,1)
    # Treinamento
    N = 64
    BS = 32
    LR = 0.001
    lstm = keras.models.Sequential([
        keras.layers.LSTM(N, input_shape=[None, X_treino.shape[2]], return_sequences=True, dropout=0.1),
        keras.layers.LSTM(N, return_sequences=False),
        keras.layers.Dense(1)
    ])
    opt = keras.optimizers.Adam(learning_rate=LR)
    lstm.compile(loss='mean_squared_error', optimizer=opt)
    historico_lstm = lstm.fit(X_treino, y_treino, epochs=50, batch_size=BS, verbose=1, shuffle=True, validation_data=[X_validacao, y_validacao])
    min_loss = np.min(historico_lstm.history['val_loss'])
    min_epoch = np.argmin(historico_lstm.history['val_loss'])
    lista.append([n_passos, min_loss, min_epoch])
    df = pd.DataFrame(data=lista, columns=['n_passos', 'min_loss', 'min_epoch'])
    df.to_excel('resultados/sac_sma_lstm.xlsx')

#%% Treinamento com hiperparametros otimos
for n_passos in [100]:
    print(n_passos)
    # Configuracoes
    vars_entrada = ['pme', 'qin', 'tmed']
    var_saida = 'res'
    treino_ini = '2013-01-01'
    treino_fim = '2017-12-31'
    validacao_ini = '2019-01-01'
    validacao_fim = '2022-05-29'
    # Normalizacao
    ds_treino = ds.loc[treino_ini:treino_fim]
    medias_norm = ds_treino.mean()
    desvios_norm = ds_treino.std()
    ds_norm = (ds - medias_norm)/desvios_norm
    # Dataset de treino
    X_treino = []
    X_validacao = []
    for var in vars_entrada:
        ds_var = []
        for i in range(0, n_passos):
            ds_var.append(ds_norm[var].shift(i).rename(f'{var}(t-{i})'))
        ds_var = pd.concat(ds_var, axis=1).iloc[:,::-1]
        x_treino = ds_var.loc[treino_ini:treino_fim].values
        x_validacao = ds_var.loc[validacao_ini:validacao_fim].values
        X_treino.append(x_treino)
        X_validacao.append(x_validacao)
    X_treino = np.stack(X_treino, axis=2)
    X_validacao = np.stack(X_validacao, axis=2)
    y_treino = ds_norm.loc[treino_ini:treino_fim, var_saida].values.reshape(-1,1)
    y_validacao = ds_norm.loc[validacao_ini:validacao_fim, var_saida].values.reshape(-1,1)
    N = 64
    BS = 32
    LR = 0.001
    lstm = keras.models.Sequential([
        keras.layers.LSTM(N, input_shape=[None, X_treino.shape[2]], return_sequences=True, dropout=0.1),
        keras.layers.LSTM(N, return_sequences=False),
        keras.layers.Dense(1)
    ])
    opt = keras.optimizers.Adam(learning_rate=LR)
    opt = keras.optimizers.Adam(learning_rate=LR)
    lstm.compile(loss='mean_squared_error', optimizer=opt)
    historico_lstm = lstm.fit(X_treino, y_treino, epochs=5, batch_size=BS, verbose=1, shuffle=True, validation_data=[X_validacao, y_validacao])

#%% Predicao
y_pred_lstm = lstm.predict(X_validacao)
res_pred = y_pred[:,0]*desvios_norm['res'] + medias_norm['res']
Q_pred = res_pred + ds.loc[validacao_ini:validacao_fim,'qsim']
Q_pred = pd.DataFrame(data=Q_pred, index=pd.date_range(validacao_ini, validacao_fim, freq='D'))
Q_pred.to_csv('resultados/q_pred_sac_sma_lstm_validacao.csv')

#%% Plot
fig1, ax1 = plt.subplots()
ax1.plot(historico_lstm.history['loss'], label='loss')
ax1.plot(historico_lstm.history['val_loss'], label='val_loss')
fig2, ax2 = plt.subplots(len(vars_entrada)+1, 1, figsize=(16,20), sharex=True)
for i, var in enumerate(vars_entrada):
    ax2[i].plot(X_validacao[:,-1,i], color='black')
    ax2[i].title.set_text(var)
ax2[-1].plot(y_validacao, color='black', label='validacao')
ax2[-1].plot(y_pred_lstm, color='red', label='pred lstm')
ax2[-1].title.set_text('predicao')


#%% Retorna
residuo = y_pred_lstm*desvios_norm['res'] + medias_norm['res']
# %%

qsim = ds.loc['2019-01-01':, 'qsim'] + residuo[:,0]
qobs = ds.loc['2019':, 'qobs_imp'].to_numpy()
log_qsim = np.log(qsim)
log_qobs = np.log(qobs)
NSE = 1 - np.sum((qsim-qobs)**2)/np.sum((qsim-np.mean(qobs))**2)
logNSE = 1 - np.sum((log_qsim-log_qobs)**2)/np.sum((log_qsim-np.mean(log_qobs))**2)
PBIAS = np.sum(qsim-qobs)/np.sum(qobs)*100
r = np.corrcoef(qsim, qobs)[0,1] 
# %%
