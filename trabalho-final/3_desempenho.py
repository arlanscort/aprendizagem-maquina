#%%
import pandas as pd
import plotly.graph_objects as go 
import numpy as np 

#%%
dados = pd.read_csv('dados.csv', parse_dates=['data']).set_index('data')

#%% Figura 1 - Desempenho
fig = go.Figure()
fig.add_trace(go.Scatter(x=dados.index, y=dados['qobs_imp'], name='Observado'))
fig.add_trace(go.Scatter(x=dados.index, y=dados['qin'], name='Qin'))
fig.add_trace(go.Scatter(x=dados.index, y=dados['qsim'], name='Qsim SAC-SMA'))
fig.write_html('figuras/desempenho.html')

# %%
# Treina 2014, 2015, 2016, 2017, 2018
# Valida 2019, 2020, 2021, 2022

#%% NSE
qsim = dados.loc['2019-01-01':, 'qsim'].to_numpy()
qobs = dados.loc['2019':, 'qobs_imp'].to_numpy()
log_qsim = np.log(qsim)
log_qobs = np.log(qobs)
NSE = 1 - np.sum((qsim-qobs)**2)/np.sum((qsim-np.mean(qobs))**2)
logNSE = 1 - np.sum((log_qsim-log_qobs)**2)/np.sum((log_qsim-np.mean(log_qobs))**2)
PBIAS = np.sum(qsim-qobs)/np.sum(qobs)*100
r = np.corrcoef(qsim, qobs)[0,1] 

# %%
qsim = pd.read_csv('Qsim_lstm_only.csv')['0'].to_numpy()
qobs = dados.loc['2019':, 'qobs_imp'].to_numpy()
log_qsim = np.log(qsim)
log_qobs = np.log(qobs)
NSE = 1 - np.sum((qsim-qobs)**2)/np.sum((qsim-np.mean(qobs))**2)
logNSE = 1 - np.sum((log_qsim-log_qobs)**2)/np.sum((log_qsim-np.mean(log_qobs))**2)
PBIAS = np.sum(qsim-qobs)/np.sum(qobs)*100
r = np.corrcoef(qsim, qobs)[0,1] 

# %%
qsim = pd.read_csv('Qsim_lstm_sacsma.csv')['0'].to_numpy()
qobs = dados.loc['2019':, 'qobs_imp'].to_numpy()
log_qsim = np.log(qsim)
log_qobs = np.log(qobs)
NSE = 1 - np.sum((qsim-qobs)**2)/np.sum((qsim-np.mean(qobs))**2)
logNSE = 1 - np.sum((log_qsim-log_qobs)**2)/np.sum((log_qsim-np.mean(log_qobs))**2)
PBIAS = np.sum(qsim-qobs)/np.sum(qobs)*100
r = np.corrcoef(qsim, qobs)[0,1] 

# %%
