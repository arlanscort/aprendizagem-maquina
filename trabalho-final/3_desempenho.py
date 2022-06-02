#%%
import pandas as pd
import plotly.graph_objects as go
import numpy as np 

#%% Figura 2
import pandas as pd 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
df = pd.read_excel('resultados/lstm_only.xlsx', index_col='#')
df[['N', 'BS', 'LR']] = df[['N', 'BS', 'LR']].astype(str)
hp = ['N', 'BS', 'LR']
titulos = ['Número de Neurônios', 'Batch Size', 'Learning Rate']
fig = go.Figure()
fig = make_subplots(rows=3, cols=1, vertical_spacing = 0.08, subplot_titles=titulos)
for i, hp in enumerate(['N', 'BS', 'LR']):
    fig.add_trace(go.Box(
        x = df.loc[:,hp], y = df.loc[:,'min_loss'], showlegend=False,
    ), row=i+1, col=1)
    fig.update_yaxes(title_text='MSE', row=i+1, col=1)
fig.update_layout(
    margin=dict(l=25, r=25, t=25, b=25),
    height=800, width=600)
fig.write_image("figuras/fig1.jpeg")
fig.show()

#%% Resultados - periodo de validacao
dados = pd.read_csv('0_dataset.csv', parse_dates=['data']).set_index('data')
Qobs = dados['qobs_imp'].loc['2019-01-01':]
Qsacsma_only = dados['qsim'].loc['2019-01-01':]
Qlstm_only = pd.read_csv('resultados/q_pred_lstm_only_validacao.csv', parse_dates=['data']).set_index('data')['q']
Qsac_sma_lstm = pd.read_csv('resultados/q_pred_sac_sma_lstm_validacao.csv', parse_dates=['data']).set_index('data')['q']

#%%
nomes = ['SAC-SMA-only', 'LSTM-only', 'SAC-SMA-LSTM']
lista = []
for i, qsim in enumerate([Qsacsma_only, Qlstm_only, Qsac_sma_lstm]):
    qsim = qsim.to_numpy()
    qobs = Qobs.to_numpy()
    log_qsim = np.log(qsim)
    log_qobs = np.log(qobs)
    NSE = 1 - np.sum((qsim-qobs)**2)/np.sum((qsim-np.mean(qobs))**2)
    logNSE = 1 - np.sum((log_qsim-log_qobs)**2)/np.sum((log_qsim-np.mean(log_qobs))**2)
    PBIAS = np.sum(qsim-qobs)/np.sum(qobs)*100
    r = np.corrcoef(qsim, qobs)[0,1]
    lista.append([nomes[i], NSE, logNSE, PBIAS])
df_indices = pd.DataFrame(lista, columns=['Experimento', 'NSE', 'logNSE', 'PBIAS'])
df_indices.to_excel('resultados/indices.xlsx', index=None)

#%% Figura 4
fig = go.Figure()
fig.add_trace(go.Scatter(x=Qobs.loc['2020-10':'2021-09'].index, y=Qobs.loc['2020-10':'2021-09'], name='Observado', line=dict(color='black'))) 
fig.add_trace(go.Scatter(x=Qsacsma_only.loc['2020-10':'2021-09'].index, y=Qsacsma_only.loc['2020-10':'2021-09'], name='SAC-SMA-only', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=Qlstm_only.loc['2020-10':'2021-09'].index, y=Qlstm_only.loc['2020-10':'2021-09'], name='LSTM-only', line=dict(color='red')))
fig.add_trace(go.Scatter(x=Qsac_sma_lstm.loc['2020-10':'2021-09'].index, y=Qsac_sma_lstm.loc['2020-10':'2021-09'], name='SAC-SMA-LSTM', line=dict(color='orange')))
fig.update_layout(
    margin=dict(l=25, r=25, t=25, b=25),
    height = 600, 
    width = 1000,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)
fig.update_xaxes(title="Data")
fig.update_yaxes(title="Vazão (m3/s)")
fig.write_image("figuras/fig4.png")
