#%%
import pandas as pd

#%% Verifice se ha nans
df = pd.read_csv('dados/peqt_jaguari_buenopolis.csv', parse_dates=['data']).set_index('data')
nans = df.apply(lambda x: x.isnull().sum(), axis=0)

#%% Imputa dados
df['qobs_imp'] = df['qobs'].interpolate(method='spline', order=3)

#%% Inspeciona
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
fig.add_trace(go.Scatter(
    x=df.index, y=df['qobs'], name='Qobs'
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df['qobs_imp'], name='Qobs - Imputados'
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df['qobs'].isnull()
), row=2, col=1) 
fig.update_layout(height=800, width=1600)
fig.write_html('dados/inspecao_falhas.html')
fig.show()

#%% Compara com dados do DAEE
def serie_daee(arq):
    df = pd.read_csv(arq, skiprows=12, sep=';', na_values='---', decimal=',')
    df.set_index('Mês/Ano', inplace=True)
    df = df[[str(i) for i in range(1,32)]]
    df = df.stack().reset_index()
    df['year'] = df['Mês/Ano'].apply(lambda x: int(x.split('/')[1]))
    df['month'] = df['Mês/Ano'].apply(lambda x: int(x.split('/')[0]))
    df['day'] = df['level_1']
    idx = pd.to_datetime(df[['year', 'month', 'day']])
    df.set_index(idx, inplace=True)
    sr = df[0]
    sr = sr.asfreq('D')
    return sr
# Dados do DAEE 3D-015 - Jaguari em Guaripocaba (Mun de Braganca Paulista)
# !!!
# Dados do DAEE 3D-009 - Jaguari em Buenopolis (Mun de Morungaba)
sr = serie_daee('dados/3D-009_Vazoes_Medias_Diarias_Serie_20220528_230935.csv')

#%%
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['qobs_imp'], name='Dados'))
fig.add_trace(go.Scatter(x=sr.index, y=sr, name='DAEE'))
fig.update_layout(height=800, width=1600)
fig.write_html('dados/compara_DAEE.html')

#%% 
df['res'] = df['qobs_imp'] - df['qsim']
df[['pme', 'etp', 'qin', 'qobs', 'tmed', 'qsim', 'qobs_imp', 'res']].to_csv('dados.csv')
df.round(2).to_csv('dados.csv')

#%%
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['qobs_imp'], name='Qobs'))
fig.add_trace(go.Scatter(x=df.index, y=df['qsim'], name='Qsim'))
fig.add_trace(go.Scatter(x=df.index, y=df['qin'], name='Qin'))
fig.update_layout(height=800, width=1400)
fig.write_html('dados/verifica.html')

# %%
