# data_inicio = '2021-08-01'
# data_fim = '2021-09-17'

# # Coleta de novos dados
# postos_vazao = pd.read_csv('postos_vazao.csv', dtype={'idStation':str}).set_index('idStation') # Cuidado! idStation eh mantido em str
# df_novo = pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))
# for idStation in postos_vazao.index:
#     # idStation = '15'
#     print(f'\nConsultando idStation = {idStation}...')
#     # Coleta
#     response = aquisicao.ssd_pcj(idStation, 3, data_inicio, data_fim) # para vazao, idParameter = 3
#     df = pd.json_normalize(response.json())
#     # Reconhece tempos, seta o indice e ordena
#     df = df.set_index(pd.to_datetime(df.dateData, utc=True)).sort_index()
#     # Elimina registros duplicados
#     df = df.loc[~df.index.duplicated(keep='first')] # Com keep='first' marca todos os duplicados como True, exceto a primeira ocorrência
#     # Agrega a serie de 10 min em 1 hr
#     sr_hr = df['valueData'].resample('H', closed='right', label='left').apply(lambda x: x.mean(skipna=True) if x.count() >= 2 else np.nan) # Note que com 2 dados ou mais, computa a vazao horaria
#     df_novo = pd.concat([df_novo, sr_hr.rename(idStation)], axis=1)


# # Salva em historico_soleitura, se estiver coletando todo o historico
# # df_novo.round(3).to_csv('historico_soleitura.csv', index_label='datahora', na_rep='NA')

# # Atualizacao do historico
# df_historico = pd.read_csv('historico_soleitura.csv', parse_dates=True, index_col='data')
# df_historico = df_novo.combine_first(df_historico)
# df_historico.to_csv('historico_atualizado.csv', index_label='data', na_rep='NA')


#%%
import requests
import pandas as pd
url = 'https://ssd.baciaspcj.org.br/api/station'
response = requests.get(url)
inventario = pd.DataFrame(response.json())
inventario.to_excel('inventario.xlsx')
# return response

#%%
url = 'https://ssd.baciaspcj.org.br/api/data'
params = dict(
    idStation = 15, 
    idParameter = 3,
    DateDataInitial = '2011-01-01',
    DateDataFinal = '2022-05-20',
    Calculation = 'Snap',
    Aggregation = 'AllData'
    )
response = requests.get(url, params=params)
df = pd.DataFrame(response.json())
df.to_csv('q_jaguari_buenopolis.csv', index=None)

#%%
df = pd.read_csv('q_jaguari_buenopolis.csv', parse_dates=['dateData']).set_index('dateData')
q = df['valueData'].rename('q')
q = q.asfreq('10T')
qhr = q.resample('H', closed='right', label='right').mean()


#%%
import numpy as np
threshold = 6
qd = qhr.resample('D', closed='right', label='left').apply(lambda x: x.mean() if x.count() >= threshold else np.nan)
qd_falhas = qhr.resample('D', closed='right', label='left').apply(lambda x: x.isna().sum())


# %%
qd_interp = qd.iloc[1:-2]
qd_interp1 = qd_interp.interpolate(method='spline', order=2)
qd_interp2 = qd_interp.interpolate(method='linear')
qd_interp3 = qd_interp.interpolate(method='polynomial', order=2)


#%%
from plotly.subplots import make_subplots
import plotly.graph_objects as go 
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
fig.add_trace(
    go.Scatter(x=qhr.index, y=qhr, name='Vazao horaria'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=qd_falhas.index, y=qd_falhas, name='Falhas'),
    row=2, col=1
)
fig.add_hline(y=threshold, row=2, col=1)
fig.add_trace(
    go.Scatter(x=qd_falhas.index, y=qd_falhas, name='Falhas'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=qd_interp1.index, y=qd_interp1, name='Spline'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=qd_interp2.index, y=qd_interp2, name='Linear'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=qd_interp3.index, y=qd_interp3, name='Polinomial'),
    row=1, col=1
)
fig.write_html('q.html')


#%%
qd_interp3.index.name = 'data'
qd_interp3.round(2).to_csv('q_consistida.csv', date_format='%Y-%m-%d')