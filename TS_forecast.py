# 4 different time series models were applied to forecast Russia losses during invasion into the Ukraine
# forecasts are plotted

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from prophet import Prophet


plt.rc("figure", figsize=(20, 8))
plt.rc("font", size=15)
plt.rc("lines", linewidth=3)

df_full = pd.read_csv("russia_losses_personnel.csv")

act = df_full[['date', 'personnel']].copy()
act.rename(columns={'personnel': 'y', 'date': 'ds'}, inplace=True)
act.ds = pd.to_datetime(act.ds)
act.set_index('ds', drop=True, inplace=True)
t = 30

tm = ThetaModel(act, period=None)
res = tm.fit()

sarima = SARIMAX(act, freq='D')
res2 = sarima.fit(disp=False)

act1 = act.squeeze().astype('float64')
ses = ETSModel(act1, freq='D')
res3 = ses.fit(maxiter=10000, disp=False)

act2 = pd.DataFrame()
act2['ds'] = act.index
act2['y'] = list(act.y)

m = Prophet()
m.fit(act2)
future = m.make_future_dataframe(t)
forecast = m.predict(future)
fcst4 = forecast['yhat'].tail(t).tolist()


first_fc_day = act.index[-1] + pd.DateOffset(days=1)
fc_period = pd.date_range(start=first_fc_day, periods=t, freq='D')
fcst = list(res.forecast(t))
fcst2 = list(res2.forecast(t))
fcst3 = list(res3.forecast(t))

fc = pd.DataFrame({'Fc_theta': fcst,
                   'Fc_sarimax': fcst2,
                   'Fc_ses': fcst3,
                   'Fc_prophet': fcst4})
fc.set_index(fc_period, inplace=True)
act_fc = pd.concat([act, fc])

act_fc['y'].plot(label='Actual')

act_fc['Fc_prophet'].plot(label='Forecast(prophet)', linestyle=":")
act_fc['Fc_theta'].plot(label='Forecast(theta)', linestyle=":")
act_fc['Fc_sarimax'].plot(label='Forecast(sarimax)', linestyle=":")
act_fc['Fc_ses'].plot(label='Forecast(ses)', linestyle=":")

plt.title("Russia Losses During Invasion Into Ukraine")
plt.legend()

for x, y in act['y'].iloc[[-14, -7, -1]].iteritems():
    plt.annotate("{:,.0f}".format(y), (x, y),
                 textcoords="offset points",
                 xytext=(0, 5), ha='right')

for x, y in fc['Fc_prophet'].iloc[[6, 13, 20, 29]].iteritems():
    plt.annotate("{:,.0f}".format(y), (x, y),
                 textcoords="offset points",
                 xytext=(0, 5), ha='right')
plt.show()
