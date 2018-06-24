import json
from pprint import pprint
from pandas import datetime
from matplotlib import pyplot
import pandas as pd
import numpy as np
import time
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

###
###    AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.
###    I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
###    MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
###


def main ():
    with open('metric_data.json') as f:
        datastore = json.load(f)
        jsonstring = json.dumps(datastore)
    
    ss = datastore["albury-h"]
    series = pd.DataFrame.from_dict(ss)

    #series[['time']] = series[['time']].apply(pd.to_datetime)
    #series[['value']] = series[['value']].apply(pd.to_numeric)
 
    #index = pd.DatetimeIndex(series)
    #series[['value']] = series[['value']].apply(pd.to_numeric)
    series = pd.DataFrame(series)
    
    # set the name of columns
    series.columns =['date','Bandwidth in Mbps']

    # set index
    series.set_index('date', inplace=True)

    # erase +11.00 from time string
    series.index = series.index.str.split('+').str[0]

    # make column datetime
    series.index = pd.to_datetime(series.index)

    # downsample data by hour (take max )
    # Too many records for Hourly so weekly
    series = series.resample('D').max()
    #series = series.resample('W').mean()

    # replace nan values by linear interpolation
    series = series.interpolate(method='time')

    #autocorrelation_plot(series)
    series.plot()
    pyplot.show()

    autocorrelation_plot(series)
    pyplot.show()
    # fit model
    # First, we fit an ARIMA(5,1,0) model. 
    # This sets the lag value to 5 for autoregression, 
    # uses a difference order of 1 to make the time series stationary, 
    # and uses a moving average model of 0.
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    print(residuals.describe())

    # USE ARIMA
    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()

    for t in range(len(test)):
	    model = ARIMA(history, order=(5,1,0))
	    model_fit = model.fit(disp=0)
	    output = model_fit.forecast()
	    yhat = output[0]
	    predictions.append(yhat)
	    obs = test[t]
	    history.append(obs)
	    print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()
    
if __name__== '__main__':
    main()