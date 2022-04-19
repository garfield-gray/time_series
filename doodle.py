#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:02:46 2022

@author: garfield
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


d = pd.read_csv("AP.csv")

d['#Passengers'].plot()
statio = adfuller(d['#Passengers'])

print(statio[1])

d['#Passengers_1diff'] = d['#Passengers'] - d['#Passengers'].shift(1)

d['#Passengers_1diff'].dropna().plot()
statiod = adfuller(d['#Passengers_1diff'].dropna())
print(statiod[1])



d['rolle'] = d['#Passengers'].rolling(window = 10).mean()

d['rolle'].dropna().plot()

d.rename(columns={'rolle':'rolled', '#Passengers':'Passengers'}, inplace=True)

print(d.head(7))

d['Month'] = pd.to_datetime(d.Month)
d.set_index('Month', inplace=True)

print(d.head(7))

plt.figure(figsize=(15,7))
plt.plot(d.Passengers)

acffig = plot_acf(d.Passengers, lags = 30) 

pacffig = plot_pacf(d.Passengers)


ddiff = d['#Passengers_1diff'].dropna()

plt.figure(figsize=(15,7))
plt.plot(ddiff)

acffigdiff = plot_acf(ddiff, lags = 100) 

pacffigdiff = plot_pacf(ddiff)


