import netCDF4
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import sys
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import geopandas
import math
import datetime
from scipy.fft import fft, fftfreq, fftshift

def get_fft(N, T, signal):
    N = int(N)
    signal[np.isnan(signal)] = 0.0
    x = np.linspace(0.0, N*T, int(N))
    y = np.abs(fft(signal).real)
    yf = 2/N * np.abs(y[0:N//2])
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    # signal[signal==0] = np.nan

    return xf, yf