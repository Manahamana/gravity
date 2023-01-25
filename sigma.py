import math
import cmath
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate
import scipy

def myfunc(m):
    wert = m**2
    return wert

def sigma(k,n,func):
    x = 0
    for i in range (k,n+1):
        xi = func(i)
        x = x + xi
    return x






