
import numpy as np

Im = 178
tal = 30
f = 0.2
n = 2

w = 2*np.pi*f
t0 = (1/w)*np.arctan(n*tal*w)
fac = np.exp(t0/tal)/(np.sin(w*t0)**n)


def current(t):
    return Im*fac*np.exp(-t/tal)*(np.sin(w*t)**n)


print(fac)
print(Im*fac)
print(1/fac)
print(Im/fac)
