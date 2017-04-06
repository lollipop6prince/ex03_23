# -*- coding: utf-8 -*-
import numpy as np
from sympy import *
import matplotlib.pyplot as plt

fig = plt.gcf()
fig.set_size_inches(8,5)

var('x')
f = lambda x: exp(-x**2/2)

x = np.linspace(-4,4,100)
y = np.array([f(v) for v in x],dtype='float')

plt.grid(True)
plt.title('Gaussian Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x,y,color='gray')
plt.fill_between(x,y,0,color='#c0f0c0')
plt.show()
