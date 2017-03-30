張瑞珊0330作業
=========================

用numpy做出常態分配(畫出常態分配的圖)。

```python
import numpy as np
from sympy import *
from IPython.display import *
init_printing(use_latex=True)
import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(8,5)
var('x')
f = lambda x: exp(-x**2/2)
display(Latex('$ \large f(x) = ' + latex(f(x)) + '$'))
x = np.linspace(-4,4,100)
y = np.array([f(v) for v in x],dtype='float')
plt.grid(True)
plt.title('Gaussian Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x,y,color='gray')
plt.fill_between(x,y,0,color='#c0f0c0')
plt.show()
```
  
利用pandas與Random配合作出亂數之散佈圖

```python
import pandas as pd
%matplotlib inline
import random
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.DataFrame()

df['x'] = random.sample(range(1, 100), 25)
df['y'] = random.sample(range(1, 100), 25)
fig=sns.lmplot('x', 'y', data=df, fit_reg=False)
fig.savefig(“output.png”)
fig.plt.show()
```  
