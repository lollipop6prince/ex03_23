# -*- coding: utf-8 -*-
#import需要的套件
import numpy as np
from sympy import *
import matplotlib.pyplot as plt

fig = plt.gcf()          #獲得當前圖表的instance
fig.set_size_inches(8,5) #設定圖表的尺寸

# 將 'x' 設為 sympy 可處理的變數
var('x')
# 將 x 正規化成lambda函數 f
f = lambda x: exp(-x**2/2)

# 產生100個介於-4 ~ 4之間,固定間距的連續數
x = np.linspace(-4,4,100)
# 依序把 x內的值 丟到 f內 將回傳的值丟回 y
y = np.array([f(v) for v in x],dtype='float')

plt.grid(True)      #顯示圖表背景格線
plt.title('Gaussian Curve') #圖表標題
plt.xlabel('X')     #X軸標籤名
plt.ylabel('Y')     #Y軸標籤名
plt.plot(x,y,color='gray')  #依 x,y 畫出灰色的線
plt.fill_between(x,y,0,color='#c0f0c0') #依 x,y 用綠色填滿到 x軸 之間的空間
plt.show()

print ("------------------------------------------------------------------")

# -*- coding: utf-8 -*-
#import需要的套件
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# subplots()會回傳兩個物件　依序是fig,ax
# fig為圖表物件
# ax為圖表內的每張圖
fig, ax = plt.subplots(1, 1)

# linspace(A,B,size)
# 會回傳 size 個，從 A 到 B 間距相同的連續數值
x = np.linspace(norm.ppf(0.01),norm.ppf(0.99), 100)

#這是紅色那條半透明的曲線
ax.plot(x, norm.pdf(x),'r-', lw=5,alpha=0.6,label='norm pdf')

#這是黑色那條不透明曲線
ax.plot(x, norm.pdf(x), 'k-', lw=2, label='frozen pdf')

#隨機產生1000個樣本數
r = norm.rvs(size=1000)

#畫半透明的直方圖
ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)

#把有標籤的曲線 統整到圖例中 並列出來
ax.legend(loc='best', frameon=False)

#把圖表show出來
plt.show()
