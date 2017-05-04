張瑞珊 Homework and Class
=========================


Homework and Class 0330
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
Homework and Class 0406
==========================

機率

```python
from __future__ import division
from collections import Counter
import math, random
from matplotlib import pyplot as plt

def random_kid():
    return random.choice(["boy", "girl"])

def uniform_pdf(x):
    return 1 if x >= 0 and x < 1 else 0

def uniform_cdf(x):
    "returns the probability that a uniform random variable is less than x"
    if x < 0:   return 0    # uniform random is never less than 0
    elif x < 1: return x    # e.g. P(X < 0.4) = 0.4
    else:       return 1    # uniform random is always less than 1

def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

def plot_normal_pdfs(plt):
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
    plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
    plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
    plt.plot(xs,[normal_pdf(x,mu=-1)   for x in xs],'-.',label='mu=-1,sigma=1')
    plt.plot(xs, [normal_pdf(x, mu=5) for x in xs], '-.', label='mu=-1,sigma=1')
    plt.plot(xs, [normal_pdf(x, mu=3) for x in xs], '-.', label='mu=-1,sigma=1')
    plt.legend()
    plt.show()      

def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2  

def plot_normal_cdfs(plt):
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
    plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
    plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
    plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
    plt.legend(loc=4) # bottom right
    plt.show()

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """find approximate inverse using binary search"""

    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    
    low_z, low_p = -10.0, 0            # normal_cdf(-10) is (very close to) 0
    hi_z,  hi_p  =  10.0, 1            # normal_cdf(10)  is (very close to) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2     # consider the midpoint
        mid_p = normal_cdf(mid_z)      # and the cdf's value there
        if mid_p < p:
            # midpoint is still too low, search above it
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            # midpoint is still too high, search below it
            hi_z, hi_p = mid_z, mid_p
        else:
            break

    return mid_z

def bernoulli_trial(p):
    return 1 if random.random() < p else 0

def binomial(p, n):
    return sum(bernoulli_trial(p) for _ in range(n))

def make_hist(p, n, num_points):
    
    data = [binomial(p, n) for _ in range(num_points)]
    
    # use a bar chart to show the actual binomial samples
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')
    
    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))

    # use a line chart to show the normal approximation
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) 
          for i in xs]
    plt.plot(xs,ys)
    plt.show()



if __name__ == "__main__":

    #
    # CONDITIONAL PROBABILITY
    #

    both_girls = 0
    older_girl = 0
    either_girl = 0

    random.seed(0)
    for _ in range(10000):
        younger = random_kid()
        older = random_kid()
        if older == "girl":
            older_girl += 1
        if older == "girl" and younger == "girl":
            both_girls += 1
        if older == "girl" or younger == "girl":
            either_girl += 1

    print "P(both | older):", both_girls / older_girl      # 0.514 ~ 1/2
    print "P(both | either): ", both_girls / either_girl   # 0.342 ~ 1/3

    print("------------------------------------------------------------------")

    def random_ball():
        return random.choice(["B", "G"])

    a1=0
    a2=0
    aboth=0
    random.seed(2)
    n=100000
    for _ in range(n):
        get1=random_ball()
        get2=random_ball()
        if get1 == "B":
            a1+=1
        if get1 == "B" and get2 =="B":
            aboth+=1
        if get2 == "B":
            a2+=1

    print "P(both):", aboth/n
    print "P(get1):", a1/ n
    print "P(get2):", a2 / n
    print "P(get1,get2):", a1*a2 /n / n
    print "P(get1|get2)=P(both)/P(get2):", (aboth/n)/(a2/n)
    print "P(get1|get2)=P(get1,get2)/P(get2)=P(get1)P(get2)/P(get2)=P(get1):", a1 / n

plot_normal_pdfs(plt)

```
  
假設檢定

```bat
from __future__ import division
from probability import normal_cdf, inverse_normal_cdf
import math, random


def normal_approximation_to_binomial(n, p):
    """finds mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma




def normal_upper_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)


def normal_lower_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)


def normal_two_sided_bounds(probability, mu=0, sigma=1):
    """returns the symmetric (about the mean) bounds
    that contain the specified probability"""
    tail_probability = (1 - probability) / 2

    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound





if __name__ == "__main__":

    p=0.99
    a=0.46
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, a)
    print "mu_0", mu_0
    print "sigma_0", sigma_0
    print "normal_two_sided_bounds("+str(p)+", mu_0, sigma_0)", normal_two_sided_bounds(p, mu_0, sigma_0)
    print

    p=0.79
    a=0.56
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, a)
    print "mu_0", mu_0
    print "sigma_0", sigma_0
    print "normal_two_sided_bounds("+str(p)+", mu_0, sigma_0)", normal_two_sided_bounds(p, mu_0, sigma_0)
    print

    p=0.69
    a=0.86
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, a)
    print "mu_0", mu_0
    print "sigma_0", sigma_0
    print "normal_two_sided_bounds("+str(p)+", mu_0, sigma_0)", normal_two_sided_bounds(p, mu_0, sigma_0)
    print
```  

報告_常態分配

```python
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


```

報告_假設檢定

```python
# coding=utf-8
from __future__ import division
import math, random


if __name__ == "__main__":
    # coding=utf-8
    #from __future__ import division
    #import math, random
    #第一題 假設檢定
    #設定mu = 97 , sigma = 10
    mu_0, sigma_0 = 97,10
    print "mu_0", mu_0
    print "sigma_0", sigma_0


    # 常態函數分布累積method
    def normal_cdf(x, mu=0, sigma=1):
        return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

    #逼近Z值method
    def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
        """find approximate inverse using binary search"""
        # if not standard, compute standard and rescale
        if mu != 0 or sigma != 1:
            return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

        low_z, low_p = -10.0, 0  # normal_cdf(-10) is (very close to) 0
        hi_z, hi_p = 10.0, 1  # normal_cdf(10)  is (very close to) 1
        while hi_z - low_z > tolerance:
            mid_z = (low_z + hi_z) / 2  # consider the midpoint
            mid_p = normal_cdf(mid_z)  # and the cdf's value there
            if mid_p < p:
                # midpoint is still too low, search above it
                low_z, low_p = mid_z, mid_p
            elif mid_p > p:
                # midpoint is still too high, search below it
                hi_z, hi_p = mid_z, mid_p
            else:
                break

        return mid_z

    #求Z值的method
    def normal_lower_bound(probability, mu=0, sigma=1):
        """returns the z for which P(Z >= z) = probability"""
        return inverse_normal_cdf(1 - probability, mu, sigma)

    #第一個問題
    # a = 15% = 0.85機率
    print "normal_lower_bound(0.95, mu_0, sigma_0)", normal_lower_bound(0.85, mu_0, sigma_0)
    print

    #第一題第二個問題
    print "normal_lower_bound(0.9, mu_0, sigma_0) = ", normal_lower_bound(0.95, mu_0, sigma_0)

    #第一題第三個問題，求顯著性
    #normal_probability_below = normal_cdf
    print "normal_probability_belowx,mu,sigma) = ", normal_cdf(81.55,98, 10)
    print ("")

    #第二題 信賴區間

    #右尾檢定
    def normal_upper_bound(probability, mu=0, sigma=1):
        """returns the z for which P(Z <= z) = probability"""
        return inverse_normal_cdf(probability, mu, sigma)

    #左尾檢定
    def normal_lower_bound(probability, mu=0, sigma=1):
        """returns the z for which P(Z >= z) = probability"""
        return inverse_normal_cdf(1 - probability, mu, sigma)

    #雙尾檢定
    def normal_two_sided_bounds(probability, mu=0, sigma=1):
        """returns the symmetric (about the mean) bounds
        that contain the specified probability"""
        tail_probability = (1 - probability) / 2

        # upper bound should have tail_probability above it
        upper_bound = normal_lower_bound(tail_probability, mu, sigma)

        # lower bound should have tail_probability below it
        lower_bound = normal_upper_bound(tail_probability, mu, sigma)

        return lower_bound, upper_bound

    print "Confidence Intervals"
    print "normal_two_sided_bounds(信賴水準,平均數,標準差) = ",normal_two_sided_bounds(0.95,4.015,0.02)

    #第三題 A/B Testing

    #計算p(期望值/平均數) sigma(標準差)
    def estimated_parameters(N, n):
        p = n / N
        sigma = math.sqrt(p * (1 - p) / N)
        return p, sigma

    #計算兩者差距
    def a_b_test_statistic(N_A, n_A, N_B, n_B):
        p_A, sigma_A = estimated_parameters(N_A, n_A)
        p_B, sigma_B = estimated_parameters(N_B, n_B)
        return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

    z = a_b_test_statistic(1500, 400, 1400, 350)
    print "PB-PA間的差距",(z)

    #計算p-value值

    normal_probability_below = normal_cdf


    # it's above the threshold if it's not below the threshold
    def normal_probability_above(lo, mu=0, sigma=1):
        return 1 - normal_cdf(lo, mu, sigma)


    def two_sided_p_value(x, mu=0, sigma=1):
        if x >= mu:
            # if x is greater than the mean, the tail is above x
            return 2 * normal_probability_above(x, mu, sigma)
        else:
            # if x is less than the mean, the tail is below x
            return 2 * normal_probability_below(x, mu, sigma)

    print "檢定兩者之間是否有差異",(two_sided_p_value(z))

```

0413 斜率相關運算

```python
from __future__ import division
from collections import Counter
from linear_algebra import distance, vector_subtract, scalar_multiply
import math, random


def sum_of_squares(v):
    """computes the sum of squared elements in v"""
    return sum(v_i ** 2 for v_i in v)


def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h


def plot_estimated_derivative():
    def square(x):
        return x * x

    def derivative(x):
        return 2 * x

    derivative_estimate = lambda x: difference_quotient(square, x, h=0.00001)

    # plot to show they're basically the same
    import matplotlib.pyplot as plt
    x = range(-10, 10)
    plt.plot(x, map(derivative, x), 'rx')  # red  x
    plt.plot(x, map(derivative_estimate, x), 'b+')  # blue +
    plt.show()  # purple *, hopefully


def partial_difference_quotient(f, v, i, h):
    # add h to just the i-th element of v
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)]


def step(v, direction, step_size):
    """move step_size in the direction from v"""
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]


def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]


def safe(f):
    """define a new function that wraps f and return it"""

    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')  # this means "infinity" in Python

    return safe_f


#
#
# minimize / maximize batch
#
#

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """use gradient descent to find theta that minimizes target function"""

    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0  # set theta to initial value
    target_fn = safe(target_fn)  # safe version of target_fn
    value = target_fn(theta)  # value we're minimizing
    no=1
    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]

        # choose the one that minimizes the error function
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)
        no+=1
        print(str(no)+","+str(next_thetas)+","+str(next_value))
        # stop if we're "converging"
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value


def negate(f):
    """return a function that for any input x returns -f(x)"""
    return lambda *args, **kwargs: -f(*args, **kwargs)


def negate_all(f):
    """the same when f returns a list of numbers"""
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]


def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0,
                          tolerance)


#
# minimize / maximize stochastic
#

def in_random_order(data):
    """generator that returns the elements of data in random order"""
    indexes = [i for i, _ in enumerate(data)]  # create a list of indexes
    random.shuffle(indexes)  # shuffle them
    for i in indexes:  # return the data in that order
        yield data[i]


def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    data = zip(x, y)
    theta = theta_0  # initial guess
    alpha = alpha_0  # initial step size
    min_theta, min_value = None, float("inf")  # the minimum so far
    iterations_with_no_improvement = 0

    # if we ever go 100 iterations with no improvement, stop
    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            # if we've found a new minimum, remember it
            # and go back to the original step size
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # otherwise we're not improving, so try shrinking the step size
            iterations_with_no_improvement += 1
            alpha *= 0.9

        # and take a gradient step for each of the data points
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))

    return min_theta


def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(negate(target_fn),
                               negate_all(gradient_fn),
                               x, y, theta_0, alpha_0)


if __name__ == "__main__":

    print "using the gradient"
    plot_estimated_derivative()

    v = [random.randint(-10, 10) for i in range(3)]
    print("start point:"+str(sum_of_squares(v))+str(v))

    tolerance = 0.0000001
    no=1
    while True:
        # print v, sum_of_squares(v)
        gradient = sum_of_squares_gradient(v)  # compute the gradient at v
        next_v = step(v, gradient, -0.01)  # take a negative gradient step
        print(str(no)+","+"next point:" + str(sum_of_squares(next_v)) + str(next_v))
        no+=1
        if distance(next_v, v) < tolerance:  # stop if we're converging
            break
        v = next_v  # continue if we're not

    print "minimum v", v
    print "minimum value", sum_of_squares(v)
    print

    print "using minimize_batch"

    v = [random.randint(-10, 10) for i in range(3)]

    v = minimize_batch(sum_of_squares, sum_of_squares_gradient, v)

    print "minimum v", v
    print "minimum value", sum_of_squares(v)


```


20170504 散佈圖

```python
xs = [random_normal() for _ in range(1000)]
ys1 = [x - 2+ random_normal() / 2 for x in xs]
ys2 = [-x + 2+ random_normal() / 2 for x in xs]


def scatter():
    plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
    plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
    plt.xlabel('xs')
    plt.ylabel('ys')
    plt.legend(loc=9)
    plt.show()

```
  
多格之散佈圖

```python
def make_scatterplot_matrix():
    # first, generate some random data

    num_points = 100

    def random_row():
        row = [None, None, None, None, None, None]
        row[0] = random_normal()
        row[1] = -5 * row[0] + random_normal()
        row[2] = row[0] + row[1] + 5 * random_normal()
        row[3] = 6 if row[2] > -2 else 0
        row[4] = 5 * row[0] + random_normal()
        row[5] = 2
        return row

    random.seed(0)
    data = [random_row()
            for _ in range(num_points)]

    # then plot it

    _, num_columns = shape(data)
    fig, ax = plt.subplots(num_columns, num_columns)

    for i in range(num_columns):
        for j in range(num_columns):

            # scatter column_j on the x-axis vs column_i on the y-axis
            if i != j:
                ax[i][j].scatter(get_column(data, j), get_column(data, i))

            # unless i == j, in which case show the series name
            else:
                ax[i][j].annotate("series " + str(i), (0.5, 0.5),
                                  xycoords='axes fraction',
                                  ha="center", va="center")

            # then hide axis labels except left and bottom charts
            if i < num_columns - 1: ax[i][j].xaxis.set_visible(False)
            if j > 0: ax[i][j].yaxis.set_visible(False)

    # fix the bottom right and top left axis labels, which are wrong because
    # their charts only have text in them
    ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    ax[0][0].set_ylim(ax[0][1].get_ylim())

    plt.show()
```  


