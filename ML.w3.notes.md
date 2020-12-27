Machine Learning
===

https://www.w3schools.com/python/python_ml_getting_started.asp
ML -> Analyzing data and predicting the outcome

# data types:
- 3 categories:
1. numerical -> numbers
 - discrete -> limited to integers.
 - continuous -> inifinite value
2. categorical -> cannot be measured against each other.
3. ordinal -> can be measured up against each other.

# mean median mode: -> used in ML
x -> a List (array)
1. mean -> average value.
 - `numpy.mean(x)`

2. median -> mid point value(value in middle after sorting all values).
 - if 2 nbrs in middle -> divise / 2
 - `numpy.median(x)`

3. mode -> most common value
 - `from scipy import stats -> stats.mode(x)`

# standard deviation (sigma)

- nb describes how spread out values are.(varnce = std dev * std dev)
- low std deviation -> most of nbrs -> close to mean(average) value.
- high std devia -> values -> spread out wider range.

ex:
- speed = [86,87,88,86,87,85,86]
 - std devi -> 0.9
 - most values -> within range of 0.9 from mean -> 86.4

- speed = [32,111,138,28,59,77,97]
 - std devi -> 37.85
 - values -> within range of 37.85 from mean -> 77.4

- std devia with numpy:
 - `import numpy -> numpy.std(x)`

# variance (sigma square): indicates how spread out values are.
- std devia = square root of variance
 - `import numpy -> numpy.var(x)`

# Percentiles

- number describes value that given percent of values are lower than.
 - `import numpy -> numpy.percentile(x, 75)`
  - what represents 75% percentile of x ar lower than.

ex:
- ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
- 75. percentile -> is 43 => means 75% of people are 43 or younger.
- age 90% are younger than => 61.0

# data distribution

- data sets are much bigger.
- create big data sets for testing
 - exp : array of 250 random floats between 0 and 5
 - `import numpy -> numpy.random.uniform(0.0, 5.0, 250)`

# histogram -> data represents (occurences)

- visualize data set -> draw histogram -> matplotlib
 - `matplotlib.pyplot.hist(x) -> matplotlib.pyplot.show()`
```
import numpy
import matplotlib.pyplot as plt

x = numpy.random.uniform(0.0, 5.0, 250)
plt.hist(x)
# plt.hist(x) -> hist with 50 bars
plt.show()
```

- big data distribution -> ex (numpy.random.uniform(0.0, 5.0, 100000))

# normal data distribution (guassian) -> bell curve

- array values concentrated around given value.
 - `numpy.random.normal(meanvalue, std dev value, 10000)`
  - values -> concentrated around meanvalue, and rarely further away std dev.

```
import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 100000)
plt.hist(x, 100) # plot histg with 50 bars
plt.show()
```

# scatter plot -> values in data represented by dot.

- using `matplotlib.pyplot.scatter(x,y)`
- needs 2 arrays -> with same length (x-axis, y-axis)
```
import matplotlib.pyplot as plt
x = [1,5,6]
y = [6,2,7]
plt.scatter(x,y)
plt.show()
```
# random data distributions -> to scatter plot
- values concentrated around meanvx on x-axis
- values concentrated around meanvy on y-axis

```
import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(meanvx, stdev, value)
y = np.random.normal(meanvy, stdev, value)

plt.scatter(x,y)
plt.show()
```

# linear regression (method of prediction)

- regression -> when find -> relationship betwn vrbles
- ML -> relationship -> used to predict outcume of future events.
- linear regre :
 - uses relatshp btwn data-pts
 - draw straight line through all them.
 - `from scipy import stats`
 - `stats.linregress(x, y)`
- line -> used to predict future values.

```
import matplotlib.pyplot as plt
form scipy import stats

x = [1,5,36,2]
y = [2,6,82,1]
slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
	return slope * x + intercept

# run each value of x array through funct
mymodel = list(map(myfunc, x))
plt.scatter(x,y)
plt.plot(x, mymodel)
plt.show()
```

# R-squared

- r-squared value -> measures relationship btwn values.
- r-squared -> ranges [-1,1]
 - 0 -> no relationship
 - 1 or -1 -> 100% related.
 - `from scipy import stats`
 - `slope, intercept, r, p, std_err = stats.linregress(x, y)`
 - the `r` value
```
from scipy import stats
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(r)
```
- result r=-0.75-> there's relationship -> we can use linear regression future prediction

# predict future values

- ex: predict speed of 10 years old car
- use :
```
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
	return slope * x + intercept
speed = myfunc(10) # predict speed of 10 years old car
print(speed)
print(r)
```
- r=0.013 -> very bad relationship -> staight line through data pts

# polynomial regression

- if not fit linear regression -> polynomial regress -> would be ideal.
- uses reltionship btwn vrbls x and y -> find best way draw line through data.
- `mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))` -> degree 3
- `myline = numpy.linspace(1, 22, 100)`

```
import numpy
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(1, 22, 100)
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()
```
- r-squared -> 0 : no relationship, 1 : 100% related
 - computed using : r2_score() from sklearn.metrics
```
import numpy
from sklearn.metrics import r2_score
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
print(r2_score(y, mymodel(x)))
```
- r = 0.94 -> good reltnship -> use polynomial regrssn in future predictions.

# predict future values

- ex: predict speed of car passing at 17pm
```
import numpy
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
speed = mymodel(17)
print(speed) -> gives 88.87
```

# multiple regression :

- when plynomial regrss -> dosn't fit
- predict value bases -> 2 or more vrbls
 - make prediction -> more accurate
- use `pandas` module -> `import pandas`
 - allows read csv -> return DataFrame object
  - datafile = `pandas.read_csv("file.csv")`
**predict co2 based on car's wight and volume**
  - X = datafile[['Weight','Volume']]
  - y = datafile['CO2']

`common : list of indepndt val => upperCase X, list dep val => lowercase y`

```
import pandas
from sklearn import linear_model

df = pandas.read_csv("cars.csv")
X = df[['Weight','Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)
predictedCO2 = regr.predict([[2300, 1300]])
print(predictedCO2)
```

# coefficient:

- factor -> describes relatinship with unknown vrb.(x, 2x -> 2 is a coeff)
- `print(regr.coef_)`
```
...
regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr.coef_)
```

# scale

https://www.w3schools.com/python/python_ml_scale.asp
