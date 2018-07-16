# Karf
 Karf is a simple Machine Learning Library implementing a variety of algorithms 

## Getting Started

Below I will show you some exemples demonstrating how to use Karf

### Prerequisites

First, you will need to install
```
python 2.7
```

next, install all the packages mentioned in requirements.txt file
```
pip install  -r requirements.txt
```

## Linear Regression
```python
import matplotlib.pyplot as plt

from Karf.LinearModel import LinearRegression
from Karf.datasets import load
from Karf.preprocessing import polynomial_variables


# load data
data = load("simple_function")

# split data into training set and testing set
train = data.iloc[:-12].sort_values(by="X")
test = data.iloc[-12:].sort_values(by="X")

# split into X and y and generate polynomial variables
Xtrain = train.drop("y", axis=1).values
Xtrain = polynomial_variables(Xtrain, poly_degree=2)
ytrain = train["y"].values

Xtest = test.drop("y", axis=1).values
ytest = test["y"].values
Xtest = polynomial_variables(Xtest, poly_degree=2)

# initialize Linear Regression model
model = LinearRegression(regularisation=0, max_iterations=1e5, alpha=1e-6)

# fitting data
model.fit(Xtrain, ytrain)

# print the score
print ("Score: ", model.score(Xtest, ytest))

# save predictions from testing set
prediction = model.predict(Xtest)

# plot the results
plt.figure(1)
plt.subplot(122)

plt.plot(Xtrain[:, 0], ytrain, label="train")
plt.plot(Xtrain[:, 0], model.predict(Xtrain), label="predicted")
plt.legend()
plt.xlabel("X")

plt.subplot(121)
plt.plot(Xtest[:, 0], ytest, label="test")
plt.plot(Xtest[:, 0], prediction, label="predicted")

plt.ylabel("y")
plt.legend()
plt.show()
```

## Logistic Regression
```python

import matplotlib.pyplot as plt
import numpy as np
from Karf.LinearModel import LogisticRegression
from Karf.datasets import load
from Karf.preprocessing import Scaler

# load Data
df = load("Iris")

# get Species Types from dataset
types = df["Species"].drop_duplicates().values

# exclude one type in order to work only with two
data = df[df.Species != types[-1]]

# shuffle data
data = data.iloc[np.random.permutation(len(data))]

# map string to numbers for encoding purposes
code_to_name = {k: types[k] for k in range(len(types))}
name_to_code = {types[k]: k for k in range(len(types))}

# split data to X and y and code Species names to numbers
X = data.drop(["Id", "Species"], axis=1).astype(np.float)
y = data["Species"].apply(lambda x: name_to_code[x])

# split data to training sets and testing sets
train_split = int(len(X) * 0.75)

Xtrain = X[:train_split].values
ytrain = y[:train_split].values
Xtest = X[train_split:].values
ytest = y[train_split:].values

# normalize data
scaler = Scaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

# initialize Logistic Regression model
model = LogisticRegression(alpha=0.001, max_iterations=2e5)

# fitting data
model.fit(Xtrain, ytrain)

# get the score
print ("Score: ", model.score(Xtest, ytest))

# plot learning Curves
plt.xlabel("iteration")
plt.ylabel("Jtrain")
plt.plot(model.Jtrain, label="error")
plt.legend()

```

### Basic Exemples

you can also find all the previous exemples inside the exemples folder
```
./exemples
```

## TODO NEXT

* Implement the Gaussian Kernel for SVM
* Improve the SMO algorithm used to fit the SVM
* Add more Algorithms
