## Midterm
### Importing Libraries 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
```
### Importing Dataset
```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame
X = np.array(data.data)
y = np.array(data.target)
```
### Define KFold
```python
def DoKFold(model, X, y, k, standardize=False, random_state=146):
    if standardize:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
   
    train_scores = []
    test_scores = []

    train_mse = []
    test_mse = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain,:]
        Xtest = X[idxTest,:]
        ytrain = y[idxTrain]
        ytest = y[idxTest]

        if standardize:
            Xtrain = ss.fit_transform(Xtrain)
            Xtest = ss.transform(Xtest)

        model.fit(Xtrain, ytrain)

        train_scores.append(model.score(Xtrain, ytrain))
        test_scores.append(model.score(Xtest, ytest))

        ytrain_pred = model.predict(Xtrain)
        ytest_pred = model.predict(Xtest)

        train_mse.append(np.mean((ytrain - ytrain_pred)**2))
        test_mse.append(np.mean((ytest - ytest_pred)**2))
        
    return train_scores, test_scores, train_mse, test_mse
```
### Question 15

### Question 16

### Question 17

### Question 18

### Question 19

### Question 20

### Question 21

### Question 22

### Question 23

### Question 24
#### Revised code
```python
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import Lasso
def DoKFold(model, X, y, k, standardize=False, random_state=146):
    from sklearn.model_selection import KFold
    
    if standardize:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    mse_train = []
    mse_test = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain, :]
        Xtest = X[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]
        
        if standardize:
            Xtrain = ss.fit_transform(Xtrain) 
            Xtest = ss.transform(Xtest)
            
        model.fit(Xtrain, ytrain)

        ytrain_predict = model.predict(Xtrain)
        ytest_predict = model.predict(Xtest)
        
        mse_train.append(mean_squared_error(ytrain,ytrain_predict))
        mse_test.append(mean_squared_error(ytest,ytest_predict))
        
    return mse_train, mse_test

a_range = np.linspace(0.001, 0.003, 101)

k = 20

avg_mse_train = []
avg_mse_test = []

for a in a_range:
    las_reg = Lasso(alpha=a)
    mse_train, mse_test = DoKFold(las_reg,X,y,k,standardize=True)
    avg_mse_test.append(np.mean(mse_test))
    
idx = np.argmin(avg_mse_test)
print('Optimal alpha value: ' + format(a_range[idx], '.5f'))
```
### Reflection
In my original code, apart from certain typo error, I also didn't standardize the data. These two mistakes end up giving me a 0.00100 optimal alpha value. I immediately found that 0.00100 is a weird result, but I just didn't realized that I missed the standardization process. After checking my code, what I found was only a typo error (in the last for loop, I mistakenly typed `las_reg` into `rid_reg`), but this didn't change the result. However, if the standardization process is added but the typo error is not removed, the optimal alpha value would be 0.00244, so the typo error is also insignificant. When typing in all the codes, I wrote the lines of code one after another, so it is quite difficult to locate the mistake after running all the codes. Maybe breaking these codes into small parts and run them little by little can make checking errors easier. I also need to review the questions carefully because I should have check the requirements of Question 20, this could have reminded me of the standardization problem.   
