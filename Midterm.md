## Midterm
### Importing Libraries 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
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
X_names = data.feature_names
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
```python
df.corr()
```
We can see from the returned table that MedInc is most correlated with the target.   

### Question 16
```python
from sklearn.preprocessing import StandardScaler as SS
ss = SS()
X1 = SS().fit_transform(X)
df1 = pd.DataFrame(X, columns=X_names)
df1['MedHouseVal'] = y
df1.corr()
```
The result is the same as Question 15.  

### Question 17
```python
X = data.data[['MedInc']]
lin_reg = LinearRegression()
np.round(lin_reg.fit(X, y).score(X, y), 2)
```
Result: 0.47   

### Question 18
Run the "define DoKFold" process and run the following code:
```python
k=20
train_scores, test_scores, train_mse, test_mse = DoKFold(lin_reg,X,y,k,True)
print(np.mean(train_scores), np.mean(test_scores))
print(np.mean(train_mse), np.mean(test_mse))
```
Results:
```
0.6063019182717753 0.6019800920504694
0.524225284184374 0.5287980265178303
```

### Question 19
```python
from sklearn.linear_model import Ridge
k = 20
rid_a_range = np.linspace(20,30,101)

rid_train=[]
rid_test=[]
rid_train_mse=[]
rid_test_mse=[]

for a in rid_a_range:
    mdl = Ridge(alpha=a)
    train, test, train_mse, test_mse = DoKFold(mdl,X,y,k,standardize = True)
    
    rid_train.append(np.mean(train))
    rid_test.append(np.mean(test))
    rid_train_mse.append(np.mean(train_mse))
    rid_test_mse.append(np.mean(test_mse))
    
idx = np.argmax(rid_test)
print('Optimal alpha value: ' + format(rid_a_range[idx], '.5f'))
print('Training score for this value: ' + format(rid_train[idx],'.5f'))
print('Testing score for this value: ' + format(rid_test[idx], '.5f'))
```
Result:
```
Optimal alpha value: 25.80000
Training score for this value: 0.60627
Testing score for this value: 0.60201
```

### Question 20
Simply change from "Ridge" to "Lasso"
```python
from sklearn.linear_model import Lasso

las_a_range = np.linspace(0.001,0.003,101)

las_train=[]
las_test=[]
las_train_mse=[]
las_test_mse=[]

for a in las_a_range:
    mdl = Lasso(alpha=a)
    train,test,train_mse,test_mse = DoKFold(mdl,X,y,k,standardize = True)
    
    las_train.append(np.mean(train))
    las_test.append(np.mean(test))
    las_train_mse.append(np.mean(train_mse))
    las_test_mse.append(np.mean(test_mse))
    
idx = np.argmax(las_test)
print('Optimal alpha value: ' + format(las_a_range[idx], '.5f'))
print('Training score for this value: ' + format(las_train[idx],'.5f'))
print('Testing score for this value: ' + format(las_test[idx], '.5f'))
```
Result:
```
Optimal alpha value: 0.00186
Training score for this value: 0.60616
Testing score for this value: 0.60213
```

### Question 21
```python
print(X_names[5])
lin = LinearRegression(); rid=Ridge(alpha=25.8); las = Lasso(alpha=0.00186)
lin.fit(X1,y); rid.fit(X1,y); las.fit(X1,y)
lin.coef_[5], rid.coef_[5], las.coef_[5]
```
Result:
```
AveOccup
(-0.039326266978148866, -0.039412573728940366, -0.03761823364553458)
```

### Question 22
```python
print(X_names[0])
lin = LinearRegression(); rid=Ridge(alpha=25.8); las = Lasso(alpha=0.00186)
lin.fit(X1,y); rid.fit(X1,y); las.fit(X1,y)
lin.coef_[0], rid.coef_[0], las.coef_[0]
```
Result:
```
MedInc
(0.82961930428045, 0.8288892465528181, 0.8200140807502059)
```

### Question 23
```python
idx = np.argmin(rid_test_mse)
print(rid_a_range[idx], rid_train[idx], rid_test[idx], rid_train_mse[idx], rid_test_mse[idx])
```
Result: 26.1 is a different value from 25.8 in Question 19
```
26.1 0.6062700593574847 0.6020111660228638 0.5242677048909694 0.5287556631434892
```

### Question 24
Run codes for Question 20 and then the following codes:
```python
idx = np.argmin(las_test_mse)
print(las_a_range[idx], las_train[idx], las_test[idx], las_train_mse[idx], las_test_mse[idx])
```
Results:
```
0.00186 0.6061563795668891 0.6021329052825213 0.524419071473502 0.5286007023316681
```
### Reflection
In my original code, apart from certain typo error, I also didn't standardize the data. These two mistakes end up giving me a 0.00100 optimal alpha value. I immediately found that 0.00100 is a weird result, but I just didn't realized that I missed the standardization process. After checking my code, what I found was only a typo error (in the last for loop, I mistakenly typed `las_reg` into `rid_reg`), but this didn't change the result. However, if the standardization process is added but the typo error is not removed, the optimal alpha value would be 0.00244, so the typo error is also insignificant. When typing in all the codes, I wrote the lines of code one after another, so it is quite difficult to locate the mistake after running all the codes. Maybe breaking these codes into small parts and run them little by little can make checking errors easier. I also need to review the questions carefully because I should have check the requirements of Question 20, this could have reminded me of the standardization problem.   
I've attached the revision based on my original codes.
```python
from sklearn.metrics import mean_squared_error
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
