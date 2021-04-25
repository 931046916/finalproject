### 1. Import
```
pns = pd.read_csv('city_persons.csv')
check_nan = pns['age'].isnull().values.any()
pns.dropna(inplace=True)

pns.reset_index(drop=True, inplace=True)

pns['age'] = pns['age'].astype(int)
pns['edu'] = pns['edu'].astype(int)

X = pns.drop(["wealthC"],axis = 1)
y = pns.wealthC
```
### 2. KNN
Before adding distance weight, I set the K_range to be (50,80) which returns an optimal value of 79 and a testing score of 0.5470961444607125. Here's the graph:
![KNN 1](https://user-images.githubusercontent.com/78099480/115983887-949f8a00-a5d6-11eb-9218-068ccb9d3604.png)
After adding distance weight, the testing score reduced to 0.4987798926305515, and the graph changes to 
![KNN distance](https://user-images.githubusercontent.com/78099480/115984011-56ef3100-a5d7-11eb-9b71-7a1dda204443.png)
### 3. Logistic Regression

