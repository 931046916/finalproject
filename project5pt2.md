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
The logistic regression produces a training score of 0.5524088541666666, and a testing score of 0.5427037579306979. These values are very close to those with KNN model, so this model is slightly more accurate than the KNN model. 
### 4. Random Forest
Unstandardized raw data:
100 trees: [training: 0.7903645833333334, testing: 0.49829184968277207]   
150 trees: [training: 0.7906901041666666, testing: 0.4968277208394339]    
1000 trees: [training: 0.7906901041666666, testing: 0.5026842362127867]   
5000 trees: [training: 0.7906901041666666, testing: 0.4968277208394339]   
The minimum number of split I found is 24.    
Standardized data:    
100 trees: [training: 0.7952473958333334, testing: 0.4899951195705222]    
500 trees: [training: 0.7952473958333334, testing: 0.4997559785261103]    
1000 trees: [training: 0.7952473958333334, testing: 0.49536359199609564]   
5000 trees: [training: 0.7952473958333334, testing: 0.4968277208394339]    
### 5. Repeat the previous steps after recoding the wealth classes 2 and 3 into a single outcome
