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
The logistic regression (with max iterations set to 1000) produces a training score of 0.5524088541666666, and a testing score of 0.5427037579306979. These values are very close to those with KNN model, so this model is slightly more accurate than the KNN model. 
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
After changing class 2 and class 3 into a single outcome, the KNN model doesn't change much. The optimal k value changes to 69 and the testing score turns into 0.552464616886286. After adding distance weight, the testing score changes to 0.5065885797950219, which shows that the model improved a little. For logistic regression, the training score becomes 0.5537109375, and the testing score becomes 0.5436798438262567, which also remains roughly the same. For random forest before standardization, all the training scores remains roughly the same but all the testing scores increased a little (as presented below). For random forest after standardization, the training scores also remains roughly the same and all the testing scores increased a little.   
Before Standardization:   
100 trees:[training: 0.7906901041666666, testing: 0.5061005368472425]   
500 trees:[training: 0.7906901041666666, testing: 0.5085407515861395]   
1000 trees:[training: 0.7906901041666666, testing: 0.5095168374816984]    
5000 trees:[training: 0.7906901041666666, testing: 0.5070766227428014]    
After Standardization:    
100 trees:[training: 0.7958984375, testing: 0.49829184968277207]    
500 trees:[training: 0.7958984375, testing: 0.49389946315275746]    
1000 trees:[training: 0.7958984375, testing: 0.4958516349438751]    
5000 trees:[training: 0.7958984375, testing: 0.5031722791605662]    
For all these models, although all the values do not change very much, KNN with distance weight is the model that improves the most. The distance metrics would improve the accuracy of this algorithm and when merging wealth classes 2 and 3, there are less variance in feature, so fewer neighbors would lead to closer neighbors. 
### 6. Results
For the models before and after putting wealth classes 2 and 3 into a single outcome, logistic regression always produces the highest testing scores, so logistic regression model has the best performance which is also neither quite overfitting nor quite underfitting. The testing score of the logistic regression model for putting wealth classes 2 and 3 into a single outcome is slightly larger than the one that considers them seperately, so it performs slightly better than the latter one. Overall, all the models produce very close testing scores so their performances are also quite similar.
