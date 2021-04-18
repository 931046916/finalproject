### Set up
```
pns = pd.read_csv('persons.csv')
check_nan = pns['age'].isnull().values.any()
pns.dropna(inplace=True)
display(pns.dtypes)
pns['age'] = pns['age'].astype(int)
pns['edu'] = pns['edu'].astype(int)
X = pns.drop(['wealthC','wealthI'],axis = 1)
y = pns.wealthC
```
### WealthC
Linear Regression MSE: 0.44281   
Linear Regression Standardized MSE: 0.44297  
Linear Regression training R^2: 0.73583    
Linear Regression testing R^2: 0.73505  
Linear Regression standardized training R^2: 0.73581   
Linear Regression Standardized testing R^2: 0.73504  
Comparison of coefficient: the linear regression coefficient changes a lot, but the values for MSE and R^2 do not vary much before and after standardization, but the scores after standardization is slightly smaller than those before standardization.    
Ridge Regression: training score: 0.73584 ; testing score: 0.73522   
Lasso Regression: training score: 0.73583 ; testing score: 0.73506   

### WealthI
Linear Regression MSE: 1750276834.9304745   
Linear Regression Standardized MSE: 1750287416.4378276  
Linear Regression training R^2: 0.82584    
Linear Regression testing R^2: 0.82501  
Linear Regression standardized training R^2: 0.82582     
Linear Regression Standardized testing R^2: 0.82501     
Comparison of coefficient: just like the feature WealthC, the values before and after standardization is nearly the same, but the scores after standardization is slightly smaller than those before standardization.    
Ridge Regression: training score: 0.82584 ; testing score: 0.82502   
Lasso Regression: training score: 0.82583 ; testing score: 0.82502  

### Which of the models produced the best results in predicting wealth of all persons throughout the smaller West African country being described? 
All the values from the three models for target "WealthC" is higher than the values for target "WealthI", which indicates that "WealthI" fit the modles better and has higher predictability, but "WealthI" has weirdly large MSE before and after standardization. As for linear regression, ridge regression, and lasso regression for "WealthC", ridge regression produces best result, although all other values are almost the same. 
