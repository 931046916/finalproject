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
Coefficient array:
```
  [ 3.01812923e-02,  1.07882853e-02, -5.57603897e-04,  8.37880684e-02,
    4.04701739e-02,  6.37198352e-02, -1.40023112e-01,  9.99896825e-02,
    1.85515805e-01, -2.49517259e-01, -2.47122665e-01, -7.30324831e-02,
    3.09612080e-01, -1.29375995e-01,  3.53607318e-01,  2.33225714e-01,
   -1.34364084e-01, -1.92558301e-01, -1.20146711e-01,  3.59279100e-02,
    1.46004504e-01, -1.81932414e-01,  1.05944573e-01,  4.00186663e-01,
    1.72822325e-01,  2.29943453e-02,  1.03043774e-01, -1.15888783e-01,
   -2.18966624e-01, -2.90949455e-01, -3.83672661e-01, -3.84737293e-01,
    3.07519898e-01,  2.55401258e-02,  2.56163113e-01,  3.95033383e-01,
    3.60442298e-01,  1.90435535e-01,  3.86891012e-01,  1.53405264e-01,
   -2.09042764e-02,  5.43122461e-02, -1.27172669e-01, -5.40268677e-01,
   -5.63637093e-01, -1.58355761e-01, -1.08923385e-01, -2.12578757e-02,
   -3.26132080e-01,  3.26132080e-01, -6.44297719e-02,  6.44297719e-02,
   -2.76390443e-01,  4.32693258e-01,  6.03439291e-02,  4.07576086e-01,
   -6.37787977e-01,  1.35651470e-02, -2.47897601e-01,  2.47897601e-01]
```
Linear Regression Standardized MSE: 0.44297  
Coefficient array:
```
  [ 1.12548658e-01,  5.24358116e-03, -1.08884589e-02,  6.92579735e-02,
    7.36951509e+10,  8.66257201e+10,  7.69209583e+10,  7.91372426e+10,
    8.45473781e+10,  7.89854838e+10,  7.88333540e+10,  8.76583681e+10,
    8.66134726e+10,  8.54267349e+10,  1.16140874e+11,  1.01070442e+11,
    7.65053798e+10,  7.51091695e+10,  8.19133567e+10,  4.80000747e+10,
    7.26531241e+10,  7.87003037e+10, -6.56609287e+09, -6.60161265e+09,
   -1.20447035e+10, -1.40140921e+10, -7.23238899e+10, -3.56557715e+10,
   -1.50138208e+10, -2.23537221e+10, -6.34833466e+10, -4.69533271e+09,
   -7.80211026e+09, -1.23673099e+10, -1.59629508e+10,  5.30991560e+10,
    2.31813714e+11,  8.95044996e+10,  4.32553262e+10,  1.67206073e+10,
    3.49539754e+11,  2.11161816e+11,  2.27988135e+11,  4.41825617e+11,
    1.81354767e+10,  2.62747629e+10,  1.66594092e+11,  3.43909538e+10,
   -2.42890031e+11, -2.42890031e+11,  1.82248582e+10,  1.82248582e+10,
    1.21238081e+10,  5.65381739e+09,  1.10197079e+10,  2.38345511e+11,
    2.39952072e+11,  3.49669495e+10, -4.32651302e+10, -4.32651302e+10]
```
Linear Regression training R^2: 0.73583    
Linear Regression testing R^2: 0.73505  
Linear Regression standardized training R^2: 0.73581   
Linear Regression Standardized testing R^2: 0.73504  
Comparison of coefficient:  
the linear regression coefficient changes a lot, but the values for MSE and R^2 do not vary much before and after standardization, but the scores after standardization is slightly smaller than those before standardization.    
Ridge Regression (alpha value range 60 to 80): training score: 0.73584 ; testing score: 0.73522   
Lasso Regression (alpha value range 0.0001 to 0.0003): training score: 0.73583 ; testing score: 0.73506   
Evaluation:  
We can see from the extremely similar training and testing scores that changing the model does not significantly inprove performance.

### WealthI
Linear Regression MSE: 1750276834.9304745   
Coefficient array:
```
[ 2.31986195e+03,  1.08192000e+03, -5.08892487e+01,  6.53283809e+03,
  3.17688859e+03,  4.03623951e+03, -9.96051610e+03,  1.12302854e+04,
  1.02336910e+04, -1.62924258e+04, -1.71918653e+04, -6.04206999e+03,
  2.08751277e+04, -9.31120042e+03,  2.41734580e+04,  1.34930387e+04,
 -6.80151578e+03, -1.25300357e+04, -9.08909982e+03,  5.48192929e+03,
  7.99367502e+03, -1.34756043e+04,  1.74439055e+04,  3.27144540e+04,
  5.76665872e+03,  3.89473708e+02,  2.46689944e+03, -1.29356339e+04,
 -1.29054696e+04, -2.77376917e+04, -2.95652191e+04, -2.65078796e+04,
  2.29944393e+04, -3.88963009e+03,  3.17656932e+04,  4.00606955e+04,
  3.66535576e+04,  9.64026616e+03,  4.80974344e+04,  9.98177625e+03,
 -1.07028288e+04, -9.12002749e+03, -1.86232403e+04, -4.61832386e+04,
 -3.14138344e+04, -7.19146447e+03, -1.55796604e+04, -5.61943537e+03,
 -3.46563978e+04,  3.46563978e+04, -3.20570735e+04,  3.20570735e+04,
  1.51485651e+03,  5.89549456e+04,  2.36376276e+04,  9.41611219e+03,
 -6.81569745e+04, -2.53665673e+04, -2.24372689e+04,  2.24372689e+04]
 ```
Linear Regression Standardized MSE: 1750287416.4378276  
Coefficient array:
```
[ 8.64993728e+03  5.31704713e+02 -1.00083919e+03  5.39975577e+03
  5.08584139e+15  5.97820436e+15  5.30846044e+15  5.46141040e+15
  5.83477406e+15  5.45093724e+15  5.44043848e+15  6.04946935e+15
  5.97735914e+15  5.89546013e+15  8.01510082e+15  6.97506186e+15
  5.27978059e+15  5.18342548e+15  5.65299528e+15  3.31257571e+15
  5.01392915e+15  5.43125643e+15 -4.53138456e+14 -4.55589743e+14
 -8.31227714e+14 -9.67138936e+14 -4.99120808e+15 -2.46067205e+15
 -1.03613210e+15 -1.54267253e+15 -4.38110551e+15 -3.24033768e+14
 -5.38438348e+14 -8.53491387e+14 -1.10163335e+15  3.66447293e+15
  1.59979017e+16  6.17687436e+15  2.98513166e+15  1.15392065e+15
  2.41223979e+16  1.45726753e+16  1.57338913e+16  3.04912193e+16
  1.25156346e+15  1.81327095e+15  1.14969726e+16  2.37338460e+15
 -1.67622992e+16 -1.67622992e+16  1.25773184e+15  1.25773184e+15
  8.36686869e+14  3.90180604e+14  7.60490828e+14  1.64486734e+16
  1.65595452e+16  2.41313515e+15 -2.98580826e+15 -2.98580826e+15]
```
Linear Regression training R^2: 0.82584    
Linear Regression testing R^2: 0.82501  
Linear Regression standardized training R^2: 0.82582     
Linear Regression Standardized testing R^2: 0.82501     
Comparison of coefficient: just like the feature WealthC, the values before and after standardization is nearly the same, but the scores after standardization is slightly smaller than those before standardization.    
Ridge Regression (alpha value range 90 to 95): training score: 0.82584 ; testing score: 0.82502   
Lasso Regression (alpha value range 1.0 to 2.0): training score: 0.82583 ; testing score: 0.82502  
Evaluation:  
Again, the training and testing scores for all these three regressionb models are very similar, so the performance is still not improved.

### Which of the models produced the best results in predicting wealth of all persons throughout the smaller West African country being described? 
All the values from the three models for target "WealthC" is higher than the values for target "WealthI", which indicates that "WealthI" fit the modles better and has higher predictability, so "WealthI“ is more correlated with the features. But "WealthI" has weirdly large MSE before and after standardization, which suggest that the MSE value for "WealthI" in linear regression model is not usable. As for linear regression, ridge regression, and lasso regression for "WealthC", ridge regression produces best result, although all other values are almost the same. However, for both "WealthC" and "WealthI", my codes produce strange plots. For ridge regression, they produce a graph that resembles a linear relationship and for lasso regression, after showing a bunch of convergence errors, the plot is also far from what it should be look like and I don't really know what's wrong with my codes. I've attached my graph anyway to illustrate my problem, but from the statistical values, we can still see that using "WealthI" produces better outcomes.
#### Ridge for WealthC
![Figure_1](https://user-images.githubusercontent.com/78099480/115155070-c7b2bc80-a0b0-11eb-92cd-bdf08075c3e3.png)
#### Ridge for WealthI
![Figure_2](https://user-images.githubusercontent.com/78099480/115155079-d305e800-a0b0-11eb-843a-5d1b4ad17b65.png)
