# Project 3
## Question 1
After importing charleston_ask.csv, I used kFold method in sklearn to split the data into 10 folds. Since most of the data are used as training data and the rest are used as testing data, we may avoid overfitting when building the model. The training scores I produced is 0.019 and the testing scores I produced is -0.047. Both the values are far from 1, so this linear regression model performs very badly. This poor performance is probably becuase the features are not in the same units and numbers of bedrooms and baths are not highly correlated with the asking price. Therefore a model based on bedrooms, baths, and area is not a good predicting model for asking price.  

## Question 2
Attempting to improve the model, I then applied standardScaler in sklearn to standardize the model. The data are still splited into 10 folds. The result shows that this standardized model also performs badly: training score is still around 0.019 and the testing score changes to 0.031, which is still far from the perfect score. This implies that maybe linear regression model does not apt to these data and we should consider using other models to predict the relationship between the features and target.  

## Question 3
The model still didn't improve much after applying ridge regression model. The new training score is 0.020 and the new testing score is -0.038. Then I standardized the data to see if the model would improve, the training score becomes 0.019 and the testing score becomes -0.034, which doesn't make much difference. Neither of these values are close to 1, so neither linear regression model nor ridge regression model (no matter standardize the data or not) are applicable. This further proves that the features listed are not quite correlated with the target, other objects should be considered as features in order to make asking price predictable. 

## Question 4
With charleston_act.csv, in linear regression model the training score is 0.004 and the testing score is -0.010, after standardization, tbe training score remains the same and the testing score changes to -0.008. In ridge regression after standardization, the training score remains 0.004 and the testing score changes to -0.055. These value are even smaller than those with charleston_ask.csv, showing that these regression models fit even poorly for this data file. 

## Question 5
Although the training and testing scores are still low, they generally increase for all the mentioned models when including zipcodes (for charleston_act.csv). I'm still using 10 folds. In linear regression model, the training score is 0.340 and the testing score is 0.245. In linear regression after standardization, the training score reduced 0.0001 but the testing score becomes a large negative number, which indicates that in this case, standadization is not a good choice. In ridge regression before standardization, the training score is 0.339 and the testing score is 0.208. After standardization, the training score is 0.333 and the testing score is 0.219. (差 predictive power) The effectiveness of linear regression model and ridge regression model is similar and the general increase in training score shows that the location (zipcode) of the house is a larger influencing fator of price than bedrooms, baths and area. 

## Question 6
