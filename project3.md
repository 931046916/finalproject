# Project 3
## Question 1
After importing charleston_ask.csv, I used kFold method in sklearn to split the data into 10 folds. Since most of the data are used as training data and the rest are used as testing data, we may avoid overfitting when building the model. The training scores I produced is 0.019 and the testing scores I produced is -0.047. Both the values are far from 1, so this linear regression model performs very badly. This poor performance is probably becuase the features are not in the same units and numbers of bedrooms and baths are not highly correlated with the asking price. Therefore a model based on bedrooms, baths, and area is not a good predicting model for asking price.  

## Question 2
Attempting to improve the model, I then applied standardScaler in sklearn to standardize the model. The data are still splited into 10 folds. The result shows that this standardized model also performs badly: training score is still around 0.019 and the testing score changes to 0.031, which is still far from the perfect score. 

## Question 3
The model still didn't improve much after applying ridge regression model. The new training score is 0.020 and the new testing score is -0.038. Then I standardized the data to see if the model would improve, the training score becomes 0.019 and the testing score becomes -0.034, which doesn't make much difference. Neither of these values are close to 1, so neither linear regression model nor ridge regression model (no matter standardize the data or not) are applicable. 
