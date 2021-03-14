# Project 3
## Question 1
After importing charleston_ask.csv, I used kFold method in sklearn to split the data into 10 folds. Since most of the data are used as training data and the rest are used as testing data, we may avoid overfitting when building the model. The training scores I produced is 0.019 and the testing scores I produced is -0.047. Both the values are far from 1, so this linear regression model performs very badly. This poor performance is probably becuase the features are not in the same units and numbers of bedrooms and baths are not highly correlated with the asking price. Therefore a model based on bedrooms, baths, and area is not a good predicting model for asking price.  

## Question 2

