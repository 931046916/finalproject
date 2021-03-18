```python
import pandas as pd

path_to_data = 'gapminder.tsv'
data = pd.read_csv(path_to_data, sep = '\t')

data_lifeExp = data['lifeExp'].min()
idx_lifeExp = data['lifeExp'] == data_lifeExp
data_lifeExp = data[idx_lifeExp]
print(data_lifeExp)

# Subset gapminder.cvs and order it from highest to lowest the results for Germany, France, Italy and Spain in 2007.
# Which of the four European countries exhibited the most significant increase in total gross domestic
# product during the previous 5-year period (to 2007)?
data['gdp'] = data['pop'] * data['gdpPercap']
euro_data = data[(data['country'].isin(['Germany', 'Italy', 'France', 'Spain'])) & (data['year'] == 2002)]
euro_sorted = euro_data.sort_values('gdp', ascending = False)
print(euro_sorted)

print(data[['country', 'continent', 'year', 'lifeExp', 'pop']])

##
### Part 1. Data types ###
##########################

# Broadly speaking, there are a few common types of data and it's important to be able to define what type of data you are working with.  This is also one of the first steps towards deciding what type of model or analysis might (or might not) be appropriate.

# **Continuous data:** Numerical data that can take on a range of values, such as the temperature, or someone's age or income.  This is probably the most common type of data you will be working with.

# **Ordinal data:** Ordinal data is also numeric, however, unlike continuous data, it is only the order of ordinal values that is important, rather than the difference between them.  A typical example of ordinal data is rankings.  For example, a list of the top 10 colleges, or the order in which people finished a race.  A key difference here is that, although these values might be numeric, they don't have the same meaning as continuous data.  For example, although as pure numbers 1+2=3, it would not make sense to add the person who finished a race in 1st place to the person that finished in 2nd place (that would not be equal to the person who finished in third place).

# **Nominal data:** Categorical data, such as labels.  Some examples might include the name of a city or country, or the job that somebody has. This type of data can definitely be used in a model, but care must be taken to ensure that it is represented in a meaningful way.  As many models are ultimately based on numbers, we will often need to find a numerical way to represent nominal data, and we will see some examples of how this can be done later on.

# For the purposes of building a model, the next step is to identify your **features** and your **target(s)**.  A mathematician might use the words **independent** and **dependent** variables, whereas in the Data Science and Machine Learning communities, you will more often hear people say "features" and "targets".

# - Features = independent variables = the inputs to your model
# - Targets = dependent variables = the output of your model, or the thing you are trying to predict

#####################################################
### Part 2. Summary statistics and visualizations ###
#####################################################

# Produce a random set of values and compute some summary statistics that describe that data

import numpy as np

# How many data points do I want?
n = 1000
# To ensure that we all get the same answers, set the 'random seed'
np.random.seed(146)
x1 = np.random.random(size=n)

# The mean is the average value. The median is the value that divides the data into two equal groups.  50% of the data is both more and less than the median.

np.mean(x1)
x1.mean()

np.median(x1)

# Paramaters for the beta distribution

# a = 0.1
a = 5.0
# b = 0.1
b = 5.0
n = 1000
np.random.seed(146)
x2 = np.random.beta(a, b, size=n)

np.mean(x2)
np.median(x2)

# A histogram is a diagram consisting of rectangles whose area is proportional to the **frequency** of a variable and whose width is equal to the class **interval**.  [Oxford]

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
# Make the histogram
plt.hist(x2)
plt.show()

# add a space between the bars

plt.figure(figsize=(8, 8))
plt.hist(x2, rwidth=0.8) # see this argument
plt.xlabel('$x_2$')
plt.ylabel('Count')
plt.show()

# consider gapminder data

import pandas as pd

df = pd.read_csv('gapminder.tsv', sep='\t')

# Get the data for 2007
idx07 = df['year'] == 2007
df07 = df[idx07]
# df07

# Compare the mean and median

np.mean(df07['gdpPercap'])

np.median(df07['gdpPercap'])

# right skewed, right tailed or left leaning

plt.figure(figsize=(8, 8))
plt.hist(df07['gdpPercap'], rwidth=0.9)
plt.xlabel('GDP per capita (US$)')
plt.ylabel('Number of countries')
plt.show()

# Variance is the average of the squared differences between each data point and the mean.  Sigma^2 is often used to represent the variance.  A squared value can be difficult to understand, so often the standard deviation is used since it can be more easily interpreted.  The **standard deviation** is just the square root of the variance

np.var(df07['gdpPercap'])

np.std(df07['gdpPercap'])

# The ratio of standard deviation / mean is called the **coefficient of variation**

np.std(df07['gdpPercap']) / np.mean(df07['gdpPercap'])

# Write a function and print some summary statistics

def SummarizeData(x):
    '''Function will take a list or array of numbers and return a DataFrame with some
    basic summary statistics'''

    import numpy as np
    import pandas as pd

    # Compute some summary statistics
    # Length of the data
    len_x = len(x)
    # Min
    min_x = np.min(x)
    # Max
    max_x = np.max(x)
    # Mean
    mean_x = np.mean(x)
    # Median
    med_x = np.median(x)
    # Standard deviation
    std_x = np.std(x)
    # CV
    cv_x = std_x / mean_x

    # Create an empty DataFrame and label the columns
    summary = pd.DataFrame(columns=['N', 'Min', 'Max', 'Mean', 'Median', 'Std', 'CV'])

    # Put the summary statistics into a list
    data = [len_x, min_x, max_x, mean_x, med_x, std_x, cv_x]

    # Insert that list into the DataFrame
    summary.loc[0] = data
    return summary

SummarizeData(df07['gdpPercap'])

SummarizeData(df07['gdpPercap']).round(2)

#############################
### 3. Using the Gapminder data, make two histograms comparing the population in the earliest and latest years in the data.
#############################

# Determine the min and max years in the data
min_year = df['year'].min()

# Identify the rows in the dataframe, that have data for these years
idx_min_year = df['year'] == min_year
idx_max_year = df['year'] == max_year

# store them as 2 new dataframes
df_min_year = df[idx_min_year]
df_max_year = df[idx_max_year]

# Make the plot!
plt.figure(figsize=(6, 6))
plt.hist(df_min_year['pop'], rwidth=0.9, label=min_year, alpha=0.5)
plt.hist(df_max_year['pop'], rwidth=0.9, label=max_year, alpha=0.5)
plt.xlabel('Population')
plt.ylabel('Number of countries')
plt.show()

# Set the bin widths (intervals) to be consistent for the two years

# Determine the min and max years in the data
min_year = df['year'].min()
max_year = df['year'].max()

# Determine the min and max populations
min_pop = df['pop'].min()
max_pop = df['pop'].max()

# How many bins do we want?
n_bins = 10

# Compute the edges of the bins
my_bins = np.linspace(min_pop, max_pop, n_bins + 1)

# Identify the rows in the dataframe, that have data for these years
idx_min_year = df['year'] == min_year
idx_max_year = df['year'] == max_year

# store them as 2 new dataframes
df_min_year = df[idx_min_year]
df_max_year = df[idx_max_year]

# Make the plot!
plt.figure(figsize=(6, 6))
plt.hist(df_min_year['pop'], rwidth=0.9,
         label=min_year, alpha=0.5, bins=my_bins)
plt.hist(df_max_year['pop'], rwidth=0.9,
         label=max_year, alpha=0.5, bins=my_bins)
plt.xlabel('Population')
plt.ylabel('Number of countries')
plt.legend()
plt.show()

# Use a **logarithmic** transformation to describe the heterogeneity in the data along their continuums and understand scale as an element.  A logorithm is a quantity representing the power to which a fixed number (the base) must be raised to produce a given number.

# Determine the min and max years in the data
min_year = df['year'].min()
max_year = df['year'].max()

# Determine the min and max populations
min_pop = np.log10(df['pop'].min())
max_pop = np.log10(df['pop'].max())

# How many bins do we want?
n_bins = 10

# Compute the edges of the bins
my_bins = np.linspace(min_pop, max_pop, n_bins + 1)

# Identify the rows in the dataframe, that have data for these years
idx_min_year = df['year'] == min_year
idx_max_year = df['year'] == max_year

# store them as 2 new dataframes
df_min_year = df[idx_min_year]
df_max_year = df[idx_max_year]

# Make the plot!
plt.figure(figsize=(6, 6))
plt.hist(np.log10(df_min_year['pop']), rwidth=0.9,
         label=min_year, alpha=0.5, bins=my_bins)
plt.hist(np.log10(df_max_year['pop']), rwidth=0.9,
         label=max_year, alpha=0.5, bins=my_bins)
plt.xlabel('$\log_{10}$ Population')
plt.ylabel('Number of countries')
plt.legend()
plt.show()


### scatter plots and correlations ###
# import gapminder and continue to consider life expectancy & GDP / capita
import pandas as pd
df = pd.read_csv('./gapminder.tsv',sep='\t')
import matplotlib.pyplot as plt
# '07 data, rows & plot
idx07 = df['year']==2007
df07 = df[idx07]

plt.scatter(df07['lifeExp'], df07['gdpPercap'])
plt.xlabel('Life Exp.')
plt.ylabel('GDP per capita (US$)')
plt.show()

# correlation coefficient
import numpy as np
np.corrcoef(df07['lifeExp'], df07['gdpPercap'])

# logarithmic transformation
df['log_lifeExp'] = np.log10(df['lifeExp'])
df['log_gdpPercap'] = np.log10(df['gdpPercap'])

df07 = df[idx07]

plt.scatter(df07['log_lifeExp'], df07['log_gdpPercap'])
plt.xlabel('$\log_{10}$ Life Exp.')
plt.ylabel('$\log_{10}$ GDP per capita (US\$)')
plt.show()

# consider the correlation
np.corrcoef(df07['log_lifeExp'], df07['log_gdpPercap'])

# pairwise correlation plot & heatmap using the seaborn library
import seaborn as sns

sns.pairplot(df07[['lifeExp','pop','gdpPercap']])
plt.show()

cols = ['lifeExp','gdpPercap','pop']
corr_matrix = np.corrcoef(df07[cols],rowvar=False)

corr_matrix

sns.heatmap(corr_matrix,xticklabels=cols,
                        yticklabels=cols,
                        vmin=-1,
                        vmax=1,
                        cmap='bwr',
                        annot=True, fmt='.2f')
plt.show()

### new dataset -- breast cancer data ###

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

type(data)

# assign features (X) & target (y)

X = data.data
X_names = data.feature_names
X_df = pd.DataFrame(data = X, columns=X_names)
y = data.target
y_names = data.target_names

# interrogate the data
sns.heatmap(X_df.corr(),vmin=-1,vmax=1,cmap='bwr')
plt.show()

### dimensionality reduction ###
# 30 measurements (variables) for each of 569 samples (observations)
# each observation also classified as benign or malignant
# reduce to two dimensions: mean radius & mean fractal dimension, and then the third dimension, tumor classification (benign or malignant)

cols = ['mean radius', 'mean fractal dimension']
colors = ['red', 'blue']
for yi in [0,1]:
    idx = y==yi
    plt.scatter(X_df.loc[idx,cols[0]], X_df.loc[idx,cols[1]],
                color=colors[yi], label=y_names[yi],alpha=0.4,ec='k')
plt.xlabel(cols[0])
plt.ylabel(cols[1])
plt.legend()
plt.show()

# Principal Component Analysis (PCA)

# What is PCA: it is method that calculates the ratio between variables to determine the exact metric of contributing "importance" in describing how the data is spread out.  You can conceptually think of a PCA as the axes of an ellipsoid, although once you add more than three variables visualizing becomes difficult.  PCA can be an effective way to identify clusters within the data that then provide direction towards interrogating subsets.

# Here's a video from stat quest that requires a bit more focus and work, but is fairly effective at introducing PCA
# StatQuest: Principal Component Analysis (PCA), Step-by-Step by Josh Starmer
# https://www.youtube.com/watch?v=FgakZw6K1QQ

# if you are interested to further explore this idea there is a R script that follows the video -- the R interpreter is available to you in PyCharm or if you prefer a stand alone IDE, RStudio is another good option.  Check out the following link on my data science book for steps on how to install R which can then be used with PyCharm or RStudio
# https://tyler-frazier.github.io/dsbook/rinstall.html

# Conduct PCA on breast cancer data

from sklearn.decomposition import PCA
pca = PCA()
X_pca = pca.fit_transform(X_df)

colors = ['red', 'blue']

# Make a scatter plot using a loop
for yi in [0,1]:
    idx = y==yi
    plt.scatter(X_pca[idx,0], X_pca[idx,1],
                color=colors[yi], label=y_names[yi],alpha=0.4,ec='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# scree plots explain the variance per principal component, which will sum to 100%

pca.explained_variance_ratio_

evr = pca.explained_variance_ratio_
plt.figure(figsize=(14,6))
plt.bar(range(1,len(evr)+1), evr)
plt.xlabel('Principal Component')
plt.ylabel('% of Variance Explained')
plt.xlim([0.3,30])
plt.xticks(range(1,len(evr)+1))
plt.show()

# This may appear to be a useful result, although we need to consider the relative scale of each variable. Some outcomes within different variables may have widely different ranges of minimum and maximum values.  Consider one variable that has a minimum value of 1000 and a maximum value of 100,000 and it is considered in combination with a second varible that has a minimum value of 0.001 and a maximum value of 0.1.  Do you think it is possible that in some instances these two different ranges of values could impact model results?

X_df

### feature scaling ###
# feature scaling is usually important whenever distance is involved.  This is because larger "steps" can have a more significance impact on the model when compared to smaller "steps."  Feature scaling transforms the data so each variable, including all of the individual outcomes within each variable are proportionately scaled to one another.

# standarization is a type of feature scaling where the data is transformed such that all of the variables have a mean of 0 and a standard deviation of 1

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_scaled = ss.fit_transform(X_df)

X_scaled_df = pd.DataFrame(X_scaled, columns=X_names)
X_scaled_df

# compare unscaled data with scaled data (standardized)

# unscaled (x & y axes units are almost the same)

cols = ['mean radius', 'mean fractal dimension']
colors = ['red', 'blue']
for yi in [0,1]:
    idx = y==yi
    plt.scatter(X_df.loc[idx,cols[0]], X_df.loc[idx,cols[1]],
                color=colors[yi], label=y_names[yi],alpha=0.4,ec='k')
plt.xlabel(cols[0])
plt.ylabel(cols[1])
plt.legend()
plt.axis('equal')
plt.show()

# scaled

cols = ['mean radius', 'mean fractal dimension']
colors = ['red', 'blue']
for yi in [0,1]:
    idx = y==yi
    plt.scatter(X_scaled_df.loc[idx,cols[0]], X_scaled_df.loc[idx,cols[1]],
                color=colors[yi], label=y_names[yi],alpha=0.4,ec='k')
plt.xlabel(cols[0])
plt.ylabel(cols[1])
plt.legend()
plt.axis('equal')
plt.show()

# scale data and apply PCA

pca_scaled = PCA()
X_scaled_pca = pca_scaled.fit_transform(X_scaled_df)

colors = ['red', 'blue']
for yi in [0,1]:
    idx = y==yi
    plt.scatter(X_scaled_pca[idx,0], X_scaled_pca[idx,1],
                color=colors[yi], label=y_names[yi],alpha=0.4,ec='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA on Standardized Data')
plt.legend()
plt.show()

# scree plot of scaled data

evr = pca_scaled.explained_variance_ratio_
plt.figure(figsize=(12,6))
plt.bar(range(1,len(evr)+1), evr)
plt.xlabel('Principal Component')
plt.ylabel('% Explained Variance')
plt.show()

### Linear Regression ###

# Ordinary Least Squares (or OLS) is a technique used to find the solution to a linear regression, optimimzation problem.

# In a OLS Linear Regression, y's are our **targets** (or dependent variables) and the x's are our features (or independent variables).  If we have N observations, then for a single problem, we will have N instances of the equation. Each instance of the equation represents a single observation, so they will all have different values for x's and y's, but they all share the same parameters (the beta's).  Once we have determined values for all of the beta's, then we can plug in new values for the x's, and the model will predict what we should expect to see for y.

# It is more unlikely than not that any line will not pass through any three points, or any plane will not pass through any four points.  The result is that we are often working with overdetermined problems, since there are too many constraints to find a perfect solution, and some error is unavoidable.

# The goal of OLS is to find the parameters (the beta's) that minimize the sum of squared errors between the model's predictions and the true, observed values.  Other types of regression models invoke other criteria (for example minimize the sum of the absolute values of the errors).

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

### boston housing data ###

data = load_boston()
X = data.data
X_names = data.feature_names
y = data.target

lin_reg = LinearRegression()
lin_reg.fit(X,y)
lin_reg.score(X,y)

y_pred = lin_reg.predict(X)

plt.scatter(y,y_pred,alpha=0.5,ec='k')
plt.plot([min(y),max(y)], [min(y), max(y)], ':k')
plt.axis('equal')
plt.xlabel('Actual median house price')
plt.ylabel('Predicted median house price')
plt.show()

idxLo = y<50
y_lo = y[idxLo]
X_lo = X[idxLo]
lin_reg_lo = LinearRegression()
lin_reg_lo.fit(X_lo, y_lo)
lin_reg_lo.score(X_lo, y_lo)

### Mean-squared error (MSE) ###

# Mean-squared error (MSE) is the mean of the squared errors, and one way to estimate the predictive power of a model with a continuous dependent variable.  MSE is most useful as a metric for comparing how effective two models performed in accurately predicting a continuous outcome.  The lower the MSE between the two (or more) models, the better.

prices = y
predicted = y_pred

summation = 0
n = len(predicted)
for i in range (0,n):
  difference = prices[i] - predicted[i]
  squared_difference = difference**2
  summation = summation + squared_difference

MSE = summation/n
print ("The Mean Squared Error is: " , MSE)

### Validation ###

# Test of a model where new data is applied to determine its predictive power

# An overfit model is typically characterized by performing extremely well on the data used to train it, while performing poorly on new data.  That is to say, the model has basically memorized the information it was given, but hasn't really discovered the underlying "pattern", so it's unable to generalize its knoweledge to new situations.

# **Internal validity** refers to how accurate a model's predictions are on the data it was trained on
# **External validity** refers to accurate a model's predictions are on data it was not trained on, i.e., how well it generalizes to new data.
# An **overfit** model will tend to have a much lower external vs internal validity.
# An **underfit** model will tend to have both types of validity being rather low.

from sklearn.datasets import load_boston
data = load_boston()
X = data.data
X_names = data.feature_names
y = data.target

from sklearn.model_selection import train_test_split as tts

Xtrain,Xtest,ytrain,ytest = tts(X,y,test_size=0.4)

lin_reg.fit(Xtrain,ytrain)

print('Internal validity (R^2): ' + format(lin_reg.score(Xtrain,ytrain), '.3f'))
print('External validity (R^2): ' + format(lin_reg.score(Xtest,ytest), '.3f'))

### K-fold validation

# This brings us to a concept called **$K$-fold cross validation**, and here's how it goes:
#
# 1. Split your data into $K$ equally sized groups (you pick this number).  These groups are called **folds**.
# 2. Use the first fold as your test data, and the remining $K-1$ folds as your training data, and then check the scores.
# 4. Use the second fold as your test data, and the remaining $K-1$ folds as your training data.
# 5. Repeat this process $K$ times, using each of the $K$ folds as your test data exactly once.
# 6. Now look at the average of the results, or perhaps a histogram of the results.  This will provide an estimate of how well you should expect your model to perform on new data.

from sklearn.model_selection import KFold
kf = KFold(n_splits = 50, random_state=146, shuffle=True)

train_scores=[]
test_scores=[]

for idxTrain, idxTest in kf.split(X):
  Xtrain = X[idxTrain, :]
  Xtest = X[idxTest, :]
  ytrain = y[idxTrain]
  ytest = y[idxTest]

  lin_reg.fit(Xtrain, ytrain)

  train_scores.append(lin_reg.score(Xtrain, ytrain))
  test_scores.append(lin_reg.score(Xtest, ytest))

min_r2 = min(min(train_scores), min(test_scores))
max_r2 = 1

n_bins = 20
my_bins = np.linspace(min_r2,max_r2, n_bins+1)

plt.hist(train_scores, label='Training',bins=my_bins,alpha=0.5,rwidth=0.9)
plt.hist(test_scores, label='Testing',bins=my_bins,alpha=0.5,rwidth=0.9)
plt.xlabel('$R^2$')
plt.ylabel('# of folds')
plt.legend()
plt.show()

print([min(test_scores), np.mean(test_scores), max(test_scores)])
print([min(train_scores), np.mean(train_scores), max(train_scores)])

plt.scatter(train_scores,test_scores,alpha=0.5,ec='k')
plt.xlabel('Training scores')
plt.ylabel('Testing scores')
plt.axis('equal')
plt.show()

# The best case scenario would be if all the dots were clustered together,
# and if the point they were clustered on was closer to 1.  In this case,
# it looks like the test scores are pretty variable, even though the training scores are pretty consistent.
# This type of result would definitely nudge me in the direction of looking for a better type of model!

### Regularization ###

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

data = load_boston()
X = data.data
X_names = data.feature_names
y = data.target

lin_reg = LinearRegression()

# StatQuest K-fold cross validation
# https://www.youtube.com/watch?v=fSytzGwwBVw&t=90s

kf = KFold(n_splits = 10, shuffle=True,random_state=146)
# kf = KFold(n_splits = 10, shuffle=True)
train_scores=[]
test_scores=[]

for idxTrain,idxTest in kf.split(X):
    Xtrain = X[idxTrain,:]
    Xtest = X[idxTest,:]
    ytrain = y[idxTrain]
    ytest = y[idxTest]
    lin_reg.fit(Xtrain,ytrain)
    train_scores.append(lin_reg.score(Xtrain,ytrain))
    test_scores.append(lin_reg.score(Xtest,ytest))

print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))

### K-fold cross validation ###

def DoKFold(model, X, y, k, standardize=False, random_state=146):
# def DoKFold(model, X, y, k, standardize=False):
    from sklearn.model_selection import KFold
    if standardize:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    # kf = KFold(n_splits=k, shuffle=True)

    train_scores = []
    test_scores = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain, :]
        Xtest = X[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]

        if standardize:
            Xtrain = ss.fit_transform(Xtrain)
            Xtest = ss.transform(Xtest)

        model.fit(Xtrain, ytrain)

        train_scores.append(model.score(Xtrain, ytrain))
        test_scores.append(model.score(Xtest, ytest))

    return train_scores, test_scores

train_scores, test_scores = DoKFold(lin_reg,X,y,10)
print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))

### Ridge regression ###
# https://www.youtube.com/watch?v=KvtGD37Rm5I

from sklearn.linear_model import Ridge

a_range = np.linspace(0, 100, 100)

k = 10

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores,test_scores = DoKFold(rid_reg,X,y,k,standardize=True)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
print('Optimal alpha value: ' + format(a_range[idx], '.3f'))
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))

plt.plot(a_range, avg_te_score, color='r', label='Testing')
plt.xlabel('$\\alpha$')
plt.ylabel('Avg. $R^2$')
plt.legend()
plt.show()

from sklearn.preprocessing import StandardScaler as SS
ss = SS()
Xs = ss.fit_transform(X)
lin_reg.fit(Xs,y)
lin_coefs = lin_reg.coef_

a_range = np.linspace(0,1000,100)

rid_coefs = []
for a in a_range:
    rid_reg = Ridge(alpha=a)
    rid_reg.fit(Xs,y)
    rid_coefs.append(rid_reg.coef_)

plt.figure(figsize=(12,6))
plt.plot(a_range, rid_coefs)
plt.scatter([0]*len(lin_coefs), lin_coefs)
plt.legend(X_names, bbox_to_anchor=[1, 0.5], loc='center left')
plt.xlabel('$\\alpha$')
plt.ylabel('Coeff. Estimates')
plt.show()

### Lasso Regression ###

from sklearn.linear_model import Lasso

a_range = np.linspace(0.01, 0.03, 100)

k = 10

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    las_reg = Lasso(alpha=a)
    train_scores,test_scores = DoKFold(las_reg,X,y,k,standardize=True)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
print('Optimal alpha value: ' + format(a_range[idx], '.3f'))
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))

plt.plot(a_range, avg_te_score, color='r', label='Testing')
plt.xlabel('$\\alpha$')
plt.ylabel('Avg. $R^2$')
plt.legend()
plt.show()

a_range = np.linspace(1e-6,10,100)

las_coefs = []
for a in a_range:
    las_reg = Lasso(alpha=a)
    las_reg.fit(Xs,y)
    las_coefs.append(las_reg.coef_)

plt.figure(figsize=(12,6))
plt.plot(a_range, las_coefs)
plt.scatter([0]*len(lin_coefs), lin_coefs)
plt.legend(X_names, bbox_to_anchor=[1, 0.5], loc='center left')
plt.xlabel('$\\alpha$')
plt.ylabel('Coeff. Estimates')
plt.show()
```
