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

# ### Building a Data Pipeline
# 
# In this example project we wil write some code to download COVID-19 data from [The Covid Tracking Project](https://covidtracking.com/) and make a plot showing some statistics for a given state.
# 
# Our main goal with this project is to see how to develop an **automated Data Pipeline**.  The purpose of a Data Pipeline is to make your life is easy as possible when, inevitably, you have to go back and redo something.  Automation is a powerful tool that can save us a lot of time!
# 
# Along the way we will also see examples of:
# - How to download files from the internet using the **requests** library
# - How to load files and work with DataFrames using the **pandas** library
# - Some simple processing including:
#     - Working with dates
#     - Subsetting/indexing **DataFrames**
#     - Sorting lists
#     - Splitting strings
# - Making plots using the **matplotlib** library
# 
# This notebook will serve as our testing space.  Once we have figured out how to write the code and generate our plot, we will move into a new notebook and write our final, cleaned up version.
# ___
# 
# ### Part 1) Get the data
# The first thing we need to do is get our data.  Somtimes this is easier, sometimes it is harder!  In this case, since we're trying to download a .csv file, it will be rather easy.
# 
# We will use the **requests** library for this:
​
# In[1]:

import requests

# Before we download anything we need to do a little setup.

# The data we are obtaining is updated (nearly) daily.  As such, we're probably going to want to get updates periodically, and also keep track of when our files were updated.  

# One approach to this is to include a timestamp in the name of the file when we download it.  We're also going to be reformatting some of the dates within the file (you'll see why shortly).  
# 
# For all of our date/time related needs, we're going to import two additional libraries:

# In[4]:

# For adding timestamps to our files and working with/formatting dates within the files:
from datetime import datetime as dt
# This next library is all about timezones... more on that soon
import pytz

# The next thing to do is specify where our data is coming from:

# In[6]:

# URL for the file we want to download
#url = "https://covidtracking.com/api/v1/states/daily.csv"
# Note: The URL has changed (8.18.2020)
url = "https://api.covidtracking.com/v1/states/daily.csv"

# Whenever we download a fresh copy of the data, I'm going to add a timestamp to the file name.  This will allow us to always access the most recent file, while also saving older copies as a backup (just in case... you never know!)
# 
# Let's see how to create this timestamp using the now() function from the datetime module:

# In[5]:


dt.now()

# In[7]:

# Same as above, but represented as a string
str(dt.now())

# Just in case we move to a different timezone, or share our code with someone somewhere across the world, etc. we'll explicitly specify to put the timestamp in UTC [(Coordinated Universal Time)](https://en.wikipedia.org/wiki/Coordinated_Universal_Time).  Otherwise, running the now() function on different computers may return different results, which is not desireable.
# 
# This is why we imported pytz:

# In[8]:

str(dt.now(tz = pytz.utc))

# I might be nitpicking here a little - but it's not generally a good idea to include spaces in file names, folder names, variable names, etc.  Basically, **try to avoid using spaces for names of things**.
# 
# Here, I'll use the replace function to replace spaces with underscores:

# In[9]:

str(dt.now(tz = pytz.utc)).replace(' ', '_')

# Great!  Now that we have that figured out, let's get our destination for the data setup.
# 
# To keep my folder clean, I'm going to save the data in a subfolder called 'data'. Let's check to see if this folder already exists, and if not, we'll create it.

# In[11]:

# Use the os library for this
import os
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# In[12]:

# Now construct the file name
file_name_short = 'ctp_' + str(dt.now(tz = pytz.utc)).replace(' ', '_') + '.csv'
file_name = os.path.join(data_folder, file_name_short)

# In[13]:

file_name

# We are now ready to download our data!
# In[15]:

# This retrieves the contents of the URL
r = requests.get(url)
# We are openining (creating) our output file for writing (w) in binary mode (b)
# Using the 'with' statement will immediately close the file once we are done writing
# (This is the kind of thing you should just google if you forget how to do it!)
with open(file_name, 'wb') as f:
    f.write(r.content)

# Now we'll import the file we just downloaded as a pandas DataFrame:

# In[16]:

import pandas as pd
df = pd.read_csv(file_name)

# We can inspect the first (head) or last (tail) rows of the DataFrame to make sure everything looks OK:

# In[18]:

df.columns

# ### Test your understanding:
# 
# 1. What is the goal of this project?  What are we trying to produce?
# 2. Why is it important to have an automated Data Pipeline?
# 3. What have we accomplished so far?

# ### Part 2) Some filtering/preprocessing
# 
# Once we have our hands on some data, the next step is to determine if any preprocessing is necessary.  Preprocessing basically refers to any adjustments/corrections that need to be made to the data before we build a model or produce our final output. We will be learning about various types of preprocessing throughout this course - for now, we just have a few simple adjustments to make.
# 
# First, we'll extract some columns of interest.  Let's get a list of the columns in the DataFrame:

# In[19]:

df.columns

# Next, create a list of the columns we want to keep:

# In[20]:

cols = ['date', 'state','positive', 'death']

# We're going to copy these columns into a new DataFrame.  **One important thing you should pay attention to here** is whether you are working with a 'copy' or a 'view' of the data.  I will have a supplementary notebook/video about this, but so we don't get too far off track for now - just trust me and don't forget to include .copy()!

# In[21]:

df_filtered = df[cols].copy()

# In[22]:

df_filtered

# We're getting close!  Let's see if we can get a prototype of our plot working.  I want to plot data for a specific state, which will be specificed by the user.
# 
# Here's how to extract the data for a certain state.  Although this can be done in a single line of code, I usually prefer to do it in a few steps:

# In[27]:

# Specify the state we are looking for
state = 'VA'
# Determine which rows have data for this state
idx_state = df_filtered['state']==state

# In[28]:

# This is a list of True/False (i.e., boolean) values, that will be True for rows containing the state we requested
idx_state

# In[25]:

# We can check how many rows matched:
sum(idx_state)

# In[29]:

# We can look at just those rows by subsetting the DataFrame with this True/False list
df_filtered[idx_state]

# Now would be a good time to make the first version of our plot and see if anything else needs to be changed!

# In[30]:

# Import a library for plotting
import matplotlib.pyplot as plt

# In[31]:

metric = 'death'
plt.plot(df_filtered['date'][idx_state], df_filtered[metric][idx_state])
plt.show()

# The main issue here is the dates. They are showing up like this because they are stored as integers. It will be much better to store them as actual date/time objects.
# 
# First, I'll extract just one date and show how we can convert it:

# In[32]:

test_date = df_filtered['date'][0]
test_date

# In[33]:

# We'll make the conversion using the 'strptime' function
# First, we need to specify the general format of the date.
# You can look up all of the possible values here by googling the 'strptime' function
#.   https://pubs.opengroup.org/onlinepubs/009695399/functions/strptime.html
date_format = '%Y%m%d'

dt.strptime(test_date, date_format)

# It's important to get used to reading error messages like this and knowing how to find your own solution.  Debugging your code is often how you will be spending a lot of your time, and it is arguably one of the most important programming skills to have!

# In[35]:

# The error is telling us that the first argument (the date) needs to be a string.  Right now, it is an integer.
type(test_date)

# In[34]:

# We can use the 'str' function:
dt.strptime(str(test_date), date_format)

# In[36]:

# The next thing to do is to convert all of the dates in the DataFrame from integers into datetime objects
# You could write a loop, but there is an easier way using the 'apply' function from pandas.

df_filtered['date'].apply(lambda d: dt.strptime(str(d), date_format))

# In[37]:

# That looks good, so we'll go ahead and replace the original dates with this version
df_filtered['date'] = df_filtered['date'].apply(lambda d: dt.strptime(str(d), date_format))

# In[38]:

df_filtered

# The bold column is called the 'index' of the DataFrame.  Just like columns have column names, the index is essentially the name of each row.  We will set this index to be the date - this has some advantages for plotting, for example, Python will automatically attempt to use the index as the x-values on a plot, which will make our code a little shorter later on.

# In[39]:
df_filtered.index = df_filtered['date']

df_filtered

# We no longer have any need for the original date column, so might as well get rid of it:

# In[41]:

df_filtered = df_filtered.drop(columns=['date'])

# In[42]:

df_filtered

# Last but not least, I'm going to rename the columns.  This is mainly for the purposes of keeping our code for plotting shorter - it will keep us from having to relabel certain parts of the plot.

# In[43]:

col_names = ['State', 'PositiveTests', 'TotalDeaths']
df_filtered.columns = col_names

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



### create your KFold function ###
​
def DoKFold(model, X, y, k, random_state=146, scaler=None):
    '''Function will perform K-fold validation and return a list of K training and testing scores, inclduing R^2 as well as MSE.
​
        Inputs:
            model: An sklearn model with defined 'fit' and 'score' methods
            X: An N by p array containing the features of the model.  The N rows are observations, and the p columns are features.
            y: An array of length N containing the target of the model
            k: The number of folds to split the data into for K-fold validation
            random_state: used when splitting the data into the K folds (default=146)
            scaler: An sklearn feature scaler.  If none is passed, no feature scaling will be performed
        Outputs:
            train_scores: A list of length K containing the training scores
            test_scores: A list of length K containing the testing scores
            train_mse: A list of length K containing the MSE on training data
            test_mse: A list of length K containing the MSE on testing data
    '''
​
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
​
    train_scores = []
    test_scores = []
    train_mse = []
    test_mse = []
​
    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain, :]
        Xtest = X[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]
​
        if scaler != None:
            Xtrain = scaler.fit_transform(Xtrain)
            Xtest = scaler.transform(Xtest)
​
        model.fit(Xtrain, ytrain)
​
        train_scores.append(model.score(Xtrain, ytrain))
        test_scores.append(model.score(Xtest, ytest))
​
        # Compute the mean squared errors
        ytrain_pred = model.predict(Xtrain)
        ytest_pred = model.predict(Xtest)
        train_mse.append(np.mean((ytrain - ytrain_pred) ** 2))
        test_mse.append(np.mean((ytest - ytest_pred) ** 2))
​
    return train_scores, test_scores, train_mse, test_mse
​
​
### new functions ###
​
def GetColors(N, map_name='rainbow'):
    '''Function returns a list of N colors from a matplotlib colormap
            Input: N = number of colors, and map_name = name of a matplotlib colormap
​
            For a list of available colormaps:
                https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    '''
    import matplotlib
    cmap = matplotlib.cm.get_cmap(name=map_name)
    n = np.linspace(0, N, N) / N
    colors = cmap(n)
    return colors
​
​
def PlotGroups(points, groups, colors, ec='black', ax='None'):
    '''Function makes a scatter plot, given:
            Input:  points (array)
                    groups (an integer label for each point)
                    colors (one rgb tuple for each group)
                    ec (edgecolor for markers, default is black)
                    ax (optional handle to an existing axes object to add the new plot on top of)
            Output: handles to the figure (fig) and axes (ax) objects
    '''
    import matplotlib.pyplot as plt
    import numpy as np
​
    # Create a new plot, unless something was passed for 'ax'
    if ax == 'None':
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
​
    for i in np.unique(groups):
        idx = (groups == i)
        ax.scatter(points[idx, 0], points[idx, 1], color=colors[i],
                   ec=ec, alpha=0.5, label='Group ' + str(i))
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(bbox_to_anchor=[1, 0.5], loc='center left')
    return fig, ax
​
​
def CompareClasses(actual, predicted, names=None):
    '''Function returns a confusion matrix, and overall accuracy given:
            Input:  actual - a list of actual classifications
                    predicted - a list of predicted classifications
                    names (optional) - a list of class names
    '''
    accuracy = sum(actual == predicted) / actual.shape[0]
​
    classes = pd.DataFrame(columns=['Actual', 'Predicted'])
    classes['Actual'] = actual
    classes['Predicted'] = predicted
​
    conf_mat = pd.crosstab(classes['Predicted'], classes['Actual'])
​
    if type(names) != type(None):
        conf_mat.index = names
        conf_mat.index.name = 'Predicted'
        conf_mat.columns = names
        conf_mat.columns.name = 'Actual'
​
    print('Accuracy = ' + format(accuracy, '.2f'))
    return conf_mat, accuracy
​
###########################################
### Classification: K-nearest neighbors ###
###########################################
​
# Our topic this week is on **classification** techniques.  Classification, like all of the methods we've seen so far, refers to types of **supervised learning**.  With supervised learning, the model is being provided with information about what the correct answers are during the model training process.  In the case of classification, the targets we are trying to predict are labels (or classifications), rather than a continuous range of outputs as in regression.  These lables will often be nominal, or ordinal variables (in the case of an ordinal target, a regression approach might also be suitable).
#
# The first technique we'll discuss is called $K$-nearest neighbors - and it is one of those methods where the name basically tells you how it works!
​
### 1. Intro to kNN ###
​
# To get started, we'll look at some simulated data!  We're also going to be looking at a lot of the same types of plots throughout these examples, so we need to write a few quick functions that we'll be re-using.
#
# In this example, we'll generate three clusters of data in 2-dimensions, plot each group with a different color, generate a new random point, and then see how k-NN would classify this new random point.
​
# Do some testing with data that is clearly in clusters
# Import some standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
​
# Make some blobs of data
from sklearn.datasets import make_blobs as mb
​
# Specify how many blobs
n_groups = 3
# Specify how many total points
n_points = 100
# Specify how many dimensions (how many features)
n_feats = 2
​
np.random.seed(146)
data = mb(n_samples = n_points, n_features=n_feats, centers = n_groups)
X = data[0]
y = data[1]
​
colors = GetColors(n_groups)
​
# Make a plot of this data, color each group separately
for yi in np.unique(y):
    # Plot the rows of data with the current value of yi
    idx = (y==yi)
    plt.scatter(X[idx,0], X[idx,1],color=colors[yi], ec='k',s=100, label = 'Group ' + str(yi))
# Create a new data point
random_point = np.random.random(size=n_feats)
# Add this point to the plot
plt.scatter(random_point[0], random_point[1], color='grey', s=150, ec='k')
plt.legend()
plt.show()
​
# Do KNN to classify the new point
from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN(n_neighbors=20)
knn.fit(X,y)
knn.predict(random_point.reshape(1,-1))
​
### second iteration ###
​
np.random.seed(146)
​
centers =[[0,0], [5,5]]
n_points = [2, 10]
data = mb(n_samples=n_points, centers=centers)
X = data[0]
y = data[1]
​
# Create a new point
new_point = np.random.random(n_feats)
​
colors = GetColors(len(np.unique(y)))
PlotGroups(X,y,colors)
plt.scatter(new_point[0], new_point[1],color='grey',s=100)
plt.title('What will the grey point be classified as?')
plt.show()
​
knn = KNN(n_neighbors=5)
knn.fit(X,y)
knn.predict(new_point.reshape(1,-1))
​
# This is the result of something called **class imbalance**.  The default value being used here of $k=5$ means that no matter what the location of the new point is, it will always be classified as red, since there are only 2 purple points.
#
# Although this example is artificially small in terms of the total amount of data, class imbalance can certainly apply to much larger data sets.  In cases like that, it might be beneficial to **undersample** an overrepresented group, or **oversample** (e.g., by bootstrapping) an underrepresented group.
#
# We won't spend too much time on that though.  Instead you might also think, "Why don't we just reduce the value of $k$?".  In this case, that would probably work:
​
knn = KNN(n_neighbors=1)
knn.fit(X,y)
knn.predict(new_point.reshape(1,-1))
​
​
# However, choosing a value for $k$ that is too small can make the algorithm too sensitive to local noise in the data.
#
# Let's take a look at a larger data set.  We'll go back to the breast cancer data set, and for the purposes of simplicity in visualization, I'll pretend we only have 2 independent variables.
​
from sklearn.datasets import load_breast_cancer as lbc
from sklearn.model_selection import train_test_split as tts
​
data = lbc()
X = data.data
X_names = data.feature_names
y = data.target
y_names = data.target_names
​
Xdf = pd.DataFrame(X, columns=X_names)
​
cols = ['mean perimeter', 'mean fractal dimension']
Xsubset = Xdf[cols]
​
# Split the data into training and testing sets
Xtrain,Xtest,ytrain,ytest = tts(Xsubset.values, y, test_size=0.5, random_state=146)
​
# Plot the training data
colors = GetColors(len(np.unique(y)))
PlotGroups(Xtrain, ytrain, colors)
plt.show()
​
### 2. Feature scaling ###
​
# Let's take a look at that same data, but this time make sure the axes are both set to the same scale.
​
colors = GetColors(len(np.unique(y)))
PlotGroups(Xtrain, ytrain, colors)
plt.axis('equal')
plt.show()
​
# It should be clear here, that if we are going to be classifying new data based on their distance from the existing data points, that we would effectively be ignoring the 'mean fractal dimension', as it is effectively contributing nothing to the distance between two points in this data.
#
# Let's look at the same data, after standardization:
​
from sklearn.preprocessing import StandardScaler as SS
ss = SS()
​
Xtrain_s = ss.fit_transform(Xtrain)
Xtest_s = ss.transform(Xtest)
​
PlotGroups(Xtrain_s, ytrain, colors)
plt.xlabel('Standardized ' + cols[0])
plt.ylabel('Standardized ' + cols[1])
plt.axis('equal')
plt.show()
​
# Much better!  Now, back to the issue of determining an optimal value for $k$.  As mentioned previously, choosing too small of a value can lead to the algorithm being too sensitive to local noise.
#
# As an example, consider classifying the black point below:
​
PlotGroups(Xtrain_s, ytrain, colors)
plt.xlabel('Standardized ' + cols[0])
plt.ylabel('Standardized ' + cols[1])
plt.axis('equal')
​
new_point = np.array([-0.01, 0.35])
plt.scatter(new_point[0], new_point[1], color='k')
plt.show()
​
knn = KNN(n_neighbors=20)
knn.fit(Xtrain_s,ytrain)
knn.predict(new_point.reshape(1,-1))
​
# Since the closest point is purple, for $k=1$, we classify the test point as purple.  As $k$ gets larger, the classification reverses.  We might also have a bit of an issue with class imbalance here, which might start to dominate the answer as $k$ gets larger.
#
# Here's how to check for class imbalance:
​
values, counts = np.unique(ytrain, return_counts = True)
​
values
​
counts
​
# So we have nearly double the number of observations from Group 1 as we do in Group 0.  As discussed previously, we could perhaps undersample from Group 1.  While this is definitely something to consider, let's just go ahead and...
​
### 3. Test the model
​
# The primary hyperparameter here is $k$, the number of neighbors to consider.  I'm sure you already know how we're going to test this!
#
# We will not be using metrics such as $R^2$ or MSE to assess a classification model - those metrics don't make any sense here, since the things we are predicting are labels.  Instead, when you call the .score() method of a classification model, you'll be looking at the overall accuracy of the model (i.e. \% correct).
#
# I am not going to do very extensive testing here, as that could be quite time consuming.  I'm only going to check a relatively small number of values for $k$, and only use a small number of folds for our cross validation.  You should experiment a little more with this example, and if this was for a real-world application, expect to spend some number of hours (or more) on testing.
​
# Determine an optimal value for the hyperparameter k
k_range = np.arange(10, 30, 1)
​
# Keep track of the percent of correct answer
train = []
test = []
​
for k in k_range:
    knn = KNN(n_neighbors=k)
​
    tr, te, _, _ = DoKFold(knn, X, y, 10, scaler=SS())
...
Collapse
 This snippet was truncated for display; see it in full


```
