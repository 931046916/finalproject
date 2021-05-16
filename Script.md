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

2021/04/12
​
### add DoKFold ###
​
def DoKFold(model ,X ,y ,k ,random_state=146 ,scaler=None):
    ''' Function will perform K-fold validation and return a list of K training and testing scores, inclduing R^2 as well as MSE.
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
    kf = KFold(n_splits=k ,shuffle=True ,random_state=random_state)
​
    train_scores =[]
    test_scores =[]
    train_mse =[]
    test_mse =[]
​
    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain ,:]
        Xtest = X[idxTest ,:]
        ytrain = y[idxTrain]
        ytest = y[idxTest]
​
        if scaler != None:
            Xtrain = scaler.fit_transform(Xtrain)
            Xtest = scaler.transform(Xtest)
​
        model.fit(Xtrain ,ytrain)
​
        train_scores.append(model.score(Xtrain ,ytrain))
        test_scores.append(model.score(Xtest ,ytest))
​
        # Compute the mean squared errors
        ytrain_pred = model.predict(Xtrain)
        ytest_pred = model.predict(Xtest)
        train_mse.append(np.mean((ytrain -ytrain_pred )**2))
        test_mse.append(np.mean((ytest -ytest_pred )**2))
​
    return train_scores ,test_scores ,train_mse ,test_mse
​
def CompareClasses(actual, predicted, names=None):
    '''Function returns a confusion matrix, and overall accuracy given:
            Input:  actual - a list of actual classifications
                    predicted - a list of predicted classifications
                    names (optional) - a list of class names
    '''
    import pandas as pd
    accuracy = sum(actual == predicted) /actual.shape[0]
    classes = pd.DataFrame(columns=['Actual' ,'Predicted'])
    classes['Actual'] = actual
    classes['Predicted'] = predicted
    conf_mat = pd.crosstab(classes['Predicted'] ,classes['Actual'])
    # Relabel the rows/columns if names was provided
    if type(names) != type(None):
        conf_mat.index =y_names
        conf_mat.index.name ='Predicted'
        conf_mat.columns =y_names
        conf_mat.columns.name = 'Actual'
    print('Accuracy = ' + format(accuracy, '.2f'))
    return conf_mat, accuracy
​
### 1.  Intro to Logistic Regression ###
​
# Logistic regression is useful a method for classification.  It is called a regression because the model performs a linear regression in the background, however the final output of the model is transformed by something called a **logistic (or sigmoid) function** which transforms the existing range of values to a new distribution with a minimum value of 0 and maximum value of 1. A typical example of a sigmoid function looks like:
​
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
x = np.linspace(-10,10,1000)
y = 1/(1+np.exp(-x))
plt.plot(x,y)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
​
# These values are meant to be interpreted as the probabilities of membership in a particular group.  For example, if determining whether a data point belongs to Group 0 or Group 1, a logistic regression that predicts a value of $y=0.7$ is interpreted as a 70\% chance that the data point belongs to Group 1 (or, equivalently, as a 30\% chance that the data point belongs to Group 0).
#
# Logistic Regression can also be applied to problems involving more than 2 classifications.  In this case, the model will output probabilities of membership in each particular group, such that the sum of these probabilities will always equal 1 (that is, there is a 100\% chance that the data belongs to one of the groups).  For example, a Logistic Regression might predict a 20\% probability for Group A, 45\% probability for Group B, and a 35\% probability for Group C (where it should be noted that 0.20 + 0.45 + 0.35 = 1).
#
# These probabilities are referred to as **soft classifications**.  They are "soft" in the sense that we are not assigning a yes/no answer for whether the data point belongs to a particular group.  You might take the largest probability, using the numbers above for example, and predict that the data point belongs to Group B.  This is referred to as a **hard classification**.  Although we are often interested in the hard classifications, examining the soft classifications can also yield some insight into how "certain" the model is about its prediction.
#
# We'll start by looking at Logistic Regression applied to a problem with only 2 classes, and with only one independent variable.
​
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
​
from sklearn.datasets import load_breast_cancer as lbc
from sklearn.linear_model import LogisticRegression
​
bc = lbc()
X = bc.data
X_names = bc.feature_names
y = bc.target
y_names = bc.target_names
​
Xdf = pd.DataFrame(X, columns=X_names)
​
# I'll select just one of the features for now.  To do so, I'll simply pick the variable that has the largest (in absolute value) correlation with the target.  I'm not suggesting that this is a good way to go about building a model, but we want to be able to visualize a simple case first before we move on to more complicated cases.
​
Xdf['y'] = y
​
Xdf.corr().loc['y'].sort_values()
​
col = 'worst concave points'
X1 = Xdf[col]
plt.scatter(X1,y,alpha=0.5,ec='k')
plt.xlabel(col)
plt.yticks([0,1], [ y_names[i] + ' (y= ' + str(i) +')' for i in [0,1]])
plt.show()
​
# First of all, it stands to reason, just by looking at the picture, that we should be able to build a somewhat decent model here.  First, we will fit our Logistic Regression to the entire data set, and visualize the probabilities that it predicts.
​
log_reg = LogisticRegression()
log_reg.fit(X1.values.reshape(-1,1), y)
y_pred_proba = log_reg.predict_proba(X1.values.reshape(-1,1))
​
y_pred_proba
​
# The first column above represents the probability of membership in Group 0 (malignant), and the second column represents the probability of membership in Group 1 (benign).  The way our data is currently setup, we'll want to plot the probabilities of being benign, since in our data, $y=1$ represents the class "benign".
#
# To visualize the curve, I'll generate a list of values between the min and max of our current feature:
​
x_range = np.linspace(min(X1), max(X1), 100)
y_pred_proba = log_reg.predict_proba(x_range.reshape(-1,1))
​
plt.scatter(X1,y,alpha=0.5,ec='k')
plt.plot(x_range,y_pred_proba[:,1], '-r')
plt.plot([min(X1), max(X1)], [0.5, 0.5], ':r')
plt.xlabel(col)
plt.yticks([0,1], [ y_names[i] + ' (y= ' + str(i) +')' for i in [0,1]])
plt.show()
​
# Next, we can convert these soft classifications into hard classifications.  For a 2 class problem, this corresponds to simply placing a threshold at $y=0.5$.  Anything above this line would be classified as Group 1, and anything below it as Group 0.
#
# There's at least two ways we can do this:
​
y_hard = log_reg.predict(x_range.reshape(-1,1))
y_hard
​
# This should be identical to what we would get if we picked the group with the highest probability:
​
y_hard2 = np.argmax(y_pred_proba, axis=1)
y_hard2
​
# Now that we understand what we are predicting, let's fit the model using all of our features and see how well it does!
#
# Note that, by default, the Logistic Regression will perform an L2 regularization (similar to the Ridge regression), therefore we should standardize our features.  Additionally, standardizing is going to speed up the convergence of the solution (if you don't believe me, try running it without the standardization).
​
from sklearn.preprocessing import StandardScaler as SS
log_reg = LogisticRegression()
k=20
tr,te,_,_ = DoKFold(log_reg,X,y,k,scaler=SS())
​
[np.mean(tr), np.mean(te)]
​
plt.scatter(tr,te,alpha=0.5,ec='k')
plt.xlabel('Training Accuracy')
plt.ylabel('Testing Accuracy')
plt.xlim([0,1.1])
plt.ylim([0,1.1])
plt.show()
​
# Let's look at a confusion matrix for a single train/test split:
from sklearn.model_selection import train_test_split as tts
​
# Create the training/testing split
Xtrain,Xtest,ytrain,ytest = tts(X,y,test_size=0.4, random_state=146)
​
# Standardize the data
ss = SS()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.transform(Xtest)
​
# Fit the model
log_reg.fit(Xtrain,ytrain)
​
# Predict y for the test set
y_pred = log_reg.predict(Xtest)
​
# Look at the confusion matrix
CompareClasses(ytest,y_pred,y_names)
​
### 2.  Logistic regression with a multiple class problem ###
​
# Next, we'll take a look at the iris dataset (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
#
# The goal is to predict one of three types of iris species (setosa, virginica, veritosa), based on 4 physical measurements of the sepals and petals of the plant.
​
from sklearn.datasets import load_iris
iris = load_iris()
​
X = iris.data
X_names = iris.feature_names
y = iris.target
y_names = iris.target_names
​
Xdf = pd.DataFrame(X, columns=X_names)
​
Xdf.head(3)
​
# Perform a K-fold validation:
​
log_reg = LogisticRegression()
k = 20
tr,te,_,_ = DoKFold(log_reg,X,y,k,scaler=SS())
[np.mean(tr), np.mean(te)]
​
plt.scatter(tr,te,alpha=0.5,ec='k')
plt.xlabel('Training Accuracy')
plt.ylabel('Testing Accuracy')
plt.xlim([0,1.1])
plt.ylim([0,1.1])
plt.show()
​
# Look at a confusion matrix for single train/test split:
​
# Create the training/testing split
Xtrain,Xtest,ytrain,ytest = tts(X,y,test_size=0.4, random_state=149)
​
# Standardize the data
ss = SS()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.transform(Xtest)
​
# Fit the model
log_reg.fit(Xtrain,ytrain)
​
# Predict y for the test set
y_pred = log_reg.predict(Xtest)
y_pred_proba = log_reg.predict_proba(Xtest)
​
# Look at the confusion matrix
CompareClasses(ytest,y_pred,y_names)
​
y_pred_proba
​
### Part 3.  Logistic regression application with images of handwritten digits ###
​
# Here's a fun one - we'll use logistic regression to classify some handwritten digits (we'll see this data again when we discuss the Multilayer Perceptron).
#
# The data consists of very low resolution image data, 8x8 = 64 pixels, with each pixel represented on 0 to 16 scale representing their grayscale intensity.  The target is, of course, one of the digits 0 through 9.
#
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
​
from sklearn.datasets import load_digits
dig = load_digits()
X = dig.data
# There are no feature names here
y = dig.target
# There are no target names
# But there are also "images"
img = dig.images
​
# Display a random image
idx = np.random.choice(len(img))
plt.figure(figsize=(4,4))
plt.imshow(img[idx], cmap='gray')
plt.title('This is supposed to be a ' + str(y[idx]))
plt.show()
​
# Let's do a K-fold validation.  Due to the nature of the data (pixel intensities between 0 and 16), I'm not sure if I want to standardize or not, so I'll just try it both ways.
​
log_reg = LogisticRegression()
k=5
tr,te,_,_ = DoKFold(log_reg,X,y,k)
​
# Watch out for convergence warnings!  This is the algorithm telling you it doesn't think it's done yet!
#
# For many problems it is not possible to explicitly find a solution, e.g. by solving an equation.  Instead, **numerical approaches** are often taken which amounts to the computer making succesively (hopefully) closer approximations to the solution (a simple example of a numerical method is Newton's method: https://en.wikipedia.org/wiki/Newton%27s_method).  It is entirely possible in some cases that, if permitted, the computer would continue to run indefinitely, i.e. an infinite loop.  Because of this, many numerical methods have an upper limit on how many iterations they are allowed to run before terminating.
#
# When you see a convergence warning, the first thing you might try is to increase this limit a bit (in sklearn, it is usually called 'max_iter').
​
log_reg = LogisticRegression(max_iter=5000)
k=5
tr,te,_,_ = DoKFold(log_reg,X,y,k)
​
[np.mean(tr), np.mean(te)]
​
plt.scatter(tr,te,alpha=0.5,ec='k')
plt.xlabel('Training Accuracy')
plt.ylabel('Testing Accuracy')
plt.xlim([0,1.1])
plt.ylim([0,1.1])
plt.show()
​
# Look at a confusion matrix on a single train/test split:
​
# Create the training/testing split
Xtrain,Xtest,ytrain,ytest = tts(X,y,test_size=0.4, random_state=149)
​
# Fit the model
log_reg.fit(Xtrain,ytrain)
​
# Predict y for the test set
y_pred = log_reg.predict(Xtest)
y_pred_proba = log_reg.predict_proba(Xtest)
​
# Look at the confusion matrix
CompareClasses(ytest,y_pred)
​
# Repeat the above, but this time standardize the data
log_reg = LogisticRegression(max_iter=5000)
k=5
tr,te,_,_ = DoKFold(log_reg,X,y,k,scaler=SS())
[np.mean(tr), np.mean(te)]
​
# Create the training/testing split
Xtrain,Xtest,ytrain,ytest,itrain,itest = tts(X,y,img,test_size=0.4, random_state=146)
​
# Standardize the data
ss = SS()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.transform(Xtest)
​
# Fit the model
log_reg.fit(Xtrain,ytrain)
​
# Predict y for the test set
y_pred = log_reg.predict(Xtest)
y_pred_proba = log_reg.predict_proba(Xtest)
​
# Look at the confusion matrix
CompareClasses(ytest,y_pred)
​
# One interesting thing to note here - the Logistic Regression has no idea about the spatial orientation of the pixels.  As far as the model is concerned, it's just a list of 64 numbers without any inherent ordering! (The images are already centered though, so that's certainly important).
#
# Let's take a look at some of the ones it got wrong.  We'll get a list of all the incorrect classifications and then select one at random to look at:
​
idxWrong = np.where(ytest != y_pred)
idxWrong[0]
​
#idx = np.random.choice(idxWrong[0])
idx = 210
plt.figure(figsize=(4,4))
plt.imshow(itest[idx], cmap='gray')
plt.title('This is supposed to be a ' + str(ytest[idx]) + '\nIt was classified as a ' + str(y_pred[idx]))
plt.show()
​
# Let's take a peek at the soft classifications:
​
plt.figure(figsize=(12,6))
plt.bar(range(10), y_pred_proba[idx])
plt.xlabel('Digit')
plt.ylabel('Predicted Probability')
plt.xticks(range(10))
plt.show()
​
Collapse


2021/04/14
### add DoKFold ###

def DoKFold(model, X, y, k, random_state=146, scaler=None):
    '''Function will perform K-fold validation and return a list of K training and testing scores, inclduing R^2 as well as MSE.

        Inputs:
            model: An sklearn model with defined 'fit' and 'score' methods
            X: An N by p array containing the features of the model.  The N Columns are features, and the p rows are observations.
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

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    train_scores = []
    test_scores = []
    train_mse = []
    test_mse = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain, :]
        Xtest = X[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]

        if scaler != None:
            Xtrain = scaler.fit_transform(Xtrain)
            Xtest = scaler.transform(Xtest)

        model.fit(Xtrain, ytrain)

        train_scores.append(model.score(Xtrain, ytrain))
        test_scores.append(model.score(Xtest, ytest))

        # Compute the mean squared errors
        ytrain_pred = model.predict(Xtrain)
        ytest_pred = model.predict(Xtest)
        train_mse.append(np.mean((ytrain - ytrain_pred) ** 2))
        test_mse.append(np.mean((ytest - ytest_pred) ** 2))

    return train_scores, test_scores, train_mse, test_mse

### 1. Using decision trees for regression ###

# In this notebook, we'll see how decision trees can be applied to regression problems.  The process is very similar to how they are used for classification, the primary difference being that we will be interested in predicting some continuous-valued target rather than a classification/group.
#
# The modifications to the model are based on 1) how predictions are made, and 2) the criteria used when determining how to split a node.
#
# When using a decision tree for regression, each node will contain some number of observations, with the targets typically being some sort of continuous variable.  The predicted values are typically the mean value of the target within each of the leaves, although using the median value might also be suitable in some applications.
#
# When using decision trees for classification, the criteria used to split a node was based on minimizing the Gini impurity of the resulting nodes.  This would of course not make sense in the context of a regression problem, since we are not predicting labels.  Instead, the criteria used to split a node is typically based on minimizing the MSE of the resulting nodes (although the are other things you might choose to optimize, such as the mean-absolute error, as one example).
#
# By default, the decision trees we use here will make their predictions based on the mean value of the target within each leaf of the tree, and the splitting criteria will be based on minimizing the MSE.
#
# To get started, I'll make the smallest possible tree for our Boston House Price data.

from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn import tree
from sklearn.datasets import load_boston
import pandas as pd

data = load_boston()
X = data.data
X_names = data.feature_names
y = data.target
X_df = pd.DataFrame(X,columns=X_names)

dt = DTR(max_depth=1, random_state=146)
dt.fit(X_df,y)
tree.plot_tree(dt,feature_names=X_names);

# Here's how to interpret the decision tree above.  Starting at the top (the root):
#
# - The root node contains all of the data (samples = 506).
# - The target in this data is the median house price (in US\\$ thousands), and the average of this value for all 506 samples is 22.533.
# - If we were to use the value above (22.533) as our prediction, then the mean squared error (MSE) on the data the model was fit to would be 84.42.
# - Splitting the data into two new nodes, based on the criteria of RM<=6.941, results in the two new nodes you see on the bottom.  If the number of rooms in a house (RM) is less than this value (i.e., if the splitting criteria is True), you go into the bottom left node, otherwise (if the splitting criteria is False), you go to the bottom right node.
# - The nodes at the bottom are referred to as the **leaves**.  This is where our predictions would come from.  So, if a house has less than 6.941 rooms, we would predict a price of 19.934.  If it has more rooms than that, then we would predict a price of 37.238.
# - Since I specified "max_depth=1", the algorithm terminates after the first split is performed.
#
# If you wanted to translate that model into some code, it might look something like this:
#
#         if RM <= 6.941:
#             predicted_value = 19.934
#         else:
#             predicted_value = 37.238
#
# and as the tree gets larger, it's basically just a big set of nested conditional statements!
#
# The splitting criteria used here is minimizing the weighted average of the mean squared errors of the two new nodes (equivalently, it's minimizing the mean squared error of the model's predictions).  In other words, splitting the data based on any other feature, or any other value of RM would result in two new nodes that have a larger overall MSE than what you see here.
#
# We should be able to compute the MSE of the model on the training data in two different ways, and verify that they agree.

mse1 = (40.273*430 + 79.729*76)/(430+76)
mse1

y_pred = dt.predict(X)
mse2 = np.mean((y-y_pred)**2)
mse2

# For smaller problems, it's usually possible to find the best split by scanning through all of the data.  For larger problems, the search might be done randomly.  That is, instead of scanning through all features, a random subset might be selected for each split - the purpose of this is to reduce the computational time required.  **Note:** Even if all features are being considered at each split, the results still may vary between runs.  This is because the features are still being selected in random order, and it is possible that two different splits give the same reduction in MSE.
#
# You should definitely consult the documentation, especially regarding settings such as max_features and random_state to ensure you understand what is being done in the background!

### 2.  Decision trees output ###

# Let's visualize the predictions of this model.  Since this tree is currently only using one feature (RM), we can plot that on the horizontal axis, and plot the actual and predicted house prices on the vertical axis.

# Make the plot
plt.scatter(X_df['RM'], y, color='k', label='Actual')
plt.scatter(X_df['RM'], y_pred, color='r', label='Predicted')
plt.title('Decision Tree Regression with max_depth=1')
plt.xlabel('RM')
plt.ylabel('Median House Price')
plt.legend(bbox_to_anchor=[1,0.5],loc='center left')
plt.show()

# Note that although we said we are doing regression, the goal of which is to predict a continuous output, we are only predicting 2 different values!  In general, a decision tree can only produce, **at most**, $2^d$ distinct values, where $d$ is the maximum depth of the tree.  This is because each additional step in the algorithm can only, at most, double the number of nodes in the tree.  I say "at most" because sometimes there is no way to meaningfully split a node (for example, if the MSE of a node was 0, or if a node contains only a single sample, then there would be no meaningful way to split the node any further).
#
# Nonetheless, because decision trees are not based on a particular equation (like a line or a polynomial curve, etc.), they can capture a tremendous amount of complexity.
#
# Let's go one step further with this tree, for a max_depth of 2.  This means we will have, at most, $2^2=4$ leaves.

dt = DTR(max_depth=2, random_state=146)
dt.fit(X_df,y)
tree.plot_tree(dt,feature_names=X_names);

# At this point, the tree is clearly still picking out the number of rooms as the most important contributor to the price of a house, but it is now also including LSTAT in one of the nodes.
#
# To visualize how the model is performing - well, I don't really like 3-d scatter plots - but we can compare the actual and predicted prices:

y_pred = dt.predict(X_df)

# Make the plot
plt.scatter(y, y_pred)
# Add a reference line
plt.plot([min(y),max(y)],[min(y),max(y)], ':k')
plt.title('Decision Tree Regression with max_depth=2')
plt.xlabel('Actual Median House Price')
plt.ylabel('Predicted Median House Price')
plt.show()

# Let's see how our MSE has changed from the previous model:

mse_depth2 = np.mean((y-y_pred)**2)
[mse2, mse_depth2]

# You might also check the $R^2$ value in the usual fashion:

dt.score(X_df,y)

# Let's do a quick K-fold validation with the current model:

dt = DTR(max_depth=2)
k=20
train_scores,test_scores,train_mse,test_mse = DoKFold(dt,X,y,k)
[np.mean(train_scores), np.mean(test_scores)]

# Looks a bit overfit.  Let's see what happens if we let the algorithm run without specifying any stopping criteria (that is, nodes will continue to be split until they cannot be split any further):

dt = DTR()
train_scores,test_scores,train_mse,test_mse= DoKFold(dt,X,y,k)
[np.mean(train_scores), np.mean(test_scores)]


# At this point, you should expect a pretty overfit model.  That's because the tree looks something like this:

dt = DTR()
dt.fit(X_df,y)
tree.plot_tree(dt,feature_names=X_names)
plt.savefig('BostonDTRegFullTree.svg',bbox_inches='tight')


# I doubt you'll be able to zoom in close enough to actually read the tree, but all of the leaves at this point are going to have an MSE of 0.  Additionally, they'll probably also contain only a single sample (unless some samples had the exact same target value).
#
# Although the model is overfit, let's split the data into training and testing sets, and take a look at the predictions on the test data only.

from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,ytrain,ytest = tts(X,y,test_size=0.5,random_state=146)

dt = DTR(random_state=146)
dt.fit(Xtrain,ytrain)
y_pred = dt.predict(Xtest)

print(dt.score(Xtest,ytest))

plt.scatter(ytest,y_pred)
plt.plot([min(ytest),max(ytest)],[min(ytest),max(ytest)], ':k')
plt.xlabel('Test Data')
plt.ylabel('Predicted')
plt.show()

# Although this decision tree is overfit - it is actually doing better on the test data (in this one example) than our previous not-as-overfit linear models (OLS, Ridge, Lasso, which were in the $R^2 \approx 0.7$ range.)

### 3. Tweaking for improved performance ###

# Decision trees have a lot of hyperparameters, all of which control some aspect of how the nodes are split, or when to stop splitting.  Here's a few, along with the names used by sklearn:
#
# - The maximum depth (max_depth)
# - The minimum number of data points a node must contain in order to be split (min_samples_split)
# - The minimum number of data points each leaf must contain (min_samples_leaf)
#
# Getting good performance out of a decision tree will often involve some experimentation with these values, just like we had to do with the regularization methods when finding optimal values for $\alpha$.
#
# Let's see if we can narrow in on a good value for max_depth:

max_max_depth = 20
d_range = np.arange(1,max_max_depth+1)

train = []
test = []
train_mse = []
test_mse = []

for d in d_range:
    dt = DTR(max_depth=d, random_state=146)
    tr,te,tr_mse,te_mse = DoKFold(dt,X,y,k)
    train.append(np.mean(tr))
    test.append(np.mean(te))
    train_mse.append(np.mean(tr_mse))
    test_mse.append(np.mean(te_mse))

idx = np.argmax(test)
print([d_range[idx], train[idx], test[idx]])

plt.plot(d_range, train, '-xk', label='Train')
plt.plot(d_range,test, '-xr',label='Test')
plt.xlabel('Max Depth')
plt.ylabel('Avg. $R^2$')
plt.title('K-fold validation with k = ' + str(k))
plt.show()

idx = np.argmin(test_mse)
print([d_range[idx], train_mse[idx], test_mse[idx]])

plt.plot(d_range, train_mse, '-xk', label='Train')
plt.plot(d_range,test_mse, '-xr',label='Test')
plt.xlabel('Max Depth')
plt.ylabel('Avg. MSE')
plt.title('K-fold validation with k = ' + str(k))
plt.show()

# You can see that the test scores don't just steadily increase or decrease, they tend to fluctuate a bit.  In this case, the general advice is to go with the simpler model.  Even if we got slightly better performance for a larger value, I'd stick with the max depth of 7.  Basically, I'm looking for where the test score is as good as possible, while also being as close as possible the training score.  You might remember this, roughly speaking, as where the curve for the test score starts to "level off".
#
# Let's refit the model to the entire data set, take a look at the tree, and plot the actual vs. predicted values.

dt = DTR(max_depth=7, random_state=146)
dt.fit(X_df,y)
tree.plot_tree(dt,feature_names=X_names);
plt.savefig('BostonDTReg_d7.svg',bbox_inches='tight')

y_pred = dt.predict(X_df)

plt.scatter(y,y_pred)
plt.plot([min(y),max(y)], [min(y), max(y)], ':k')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# Moral of the story here:
# - The maximum depth was 7, so the model can predict up to $2^7 = 128$ values.  Even though this isn't a continuous range, it certainly looks close in the plot.
# - The tree itself is hard to interpret.  Although we have what is essentially a flowchart telling us how the predictions are made, it's not the kind of thing you'll be able to keep in your head or make much sense out of (even if we could print it very large and hang it on the wall)
# - We did end up with what looks like an overfit model... but it's not _too_ bad.
#
# ___
#
# Now, keep in mind that we only varied one of the model's hyperparameters.  Let's do a grid search now, by varying both the max depth as well as the minimum number of samples required to split a node.  Since the hyperparameters here are integers, doing a grid search won't be quite as bad (slow) as in the previous models where the hyperparameters were continuous.

s_range = np.arange(2,11)
d_range = np.arange(2,11)

best_test_score = -np.inf
for s in s_range:
    for d in d_range:
        dt = DTR(max_depth=d,min_samples_split=s,random_state=146)
        tr,te,_,_ = DoKFold(dt,X,y,k)
        if np.mean(te)>best_test_score:
            best_test_score = np.mean(te)
            best_s = s
            best_d = d
[best_test_score, best_s, best_d]

# Notice how the best depth here is not the same as before.  In this case, using a max_depth=5 and min_samples_split=4 gave us slightly better results.
#
# OK! Next, let's do one additional training / testing split with these settings, and compare the results on the test data.

Xtrain,Xtest,ytrain,ytest = tts(X,y,test_size=0.5, random_state=146)
dt = DTR(max_depth = 5, min_samples_split = 4, random_state=146)

dt.fit(Xtrain,ytrain)
y_pred = dt.predict(Xtest)

plt.scatter(ytest,y_pred)
plt.plot([min(ytest), max(ytest)], [min(ytest), max(ytest)], ':k')
plt.xlabel('Test Data')
plt.ylabel('Predicted')
plt.show()
[dt.score(Xtrain,ytrain), dt.score(Xtest,ytest)]

# Finally, let's compare training and testing scores for an entire K-fold validation, using these same settings:

tr,te,_,_ = DoKFold(dt,X,y,k)

plt.scatter(tr,te,alpha=0.5,ec='k')
plt.xlabel('Training ($R^2$)')
plt.ylabel('Testing ($R^2$)')
plt.axis('equal')
plt.show()

# Overfit, yes - but we are still achieving a higher external validity (on the test data) than the linear models!  All in all, I think that this is not a very bad model for this data (and we may be able to improve it a bit further...)

### 4. Random Forest ###

# At this point, we probably don't need much of an introduction.  This is going to be very similar to random forest classification, where we will build many trees (each based on a bootstrapped version of the original data), and the predictions made by the model will be the average of the individual predictions made by each tree in the forest.

from sklearn.ensemble import RandomForestRegressor as RFR

n_trees = 10
rfr = RFR(random_state=146, n_estimators=n_trees)

rfr.fit(X,y)
rfr.score(X,y)

# We can take a look at the individual trees in the forest:

# Pick a tree, any tree!
idx = np.random.choice(n_trees)
t = rfr.estimators_[idx]
tree.plot_tree(t,feature_names=X_names);

# That tree looks like one of those "probably overfit" trees!
#
# Let's do some cross-validation though and see how the entire forest performs.

k=5
train,test,train_mse,test_mse = DoKFold(rfr,X,y,k)

[np.mean(train), np.mean(test)]


[np.mean(train_mse), np.mean(test_mse)]


# It is a bit overfit, but not too bad for our first try, using nothing but default settings.
#
# What happens if we add more trees?

n_trees = 100
rfr = RFR(random_state=146,n_estimators=n_trees)
train,test,train_mse,test_mse = DoKFold(rfr,X,y,k)

[np.mean(train), np.mean(test)]


[np.mean(train_mse), np.mean(test_mse)]

# And, of course, we could do a more methodical search:

n_trees_range = [10, 100, 500, 1000]

train = []
test = []
train_mse = []
test_mse = []
for n_trees in n_trees_range:
    rfr = RFR(random_state=146, n_estimators=n_trees)
    tr, te, tr_mse, te_mse = DoKFold(rfr, X, y, k)

    train.append(np.mean(tr))
    test.append(np.mean(te))
    train_mse.append(np.mean(tr_mse))
    test_mse.append(np.mean(te_mse))

#plt.plot(n_trees_range, train, '-xk', label='Train')
plt.plot(n_trees_range, test, '-xr', label='Test')
plt.xlabel('# of trees')
plt.ylabel('$R^2$')
plt.legend()
plt.show()

#plt.plot(n_trees_range, train_mse, '-xk', label='Train')
plt.plot(n_trees_range, test_mse, '-xr', label='Test')
plt.xlabel('# of trees')
plt.ylabel('MSE')
plt.legend()
plt.show()

# What if, instead of letting the trees all be fully expanded, we use the "best" values we found before for the individual decision tree?
#
# Those were, max_depth = 5, and min_samples_split=4

rfr = RFR(n_estimators=100, max_depth=5, min_samples_split=4,random_state=146)
train,test,train_mse,test_mse = DoKFold(rfr,X,y,k)

print([np.mean(train), np.mean(test)])
print([np.mean(train_mse), np.mean(test_mse)])


# And - while our performance has gotten a bit worse on both training and testing data, the model also looks a little less overfit than the 100 tree model with no additional stopping criteria.
#
# Of course, you would probably want to do a more systematic search, but this will take quite a bit longer. This time we will keep track of all of the results, not just the best ones.  This is a good thing to do, especially if the code might take awhile to run, so that you don't have to go back and recompute anything later.
#
# Additionally, we'll also keep track of how long this takes.  It's often a good idea to run smaller, quicker tests first, so you can estimate how long the "final version" might take to run.

from datetime import datetime as dt
tStart = dt.now()
z = 0
for i in range(1000):
    z+=1
print(dt.now()-tStart)

rng_trees = [10,100,200]
rng_depth = np.arange(2,11)
rng_samples = np.arange(2,11)
k = 5

results = []

# Keep track of how long this takes
tStart = dt.now()
for t in rng_trees:
    for d in rng_depth:
        for s in rng_samples:
            settings = [t,d,s]
            print(settings)
            rfr = RFR(random_state=146, n_estimators=t, max_depth = d, min_samples_split=s)
            tr,te,tr_mse,te_mse = DoKFold(rfr,X,y,k)
            results.append([*settings,tr,te,tr_mse,te_mse])
print(dt.now()-tStart)


# Next up, let's go find the settings that gave us the best results.  Have a look at one row of our results first:

results[0]


# Suppose we're looking for the settings that gave us the lowest average MSE on the test data.  We can determine this in just a few lines:

# The test MSEs are element 6 of each row
mean_test_mse = [np.mean(r[6]) for r in results]

mean_test_mse[0]

# Verify for one (or more) row of results
#results[0]
#results[0][6]
np.mean(results[0][6])

# Get the smallest value(s)
min_test_mse = min(mean_test_mse)
min_test_mse

idx = np.where(mean_test_mse == min_test_mse)
idx

best_results = results[idx[0][0]]
best_results

# Well, if you go back and compare with our previous tests, it looks like the model with 10 trees and no additional stopping criteria was still the best, at least in terms of minimum average MSE on the test data.  At least at this point we could feel a little more confident about that!
#
# Last but not least, let's visualize what the model is predicting vs. the actual values.
#
# To do this, I will fit one additional model using our "best" settings on a single training/testing split of the data.
#
# We'll also step up our plotting game a bit. For this example, I'll put two plots side by side, one for the testing data, and the other for the training data.

from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,ytrain,ytest = tts(X,y,test_size=0.5, random_state=146)

rfr = RFR(n_estimators=100, random_state=146)
rfr.fit(Xtrain,ytrain)
y_pred = rfr.predict(Xtest)
y_train_pred = rfr.predict(Xtrain)

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5),sharey=True)
ax1.scatter(ytest,y_pred,alpha=0.5,ec='k')
ax1.plot([min(ytest),max(ytest)], [min(ytest),max(ytest)], ':k')
ax1.set_xlabel('Test Data')
ax1.set_ylabel('Predicted')

ax2.scatter(ytrain,y_train_pred,alpha=0.5,ec='k')
ax2.plot([min(ytest),max(ytest)], [min(ytest),max(ytest)], ':k')
ax2.set_xlabel('Training Data')
# Really no need for the redundant label on the y axis
#ax2.set_ylabel('Predicted')

plt.show()

# Comparing the two plots here also gives you a visual sense of how the model is a bit overfit!

```
```
### import libraries ###
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
​
### import functions ###
​
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.datasets import make_blobs as mb
​
### write functions ###
​
def GetColors(N, map_name='rainbow'):
    cmap = matplotlib.cm.get_cmap(map_name)
    n = np.linspace(0, N, N) / N
    return cmap(n)
​
def PlotGroups(points, groups, colors, ec='black', ax='None'):
    if ax == 'None':
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
​
    for i in np.unique(groups):
        idx = (groups == i)
        ax.scatter(points[idx, 0], points[idx, 1],
                   color=colors[i], edgecolor=ec,
                   label='Group ' + str(i), alpha=0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(bbox_to_anchor=[1, 0.5], loc='center left')
    return fig, ax
​
def CompareClasses(actual, predicted, names=None):
    accuracy = sum(actual == predicted) / actual.shape[0]
    classes = pd.DataFrame(columns=['Actual', 'Predicted'])
    classes['Actual'] = actual
    classes['Predicted'] = predicted
    conf_mat = pd.crosstab(classes['Predicted'], classes['Actual'])
    if type(names) != type(None):
        conf_mat.index = y_names
        conf_mat.index.name = 'Predicted'
        conf_mat.columns = y_names
        conf_mat.columns.name = 'Actual'
    print('Accuracy = ' + format(accuracy, '.2f'))
    return conf_mat, accuracy
​
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel('Data Point')
    plt.ylabel('Distance')
​
### Part 1.  Intro to Agglomerative Heirarchical Clustering ###
​
# Heirarchical clustering refers to appraoches which gradually cluster the data.  In other words, they build a heirarchy of clusters! There's two main approaches to heirarchical clustering:
​
# 1. Agglomerative clustering: Build the clusters from the bottom-up.  In other words, we start with each data point being its own cluster, and then we succesively merge clusters that are "close" to each other.
#
# 2. Divisive clustersing: Build the clusters from the top-down.  In other words, start with all data points being part of a single cluster, then succesively split into smaller clusters.
​
# We're going to be looking at **Agglomerative Heirarchical Clustering (AHC)**.  Unlike $k$-means, with AHC we don't have to specify anything about an expected number of clusters, however we will need to consider what we mean for two clusters of points to be "close" - this concept is referred to as **linkage**.
​
# AHC works by:
#
# 1. Starting off with each data point in its own cluster
# 2. Beginning with a distance of zero, we gradually increase the distance, and when two clusters (which may only contain one point initially) fall within that distance of each other, they are merged into a single cluster.
# 3. What we mean by "distance between clusters" depends on the type of linkage being used.
# 4. This process continues until all the data has been merged into a single cluster.
#
# The results of this type of clustering are often summarized using **dendrograms**, which can then be inspected to determine the optimal number of clusters.  I've included a function above called plot_dendrogram that we can use to make these plots.  Let's do a quick example and look at the dendrogram:
​
X = np.array([[0,0],
              [0,0.1],
              [1,1],
              [1,1.5]])
plt.scatter(X[:,0], X[:,1])
plt.show()
​
ac = AC(n_clusters=None,distance_threshold=0)
ac.fit(X)
plot_dendrogram(ac)
plt.show()
​
# To determine the optimal number of clusters, you'll want to start at distance=0, and then go up until you find vertical lines that are much longer than the rest. This indicates where we had to start considering much longer distances between clusters before merging them - in other words - clusters that were separated by a much larger distance than the rest of the points.
#
# In the above example, the blue lines are quite a bit longer than the rest, and if you imagine drawing a horizontal line through them, you'd intersect the graph twice.  This is indicating that there are two clusters in the data.
#
# To get the cluster labels, you can refit the model by specifying the number of clusters:
​
n = 2
ac = AC(n_clusters=n)
clusters = ac.fit_predict(X)
​
colors = GetColors(n)
PlotGroups(X,clusters,colors)
plt.show()
​
​
​
np.random.seed(147)
n_groups = 4
n_pts = 100
data = mb(n_samples=n_pts, n_features=2, centers=n_groups)
X = data[0]
y = data[1]
​
ac = AC(n_clusters=None,distance_threshold=0)
ac.fit(X)
​
plt.figure(figsize=(18,6))
plot_dendrogram(ac)
plt.xticks([])
plt.show()
​
colors = GetColors(4)
PlotGroups(X,y,colors)
plt.show()
​
### Part 2.  Exploring different linkage types ###
​
np.random.seed(146)
n1 = 100
x11 = 6.3*np.random.random(size=(n1,1))
x12 = np.sin(x11)+np.random.normal(scale=0.1,size=(n1,1))
​
n2 = 150
r = 0.5; center = (5,0.5)
theta = 2 * np.pi * np.random.random(size=(n2,1))
x21 = r*np.cos(theta) + center[0] + np.random.normal(scale=0.1,size=(n2,1))
x22 = r*np.sin(theta) + center[1] + np.random.normal(scale=0.1,size=(n2,1))
​
x1 = np.concatenate([x11,x21],axis=0)
x2 = np.concatenate([x12,x22],axis=0)
​
X = np.concatenate([x1,x2],axis=1)
plt.scatter(x1,x2,ec='k',alpha=0.5)
plt.axis('equal')
plt.show()
​
ac = AC(n_clusters=None,distance_threshold=0)
ac.fit(X)
plot_dendrogram(ac)
plt.xticks([])
plt.show()
​
n = 2
ac = AC(n_clusters = n)
clusters = ac.fit_predict(X)
​
colors = GetColors(n)
PlotGroups(X,clusters,colors)
plt.axis('equal')
plt.show()
​
​
n = 2
ac = AC(n_clusters = n, linkage='single')
clusters = ac.fit_predict(X)
​
colors = GetColors(n)
PlotGroups(X,clusters,colors)
plt.axis('equal')
plt.show()
​
​
### Part 3. AHC - Additional Examples ###
​
from sklearn.datasets import load_breast_cancer as lbc
bc = lbc()
X = bc.data
y = bc.target
​
ac = AC(n_clusters=None, distance_threshold=0)
ac.fit(X)
​
plt.figure(figsize=(12,6))
plot_dendrogram(ac)
plt.xticks([])
plt.show()
​
n = 2
ac = AC(n_clusters=n)
clusters = ac.fit_predict(X)
​
from sklearn.manifold import TSNE
tsne = TSNE(random_state=146)
Xt = tsne.fit_transform(X)
​
colors = GetColors(n)
PlotGroups(Xt,clusters,colors)
plt.xlabel('$tSNE_1$')
plt.ylabel('$tSNE_2$')
plt.show()
​
from sklearn.preprocessing import StandardScaler as SS
ss = SS()
Xs = ss.fit_transform(X)
Xt = tsne.fit_transform(Xs)
​
ac = AC(n_clusters=None, distance_threshold=0)
ac.fit(Xs)
​
plot_dendrogram(ac)
plt.show()
​
n = 2
ac = AC(n_clusters=n)
clusters = ac.fit_predict(Xs)
​
colors = GetColors(n)
PlotGroups(Xt,clusters,colors)
plt.xlabel('$tSNE_1$')
plt.ylabel('$tSNE_2$')
plt.show()
​
CompareClasses(y,clusters)
```
```
# import libraries

import numpy as np
import matplotlib.pyplot as plt

# create functions

def DoKFold(model, X, y, k, random_state=146, scaler=None):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    train_scores = []
    test_scores = []
    train_mse = []
    test_mse = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain, :]
        Xtest = X[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]

        if scaler != None:
            Xtrain = scaler.fit_transform(Xtrain)
            Xtest = scaler.transform(Xtest)

        model.fit(Xtrain, ytrain)

        train_scores.append(model.score(Xtrain, ytrain))
        test_scores.append(model.score(Xtest, ytest))

        ytrain_pred = model.predict(Xtrain)
        ytest_pred = model.predict(Xtest)
        train_mse.append(np.mean((ytrain - ytrain_pred) ** 2))
        test_mse.append(np.mean((ytest - ytest_pred) ** 2))

    return train_scores, test_scores, train_mse, test_mse
def GetColors(N, map_name='rainbow'):
    import matplotlib
    import numpy as np
    cmap = matplotlib.cm.get_cmap(map_name)
    n = np.linspace(0, N, N) / N
    return cmap(n)
def PlotGroups(points, groups, colors, ec='black', ax='None'):
    import matplotlib.pyplot as plt

    if ax == 'None':
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    for i in np.unique(groups):
        idx = (groups == i)
        ax.scatter(points[idx, 0], points[idx, 1],
                   color=colors[i], edgecolor=ec,
                   label='Group ' + str(i), alpha=0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(bbox_to_anchor=[1, 0.5], loc='center left')
    return fig, ax
def CompareClasses(actual, predicted, names=None):
    import pandas as pd
    accuracy = sum(actual == predicted) / actual.shape[0]
    classes = pd.DataFrame(columns=['Actual', 'Predicted'])
    classes['Actual'] = actual
    classes['Predicted'] = predicted
    conf_mat = pd.crosstab(classes['Predicted'], classes['Actual'])

    if type(names) != type(None):
        conf_mat.index = y_names
        conf_mat.index.name = 'Predicted'
        conf_mat.columns = y_names
        conf_mat.columns.name = 'Actual'
    print('Accuracy = ' + format(accuracy, '.2f'))
    return conf_mat, accuracy
def MakeGrid(x_range, y_range):
    import numpy as np
    xx, yy = np.meshgrid(x_range, y_range)
    points = np.vstack([xx.ravel(), yy.ravel()]).T
    return points

### Part 0.  Some introductory comments about clustering ###

# The goal of clustering is very similar to classification - that is - we are trying to train the computer to assign labels to data points.  The main difference between clustering and classification is that **clustering refers to unsupervised learning** techniques.

# In unsupervised learning, the algorithm does not have any knowledge of actual labels or classifications of any of the data points.  Clustering is therefore often an exploratory approach, the goal of which is to identify groups or "clusters" within the data, i.e. sets of data points which are more similar to each other than they are to the rest of the data.

# In order to test the accuracy of our methods, in class we will be making use of datasets that do have labels. However, this information will not be provided to any of the models during training, we will be using it solely for the purposes of assessing our accuracy afterwards.  In practice, when you are doing clustering it will often be because this type of information is not available to you.

### Part 1.  Introduction to Clustering with DBSCAN ###

# **Density-based spatial clustering of applications with noise (DBSCAN)** is a type of clustering technique which is based on the idea that "clusters" in data are regions of high density surrounded by regions of low density.

# This idea is implemented by first definining two hyperparameters:

# $\epsilon$: The radius of a "neighborhood".  If two points lie within a distance $\epsilon$ of each other, then they are considered neighbors.
#
# $m$: The minimum number of neighbors a point must have (including itself) in order to be considered a **core point**.

# The algorithm is then defined as follows:

# 1. Any points that lie within a distance $\epsilon$ of a point $p$ are said to be "reachable" from $p$.
# 2. Any point $p$ with at least $m$ neighbors is considered a core point.
# 3. Clusters are defined as core points along with all points reachable from those cores.
# 4. Any points which are neither cores nor reachable from a core are labeled as noise, which simply means that they are not considered to be part of any cluster.

# The ability to label points as noise makes DBSCAN stand out against other methods we will see such as $k$-means and AHC, which will always assign every data point to a cluster.

# There are, of course, a couple downsides:
#
# 1. If the clusters have different densities, then DBSCAN might not perform well since the choice of $\epsilon$ and $m$ apply to the entire dataset.
# 2. Choosing the best values for these hyperparameters can also be difficult as it depends on the relative scale of the data.
#

# let's start with an example

np.random.seed(146)
n1 = 100
x11 = 6.3*np.random.random(size=(n1,1))
x12 = np.sin(x11)+np.random.normal(scale=0.1,size=(n1,1))

n2 = 150
r = 0.5; center = (5,0.5)
theta = 2 * np.pi * np.random.random(size=(n2,1))
x21 = r*np.cos(theta) + center[0] + np.random.normal(scale=0.1,size=(n2,1))
x22 = r*np.sin(theta) + center[1] + np.random.normal(scale=0.1,size=(n2,1))

x1 = np.concatenate([x11,x21],axis=0)
x2 = np.concatenate([x12,x22],axis=0)

X = np.concatenate([x1,x2],axis=1)
plt.scatter(x1,x2,ec='k',alpha=0.5)
plt.axis('equal')
plt.show()

from sklearn.cluster import DBSCAN
e = 1
m = 4

db = DBSCAN(eps=e, min_samples=m)
clusters = db.fit_predict(X)

n = len(np.unique(clusters))
colors = GetColors(n)
PlotGroups(X, clusters, colors)
plt.axis('equal')
plt.show()

### Part 2.  Some examples ###

from sklearn.datasets import load_wine
data = load_wine()
X = data.data
X_names = data.feature_names
y = data.target

import pandas as pd
Xdf = pd.DataFrame(X, columns=X_names)
Xdf.head(3)

# Next we'll apply DBSCAN to this dataset.  In order to visualize our results, we'll do a dimensionality reduction after the clustering is complete.

e = 1
m = 4

db = DBSCAN(eps=e, min_samples=m)
clusters = db.fit_predict(X)

from sklearn.decomposition import PCA
pca = PCA()
Xp = pca.fit_transform(X)

n = len(np.unique(clusters))
colors = GetColors(n)
PlotGroups(Xp[:, 0:2], clusters, colors)
plt.axis('equal')
plt.xlabel('$PC_1$')
plt.ylabel('$PC_2$')
plt.show()

# Our PCA plot is telling us something very important here.  See how all the data is basically all on a line?  This means that, in the original full-dimensional space, the data was basically all on a line!
#
# Based on this we should be able to reason that we are not going to find any meaningful clusters here, since the data basically just lies along one line!
#
# Do some feature scaling!

from sklearn.preprocessing import StandardScaler as SS
ss = SS()

Xs = ss.fit_transform(X)
Xp = pca.fit_transform(Xs)

clusters = db.fit_predict(Xs)

n = len(np.unique(clusters))
colors = GetColors(n)
PlotGroups(Xp[:, 0:2], clusters, colors)
plt.axis('equal')
plt.xlabel('$PC_1$')
plt.ylabel('$PC_2$')
plt.show()

# This is looking a little better, but if there are in fact clusters in this data, I'd like to see if I can find a better view of them.  I'd like to "see" them, before I continue with my clustering analysis!
#
# Keep in mind that we aren't "supposed" to know yet that there are three groups in our data.  So, we should have the mindset right now that there may or may not be any clusters to be found.  We are just doing some exploratory analysis right now!
#
# I'll try a tSNE transformation, just to see if anything pops out.

from sklearn.manifold import TSNE
tsne = TSNE(random_state=146)

Xt = tsne.fit_transform(Xs)

PlotGroups(Xt,clusters, colors)
plt.xlabel('$tSNE_1$')
plt.ylabel('$tSNE_2$')
plt.axis('equal')
plt.show()

# How neat! I feel like I see three somewhat distinct groups here.  Maybe it's something, maybe it isn't!
#
# Let's play around with those hyperparameters now and keep in mind - we are doing the clustering in the full-dimensional space, and then viewing the results in a lower dimensional representation.  So, it may or may not work out to be the case that our clustering will actually detect what appears to be three groups above.

e = 2
m = 4

db = DBSCAN(eps=e, min_samples=m)
clusters = db.fit_predict(Xs)

n = len(np.unique(clusters))
colors = GetColors(n)

PlotGroups(Xt,clusters, colors)
plt.xlabel('$tSNE_1$')
plt.ylabel('$tSNE_2$')
plt.axis('equal')
plt.show()

# What if we do the clustering on the version of the data that has had its dimensionality reduced?  I mean, I can pretty well _see_ the three clusters in the plot!

e = 2
m = 4

db = DBSCAN(eps=e, min_samples=m)
clusters = db.fit_predict(Xt)

n = len(np.unique(clusters))
colors = GetColors(n)

PlotGroups(Xt,clusters, colors)
plt.xlabel('$tSNE_1$')
plt.ylabel('$tSNE_2$')
plt.axis('equal')
plt.show()

# Now let's "cheat" and take a look at the actual labels

PlotGroups(Xt,y, colors)
plt.xlabel('$tSNE_1$')
plt.ylabel('$tSNE_2$')
plt.axis('equal')
plt.show()

# Finally, I'll check a confusion matrix.  Note that the labels here will not necessarily correspond to the actual labels in the data (e.g., Group 1 assigned by DBSCAN has nothing to do with the actual label y=1), so this sort of analysis may require you to reorganize/relabel the output of the model.

CompareClasses(y, clusters)

### Part 3.  The curse of dimensionality ###

# What we've seen in the last example is known as the **Curse of Dimensionality**.  Now, wouldn't you think that having more information is better?  How is it that reducing our data to a lower dimension can actually help us achieve better results?

# Well, especially when considering the distances between points, things get a little strange with high-dimensional data.

# 1. Suppose you measure the height of two people, and they are both 5.5 feet.  You can imagine this representation of those two people as a point on a number line.
# 2. With this very simple representation, these two people appear to be identical - the distance between the points representing them is 0.
# 3. Now add another variable, perhaps their age.  Suppose one person is 19 and the other person is 20.  You can imagine representing this data as points in a 2-d plane.
# 4.  At this point, the people look a little different - the distance between the points that represent them is now 1.
# 5. Now add 100 more variables.  What did they eat for lunch yesterday? How long is their hair? And so on, and so on.
# 6. It's hard (impossible!) to imagine data living in a 100 dimensional space, but you can see from the way this example has gone so far, that the points representing these people in a 100 dimensional space is going to be quite large.

# As you measure more and more variables, the "volume" of the space needed to represent them grows exponentially.  Unless you are also collecting exponentially more data every time you measure a new variable, your data is going to become very sparse, relative to the space it is represented in.  This can render measurements such as the distance between points almost useless, which can in turn hinder the performance of models that use distances to derive conclusions about the data. Thus, reducing the dimensionality of your data can help improve the performance of techniques like this!

from sklearn.datasets import load_breast_cancer as lbc
bc = lbc()
X = bc.data
X_names = bc.feature_names
y = bc.target
y_names = bc.target_names

Xs = ss.fit_transform(X)
Xt = tsne.fit_transform(Xs)

e = 5
m = 15

db = DBSCAN(eps=e, min_samples=m)
clusters = db.fit_predict(Xt)

n = len(np.unique(clusters))
colors = GetColors(n)

PlotGroups(Xt,clusters, colors)
plt.xlabel('$tSNE_1$')
plt.ylabel('$tSNE_2$')
plt.axis('equal')
plt.show()

CompareClasses(y,clusters)

# After all this manual testing, you might be wondering if there is some way to automate the search for the hyperparamters.  The answer is usually yes, so long as you can devise some meaningful way to "score" the model.  We obviously can't use the actual classifications or classification accuracy (we are doing _unsupervised_ learning after all), but I'll propose that we look at how the average within-cluster distance is changing as we vary $\epsilon$ (I won't worry about $m$ for now).
#
# Basically, if there was some point where we went from having some well-separated clusters to the data suddenly merging into larger clusters, this quantity should suddenly increase.  On the other hand, if the only thing that is changing is that the clusters are slowly gaining new points, then this quantity should only gradually increase.

from sklearn.metrics import pairwise_distances as pdist

dist_matrix = pdist(Xt)
print(dist_matrix)

print(np.triu(dist_matrix))

idx = np.triu_indices_from(dist_matrix,k=1)
dist_matrix[idx]

# You may want to verify that the number of distances we are computing is correct.
#
# If there are $N$ data points, then the number of ways to select $k$ of them is:
#
# $$C(N,k) = \frac{N!}{(N-k)!k!}$$
#
# (here, we are interested in the case where $k=2$, that is, how many pairs of points are there in the data)

dist_matrix[idx].shape

Xt.shape

from math import comb

comb(Xt.shape[0], 2)

def WithinClusterDistances(points, labels):
    dist = []
    for i in np.unique(labels):
        # Ignore points labeled as noise
        if i != -1:
            idx = (labels == i)
            cur_points = points[idx, :]
            dist_matrix = pdist(cur_points)
            dist_list = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            dist.append(dist_list)

    # Determine the number of distances computed
    n_pairs = sum([len(row) for row in dist])

    # Compute the weighted average
    avg_dist = 0
    for row in dist:
        avg_dist += sum(row) * len(row) / n_pairs

    return avg_dist

e_range = np.linspace(1,6,100)
m = 15

adist = []

for e in e_range:
    d = DBSCAN(eps = e, min_samples=m)
    clusters = d.fit_predict(Xt)
    adist.append(WithinClusterDistances(Xt,clusters))

plt.figure(figsize=(12,6))
plt.plot(e_range, adist, ':xk')
plt.xlabel('$\\epsilon$')
plt.ylabel('Avg Within Cluster Distance')
plt.show()

# That big jump is what we are interested in.  As we increased $\epsilon$ past that point, something about our answers drastically changed.  This is most likely due to several clusters all of a sudden being merged into a larger cluster.  Therefore, we are probably going to want to take the value of $\epsilon$ from just before that jump.

dist_diff = [adist[i+1] - adist[i] for i in range(len(adist)-1)]
idx_max_jump = np.argmax(dist_diff)
[e_range[idx_max_jump], dist_diff[idx_max_jump]]

plt.figure(figsize=(12,6))
plt.plot(e_range[1:], dist_diff, ':xk')
plt.xlabel('$\\epsilon$')
plt.ylabel('$\\Delta$ Avg Within Cluster Distance')
plt.show()

d = DBSCAN(min_samples=m,eps = e_range[idx_max_jump])
clusters = d.fit_predict(Xt)

CompareClasses(y,clusters)

# Now, let's say we're somewhat confident in these results and we want to take things one step farther.  Perhaps we want to train some type of classifier.  I can use the clusters predicted from the clustering as the input to something like, say KNN.
#
# Let's see how such a model might perform on this data.

from sklearn.neighbors import KNeighborsClassifier as KNN

rng = np.linspace(-30,40,75)
points = MakeGrid(rng,rng)

knn = KNN(n_neighbors = 10)

knn.fit(Xt, clusters)
regions = knn.predict(points)

front_colors = ['red', 'blue', 'yellow']
back_colors = ['magenta', 'cyan', 'orange']

fig,ax = PlotGroups(points,regions,back_colors, ec=None)
PlotGroups(Xt, clusters, front_colors, ax=ax)
ax.get_legend().remove()
plt.xlabel('$tSNE_1$')
plt.ylabel('$tSNE_2$')
plt.show()

# Next up, we'll "cheat" again, and run a training/testing split on this data.
#
# We'll train it using the DBSCAN labels - and we'll "cheat" by scoring the model on the actual labels for the test data.  Sort of a "what if we actually did this" scenario.

from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,ytrain,ytest = tts(Xt,y,test_size=0.41, random_state=146)

# Fit the DBSCAN model using the training data
clusters = d.fit_predict(Xtrain)

# Train the KNN model using these clusters
knn = KNN(n_neighbors=10)
knn.fit(Xtrain,clusters)

y_pred = knn.predict(Xtest)

# Then sometime in the future, the true answers became known to us
CompareClasses(ytest, y_pred)
```
