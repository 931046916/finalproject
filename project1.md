# 1.	
Package is a directory that contains a ```__init__.py``` file which may include multiple objects; it can be downloaded and installed by installer. Libraries are bundle of codes that have specific functionalities and can be reused later in a project; libraries are naturally installed in python and don’t need to be downloaded. Click on Python Interpreter in PyCharm preference and click on the “+” sign to install package. ```import pandas as pd```. Using abbreviation is more convenient when conducting later coding. ```from datetime import datetime``` would import datetime function from datetime library.

#2
Data frame is a table with labeled columns of different data. “Pandas” library is most useful with data frames. Use the ```read_() ``` command from pandas library to specify the library and path to file to locate a file in the operating system. Since data is stored in different types, we need to specify which type of data we re reading (html, csv, etc.). To inform pandas that we are using a tab-separated file instead of a default comma-separated file, use the command ```new_object = library_name.function_name(path_to_file_object, sep = '\t')```
Use the command ```data.describe()``` to decribe the data.
For "gapminder.csv" file, there are 6 columns and 1704 rows. 
To return a list of just the column names, use ```list(data.columns)```, or the``` .describe()``` command.

#3
The year variable has a regular 5 year interval. Data from year 2012 and 2017 should be added to make it more current. A total of 284 data should be added.

#4
The country with lowest life expectancy is Rwanda with a life expectancy of 23.599 in 1992. This is because of the genocide activities against the Tutsis in the 1990s which killed up to a million and left Rwanda one of the poorest and sickest countries in the world at that time.

#5
In 2007, the GDP of these fou countries are:
| country | continent |      gdp     |
| ------- | :-------: | :----------- |
| Germany |  Europe   | 2.650871e+12 |
| France  |  Europe   | 1.861228e+12 |
| Italy   |  Europe   | 1.661264e+12 |
| Spain   |  Europe   | 1.165760e+12 |
In 2002, the GDP of these fou countries are:
| country | continent |      gdp     |
| ------- | :-------: | :----------- |
| Germany |  Europe   | 2.473468e+12 |
| France  |  Europe   | 1.733394e+12 |
| Italy   |  Europe   | 1.620108e+12 |
| Spain   |  Europe   | 9.972067e+11 |
Spain has greatest increase in total GDP: 1.77403e+11

#6
"==" indicates that the statement of one variable equalling to a certain value is true. For example, ```data['continent']=='Asia'``` means this is the data subset of the continent Asia. "&" is the "and" symbol that is used when both statements are true in a given programming context. For example, when looking for the latest data of from a Asia,  use the command ```data_asia = data[(data['continent']=='Asia') & (data['year'] == data['year'].max())]```
The "|" symbol is the "or" operator that would return true when at least one of the statements is correct. Foe example, this would return true: ```(1 + 1 == 2) | (1 + 2 > 3)```. " ^ " is the exclusive or symbol. The computer would only return true if only one statement is true while the other one is flase. It would return false if both statement are true or both are false. For example, ```(1 + 1 == 2) ^ ("cat" == "dog")```would return true but ```(1 + 1 > 2) ^ ("cat" == "dog")``` would return false. 

#7

