# Codes for Question 1: 
import pandas as pd

path_to_data = 'gapminder.tsv'
data = pd.read_csv(path_to_data, sep = '\t')

print(data['year'])
print(data['year'].unique())
print(len(data['year'].unique()))

# Codes for Question 2:
data_pop = data['pop'].max()
print(data_pop)
idx_pop = data['pop'] == data_pop
data_pop = data[idx_pop]
print(data_pop)

# Codes for Question 3:
idx_europe = data['continent'] == 'Europe'
data_europe = data[idx_europe]
print(data_europe)

def GetMinPopInEurope(data_europe, year):
    idxYear = data_europe['year'] == year
    temp = data_europe[idxYear]
    pop = temp['pop'].min()
    idxPop = temp['pop'] == pop
    return temp[idxPop]

MinPop_europe = GetMinPopInEurope(data_europe, 1977)
print(MinPop_europe)

idx_iceland = data_europe['country'] == 'Iceland'
data_iceland = data_europe[idx_iceland]
print(data_iceland)

def GetPopInIceland(data_iceland, year):
    idxYear = data_iceland['year'] == year
    temp = data_iceland[idxYear]
    pop = temp['pop']
    idxPop = temp['pop'] == pop
    return temp[idxPop]

IcelandPop = GetPopInIceland(data_iceland, 2007)
print(IcelandPop)
