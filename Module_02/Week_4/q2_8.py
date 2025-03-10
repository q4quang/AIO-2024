import pandas as pd

data = pd.read_csv("Module_02/Week_4/advertising.csv")
print(data.corr())