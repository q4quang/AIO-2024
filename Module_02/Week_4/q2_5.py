import pandas as pd

data = pd.read_csv("Module_02/Week_4/advertising.csv")

def correlation(x, y):
    return round(x.corr(y), 2)

# Example usage:
x = data['TV']
y = data['Radio']
corr_xy = correlation(x, y)
print(f"Correlation between TV and Sales: {round(corr_xy, 2)}")