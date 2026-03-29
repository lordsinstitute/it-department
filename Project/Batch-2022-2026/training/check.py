import pandas as pd

df = pd.read_csv("../dataset/labels.csv")
print("Columns:", df.columns)
print(df.head())