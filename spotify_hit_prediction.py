import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset-of-00s.csv')

# Verifying if the column names adhere to the correct formatting.
print(df.columns)

# Show fundamental details about the dataset.
print(df.info())

# Summary statistics of the columns
print(df.describe())

# Check for missing values in the dataset
print(df.isnull().sum())

# shape
print(f'The shape of the dataset is {df.shape}')

# data types of columns
print(df.dtypes)
