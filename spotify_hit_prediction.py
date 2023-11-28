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

#Checking for duplicate elements

def extract_id(text):
    return text.split(':')[-1]  
df['uri'] = df['uri'].apply(extract_id)

print(df['uri'].nunique(),) 
print(df['uri'].value_counts())
print(df['uri'].value_counts().unique())

duplicate_values = df['uri'].value_counts()==2
duplicate_index = duplicate_values[duplicate_values]
print(duplicate_index.value_counts,  duplicate_index.shape) 

duplicate_index  = duplicate_index.index
duplicate_index = duplicate_index.tolist()
print(duplicate_index)

remove_duplicate_index = df[df.duplicated(subset='uri', keep=False)].index.tolist() 

# cleaned dataset after removing duplicate recors
df.drop(remove_duplicate_index,axis=0,inplace=True)
print(df.shape)

#removing the columns which we dont need
df = df.drop(['track', 'artist', 'uri'], axis=1)

# understanding the ratio of target values to have no bias in the classifier
print(df.target.value_counts(normalize=True))
colors = ["#3498db", "#e74c3c"]  

sns.countplot(x='target', data=df, hue='target', palette=colors, legend=False)
plt.show()

# understanding the distribution of all features
fig, ax = plt.subplots(5, 3, figsize=(20, 20))

def hist_plot(axis, data, variable, binsnum=10, color='r'):
    axis.hist(data[variable], bins=binsnum, color=color)
    axis.set_title(f'{variable.capitalize()} Histogram')

column_features = df.columns.tolist() 
column_features = column_features[:min(len(column_features), 15)]  

for i, j in enumerate(column_features):
    row = i // 3  
    col = i % 3   
    hist_plot(ax[row, col], df, j)

if len(column_features) < 15:
    for i in range(len(column_features), 15):
        row = i // 3
        col = i % 3
        ax[row, col].axis('off')

plt.tight_layout()
plt.show()
