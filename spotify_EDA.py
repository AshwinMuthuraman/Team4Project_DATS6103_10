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

#Checking for duplicate elementa

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

#Univariate analysis
# understanding stats of features between hits and flops
hit_songs = df.drop('target', axis=1).loc[df['target'] == 1]
flop_songs = df.drop('target', axis=1).loc[df['target'] == 0]

mean_of_hits = pd.DataFrame(hit_songs.describe().loc['mean'])
mean_of_flops = pd.DataFrame(flop_songs.describe().loc['mean'])

combined_means = pd.concat([mean_of_hits,mean_of_flops, (mean_of_hits-mean_of_flops)], axis = 1)
combined_means.columns = ['mean_of_hits', 'mean_of_flops', 'difference_of_means']
print(combined_means)

# F-test to understand strong features against target
from sklearn.feature_selection import f_classif

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

f_stat, p_value = f_classif(X, y)

feature_list = df.iloc[:, :-1].columns.tolist()

df_stats = pd.DataFrame({
    'features': feature_list,
    'f_stat': f_stat,
    'p_value': p_value
})

df_stats_sorted = df_stats.sort_values(by='p_value')

print(df_stats_sorted)

# Understanding the range distribution of the strong features 
features = ['danceability','loudness', 'valence','acousticness','instrumentalness']
fig, axes = plt.subplots(nrows=1, ncols=len(features), figsize=(15, 5))

for i, column in enumerate(features):
    sns.boxplot(x='target', y=column, data=df, ax=axes[i])
    axes[i].set_title(f'{column.capitalize()} vs Target')

plt.tight_layout()
plt.show()

#Removing negative outliers values in loudness
loudness_outliers = df[df['loudness']>0].index
print(loudness_outliers)

df.drop(loudness_outliers,axis=0, inplace=True)
print(df.shape)

#Bivariate analysis
#correlation
pearson_corr = df.corr(method='pearson')

plt.figure(figsize=(12, 10))
plt.title("Absolute Pearson's Correlation Coefficient")

sns.heatmap(
    pearson_corr.abs(),
    cmap="coolwarm",
    square=True,
    vmin=0,
    vmax=1,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 10},
    linewidths=0.5,
    linecolor='black',
    cbar_kws={"shrink": 0.8}
)

plt.xlabel("Features")
plt.ylabel("Features")
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

#trying to understand the corrleation between danceability and energy
plt.figure(figsize=(8, 6))
sns.scatterplot(x='danceability', y='energy', data=df, alpha=0.7, marker='o', color='blue')
plt.title('Energy vs Danceability')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.grid(True)
plt.show()

# understanding how mode and speechiness affect the popularity of a song

plt.figure(figsize=(10, 6))
custom_palette = "Set2"  # Change the palette to suit your preference

sns.violinplot(x='mode', y='speechiness', hue='target', data=df, palette=custom_palette)
plt.title('Speechiness across Modes by Target')
plt.xlabel('Mode')
plt.ylabel('Speechiness')
plt.legend(title='Target', loc='upper right')
plt.grid(axis='y')
plt.show()

# Understanding how danceability affected the songs popularity
dance_hit = df[df['target'] ==1]['danceability'].mean()
print("The mean of danceability of songs that were hits", dance_hit)

# Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming 'df' is the cleaned dataset

# Splitting the data into features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating the logistic regression model
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)

# Making predictions on the test set
y_pred = logistic_model.predict(X_test_scaled)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
# Regression Ended


