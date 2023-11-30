# %%
# EDA code by Ashwin

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

sns.countplot(x='target', data=df, hue='target', palette=colors)
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



# %%
# Decision Tree Model by Kanishk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from  sklearn import tree # for visualization
import warnings
warnings.filterwarnings('ignore')


# %%
#Define dependent & independent features
dep_feature = df.iloc[:,-1]
indep_features = df.iloc[:,0:15]

#Train test split
X_train, X_test, y_train, y_test = train_test_split(indep_features, dep_feature, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
model0 = DecisionTreeClassifier()
model0.fit(X_train, y_train)
print("The max depth of decision tree is {}".format(model0.tree_.max_depth))

# %%
# Using max depth
hyperparameter_depth = 0
diff=5
#validation_score=list()
#training_score=list()
for d in range(1, 23):
    model1 = DecisionTreeClassifier(max_depth= d, random_state=42)
    model1.fit(X_train, y_train)
    tr = model1.score(X_train, y_train)
    val = model1.score(X_val, y_val)
    #training_score.append(tr)
    #validation_score.append(val)
    if tr-val < diff:
        diff = tr-val
        hyperparameter_depth=d

print(hyperparameter_depth)

final_model1 = DecisionTreeClassifier(max_depth=hyperparameter_depth, random_state=42)
final_model1.fit(X_train, y_train)

from sklearn import tree
plt.figure(figsize=(80,80))
tree.plot_tree(final_model1, feature_names=X_train.columns, filled=True)

from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = model1.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

# Using max leaf nodes
model2 = DecisionTreeClassifier(max_leaf_nodes=30, random_state=42)
model2.fit(X_train, y_train)
print(model2.score(X_train, y_train))
print(model2.score(X_val, y_val))
tree.plot_tree(model2)

from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=50, n_jobs=-1)
model3.fit(X_train, y_train)
print(model3.score(X_train, y_train))
print(model3.score(X_val, y_val))

# Using max_features
for mf in [None, 'log2', 'sqrt']:
    model4 = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_features=mf)
    model4.fit(X_train, y_train)
    print("\nFor max_features={}".format(mf))
    print(model4.score(X_train, y_train))
    print(model4.score(X_val, y_val))
    

# Using min_sample_split
# It holds the minimum sample required to split
for ms in (10,20,30,40,50,60,70,80,90,100):
    model5 = RandomForestClassifier(n_estimators=50, n_jobs=-1, min_samples_split=ms, bootstrap=False)
    model5.fit(X_train, y_train)
    print("\nFor min_sample={}".format(ms))
    print(model5.score(X_train, y_train))
    print(model5.score(X_val, y_val))


model6=DecisionTreeClassifier(random_state=42, min_samples_leaf=50)
model6.fit(X_train, y_train)
print(model6.score(X_train, y_train))
print(model6.score(X_val, y_val))

# Final model with all hyperparameters
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
final_model = RandomForestClassifier(random_state=42)
param={
    'n_estimators':[30,60,90],
    'criterion':['gini','entropy', 'log_loss'],
    'max_depth':[1,2,3,4,5],
    'max_features':[None,'sqrt','log2'],
    'min_samples_split':[50,60,70,80,90,100],
    'max_leaf_nodes':[40,50,60,70,80,90,100]
}

#Using grid_search_cv
#grid = GridSearchCV(estimator=final_model, param_grid=param, scoring=accuracy_score, n_jobs=-1, cv=5)
grid = RandomizedSearchCV(n_iter=500, estimator=final_model, param_distributions=param, scoring=accuracy_score, n_jobs=-1, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)

#n_estimators': 60, 'min_samples_split': 100, 'max_leaf_nodes': 90, 'max_features': 'log2', 'max_depth': 3, 'criterion': 'entropy'
m = RandomForestClassifier(n_estimators=60, min_samples_split=100, max_leaf_nodes=90, max_features='log2', max_depth=3, criterion='entropy')
m.fit(X_train, y_train)
y_pred=m.predict(X_test)
accuracy_score(y_test, y_pred)

                           

