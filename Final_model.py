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


## DECISION TREES ##

# %%
# Importing all required libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree # for visualization
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

#%%
# Function to print metrics
def print_metric(y_pred, y_test):
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("The accuracy score for the model is: {}".format(round(acc, 2)))
    print("The precision for the model is: {}".format(round(pre,2)))
    print("The f1 score for the model is: {}".format(round(f1,2)))

    return acc, pre, f1

# %%
# Define dependent & independent features
dep_feature = df.iloc[:,-1]
indep_features = df.iloc[:,0:15]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(indep_features, dep_feature, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

#%%
# Generic model 
model0 = DecisionTreeClassifier()
model0.fit(X_train, y_train)
y_pred=model0.predict(X_test)
print("The maximum depth of decision tree is {}".format(model0.tree_.max_depth))
model0_acc, model0_pre, model0_f1 = print_metric(y_pred, y_test)
tree.plot_tree(model0)


#%%
# Using max depth
hyperparameter_depth = 0
diff=5

for d in range(1, 23):
    model1 = DecisionTreeClassifier(max_depth= d, random_state=42)
    model1.fit(X_train, y_train)
    tr = model1.score(X_train, y_train)
    val = model1.score(X_val, y_val)
    if tr-val < diff:
        diff = tr-val
        hyperparameter_depth=d

print("The depth with maximum accuracy is {}".format(hyperparameter_depth))

model1 = DecisionTreeClassifier(max_depth=hyperparameter_depth, random_state=42)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
model1_acc, model1_pre, model1_f1 = print_metric(y_pred, y_test)

# Printing tree with depth = 3
plt.figure(figsize=(80,80))
tree.plot_tree(model1, feature_names=X_train.columns, filled=True)

# Print confusion matrix
cnf=confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cnf, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=["Predicted 0", "Predicted 1"], 
            yticklabels=["Actual 0", "Actual 1"])
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
avg_precision = average_precision_score(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall Curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()


# %%
# We can do this using any of these hyperparameters:
# max_leaf_nodes
# max_features
# min_sample_split
# min_sample_leaf
# min_impurity_decrease  
# bootstrap


# %%
# Random Forest
model2 = RandomForestClassifier(n_estimators=50, n_jobs=-1)
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)
model2_acc, model2_pre, model2_f1 = print_metric(y_test, y_pred)

plt.figure(figsize=(80,80))
tree.plot_tree(model2, feature_names=X_train.columns, filled=True)

cnf=confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cnf, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=["Predicted 0", "Predicted 1"], 
            yticklabels=["Actual 0", "Actual 1"])
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
avg_precision = average_precision_score(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall Curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()


# %%
# Hyperparameter tuning using random search cross validation and random forests

param={
    'n_estimators':[30,60,90],
    'criterion':['gini','entropy','log_loss'],
    'max_depth':[1,2,3,4,5],
    'max_features':['sqrt','log2'],
    'min_samples_split':[50,60,70,80,90,100],
    'max_leaf_nodes':[40,50,60,70,80,90,100]
}

grid = RandomizedSearchCV(n_iter=500, estimator=RandomForestClassifier(random_state=42), param_distributions=param, scoring='accuracy', n_jobs=-1, cv=5)
grid.fit(X_train, y_train)

print("The best params chosen are {}".fomrat(grid.best_params_))
print("The best score is {}".fomrat(grid.best_score_))

# %%
# Best model so far with best_params
best_params=grid.best_params_
final_model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
    criterion=best_params['criterion'],
    max_depth=best_params['max_depth'],
    max_features=best_params['max_features'],
    min_samples_split=best_params['min_samples_split'],
    max_leaf_nodes=best_params['max_leaf_nodes'],
    random_state=42)
final_model.fit(X_train, y_train)
y_pred=final_model.predict(X_test)

final_model_acc, final_model_pre, final_model_f1 = print_metric(y_test, y_pred)
cnf=confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cnf, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=["Predicted 0", "Predicted 1"], 
            yticklabels=["Actual 0", "Actual 1"])
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
avg_precision = average_precision_score(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall Curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()

