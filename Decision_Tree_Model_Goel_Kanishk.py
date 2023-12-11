## DECISION TREES by Goel_Kanishk##

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
