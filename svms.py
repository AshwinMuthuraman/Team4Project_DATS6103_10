# %%
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
# %%
df = pd.read_csv("./dataset-of-00s.csv")
df = df.drop(['track', 'artist', 'uri'], axis=1)
# %%
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for k in ["linear", "sigmoid", "rbf", "poly"]:
    svm_model = SVC(kernel=k, C=1.0)
    svm_model.fit(X_train_scaled, y_train)

    y_train_predictions = svm_model.predict(X_train_scaled)
    y_test_predictions = svm_model.predict(X_test_scaled)

    print("Evaluation metrics for SVM with kernel", k)
    print("train accuracy", accuracy_score(y_train, y_train_predictions), end = ", ")
    print("test accuracy", accuracy_score(y_test, y_test_predictions))
    print("classification_report", classification_report(y_test, y_test_predictions))
    
    confusion_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_test_predictions)
    plt.subplots(figsize=(4, 4))
    sns.heatmap(confusion_matrix, annot = True, fmt = "d")
    
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.title("confusion Matrix")
    plt.show()
    
    # print("confusion matrix", confusion_matrix(y_test, y_test_predictions))
    # print("classification_report", classification_report(y_test, y_test_predictions))
    print("=======================================================")
# %%
param_grid = {
    'C': [0.1, 1, 10,],
    'kernel': ['rbf'],
    'gamma': [0.1, 0.5, 1]
}

svm_model = SVC()
grid_search = GridSearchCV(svm_model, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
y_pred = grid_search.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("best hyperparameters:", best_params)
print("Accuracy: ", accuracy)