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
# %%
best_model = grid_search.best_estimator_

X_test_scaled = scaler.transform(X_test)
y_pred = best_model.predict(X_test_scaled)

confusion_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
plt.subplots(figsize=(4, 4))
sns.heatmap(confusion_matrix, annot = True, fmt = "d")

plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("confusion Matrix")
plt.show()

# %%
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

svm_model = SVC(kernel="rbf", C=1, gamma=0.1)
svm_model.fit(X_train_pca, y_train)
y_train_predictions = svm_model.predict(X_train_pca)
y_test_predictions = svm_model.predict(X_test_pca)

print(f"Evaluation metrics for SVM with kernel {k} and PCA (n= 3 features)")
print("train accuracy", accuracy_score(y_train, y_train_predictions), end = ", ")
print("test accuracy", accuracy_score(y_test, y_test_predictions))

confusion_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_test_predictions)
plt.subplots(figsize=(4, 4))
sns.heatmap(confusion_matrix, annot = True, fmt = "d")

plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("confusion Matrix")
plt.show()
#%%
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['blue', 'red']
for label, color in zip(np.unique(y_train), colors):
    indices = np.where(y_train == label)
    ax.scatter(X_train_pca[indices, 0], X_train_pca[indices, 1], X_train_pca[indices, 2], c=color, label=str(label))

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA Plot')

plt.legend()  
plt.show()
#%%
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


def fit_and_visualize_svm(X, y, C):
    clf = svm.SVC(kernel="linear")
    clf.fit(X, y)
    y_pred = clf.predict(X_test_pca)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k', label='Training Points')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    
    # Plot decision boundary and margins
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'], label='Decision Boundary')
    
    # Highlight the support vectors
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'SVM with C={C}, Accuracy={accuracy_score(y_test, y_pred):.2f}')

    plt.legend()
    
    plt.show()

fit_and_visualize_svm(X_train_pca, y_train, C=0.1)  # weak margin
fit_and_visualize_svm(X_train_pca, y_train, C=1.0)  # Default margin
fit_and_visualize_svm(X_train_pca, y_train, C=10.0)  # Strong margin
