# Decision Tree Model by Kanishk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from  sklearn import tree # for visualization
import warnings
warnings.filterwarnings('ignore')


#Define dependent & independent features
dep_feature = df.iloc[:,-1]
indep_features = df.iloc[:,0:15]

#Train test split
X_train, X_test, y_train, y_test = train_test_split(indep_features, dep_feature, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
model0 = DecisionTreeClassifier()
model0.fit(X_train, y_train)
print("The max depth of decision tree is {}".format(model0.tree_.max_depth))


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

                           

